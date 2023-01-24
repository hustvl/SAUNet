import torch.nn as nn
import torch
import torch.nn.functional as F
from timm.models.layers import DropPath
import math
import warnings
from torch.utils.checkpoint import checkpoint


def A(x, Phi):
    temp = x * Phi
    y = torch.sum(temp, 1)
    return y


def At(y, Phi):
    temp = torch.unsqueeze(y, 1).repeat(1, Phi.shape[1], 1, 1)
    x = temp * Phi
    return x


def shift_3d(inputs, step=2):
    [_, nC, _, _] = inputs.shape
    for i in range(nC):
        inputs[:, i, :, :] = torch.roll(inputs[:, i, :, :], shifts=step * i, dims=2)
    return inputs


def shift_back_3d(inputs, step=2):
    [_, nC, _, _] = inputs.shape
    for i in range(nC):
        inputs[:, i, :, :] = torch.roll(inputs[:, i, :, :], shifts=(-1) * step * i, dims=2)
    return inputs


def _no_grad_trunc_normal_(tensor, mean, std, a, b):

    def norm_cdf(x):
        return (1. + math.erf(x / math.sqrt(2.))) / 2.

    if (mean < a - 2 * std) or (mean > b + 2 * std):
        warnings.warn(
            "mean is more than 2 std from [a, b] in nn.init.trunc_normal_. "
            "The distribution of values may be incorrect.",
            stacklevel=2)
    with torch.no_grad():
        l = norm_cdf((a - mean) / std)
        u = norm_cdf((b - mean) / std)
        tensor.uniform_(2 * l - 1, 2 * u - 1)
        tensor.erfinv_()
        tensor.mul_(std * math.sqrt(2.))
        tensor.add_(mean)
        tensor.clamp_(min=a, max=b)
        return tensor


def trunc_normal_(tensor, mean=0., std=1., a=-2., b=2.):
    # type: (Tensor, float, float, float, float) -> Tensor
    return _no_grad_trunc_normal_(tensor, mean, std, a, b)


class GELU(nn.Module):

    def forward(self, x):
        return F.gelu(x)


class LayerNorm(nn.Module):

    def __init__(self, normalized_shape, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.normalized_shape = (normalized_shape, )

    def forward(self, x):
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        x = self.weight[:, None, None] * x + self.bias[:, None, None]
        return x


class ConvReLU(nn.Sequential):

    def __init__(self, in_planes, out_planes, kernel_size=3, stride=1, groups=1, act_layer=None):
        padding = (kernel_size - 1) // 2
        if act_layer is None:
            act_layer = nn.Hardswish(inplace=True)
        super(ConvReLU,
              self).__init__(nn.Conv2d(in_planes, out_planes, kernel_size, stride, padding, groups=groups, bias=False),
                             act_layer)


class PreNorm(nn.Module):

    def __init__(self, dim, fn):
        super(PreNorm, self).__init__()
        self.fn = fn
        self.norm = LayerNorm(dim)

    def forward(self, x, *args, **kwargs):
        x = self.norm(x)
        return self.fn(x, *args, **kwargs)


class CMB(nn.Module):

    def __init__(self, dim, kernel_size=7):
        super(CMB, self).__init__()
        padding = (kernel_size - 1) // 2

        self.norm = LayerNorm(dim, eps=1e-6)
        self.a = nn.Sequential(nn.Conv2d(dim, dim, 1), nn.GELU(), nn.Conv2d(dim, dim, kernel_size, padding=padding, groups=dim))

        self.v = nn.Conv2d(dim, dim, 1)
        self.proj = nn.Conv2d(dim, dim, 1)

    def forward(self, x):
        x = self.norm(x)
        a = self.a(x)
        x = a * self.v(x)
        x = self.proj(x)
        return x


class DwFFN(nn.Module):

    def __init__(self, dim, kernel_size=3, ffn_ratio=4., act_layer=nn.GELU()):
        super(DwFFN, self).__init__()
        hidden_dim = int(round(dim * ffn_ratio))
        self.fc = ConvReLU(dim, hidden_dim, kernel_size=1, stride=1, act_layer=act_layer)

        self.dwconv = ConvReLU(hidden_dim,
                               hidden_dim,
                               stride=1,
                               groups=hidden_dim,
                               kernel_size=kernel_size,
                               act_layer=act_layer)
        self.ffn = nn.Conv2d(hidden_dim, dim, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        out = self.fc(x)
        out = self.dwconv(out)
        x = self.ffn(out)
        return x


class CAB(nn.Module):

    def __init__(self, dim, dpr, cmb_kernel=7, dw_kernel=3, ffn_ratio=4., act_layer=nn.GELU()):
        super(CAB, self).__init__()
        self.attn = CMB(dim=dim, kernel_size=cmb_kernel)
        self.dwffn = PreNorm(dim, DwFFN(dim, dw_kernel, ffn_ratio, act_layer))
        self.drop_path = DropPath(dpr) if dpr > 0. else nn.Identity()

    def forward(self, x):
        x = self.drop_path(self.attn(x)) + x  #a shortcut in attn module
        x = self.drop_path(self.dwffn(x)) + x  #a shortcut in dwffn module
        return x


class CABs(nn.Module):

    def __init__(self, dim, dpr, num_blocks=3, cmb_kernel=7, dw_kernel=3, ffn_ratio=4):
        super(CABs, self).__init__()

        blocks = []

        if num_blocks == 1:  #for a CAB with drop-path
            dpr = [dpr]
        else:
            dpr = [x.item() for x in torch.linspace(0, dpr, num_blocks)]  # for mulit CABs with drop-path

        for _ in range(num_blocks):
            blocks.append(CAB(dim=dim, dpr=dpr[_], cmb_kernel=cmb_kernel, dw_kernel=dw_kernel, ffn_ratio=ffn_ratio))
        self.blocks = nn.Sequential(*blocks)

    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        return x


class CMFormer(nn.Module):

    def __init__(self,
                 in_dim,
                 dim=28,
                 cdpr=0.,
                 bdpr=0.,
                 num_blocks=[1, 1, 3],
                 cmb_kernel=7,
                 dw_kernel=3,
                 ffn_ratio=4,
                 with_cp=False):
        super(CMFormer, self).__init__()
        self.stage = len(num_blocks) - 1
        self.dim = dim
        self.with_cp = with_cp

        # Input stem
        self.embedding = nn.Sequential(nn.Conv2d(in_dim, self.dim, 3, 1, 1, bias=False))  # 3x3 conv

        # Encoder Layer
        self.encoder_layers = nn.ModuleList([])
        dim_stage = self.dim

        for i in range(self.stage):
            self.encoder_layers.append(
                nn.ModuleList([
                    CABs(dim=dim_stage,
                         dpr=cdpr,
                         num_blocks=num_blocks[i],
                         cmb_kernel=cmb_kernel,
                         dw_kernel=dw_kernel,
                         ffn_ratio=ffn_ratio),
                    nn.Conv2d(dim_stage, dim_stage * 2, 4, 2, 1, bias=False),
                ]))
            dim_stage *= 2

        # Bottleneck
        self.bottleneck = CABs(dim=dim_stage,
                               dpr=bdpr,
                               num_blocks=num_blocks[-1],
                               cmb_kernel=cmb_kernel,
                               dw_kernel=dw_kernel,
                               ffn_ratio=ffn_ratio)

        # Decoder
        self.decoder_layers = nn.ModuleList([])
        for i in range(self.stage):
            self.decoder_layers.append(
                nn.ModuleList([
                    nn.ConvTranspose2d(dim_stage, dim_stage // 2, stride=2, kernel_size=2, padding=0, output_padding=0),
                    nn.Conv2d(dim_stage, dim_stage // 2, 1, 1, bias=False),
                    CABs(dim=dim_stage // 2, dpr=cdpr, num_blocks=num_blocks[self.stage - 1 - i], ffn_ratio=ffn_ratio),
                ]))
            dim_stage //= 2

        # Output Mapping
        self.mapping = nn.Sequential(nn.Conv2d(self.dim, 28, 3, 1, 1, bias=False)  # 3x3 conv
                                     )

        for m in self.modules():
            if isinstance(m, nn.Linear):
                trunc_normal_(m.weight, std=.02)
                if isinstance(m, nn.Linear) and m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, (nn.LayerNorm, LayerNorm)):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        # Embedding
        b, c, h_inp, w_inp = x.shape
        hb, wb = 16, 16
        pad_h = (hb - h_inp % hb) % hb
        pad_w = (wb - w_inp % wb) % wb
        x = F.pad(x, [0, pad_w, 0, pad_h], mode='reflect')

        # Embedding
        fea = self.embedding(x)
        x = x[:, :28, :, :]

        # Encoder
        fea_encoder = []
        for (AttenBlocks, downsample) in self.encoder_layers:
            # Attention
            if self.with_cp:
                fea = checkpoint(AttenBlocks, fea)
            else:
                fea = AttenBlocks(fea)
            fea_encoder.append(fea)

            # Downsample
            fea = downsample(fea)

        # Bottleneck
        if self.with_cp:
            fea = checkpoint(self.bottleneck, fea)
        else:
            fea = self.bottleneck(fea)

        # Decoder
        for i, (FeaUpSample, Fution, AttenBlocks) in enumerate(self.decoder_layers):
            # Upsample
            fea = FeaUpSample(fea)
            # Fusion
            fea = Fution(torch.cat([fea, fea_encoder[self.stage - i - 1]], dim=1))
            # Attention
            if self.with_cp:
                fea = checkpoint(AttenBlocks, fea)
            else:
                fea = AttenBlocks(fea)
        # Mapping
        out = self.mapping(fea) + x

        return out[:, :, :h_inp, :w_inp]


class HyPaNet(nn.Module):

    def __init__(self, in_nc=29, out_nc=28, channel=64):
        super(HyPaNet, self).__init__()
        self.fution = nn.Conv2d(in_nc, channel, 1, 1, 0, bias=True)
        self.down_sample = nn.Conv2d(channel, channel, 3, 2, 1, bias=True)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.mlp = nn.Sequential(nn.Conv2d(channel, channel, 1, padding=0, bias=True), nn.ReLU(inplace=True),
                                 nn.Conv2d(channel, channel, 1, padding=0, bias=True), nn.ReLU(inplace=True),
                                 nn.Conv2d(channel, out_nc, 1, padding=0, bias=True), nn.Softplus())
        self.relu = nn.ReLU(inplace=True)
        self.out_nc = out_nc

    def forward(self, x):
        x = self.down_sample(self.relu(self.fution(x)))
        x = self.avg_pool(x)
        x = self.mlp(x) + 1e-6
        return x[:, :self.out_nc // 2, :, :], x[:, self.out_nc // 2:, :, :]


class SAUNet(nn.Module):

    def __init__(self,
                 num_iterations=1,
                 cdpr=0.,
                 bdpr=0.,
                 num_blocks=[1, 1, 3],
                 cmb_kernel=7,
                 dw_kernel=3,
                 ffn_ratio=4,
                 with_cp=False):
        super(SAUNet, self).__init__()
        self.para_estimator = HyPaNet(in_nc=28, out_nc=num_iterations * 2)
        self.fution = nn.Conv2d(56, 28, 1, padding=0, bias=True)
        self.num_iterations = num_iterations
        self.gamma = nn.ParameterList([nn.Parameter(torch.tensor(0., dtype=torch.float32)) for _ in range(num_iterations - 1)])
        self.denoisers = nn.ModuleList([])
        for _ in range(num_iterations):
            self.denoisers.append(
                CMFormer(in_dim=29,
                         dim=28,
                         cdpr=cdpr,
                         bdpr=bdpr,
                         num_blocks=num_blocks,
                         cmb_kernel=cmb_kernel,
                         dw_kernel=dw_kernel,
                         ffn_ratio=ffn_ratio,
                         with_cp=with_cp))

    def initial(self, y, Phi):
        """
        :param y: [b,256,310]
        :param Phi: [b,28,256,310]
        :return: temp: [b,28,256,310]; alpha: [b, num_iterations]; beta: [b, num_iterations]
        """
        nC, step = 28, 2
        y = y / nC * 2
        bs, row, col = y.shape
        y_shift = torch.zeros(bs, nC, row, col).cuda().float()
        for i in range(nC):
            y_shift[:, i, :, step * i:step * i + col - (nC - 1) * step] = y[:, :, step * i:step * i + col - (nC - 1) * step]
        z = self.fution(torch.cat([y_shift, Phi], dim=1))
        alpha, beta = self.para_estimator(self.fution(torch.cat([y_shift, Phi], dim=1)))
        return z, alpha, beta

    def forward(self, y, input_mask=None):
        """
        :param y: [b,256,310]
        :param Phi: [b,28,256,310]
        :param Phi_PhiT: [b,256,310]
        :return: z_crop: [b,28,256,256]
        """
        out_shape = y.shape[-2]
        Phi, Phi_s = input_mask
        z, alphas, betas = self.initial(y, Phi)
        b = torch.zeros_like(Phi)  # [b, 28, 256, 310]
        for i in range(self.num_iterations):
            # alpha [b, 1, 1]    beta [b, 1, 1, 1]
            alpha, beta = alphas[:, i, :, :], betas[:, i:i + 1, :, :]
            Phi_z = A(z + b, Phi)  # [b, 256, 310]
            x = z + b + At(torch.div(y - Phi_z, alpha + Phi_s), Phi)
            x1 = x - b
            x1 = shift_back_3d(x1)
            beta_repeat = beta.repeat(1, 1, x1.shape[2], x1.shape[3])
            z = self.denoisers[i](torch.cat([x1, beta_repeat], dim=1))
            if i < self.num_iterations - 1:
                z = shift_3d(z)
                b = b - self.gamma[i] * (x - z)
        return z[:, :, :, 0:out_shape]
