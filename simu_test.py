import os
import time
import datetime
import random
import scipy.io as scio

import torch
import numpy as np
from torch.autograd import Variable

from utils import simu_par_args, torch_psnr, torch_ssim
from simu_data import init_mask, LoadTest, init_meas
from model import model_generator

try:
    from fvcore.nn.flop_count import FlopCountAnalysis
except:
    pass

opt = simu_par_args()

# Device Init
os.environ["CUDA_DEVICE_ORDER"] = 'PCI_BUS_ID'
os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpu_id
torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True
if not torch.cuda.is_available():
    raise Exception('NO GPU!')

# Mask Init for train and test
mask3d_batch, input_mask = init_mask(opt.mask_path, opt.input_mask, 10)

# Dataset Init
test_data = LoadTest(opt.test_path)

# Saving path Init
if not os.path.exists(opt.outf):
    os.makedirs(opt.outf)

# Model Init
model = model_generator(opt.method, opt.pretrained_model_path)
print("Model init.")


def test(model):
    psnr_list, ssim_list = [], []
    test_data = LoadTest(opt.test_path)
    test_gt = test_data.cuda().float()
    input_meas = init_meas(test_gt, mask3d_batch, opt.input_setting)
    model.eval()
    with torch.no_grad():
        model_out = model(input_meas, input_mask)
    for k in range(test_gt.shape[0]):
        psnr_val = torch_psnr(model_out[k, :, :, :], test_gt[k, :, :, :])
        ssim_val = torch_ssim(model_out[k, :, :, :], test_gt[k, :, :, :])
        psnr_list.append(psnr_val.detach().cpu().numpy())
        ssim_list.append(ssim_val.detach().cpu().numpy())
    pred = np.transpose(model_out.detach().cpu().numpy(), (0, 2, 3, 1)).astype(np.float32)
    truth = np.transpose(test_gt.cpu().numpy(), (0, 2, 3, 1)).astype(np.float32)
    psnr_mean = np.mean(np.asarray(psnr_list))
    ssim_mean = np.mean(np.asarray(ssim_list))

    print("PSNR_results -> \n", [float(i) for i in psnr_list])
    print("SSIM_results -> \n", [float(i) for i in ssim_list])
    print('===> Test: testing psnr = {:.2f}, ssim = {:.3f}'.format(psnr_mean, ssim_mean))
    return pred, truth


def main():
    # model
    model = model_generator(opt.method, opt.pretrained_model_path).cuda()
    pred, truth = test(model)
    name = opt.outf + 'Test_result.mat'
    print(f'Save reconstructed HSIs as {name}.')
    scio.savemat(name, {'truth': truth, 'pred': pred})


if __name__ == '__main__':
    main()
