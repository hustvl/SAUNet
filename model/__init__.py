import re
import torch
from .SAUNet import SAUNet


def param_setting(num_iterations, with_cp):
    bdpr, cdpr = 0, 0

    # Recommended hyper-parameters in simulation experiments in CAVE and KAIST datasets.
    if num_iterations == 1:
        bdpr, cdpr = 0, 0
    elif num_iterations == 2:
        bdpr, cdpr = 0.1, 0.1
    elif num_iterations == 3:
        bdpr, cdpr = 0.2, 0.1
    elif num_iterations == 5:
        bdpr, cdpr = 0.2, 0.1
    elif num_iterations == 9:
        bdpr, cdpr = 0.3, 0.0
    elif num_iterations == 13:
        bdpr, cdpr = 0.2, 0.1
    param = {
        'num_iterations': num_iterations,
        'cdpr': cdpr,
        'bdpr': bdpr,
        'num_blocks': [1, 1, 3],
        'cmb_kernel': 7,  # Convolutional Modulational Block kernel size, setting larger value with the pursuit of accuary
        'dw_kernel': 3,
        'ffn_ratio': 4,
        'with_cp': with_cp
    }
    return param


def model_generator(method, pretrained_model_path=None, with_cp=False):
    if 'saunet' in method:
        num_iterations = int(re.findall("\d+", method)[0])
        param = param_setting(num_iterations, with_cp)
        model = SAUNet(**param).cuda()
    else:
        print(f'Method {method} is not defined !!!!')

    if pretrained_model_path is not None:
        print(f'load model from {pretrained_model_path}')
        checkpoint = torch.load(pretrained_model_path)
        model.load_state_dict({k.replace('module.', ''): v for k, v in checkpoint.items()}, strict=True)
    return model