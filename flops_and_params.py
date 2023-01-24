import os
import datetime
import random

import torch
import numpy as np

from utils import simu_par_args, time2file_name, count_param
from model import model_generator

# ArgumentParser
opt = simu_par_args()

# device
os.environ["CUDA_DEVICE_ORDER"] = 'PCI_BUS_ID'
os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpu_id
torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True
if not torch.cuda.is_available():
    raise Exception('NO GPU!')

# model
model = model_generator(opt.method, opt.pretrained_model_path if opt.resume_path is None else opt.resume_path).cuda()
print("Model init.")


def Param_FLOPs_test():

    input_meas = torch.rand(1, 256, 310).cuda()
    input_mask_test = (torch.rand(1, 28, 256, 310).cuda(), torch.rand(1, 256, 310).cuda())

    from fvcore.nn.flop_count import FlopCountAnalysis
    flops = FlopCountAnalysis(model, (input_meas, input_mask_test))
    print("FLOPS total: {:.3f}".format(flops.total() / 1e9))

    print("Params: {}".format(count_param(model)))


def main():
    model.eval()
    Param_FLOPs_test()


if __name__ == '__main__':
    main()
