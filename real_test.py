import os
import time
import datetime
import random
import scipy.io as scio

import torch
import torch.nn as nn
import numpy as np
from torch.autograd import Variable

import torch.utils.data as tud

from real_data import *
from utils import real_par_args
from model import model_generator

# Argument Init
opt = real_par_args()

# device
os.environ["CUDA_DEVICE_ORDER"] = 'PCI_BUS_ID'
os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpu_id
torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True
if not torch.cuda.is_available():
    raise Exception('NO GPU!')

# random seed
if opt.seed is None:
    opt.seed = np.random.randint(2**31)

random.seed(opt.seed)
np.random.seed(opt.seed)
torch.manual_seed(opt.seed)
torch.cuda.manual_seed_all(opt.seed)
if opt.deterministic:
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# load test data
HR_HSI = prepare_test_data(opt.test_path, 5)
mask_3d_shift, mask_3d_shift_s = load_test_mask(opt.mask_path)

if not os.path.exists(opt.outf):
    os.makedirs(opt.outf)

model = model_generator(opt.method, opt.pretrained_model_path).cuda()
print("Model init.")


def test(HR_HSI, mask_3d_shift, mask_3d_shift_s):
    pred = []
    model.eval()
    begin = time.time()

    for j in range(5):
        with torch.no_grad():
            meas = HR_HSI[:, :, j]
            meas = meas / meas.max() * 0.8
            meas = torch.FloatTensor(meas)
            input = meas.unsqueeze(0)
            input = Variable(input)
            input = input.cuda()
            mask_3d_shift = mask_3d_shift.cuda()
            mask_3d_shift_s = mask_3d_shift_s.cuda()
            out = model(input, (mask_3d_shift, mask_3d_shift_s))

            result = out
            result = result.clamp(min=0., max=1.)
            res = result.cpu().permute(2, 3, 1, 0).squeeze(3).numpy()
            pred.append(res)  # H W C

    end = time.time()
    print('===> Epoch {}: time: {:.2f}'.format(0, (end - begin)))
    return pred


def main():

    pred = test(HR_HSI, mask_3d_shift, mask_3d_shift_s)

    name = os.path.join(opt.outf, 'Test_results.mat')
    scio.savemat(name, {'pred': pred})


if __name__ == '__main__':
    main()