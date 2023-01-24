# Run 1000 measurements, warm-up 200 and the rest for computing fps (forward_time).

import os
import datetime
import random

import torch
import numpy as np

from utils import simu_par_args, time2file_name, gen_log, get_elaspe_time
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


date_time = str(datetime.datetime.now())
date_time = time2file_name(date_time)
result_path = opt.outf + date_time + '/result/'
if not os.path.exists(result_path):
    os.makedirs(result_path)

# model
model = model_generator(opt.method, opt.pretrained_model_path if opt.resume_path is None else opt.resume_path).cuda()
print("Model init.")


def fps_test(logger):
    test_begin = torch.cuda.Event(enable_timing=True)
    test_end = torch.cuda.Event(enable_timing=True)
    forward_times = 0
    load_data_times = 0
    times = 0

    input_meas = torch.rand(1, 256, 310).cuda()
    input_mask_test = (torch.rand(1, 28, 256, 310).cuda(), torch.rand(1, 256, 310).cuda())

    try:
        from fvcore.nn.flop_count import FlopCountAnalysis
        flops = FlopCountAnalysis(model, (input_meas, input_mask_test))
        print("FLOPS total: {:.3f}".format(flops.total() / 1e9))
    except:
        pass

    for i in range(1, 1001):
        with torch.no_grad():
            torch.cuda.synchronize()
            test_begin.record()
            _ = model(input_meas, input_mask_test)
            torch.cuda.synchronize()
            test_end.record()
            forward_time = get_elaspe_time(test_begin, test_end)
            if i > 200:
                forward_times += forward_time
                times += 1
    logger.info("===> Epoch {} Sences {} Complete: forward_time: {:.3f} data_load_time: {:.3f}, all_time: {:.3f}".format(
        0, times, forward_times, load_data_times, forward_times + load_data_times))
    return forward_times, load_data_times, forward_times + load_data_times, times


def main():

    logger = gen_log(result_path)
    model.eval()

    (epoch_forward, epoch_load, epoch_all_time, sences) = fps_test(logger)

    logger.info("forward all time : {:.3f} , load all time :  {:.3f} , all time {:.3f}".format(
        epoch_forward, epoch_load, epoch_all_time))
    logger.info("FPS: {:.3f} , all_time_FPS {:.3f}".format(sences / epoch_forward, sences / epoch_all_time))


if __name__ == '__main__':
    main()
