import argparse
import os
import numpy as np

parser = argparse.ArgumentParser(description="Training time compute")

parser.add_argument('log_path', type=str, help='the analysis log path')
opt = parser.parse_args()

assert os.path.exists(opt.log_path), 'Please input an exisit training log'

with open(opt.log_path, 'r') as f:
    data = f.read().splitlines()
    data = [float(i.split()[-1]) for i in data if 'Loss' in i]
    data = np.array(data)
    print("all_time", data.mean() * 300 / 3600, "hours")  # training 300 epochs

with open(opt.log_path, 'r') as f:
    data = f.read().splitlines()
    data = [float(i.split()[-1]) for i in data if 'train_time' in i]
    data = np.array(data)
    #if 'birnat' in opt.log_path:
    #    print("GPU_hours", data.mean() * 100 / 3600, "hours")
    #else:
    print("GPU_hours", data.mean() * 300 / 3600, "hours")
