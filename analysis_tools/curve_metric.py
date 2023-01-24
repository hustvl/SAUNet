import argparse
import os
import numpy as np
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser(description="write the SSIM or PSNR - epoch curve")

parser.add_argument('work_dir', type=str, default='/home/jywang/HSI/HSITrans/exp/', help='the work_dir of ')
parser.add_argument('-lps', '--log_paths', type=str, nargs='+', help='the analysis log paths')
parser.add_argument('-l', '--labels', type=str, nargs='+', help='the labels corresponding to logs to print in fig')
parser.add_argument('-mt', '--metric_type', choices=['Loss', 'psnr', 'ssim'])
parser.add_argument('-b', '--begin', type=int, default=0, help='the begin epoch')
parser.add_argument('-e', '--end', type=int, default=-1, help='the end epoch')
parser.add_argument('-f', '--format', type=str, default='svg', choices=['jpg', 'png', 'svg'], help='save the image format')
parser.add_argument('-hg', '--hguides', type=float, nargs='+', help='draw the horizontal guides in an image')
parser.add_argument('-vg', '--vguides', type=float, nargs='+', help='draw the vertical guides in an image')

opt = parser.parse_args()

pos_indexes = {'Epoch': 6, 'Loss': -3, 'psnr': -6, 'ssim': -3}

# check input
assert len(opt.log_paths) == len(opt.labels), 'the number of logs are not the same as the labels!'

for i in opt.log_paths:
    assert os.path.exists(i), f"the log path {i} does not exsist."

# create work_dir
if not os.path.exists(opt.work_dir):
    os.makedirs(opt.work_dir)


# load data
def load_value(log_path):
    with open(log_path, 'r') as f:
        data = f.read().splitlines()
        y_data = [float(i.split()[pos_indexes[opt.metric_type]].split(',')[0]) for i in data if opt.metric_type in i]
        x_data = [int(i.split()[pos_indexes['Epoch']].split(':')[0]) for i in data if opt.metric_type in i]
    return x_data, y_data


x_data_list = []
y_data_list = []
for i in opt.log_paths:
    x_data, y_data = load_value(i)
    x_data_list.append(np.array(x_data, dtype=np.int32))
    y_data_list.append(np.array(y_data))

# draw curves
fig, ax = plt.subplots(1, 1)
ax.set_title(opt.metric_type.lower() + '-epoch Curve')
ax.set_xlabel('epoch')
ax.set_ylabel(opt.metric_type.lower())

for epoch, data, label in zip(x_data_list, y_data_list, opt.labels):
    if opt.end == -1 or len(epoch) <= opt.end:

        ax.plot(epoch[opt.begin:], data[opt.begin:], label=label)
        # draw the max points
        #max_point_index = np.argmax(data[opt.begin:])
        #str_ = "(" + str(epoch[opt.begin:][max_point_index]) + " " + str(data[opt.begin:][max_point_index]) + ")"
        #ax.annotate(str_,
        #            xy=(max_point_index, data[opt.begin:][max_point_index]),
        #            xytext=(max_point_index, data[opt.begin:][max_point_index]))
    else:
        ax.plot(epoch[opt.begin:opt.end + 1], data[opt.begin:opt.end + 1], label=label)
        #max_point_index = np.argmax(data[opt.begin:opt.end + 1])
        #str_ = "(" + str(epoch[opt.begin:opt.end + 1][max_point_index]) + " " + str(
        #    data[opt.begin:opt.end + 1][max_point_index]) + ")"
        #ax.annotate(str_,
        #            xy=(max_point_index, data[opt.begin:opt.end + 1][max_point_index]),
        #            xytext=(max_point_index, data[opt.begin:opt.end + 1][max_point_index]))

# draw guides
if opt.hguides is not None:
    for y in opt.hguides:
        ax.axhline(y, c='k', alpha=0.5, ls='--', linewidth=0.5)
        ax.annotate(str(y), xy=(0, y), xytext=(0, y), fontsize='xx-small')

if opt.vguides is not None:
    for x in opt.vguides:
        ax.axvline(x, c='k', alpha=0.5, ls='--', linewidth=0.5)
        ax.annotate(str(x), xy=(x, 0), xytext=(x, 0), fontsize='xx-small')

legend = ax.legend(loc='best', shadow=True)

plt.savefig(os.path.join(opt.work_dir, opt.metric_type.lower() + '_results.' + opt.format), format=opt.format)
plt.show()
plt.close()