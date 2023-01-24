import os
import time
import datetime
import random
import scipy.io as scio

import torch
import numpy as np
from torch.autograd import Variable

from utils import simu_par_args, time2file_name, gen_log, checkpoint, get_elaspe_time, count_param, torch_psnr, torch_ssim
from simu_data import init_mask, LoadTraining, LoadTest, shuffle_crop, init_meas
from model import model_generator

try:
    from fvcore.nn.flop_count import FlopCountAnalysis
except:
    pass

# ------------------------------- Init for training   ---------------------------------
# Argument Init
opt = simu_par_args()

# Device Init
os.environ["CUDA_DEVICE_ORDER"] = 'PCI_BUS_ID'
os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpu_id
torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True
if not torch.cuda.is_available():
    raise Exception('NO GPU!')

# Random seed Init
if opt.seed is None:
    opt.seed = np.random.randint(2**31)

random.seed(opt.seed)
np.random.seed(opt.seed)
torch.manual_seed(opt.seed)
torch.cuda.manual_seed_all(opt.seed)
if opt.deterministic:
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# Mask Init for train and test
mask3d_batch_train, input_mask_train = init_mask(opt.mask_path, opt.input_mask, opt.batch_size)
mask3d_batch_test, input_mask_test = init_mask(opt.mask_path, opt.input_mask, 10)

# Dataset Init
train_set = LoadTraining(opt.data_path, opt.debug)
test_data = LoadTest(opt.test_path)

# Saving path Init
if opt.resume_path is None:
    date_time = str(datetime.datetime.now())
    date_time = time2file_name(date_time)
    result_path = opt.outf + date_time + '/result/'
    model_path = opt.outf + date_time + '/model/'
    if not os.path.exists(result_path):
        os.makedirs(result_path)
    if not os.path.exists(model_path):
        os.makedirs(model_path)
else:
    result_path = os.path.dirname(opt.resume_path).split('/model')[0] + '/result/'
    model_path = os.path.dirname(opt.resume_path)

# Model Init
model = model_generator(opt.method, opt.pretrained_model_path if opt.resume_path is None else opt.resume_path, with_cp=opt.cp)
print("Model init.")

# Optimizer Init
if opt.optim == 'Adam':
    optimizer = torch.optim.Adam(model.parameters(), lr=opt.learning_rate, betas=(0.9, 0.999))
elif opt.optim == 'AdamW':
    optimizer = torch.optim.AdamW(model.parameters(), lr=opt.learning_rate, betas=(0.9, 0.999))

# Optimizing scheduler Init
if opt.scheduler == 'MultiStepLR':
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=opt.milestones, gamma=opt.gamma)
elif opt.scheduler == 'CosineAnnealingLR':
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, opt.max_epoch, eta_min=1e-6)
if opt.resume_path is not None:
    scheduler.step(opt.beg_epoch)
mse = torch.nn.MSELoss().cuda()
print("optimizer init.")


def train(epoch, logger):
    epoch_loss = 0
    begin = time.time()
    batch_num = 10 if opt.debug else int(np.floor(opt.epoch_sam_num / opt.batch_size))
    model.train()
    train_begin = torch.cuda.Event(enable_timing=True)
    train_end = torch.cuda.Event(enable_timing=True)
    forward_times = 0
    backward_times = 0

    for i in range(batch_num):
        # When epoch == 1 and pretrain，warm-up.
        if epoch == 1 and opt.pretrained_model_path is not None:
            lr = opt.learning_rate * (1 + i) / batch_num
            if i % 50 == 0:
                print(f'warm_up, learning rate is {lr}')
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr
        gt_batch = shuffle_crop(train_set, opt.batch_size)
        gt = Variable(gt_batch).cuda().float()
        input_meas = init_meas(gt, mask3d_batch_train, opt.input_setting)

        optimizer.zero_grad()

        train_begin.record()
        model_out = model(input_meas, input_mask_train)
        loss = torch.sqrt(mse(model_out, gt))
        train_end.record()
        forward_times += get_elaspe_time(train_begin, train_end)

        epoch_loss += loss.data

        train_begin.record()
        loss.backward()
        train_end.record()
        backward_times += get_elaspe_time(train_begin, train_end)

        optimizer.step()
    end = time.time()

    logger.info("===> Epoch {} Complete: Avg. Loss: {:.6f} time: {:.2f}".format(epoch, epoch_loss / batch_num, (end - begin)))
    logger.info("===> Epoch {} Complete: Avg. forward_time: {:.3f} backward_time: {:.3f} train_time: {:.3f}".format(
        epoch, forward_times, backward_times, (forward_times + backward_times)))
    return 0


def test(epoch, logger):
    psnr_list, ssim_list = [], []
    test_gt = test_data.cuda().float()
    input_meas = init_meas(test_gt, mask3d_batch_test, opt.input_setting)
    model.eval()
    begin = time.time()

    with torch.no_grad():
        model_out = model(input_meas, input_mask_test)

    end = time.time()
    for k in range(test_gt.shape[0]):
        psnr_val = torch_psnr(model_out[k, :, :, :], test_gt[k, :, :, :])
        ssim_val = torch_ssim(model_out[k, :, :, :], test_gt[k, :, :, :])
        psnr_list.append(psnr_val.detach().cpu().numpy())
        ssim_list.append(ssim_val.detach().cpu().numpy())
    pred = np.transpose(model_out.detach().cpu().numpy(), (0, 2, 3, 1)).astype(np.float32)
    truth = np.transpose(test_gt.cpu().numpy(), (0, 2, 3, 1)).astype(np.float32)
    psnr_mean = np.mean(np.asarray(psnr_list))
    ssim_mean = np.mean(np.asarray(ssim_list))
    logger.info('===> Epoch {}: testing psnr = {:.2f}, ssim = {:.3f}, time: {:.2f}'.format(
        epoch, psnr_mean, ssim_mean, (end - begin)))
    model.train()
    return pred, truth, psnr_list, ssim_list, psnr_mean, ssim_mean


def main():
    logger = gen_log(model_path)
    logger.info("Learning rate:{}, batch_size:{}.\n".format(opt.learning_rate, opt.batch_size))
    psnr_max = 0

    # Caclulate flops and params
    params = count_param(model)
    try:
        gt_batch = shuffle_crop(train_set, opt.batch_size)
        gt = Variable(gt_batch).cuda().float()
        input_meas = init_meas(gt, mask3d_batch_train, opt.input_setting)

        flops = FlopCountAnalysis(model, (input_meas, input_mask_train))
        logger.info("parms:{}, GFLOPs: {}".format(params, flops.total() / 1e9 / opt.batch_size))
    except:
        pass

    # Begin to training
    for epoch in range(1 if opt.resume_path is None else opt.beg_epoch + 1, opt.max_epoch + 1):
        train(epoch, logger)
        (pred, truth, psnr_all, ssim_all, psnr_mean, ssim_mean) = test(epoch, logger)
        scheduler.step()
        if psnr_mean > psnr_max:
            psnr_max = psnr_mean
            if psnr_mean > 30:
                name = result_path + '/' + 'Test_{}_{:.2f}_{:.3f}'.format(epoch, psnr_max, ssim_mean) + '.mat'
                scio.savemat(name, {'truth': truth, 'pred': pred, 'psnr_list': psnr_all, 'ssim_list': ssim_all})
                checkpoint(model, epoch, model_path, logger)
        torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
