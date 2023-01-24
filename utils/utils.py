import torch
import logging
import os


# ----- time formate -------
def time2file_name(time):
    year = time[0:4]
    month = time[5:7]
    day = time[8:10]
    hour = time[11:13]
    minute = time[14:16]
    second = time[17:19]
    time_filename = year + '_' + month + '_' + day + '_' + hour + '_' + minute + '_' + second
    return time_filename


# ----- log ---------
def gen_log(model_path, stream_path=True, log_name='log.txt'):
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s - %(levelname)s: %(message)s")

    log_file = os.path.join(model_path, log_name)
    fh = logging.FileHandler(log_file, mode='a')
    fh.setLevel(logging.INFO)
    fh.setFormatter(formatter)

    logger.addHandler(fh)

    if stream_path:
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        ch.setFormatter(formatter)
        logger.addHandler(ch)

    return logger


# ----- load and save model --------
def checkpoint(model, epoch, model_path, logger):
    model_out_path = model_path + "/model_epoch_{}.pth".format(epoch)
    torch.save(model.state_dict(), model_out_path)
    logger.info("Checkpoint saved to {}".format(model_out_path))


#--------------------------compute the Params and training time ----------------------
def count_param(model):
    param_count = 0
    state_dict_model = model.state_dict()
    for k, v in state_dict_model.items():
        param_count += v.numel()
    return param_count


# Record the training GPU-time.
# code referecnce : https://github.com/YIWEI-CHEN/pc_darts/blob/024b896fc22e232a31dbfd0ae5c6425b556151e7/utils.py#L167
def get_elaspe_time(begin, end):
    torch.cuda.synchronize()
    return begin.elapsed_time(end) / 1000.0