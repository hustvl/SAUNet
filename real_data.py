import torch.utils.data as tud
import random
import torch
import numpy as np
import scipy.io as sio
import os


def prepare_data_cave(path, file_num):
    HR_HSI = np.zeros((((512, 512, 28, file_num))))
    file_list = os.listdir(path)
    # for idx in range(1):
    for idx in range(file_num):
        print(f'loading CAVE {idx}')
        ####  read HrHSI
        HR_code = file_list[idx]
        path1 = os.path.join(path) + HR_code
        data = sio.loadmat(path1)
        HR_HSI[:, :, :, idx] = np.ascontiguousarray(data['data_slice']) / 65535.0
        HR_HSI[HR_HSI < 0] = 0
        HR_HSI[HR_HSI > 1] = 1
    return HR_HSI


def prepare_data_KAIST(path, file_num):
    HR_HSI = np.zeros((((2704, 3376, 28, file_num))))
    file_list = os.listdir(path)
    # for idx in range(1):
    for idx in range(file_num):
        print(f'loading KAIST {idx}')
        ####  read HrHSI
        HR_code = file_list[idx]
        path1 = os.path.join(path) + HR_code
        data = sio.loadmat(path1)
        HR_HSI[:, :, :, idx] = np.ascontiguousarray(data['HSI'])
        HR_HSI[HR_HSI < 0] = 0
        HR_HSI[HR_HSI > 1] = 1
    return HR_HSI


def prepare_test_data(path, file_num):
    HR_HSI = np.zeros((((660, 714, file_num))))
    for idx in range(file_num):
        ####  read HrHSI
        path1 = os.path.join(path) + 'scene' + str(idx + 1) + '.mat'
        data = sio.loadmat(path1)
        HR_HSI[:, :, idx] = np.ascontiguousarray(data['meas_real'])
        HR_HSI[HR_HSI < 0] = 0.0
        HR_HSI[HR_HSI > 1] = 1.0
    return HR_HSI


def load_test_mask(path, size=660):
    ## load mask
    data = sio.loadmat(path)
    mask = np.ascontiguousarray(data['mask'])
    mask_3d = np.tile(mask[:, :, np.newaxis], (1, 1, 28))
    mask_3d_shift = np.zeros((size, size + (28 - 1) * 2, 28))
    mask_3d_shift[:, 0:size, :] = mask_3d
    for t in range(28):
        mask_3d_shift[:, :, t] = np.roll(mask_3d_shift[:, :, t], 2 * t, axis=1)
    mask_3d_shift_s = np.sum(mask_3d_shift**2, axis=2, keepdims=False)
    mask_3d_shift_s[mask_3d_shift_s == 0] = 1
    mask_3d_shift = torch.FloatTensor(mask_3d_shift.copy()).permute(2, 0, 1)
    mask_3d_shift_s = torch.FloatTensor(mask_3d_shift_s.copy())
    return mask_3d_shift.unsqueeze(0), mask_3d_shift_s.unsqueeze(0)


def init_mask(mask, Phi, Phi_s, mask_type):
    if mask_type == 'Phi':
        input_mask = Phi
    elif mask_type == 'Phi_PhiPhiT':
        input_mask = (Phi, Phi_s)
    elif mask_type == 'Mask':
        input_mask = mask
    elif mask_type == None:
        input_mask = None
    return input_mask

class dataset(tud.Dataset):
    def __init__(self, opt, CAVE, KAIST):
        super(dataset, self).__init__()
        self.isTrain = opt.isTrain
        self.debug = opt.debug
        self.size = opt.size
        # self.path = opt.data_path
        if self.isTrain == True:
            self.num = opt.trainset_num
        else:
            self.num = opt.testset_num
        self.CAVE = CAVE
        self.KAIST = KAIST
        ## load mask
        data = sio.loadmat(opt.mask_path)
        self.mask = np.ascontiguousarray(data['mask'])
        self.mask_3d = np.tile(self.mask[:, :, np.newaxis], (1, 1, 28))

    def __getitem__(self, index):
        if self.isTrain == True:
            # index1 = 0
            index1 = random.randint(0, 2 if self.debug else 29)
            d = random.randint(0, 1)
            if d == 0:
                hsi = self.CAVE[:, :, :, index1]
            else:
                hsi = self.KAIST[:, :, :, index1]
        else:
            index1 = index
            hsi = self.HSI[:, :, :, index1]
        shape = np.shape(hsi)

        px = random.randint(0, shape[0] - self.size)
        py = random.randint(0, shape[1] - self.size)
        label = hsi[px:px + self.size:1, py:py + self.size:1, :]
        # while np.max(label)==0:
        #     px = random.randint(0, shape[0] - self.size)
        #     py = random.randint(0, shape[1] - self.size)
        #     label = hsi[px:px + self.size:1, py:py + self.size:1, :]
        #     print(np.min(), np.max())

        pxm = random.randint(0, 660 - self.size)
        pym = random.randint(0, 660 - self.size)
        mask_3d = self.mask_3d[pxm:pxm + self.size:1, pym:pym + self.size:1, :]

        mask_3d_shift = np.zeros((self.size, self.size + (28 - 1) * 2, 28))
        mask_3d_shift[:, 0:self.size, :] = mask_3d
        for t in range(28):
            mask_3d_shift[:, :, t] = np.roll(mask_3d_shift[:, :, t], 2 * t, axis=1)
        mask_3d_shift_s = np.sum(mask_3d_shift**2, axis=2, keepdims=False)
        mask_3d_shift_s[mask_3d_shift_s == 0] = 1

        if self.isTrain == True:

            rotTimes = random.randint(0, 3)
            vFlip = random.randint(0, 1)
            hFlip = random.randint(0, 1)

            # Random rotation
            for j in range(rotTimes):
                label = np.rot90(label)

            # Random vertical Flip
            for j in range(vFlip):
                label = label[:, ::-1, :].copy()

            # Random horizontal Flip
            for j in range(hFlip):
                label = label[::-1, :, :].copy()

        temp = mask_3d * label
        temp_shift = np.zeros((self.size, self.size + (28 - 1) * 2, 28))
        temp_shift[:, 0:self.size, :] = temp
        for t in range(28):
            temp_shift[:, :, t] = np.roll(temp_shift[:, :, t], 2 * t, axis=1)
        meas = np.sum(temp_shift, axis=2)
        input = meas / 28 * 2 * 1.2

        QE, bit = 0.4, 2048
        input = np.random.binomial((input * bit / QE).astype(int), QE)
        input = np.float32(input) / np.float32(bit)

        label = torch.FloatTensor(label.copy()).permute(2, 0, 1)
        input = torch.FloatTensor(input.copy())
        mask_3d_shift = torch.FloatTensor(mask_3d_shift.copy()).permute(2, 0, 1)
        mask_3d_shift_s = torch.FloatTensor(mask_3d_shift_s.copy())
        return input, label, mask_3d, mask_3d_shift, mask_3d_shift_s

    def __len__(self):
        return self.num
