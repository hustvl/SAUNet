import argparse


def simu_par_args():
    parser = argparse.ArgumentParser(description="HyperSpectral Image Reconstruction for Simulation Experiment")

    # Hardware specifications
    parser.add_argument("--gpu_id", type=str, default='0')

    # Data specifications
    parser.add_argument('--data_root', type=str, default='./datasets/', help='dataset directory')

    # Saving specifications
    parser.add_argument('--outf', type=str, default='./exp/saunet_1stg/', help='saving_path')

    # Model specifications
    parser.add_argument('--method', type=str, default='saunet_1stg', help='method name')
    parser.add_argument('--pretrained_model_path', type=str, default=None, help='pretrained model directory')
    parser.add_argument(
        "--input_setting", type=str, default='Y',
        help='the input measurement of the network: H, HM or Y')  # H-> shift_back , HM -> shift_back * mask , Y-> measuement
    parser.add_argument("--input_mask",
                        type=str,
                        default='Phi_PhiPhiT',
                        help='the input mask of the network: Phi, Phi_PhiPhiT, Mask or None'
                        )  # Phi: shift_mask   Mask: mask , Phi_PhiT \Phi\Phi^top

    # Training specifications
    parser.add_argument('--batch_size', type=int, default=5, help='the number of HSIs per batch')
    parser.add_argument("--max_epoch", type=int, default=300, help='total epoch')
    parser.add_argument('--optim', type=str, default='Adam', help='the optimizer choice')
    parser.add_argument("--scheduler", type=str, default='MultiStepLR', help='MultiStepLR or CosineAnnealingLR')
    parser.add_argument("--milestones", type=int, default=[50, 100, 150, 200, 250], help='milestones for MultiStepLR')
    parser.add_argument("--gamma", type=float, default=0.5, help='learning rate decay for MultiStepLR')
    parser.add_argument("--epoch_sam_num", type=int, default=5000, help='the number of samples per epoch')
    parser.add_argument("--learning_rate", type=float, default=0.0004)
    parser.add_argument('--seed', type=int, default=None, help='random seed')
    parser.add_argument('--deterministic', action='store_true', help='whether to set deterministic options for CUDNN backend.')
    parser.add_argument('--debug', action='store_true',
                        help='debug the code')  # Loading 5 scence and training 10 batchsize for a epoch
    parser.add_argument('--cp', action='store_true', help='use checkpoint when training model out of memery')

    ## -----Specifications for Resume training ---------
    parser.add_argument('--resume_path', '-r', type=str, default=None, help='resume model path')
    parser.add_argument('--beg_epoch', type=int, default=0, help='begin epoch path')

    opt = parser.parse_args()

    # dataset
    opt.data_path = f"{opt.data_root}/cave_1024_28/"
    opt.mask_path = f"{opt.data_root}/TSA_simu_data/"
    opt.test_path = f"{opt.data_root}/TSA_simu_data/Truth/"

    return opt
