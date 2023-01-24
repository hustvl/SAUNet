import argparse


def real_par_args():
    parser = argparse.ArgumentParser(description="Real HyperSpectral Image Reconstruction")

    # Hardware specifications
    parser.add_argument("--gpu_id", type=str, default='0')

    # Data specifications
    parser.add_argument('--data_root', type=str, default='./datasets/', help='dataset directory')
    parser.add_argument('--data_path_CAVE', default='./datasets/cave_512_28/', type=str, help='path of data')
    parser.add_argument('--data_path_KAIST', default='./datasets/KAIST_CVPR2021/', type=str, help='path of data')
    parser.add_argument('--mask_path', default='./datasets/TSA_real_data/mask.mat', type=str, help='path of mask')
    parser.add_argument('--test_path', default='./datasets/TSA_real_data/Measurements/', type=str, help='path of test')
    # Saving specifications
    parser.add_argument('--outf', type=str, default='./exp/saunet_3stg/', help='saving_path')

    # Model specifications
    parser.add_argument('--method', type=str, default='saunet_3stg', help='method name')
    parser.add_argument('--pretrained_model_path', type=str, default=None, help='pretrained model directory')
    parser.add_argument("--input_setting", type=str, default='Y', help='the input measurement of the network: H, HM or Y')
    parser.add_argument("--input_mask",
                        type=str,
                        default='Phi_PhiPhiT',
                        help='the input mask of the network: Phi, Phi_PhiPhiT, Mask or None')  # Phi: shift_mask   Mask: mask

    # Training specifications
    parser.add_argument("--size", default=384, type=int, help='cropped patch size')
    parser.add_argument("--epoch_sam_num", default=5000, type=int, help='total number of trainset')
    parser.add_argument('--optim', type=str, default='Adam', help='the optimizer choice')
    parser.add_argument("--seed", default=None, type=int, help='Random_seed')
    parser.add_argument('--batch_size', type=int, default=1, help='the number of HSIs per batch')
    parser.add_argument("--isTrain", action='store_true', help='train or test')
    parser.add_argument("--max_epoch", type=int, default=300, help='total epoch')
    parser.add_argument("--scheduler", type=str, default='MultiStepLR', help='MultiStepLR or CosineAnnealingLR')
    parser.add_argument("--milestones", type=int, default=[50, 100, 150, 200, 250], help='milestones for MultiStepLR')
    parser.add_argument("--gamma", type=float, default=0.5, help='learning rate decay for MultiStepLR')
    parser.add_argument("--learning_rate", type=float, default=0.0004)
    parser.add_argument('--deterministic', action='store_true', help='whether to set deterministic options for CUDNN backend.')
    parser.add_argument('--debug', action='store_true', help='debug the code')
    parser.add_argument("--testset_num", default=5, type=int, help='total number of testset')

    opt = parser.parse_args()

    opt.trainset_num = 10 if opt.debug else 20000 // ((opt.size // 96)**2)
    print("trainset_num : ", opt.trainset_num)

    return opt
