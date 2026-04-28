import os
import argparse
from exp import Exp
import warnings
warnings.filterwarnings('ignore')

def create_parser():
    parser = argparse.ArgumentParser()
    # Set-up parameters
    parser.add_argument('--device', default='npu', type=str, help='Name of device to use for tensor computations (npu/cuda)')
    parser.add_argument('--res_dir', default='./results', type=str)
    parser.add_argument('--ex_name', default='Debug', type=str)
    parser.add_argument('--use_gpu', default=False, type=bool)
    parser.add_argument('--use_npu', default=True, type=bool)
    parser.add_argument('--npu', default=1, type=int)
    parser.add_argument('--seed', default=1, type=int)
    parser.add_argument('--test', action='store_true', help='Test mode, do not train')

    # mask options
    parser.add_argument('--use_mask', action='store_true', help='Enable loading masks for training')
    parser.add_argument('--alpha', default=0, type=float, help='Alpha weight for mask==0 regions in loss')
    # augmentation
    parser.add_argument('--augment_flips', action='store_true', help='Enable horizontal and vertical flips to triple training set')
    parser.add_argument('--debug', action='store_true', help='Enable debug logging for numeric ranges')
    # dataset parameters
    parser.add_argument('--batch_size', default=1, type=int, help='Batch size')
    parser.add_argument('--val_batch_size', default=1, type=int, help='Batch size')
    parser.add_argument('--data_root', default='./data/')
    parser.add_argument('--dataname', default='aifund', choices=['mmnist', 'taxibj', 'aifund'])
    parser.add_argument('--num_workers', default=8, type=int)
    parser.add_argument('--data_type', default='PEEQ', choices=['PEEQ', 'MISES'], type=str, help='Type of data to load (PEEQ or MISES)')
    parser.add_argument('--offset', default=0, type=int, help='Offset for deterministic indexing in AIFund dataset')

    # normalization: zscore -> (x-mean)/std ; minmax -> (x-min)/(max-min) ; none -> no normalization
    parser.add_argument('--normalize', default='zscore', choices=['zscore', 'minmax', 'none'], help='Normalization method for dataset')
    # model parameters
    parser.add_argument('--input_frames', default=[20], type=int, nargs='*')
    parser.add_argument('--output_frames', default=[i for i in range(21,41)], type=int, nargs='*')
    parser.add_argument('--channel', default=1, type=int)
    parser.add_argument('--shape', default=[192, 288], type=int, nargs='*')
    parser.add_argument('--hid_S', default=64, type=int)
    parser.add_argument('--hid_T', default=512, type=int)
    parser.add_argument('--N_S', default=4, type=int)
    parser.add_argument('--N_T', default=8, type=int)
    parser.add_argument('--groups', default=1, type=int)
    parser.add_argument('--use_physical_prior', action='store_true', help='Enable Physical Prior Module in PSN model')
    parser.add_argument('--mises_percentile', default=99.35, type=float, help='Percentile for MISES normalization')
    parser.add_argument('--peeq_percentile', default=99.95, type=float, help='Percentile for PEEQ normalization')

    # Training parameters
    parser.add_argument('--epochs', default=1, type=int)
    parser.add_argument('--log_step', default=1, type=int)
    parser.add_argument('--lr', default=0.001, type=float, help='Learning rate')

    # ==========================================
    # Loss function parameters (新增部分)
    # ==========================================
    parser.add_argument('--l1_weight', default=1.0, type=float, help='Weight for L1 (Pixel) Loss')
    parser.add_argument('--ssim_weight', default=0.0, type=float, help='Weight for SSIM Loss')
    parser.add_argument('--grad_weight', default=0.0, type=float, help='Weight for Gradient Loss (Sobel)')
    parser.add_argument('--vgg_weight', default=0.0, type=float, help='Weight for VGG Perceptual Loss')
    parser.add_argument('--lpips_weight', default=0.0, type=float, help='Weight for LPIPS Loss')
    parser.add_argument('--fft_weight', default=0.0, type=float, help='Weight for FFT Loss')

    return parser

if __name__ == '__main__':
    args = create_parser().parse_args()
    args.image_height = args.shape[0]
    args.image_width = args.shape[1]
    args.n_frames_input = args.input_frames.__len__()
    args.n_frames_output = args.output_frames.__len__()
    args.n_channels = args.channel
    # 为兼容 `exp._build_model`，构建 `in_shape`/`out_shape`（[T, C, H, W]）
    args.in_shape = [args.n_frames_input, args.n_channels, args.image_height, args.image_width]
    args.out_shape = [args.n_frames_output, args.n_channels, args.image_height, args.image_width]
    config = args.__dict__

    exp = Exp(args)
    if args.test:
        print('>>>>>>>>>>>>>>>>>>>>>>>>>>>> testing <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<')
        mse = exp.test(args)
        print(f'Test MSE: {mse}')
        exit(0)
    else:
        print('>>>>>>>>>>>>>>>>>>>>>>>>>>>>  start <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<')
        exp.train(args)
        print('>>>>>>>>>>>>>>>>>>>>>>>>>>>> testing <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<')
        mse = exp.test(args)