from pathlib import Path
from argparse import ArgumentParser
import numpy as np
import torch
import torch.nn as nn
from Models import UNet, SpatialTransform
from Functions import ValidationDataset
import torch.utils.data as Data
from natsort import natsorted
from skimage import io, img_as_ubyte

parser = ArgumentParser()
parser.add_argument("--lr", type=float,
                    dest="lr", default=1e-4, help="learning rate")
parser.add_argument("--bs", type=int,
                    dest="bs", default=1, help="batch_size")
parser.add_argument("--iteration", type=int,
                    dest="iteration", default=320001,
                    help="number of total iterations")
parser.add_argument("--local_ori", type=float,
                    dest="local_ori", default=1000.0,
                    help="Local Orientation Consistency loss: suggested range 1 to 1000")
parser.add_argument("--magnitude", type=float,
                    dest="magnitude", default=0.001,
                    help="magnitude loss: suggested range 0.001 to 1.0")
parser.add_argument("--mask_labda", type=float,
                    dest="mask_labda", default=0.25,
                    help="mask_labda loss: suggested range 0.1 to 10")
parser.add_argument("--data_labda", type=float,
                    dest="data_labda", default=0.02,
                    help="data_labda loss: suggested range 0.1 to 10")
parser.add_argument("--smth_labda", type=float,
                    dest="smth_labda", default=5.0,
                    help="labda loss: suggested range 0.1 to 10")
parser.add_argument("--checkpoint", type=int,
                    dest="checkpoint", default=4000,
                    help="frequency of saving models")
parser.add_argument("--start_channel", type=int,
                    dest="start_channel", default=8,
                    help="number of start channels")
parser.add_argument("--datapath", type=str,
                    dest="datapath",
                    default='../data',
                    help="data path for test images")
parser.add_argument("--trainingset", type=int,
                    dest="trainingset", default=4,
                    help="1 Half : 200 Images, 2 The other Half 200 Images 3 All 400 Images")
parser.add_argument("--using_l2", type=int,
                    dest="using_l2",
                    default=1,
                    help="using l2 or not")
opt = parser.parse_args()


def test(model_dir):
    bs = 1
    model = UNet(4, 2, opt.start_channel).cuda()

    model_idx = -1
    model_path = Path(model_dir)
    model_files = natsorted(model_path.iterdir())
    print(f'Best model: {model_files[model_idx].name}')
    best_model = torch.load(model_files[model_idx])
    model.load_state_dict(best_model)

    torch.backends.cudnn.benchmark = True
    transform = SpatialTransform().cuda()
    model.eval()
    transform.eval()
    use_cuda = True
    device = torch.device("cuda" if use_cuda else "cpu")
    test_set = ValidationDataset(opt.datapath, img_file='test_list.txt')
    test_generator = Data.DataLoader(dataset=test_set, batch_size=bs, shuffle=False, num_workers=2)

    output_dir = Path('inference_outputs')
    output_dir.mkdir(exist_ok=True)

    sample_idx = 0
    for mov_img, fix_img in test_generator:
        with torch.no_grad():
            mov_img_gpu = mov_img.float().to(device)
            fix_img_gpu = fix_img.float().to(device)

            V_xy = model(mov_img_gpu, fix_img_gpu)
            _, warped_mov_img = transform(mov_img_gpu, V_xy.permute(0, 2, 3, 1))

            for bs_index in range(bs):
                try:
                    moving_np = mov_img[bs_index].cpu().numpy()
                    fixed_np = fix_img[bs_index].cpu().numpy()
                    warped_np = warped_mov_img[bs_index, 0].cpu().numpy()
                    moving_np = img_as_ubyte(moving_np)
                    fix_img = img_as_ubyte(fixed_np)
                    warped_np = img_as_ubyte(warped_np)
                except Exception as e:
                    print(f'Error processing sample {sample_idx}: {e}')
                    print(moving_np.dtype, moving_np.min(), moving_np.max())
                    print(fixed_np.dtype, fixed_np.min(), fixed_np.max())
                    print(warped_np.dtype, warped_np.min(), warped_np.max())
                io.imsave(output_dir / f'sample_{sample_idx:03d}_moving.png', moving_np)
                io.imsave(output_dir / f'sample_{sample_idx:03d}_fixed.png', fixed_np)
                io.imsave(output_dir / f'sample_{sample_idx:03d}_warped.png', warped_np)

                sample_idx += 1

    print(f'Saved {sample_idx} image sets to {output_dir}')


if __name__ == '__main__':
    model_dir = Path(f'L2ss_{opt.using_l2}_Chan_{opt.start_channel}_Smth_{opt.smth_labda}_Set_{opt.trainingset}_LR_{opt.lr}_Pth')
    print(model_dir)
    test(model_dir)