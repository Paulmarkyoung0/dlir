from argparse import ArgumentParser
import numpy as np
import torch
from Models import UNet, MSE, SAD, NCC, Dice, smoothloss, SpatialTransform
from Functions import TrainDataset, ValidationDataset
import torch.utils.data as Data
import matplotlib.pyplot as plt
from natsort import natsorted
import csv
from pathlib import Path

from pytorch_msssim import MS_SSIM
import logging
logging.getLogger('tifffile').setLevel(logging.ERROR)


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
                    help="data path for training images")
parser.add_argument("--trainingset", type=int,
                    dest="trainingset", default=4,
                    help="1 Half : 200 Images, 2 The other Half 200 Images 3 All 400 Images")
parser.add_argument("--using_l2", type=int,
                    dest="using_l2",
                    default=1,
                    help="using l2 or not")
parser.add_argument("--loss", type=str,
                    dest="loss",
                    default='dice',
                    help="truth or dice")
opt = parser.parse_args()

lr = opt.lr
bs = opt.bs
iteration = opt.iteration
start_channel = opt.start_channel
local_ori = opt.local_ori
magnitude = opt.magnitude
n_checkpoint = opt.checkpoint
smooth = opt.smth_labda
datapath = opt.datapath
mask_labda = opt.mask_labda
data_labda = opt.data_labda
trainingset = opt.trainingset
using_l2 = opt.using_l2
loss_type = opt.loss

def dice(pred1, truth1):
    dice_35=0
    
    mask4_value1 = np.unique(pred1)
    mask4_value2 = np.unique(truth1)
    mask_value4 = list(set(mask4_value1) & set(mask4_value2))
    for k in mask_value4[1:]:
        truth = truth1.copy()
        pred = pred1.copy()
        truth[truth!=k]=0
        pred[pred!=k]=0
        truth=truth/k
        pred=pred/k
        intersection = np.sum(pred[truth==1.0]) * 2.0
        dice_35 = dice_35 + intersection / (np.sum(pred) + np.sum(truth))
    return dice_35/(len(mask_value4)-1)

def save_checkpoint(state, save_dir, save_filename, max_model_num=10):
    torch.save(state, Path(save_dir) / save_filename)
    model_lists = natsorted(Path(save_dir).iterdir())
    while len(model_lists) > max_model_num:
        model_lists.pop(0).unlink()

def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = UNet(4, 2, start_channel).to(device)
    if using_l2 == 1:
        loss_similarity = MSE().loss
    elif using_l2 == 0:
        loss_similarity = SAD().loss
    elif using_l2 == 2:
        loss_similarity = NCC(win=9)
    elif using_l2 == 3:
        ms_ssim_module = MS_SSIM(data_range=1, size_average=True, channel=1, win_size=9)
        loss_similarity = SAD().loss
    loss_smooth = smoothloss

    transform = SpatialTransform().to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    lossall = np.zeros((3, iteration))
    train_set = TrainDataset(datapath,img_file='train_list.txt', supervision=loss_type)
    training_generator = Data.DataLoader(dataset=train_set, batch_size=bs, shuffle=True, num_workers=4)
    test_set = ValidationDataset(datapath,img_file='val_list.txt', supervision=loss_type)
    test_generator = Data.DataLoader(dataset=test_set, batch_size=bs, shuffle=False, num_workers=2)
    base_name = f'L2ss_{using_l2}_Chan_{start_channel}_Smth_{smooth}_Set_{trainingset}_LR_{lr}'
    model_dir = Path(base_name)
    model_dir_pth = Path(base_name + '_Pth')
    csv_name = base_name + '.csv'
    with open(csv_name, 'w') as f:
        fnames = ['Index','Dice','Sim','Smooth']
        writer = csv.DictWriter(f, fieldnames=fnames)
        writer.writeheader()

    model_dir.mkdir(parents=True, exist_ok=True)
    model_dir_pth.mkdir(parents=True, exist_ok=True)
        
    step = 1

    while step <= iteration:
        for batch in training_generator:
            mov_img, fix_img, *truths = batch
            # Move inputs to device in all cases
            mov_img = mov_img.to(device).float()
            fix_img = fix_img.to(device).float()
            if loss_type == 'dice':
                mov_lab, fix_lab = truths
                mov_lab = mov_lab.to(device).float()
                fix_lab = fix_lab.to(device).float()
            else:
                truth_img = truths[0].to(device).float()

            f_xy = model(mov_img, fix_img)

            grid, warped_mov = transform(mov_img, f_xy.permute(0, 2, 3, 1))
            if loss_type == 'dice':
                _, warped_lab = transform(mov_lab, f_xy.permute(0, 2, 3, 1), mod='nearest')

            # Ensure missing loss components are tensors so .item() works
            loss1 = loss_similarity(truth_img, warped_mov) if loss_type == 'truth' else torch.tensor(0.0, device=device)
            loss2 = Dice().loss(fix_lab, warped_lab) if loss_type == 'dice' else torch.tensor(0.0, device=device)
            loss5 = loss_smooth(f_xy)
            
            loss = loss1 + mask_labda * loss2 + smooth * loss5
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            lossall[:,step] = np.array([loss.item(),loss2.item(),loss5.item()])
            print(f'step "{step}" -> training loss "{loss:.4f}" - sim "{loss2:.4f}" -smo "{loss5:.4f}" ', end='\r', flush=True)

            if (step % n_checkpoint == 0):
                with torch.no_grad():
                    model.eval()
                    Dices_Validation = []
                    # Run validation with same loss structure as training
                    Dices_Validation = []
                    val_sim_losses = []
                    val_dice_losses = []
                    val_smooth_losses = []
                    for batch in test_generator:
                        mov_img, fix_img, *truths = batch
                        mov_img = mov_img.to(device).float()
                        fix_img = fix_img.to(device).float()
                        if loss_type == 'dice':
                            mov_lab, fix_lab = truths
                            mov_lab = mov_lab.to(device).float()
                            fix_lab = fix_lab.to(device).float()
                        else:
                            truth_img = truths[0].to(device).float()

                        V_xy = model(mov_img, fix_img)
                        _, warped_mov = transform(mov_img, V_xy.permute(0, 2, 3, 1))

                        # default zero tensors for missing components
                        v_loss1 = torch.tensor(0.0, device=device)
                        v_loss2 = torch.tensor(0.0, device=device)
                        v_loss5 = loss_smooth(V_xy)

                        if loss_type == 'dice':
                            _, warped_lab = transform(mov_lab, V_xy.permute(0, 2, 3, 1), mod='nearest')
                            v_loss2 = Dice().loss(fix_lab, warped_lab)
                            # compute per-sample Dice metric for logging
                            for bs_index in range(mov_img.size(0)):
                                warped_np = warped_lab[bs_index,...].data.cpu().numpy().copy()
                                fix_np = fix_lab[bs_index,...].data.cpu().numpy().copy()
                                dice_bs = dice(warped_np, fix_np)
                                Dices_Validation.append(dice_bs)
                        else:
                            v_loss1 = loss_similarity(truth_img, warped_mov)

                        val_sim_losses.append(v_loss1.item())
                        val_dice_losses.append(v_loss2.item())
                        val_smooth_losses.append(v_loss5.item())

                    # aggregate validation metrics
                    csv_dice = np.mean(Dices_Validation) if len(Dices_Validation) > 0 else float('nan')
                    csv_sim = float(np.mean(val_sim_losses)) if len(val_sim_losses) > 0 else float('nan')
                    csv_smooth = float(np.mean(val_smooth_losses)) if len(val_smooth_losses) > 0 else float('nan')

                    if loss_type == 'dice':
                        modelname = 'DiceVal_{:.4f}_Step_{:09d}.pth'.format(csv_dice, step)
                    else:
                        modelname = 'SimVal_{:.4f}_Step_{:09d}.pth'.format(csv_sim, step)

                    save_checkpoint(model.state_dict(), model_dir_pth, modelname)
                    np.save(model_dir / 'Loss.npy', lossall)
                    f = open(csv_name, 'a')
                    with f:
                        writer = csv.writer(f)
                        writer.writerow([step, csv_dice, csv_sim, csv_smooth])
                model.train()
            if (step * 10 % n_checkpoint == 0):
                sample_path = model_dir / '{:08d}-images.jpg'.format(step)
                save_flow(mov_img, fix_img, warped_mov, grid.permute(0, 3, 1, 2), sample_path)
            step += 1

            if step > iteration:
                break
        print("one epoch pass")
    np.save(model_dir / 'Loss.npy', lossall)

def save_flow(X, Y, X_Y, f_xy, sample_path):
    x = X.data.cpu().numpy()
    y = Y.data.cpu().numpy()
    x_pred = X_Y.data.cpu().numpy()
    x_pred = x_pred[0]
    x = x[0]
    y = y[0]
    
    flow = f_xy.data.cpu().numpy()
    op_flow =flow[0]

    plt.subplots(figsize=(7, 4))
    plt.subplot(231)
    plt.imshow(x[0], cmap='gray', vmin=0, vmax=1)
    plt.axis('off')
    plt.subplot(232)
    plt.imshow(y[0], cmap='gray', vmin=0, vmax=1)
    plt.axis('off')
    plt.subplot(233)
    plt.imshow(x_pred[0], cmap='gray', vmin=0, vmax=1)
    plt.axis('off')
    plt.subplot(234)

    interval = 7
    for i in range(0,op_flow.shape[1]-1,interval):
        plt.plot(op_flow[0,i], op_flow[1,i],c='g',lw=1)
    #plot the vertical lines
    for i in range(0,op_flow.shape[2]-1,interval):
        plt.plot(op_flow[0,:,i], op_flow[1,:,i],c='g',lw=1)
    plt.xlim(-1, 1)
    plt.ylim(-1, 1)
    plt.axis('off')
    plt.subplot(235)
    plt.imshow(abs(x[0]-y[0]), cmap='gray', vmin=0, vmax=1)
    plt.axis('off')
    plt.subplot(236)
    plt.imshow(abs(x_pred[0]-y[0]), cmap='gray', vmin=0, vmax=1)
    plt.axis('off')
    plt.savefig(sample_path,bbox_inches='tight')
    plt.close()
train()
