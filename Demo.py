from __future__ import print_function
import matplotlib
matplotlib.use('TkAgg') 
import matplotlib.pyplot as plt
# %matplotlib inline

import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '3'
from models.Unet_new import UNet_new

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

import numpy as np
from models import *

import torch
import torch.optim
import torch.nn.functional as F

from scipy.io import loadmat
from scipy.io import savemat
import scipy.io as sio

# from skimage.measure import compare_psnr
from skimage.metrics import peak_signal_noise_ratio as compare_psnr
from utils.sr_utils import *
from utils.denoising_utils import *
from utils.de_artifacts import *
import cv2


torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark =True
dtype = torch.cuda.FloatTensor
# dtype = torch.FloatTensor

imsize = -1
PLOT = True

fname1 = 'data/demo/simulation_cell_test/supplemnet sim/Ph_replica_10.mat'
fname1 = loadmat(fname1)
mat_data1 = fname1['Ph_replica']
img_np1 = np.array(mat_data1)
img_np1 = img_np1.astype(np.float32)
img_np1_show = np.expand_dims(img_np1, axis=0)

fname2 = 'data/demo/simulation_cell_test/supplemnet sim/Phase_re_deconv.mat'
fname2 = loadmat(fname2)
mat_data2 = fname2['Phase_re_deconv']
img_np2 = np.array(mat_data2)
img_np2 = img_np2.astype(np.float32)
phase_re_deconv = np_to_torch(img_np2).unsqueeze(0)

phase_ge = np_to_torch(img_np1).unsqueeze(0)

distance = np.array([0, 10])
## setup
INPUT = 'noise'  # 'meshgrid'
pad = 'reflection'
OPT_OVER = 'net'  # 'net,input'

reg_noise_std = 1. / 30.
LR = 0.01
tv_weight = 0.00000 # 0.00001

OPTIMIZER = 'adam'  # 'LBFGS'
show_every = 100
check_every = 100
exp_weight = 0.0

num_iter = 2000
input_depth = 1
figsize = 5
row = img_np1.shape
row = row[1]
col = img_np1.shape
col = col[1]

## net
net = get_net(input_depth, 'skip', pad,
              skip_n33d=128,
              skip_n33u=128,
              skip_n11=4,
              num_scales=5,
              upsample_mode='bilinear').type(dtype)
net_input = get_noise(input_depth, INPUT, (row, col)).type(dtype).detach()
# net_input = phase_re_deconv.type(dtype).detach()

s = sum([np.prod(list(p.size())) for p in net.parameters()])
print('Number of params: %d' % s)

# Loss
mse = torch.nn.MSELoss().type(dtype)
l1_loss = torch.nn.L1Loss().type(dtype)

## Optimize process
net_input_saved = net_input.detach().clone()
noise = net_input.detach().clone()
out_avg = None
last_net = None
psrn_ge_last = 0
i = 0

phase_ge_GPU = phase_ge.type(dtype).detach()
distance_torch = np_to_torch(distance).type(dtype).detach()
row_torch = torch.tensor(row, dtype=torch.float32).type(dtype).detach()
col_torch = torch.tensor(col, dtype=torch.float32).type(dtype).detach()
def closure():
    global i, out_avg, psrn_ge_last, last_net, net_input

    if reg_noise_std > 0:
        net_input = net_input_saved + (noise.normal_() * reg_noise_std)

    out = net(net_input)
    if out_avg is None:
        out_avg = out.detach()
    else:
        out_avg = out_avg * exp_weight + out.detach() * (1 - exp_weight)

    result_forward = Forward_operator_double(out, distance_torch, row_torch, col_torch)

    # total_loss = mse(result_forward, phase_ge_GPU)
    total_loss = 0.9 * mse(result_forward, phase_ge_GPU) + 0.1 * l1_loss(result_forward, phase_ge_GPU)
    # total_loss = l1_loss(result_forward, phase_ge_GPU)

    if tv_weight > 0:
        total_loss += tv_weight * tv_loss(out)

    total_loss.backward()
    img_np1_e = np.expand_dims(img_np1, axis=0)
    # img_np2_e = np.expand_dims(img_np2, axis=0)
    psrn_ge = compare_psnr(img_np1_e, result_forward.detach().cpu().numpy()[0], data_range=1)
    # psrn_gt = compare_psnr(img_np1_e, out.detach().cpu().numpy()[0], data_range=1)
    # psrn_gt_sm = compare_psnr(img_np1_e, out_avg.detach().cpu().numpy()[0], data_range=1)

    print('Iteration %05d    Loss %f   PSNR_ge: %f  ' % (
        i, total_loss.item(), psrn_ge))

    if PLOT and i % show_every == 0:
        out_np = torch_to_np(out)
        plot_image_grid([np.clip(out_np, 0, 1),
                         np.clip(torch_to_np(out_avg), 0, 1)], factor=figsize, nrow=1)

    # Backtracking
    if i % check_every:
        if psrn_ge - psrn_ge_last < -5:
            print('Falling back to previous checkpoint.')

            for new_param, net_param in zip(last_net, net.parameters()):
                net_param.data.copy_(new_param.cuda())

            return total_loss * 0
        else:
            last_net = [x.detach().cpu() for x in net.parameters()]
            psrn_ge_last = psrn_ge

    i += 1

    return total_loss

p = get_params(OPT_OVER, net, net_input)
optimize(OPTIMIZER, p, closure, LR, num_iter)
out_np = torch_to_np(net(net_input))
out_np_save = out_np.squeeze(0)
savemat('./result/simulation_cell_test/New/0-100/result.mat', {'result': out_np_save})  # New/0-1
plotout_p = (out_np_save - np.min(out_np_save)) / (np.max(out_np_save) - np.min(out_np_save))
phase = np.array(plotout_p)
phase = phase.astype('float32') * 255
cv2.imwrite("./result/simulation_cell_test/New/0-100/result.bmp", phase)

q = plot_image_grid_final([np.clip(out_np, 0, 1), img_np1_show], factor=13)




