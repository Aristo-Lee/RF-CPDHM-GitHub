import math
import torch
import torch.optim
import torch.nn.functional as F
import cv2
import numpy as np
import scipy.sparse as sparse
import scipy.sparse.linalg as sl
import matplotlib.pyplot as plt
import cupy as cp
import scipy.sparse as sparse
import scipy.sparse.linalg as sl

dtype = torch.cuda.FloatTensor


def shift_operator(dis, row, col):
    # 生成Nr和Nc的范围
    Nr = np.fft.ifftshift(np.arange(-np.floor(row / 2), np.ceil(row / 2)))
    Nc = np.fft.ifftshift(np.arange(-np.floor(col / 2), np.ceil(col / 2)))

    # 创建网格
    Nc, Nr = np.meshgrid(Nc, Nr)

    # 计算shift操作
    shift = np.exp(1j * 2 * np.pi * ((-dis[0]) * Nr / row + (-dis[1]) * Nc / col))

    # 计算Forward_op
    Forward_op = 1 - shift

    # 对Forward_op应用fftshift
    shift_out = np.fft.fftshift(Forward_op)

    # 输出一个包含实部和虚部的三维数组，形状为(row, col, 2)
    shift_out = np.stack((shift_out.real, shift_out.imag), axis=-1)

    return shift_out

def Forward_operator(input_tensor, dis, row, col):
    device = input_tensor.device
    dis = dis.squeeze(0)

    # 生成 GPU 上的网格
    Nr = torch.fft.ifftshift(torch.arange(
        -torch.floor(row/2), torch.ceil(row/2),
        dtype=torch.float32, device=device
    ))
    Nc = torch.fft.ifftshift(torch.arange(
        -torch.floor(col/2), torch.ceil(col/2),
        dtype=torch.float32, device=device
    ))
    Nr, Nc = torch.meshgrid(Nr, Nc, indexing='ij')

    # 计算复数 shift
    shift = torch.exp(1j * 2 * torch.pi * ((-dis[0]) * Nr / row + (-dis[1]) * Nc / col))
    Forward_op = 1 - shift
    # Forward_op = torch.fft.fftshift(Forward_op)

    # # FFT 和滤波
    complex_tensor = torch.complex(input_tensor, torch.zeros_like(input_tensor))
    input_F = torch.fft.fftshift(torch.fft.fft2(complex_tensor.squeeze(0)))
    input_ge_F = input_F * Forward_op.unsqueeze(0)
    #
    # # 逆 FFT 并取实部
    output = torch.fft.ifft2(torch.fft.ifftshift(input_ge_F))
    output = torch.real(output)  # 使用函数形式避免警告

    return output.unsqueeze(0)

def Forward_operator_double(input_tensor, dis, row, col):
    device = input_tensor.device
    dis = dis.squeeze(0)

    # 生成 GPU 上的网格
    Nr = torch.fft.ifftshift(torch.arange(
        -torch.floor(row/2), torch.ceil(row/2),
        dtype=torch.float32, device=device
    ))
    Nc = torch.fft.ifftshift(torch.arange(
        -torch.floor(col/2), torch.ceil(col/2),
        dtype=torch.float32, device=device
    ))
    Nr, Nc = torch.meshgrid(Nr, Nc, indexing='ij')

    # 计算复数 shift
    shift = torch.exp(1j * 2 * torch.pi * ((dis[0]) * Nr / row + (dis[1]) * Nc / col))
    Forward_op = 1 - shift
    Forward_op = torch.fft.fftshift(Forward_op)

    # # FFT 和滤波
    complex_tensor = torch.complex(input_tensor, torch.zeros_like(input_tensor))
    input_F = torch.fft.fftshift(torch.fft.fft2(complex_tensor.squeeze(0)))
    input_ge_F = input_F * Forward_op.unsqueeze(0)
    #
    # # 逆 FFT 并取实部
    output = torch.fft.ifft2(torch.fft.ifftshift(input_ge_F))
    output = torch.real(output)  # 使用函数形式避免警告

    return output.unsqueeze(0)


# def Forward_operator(input_tensor, dis, row, col):
#     # 提取 dis 的实部和虚部
#     dis = dis.squeeze(0)  # 确保 dis 的形状是 (2, )
#
#     # 生成 Nr 和 Nc 的范围
#     Nr = torch.fft.ifftshift(torch.arange(-torch.floor(row / 2), torch.ceil(row / 2), dtype=torch.float32).type(dtype))
#     Nc = torch.fft.ifftshift(torch.arange(-torch.floor(col / 2), torch.ceil(col / 2), dtype=torch.float32).type(dtype))
#
#     # 创建网格
#     Nr, Nc = torch.meshgrid(Nr, Nc, indexing='ij')
#
#     # 计算 shift
#     shift = torch.exp(1j * 2 * np.pi * ((-dis[0]) * Nr / row + (-dis[1]) * Nc / col))
#
#     # Forward 操作
#     Forward_op = 1 - shift
#
#     # 对 input_tensor 执行 FFT 并 fftshift
#     input_F = torch.fft.fftshift(torch.fft.fft2(input_tensor.squeeze(0)))
#
#     # 将 input_F 与 Forward_op 进行乘法运算
#     input_ge_F = input_F * Forward_op.unsqueeze(0)
#
#     # 对结果执行逆 FFT 并 fftshift
#     output = torch.fft.ifft2(torch.fft.ifftshift(input_ge_F))
#
#     # 取实部
#     output = output.real
#
#     # 重新添加 batch 和 channel 维度
#     output = output.unsqueeze(0)
#
#     return output