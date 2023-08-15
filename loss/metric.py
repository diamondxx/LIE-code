import torch
from skimage.metrics import peak_signal_noise_ratio as PSNR
from skimage.metrics import structural_similarity as SSIM
import numpy as np
import cv2
import torch.nn.functional as F
from math import exp
from loss.loss import l2_loss
from sklearn.metrics import mean_squared_error


def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size//2)**2/float(2*sigma**2)) for x in range(window_size)])
    return gauss/gauss.sum()


def create_window_3d(window_size, channel=1):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t())
    _3D_window = _2D_window.unsqueeze(2) @ (_1D_window.t())
    window = _3D_window.expand(1, channel, window_size, window_size, window_size).contiguous().cuda()
    return window


def ssim_matlab(img1, img2, window_size=11, window=None, size_average=True, full=False, val_range=None):
    # Value range can be different from 255. Other common ranges are 1 (sigmoid) and 2 (tanh).
    if val_range is None:
        if torch.max(img1) > 128:
            max_val = 255
        else:
            max_val = 1

        if torch.min(img1) < -0.5:
            min_val = -1
        else:
            min_val = 0
        L = max_val - min_val
    else:
        L = val_range

    padd = 0
    (_, _, height, width) = img1.size()
    if window is None:
        real_size = min(window_size, height, width)
        window = create_window_3d(real_size, channel=1).to(img1.device)

    img1 = img1.unsqueeze(1)
    img2 = img2.unsqueeze(1)

    mu1 = F.conv3d(F.pad(img1, (5, 5, 5, 5, 5, 5), mode='replicate'), window, padding=padd, groups=1)
    mu2 = F.conv3d(F.pad(img2, (5, 5, 5, 5, 5, 5), mode='replicate'), window, padding=padd, groups=1)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv3d(F.pad(img1 * img1, (5, 5, 5, 5, 5, 5), 'replicate'), window, padding=padd, groups=1) - mu1_sq
    sigma2_sq = F.conv3d(F.pad(img2 * img2, (5, 5, 5, 5, 5, 5), 'replicate'), window, padding=padd, groups=1) - mu2_sq
    sigma12 = F.conv3d(F.pad(img1 * img2, (5, 5, 5, 5, 5, 5), 'replicate'), window, padding=padd, groups=1) - mu1_mu2

    C1 = (0.01 * L) ** 2
    C2 = (0.03 * L) ** 2

    v1 = 2.0 * sigma12 + C2
    v2 = sigma1_sq + sigma2_sq + C2
    cs = torch.mean(v1 / v2)  # contrast sensitivity

    ssim_map = ((2 * mu1_mu2 + C1) * v1) / ((mu1_sq + mu2_sq + C1) * v2)

    if size_average:
        ret = ssim_map.mean()
    else:
        ret = ssim_map.mean(1).mean(1).mean(1)

    if full:
        return ret, cs
    return ret


def quantize(img, rgb_range=255):
    return img.mul(255 / rgb_range).clamp(0, 255).round()


# #######################################
# v2 adapted RAMNet
def ssim(output, target):
    y_input = output.data.cpu().numpy()
    y_target = target.cpu().numpy()

    N, C, H, W = y_input.shape
    assert(C == 1 or C == 3)
    # N x C x H x W -> N x W x H x C -> N x H x W x C
    y_input = np.swapaxes(y_input, 1, 3)
    y_input = np.swapaxes(y_input, 1, 2)
    y_target = np.swapaxes(y_target, 1, 3)
    y_target = np.swapaxes(y_target, 1, 2)
    sum_structural_similarity_over_batch = 0.
    for i in range(N):
        if C == 3:
            sum_structural_similarity_over_batch += SSIM(
                y_input[i, :, :, :], y_target[i, :, :, :], multichannel=True, gaussian_weights=True)
        else:
            sum_structural_similarity_over_batch += SSIM(
                y_input[i, :, :, 0], y_target[i, :, :, 0])

    return sum_structural_similarity_over_batch / float(N)
##########################################################


def psnr(output, target):
    with torch.no_grad():
        output_ = output.data.cpu().numpy()
        target_ = target.cpu().numpy()
        res = 0
        for i in range(target.shape[0]):
            res += PSNR(output_[i], target_[i])

        return res / target.shape[0]












