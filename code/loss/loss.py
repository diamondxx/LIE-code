import torch.nn.functional as F
import torch.nn
import numpy as np
from skimage.metrics import mean_squared_error
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import peak_signal_noise_ratio as PSNRp
from skimage.metrics import structural_similarity as SSIM_skimage
import loss.lpips.lpips as lpips__


def psnr_loss(output, target):
    return -psnr(output, target)


def mse_loss(output, target):
    # l2 loss
    return mean_squared_error(output, target)


def MSE_loss(output, target):
    mse = torch.nn.MSELoss()
    return mse(output, target)


def l1_loss(output, target):
    return torch.mean(torch.abs(output - target))


def l2_loss(output, target):
    return torch.mean(torch.square(output - target))


def SSIM_MSE(output, target):
    output_ = output.transpose(1, 3).transpose(1, 2).detach().cpu().numpy()
    target_ = target.transpose(1, 3).transpose(1, 2).detach().cpu().numpy()
    mse = torch.nn.MSELoss()
    return mse(output, target) + (1 - SSIM_skimage(output_, target_, multichannel=True))


def lpips(output, target):
    loss_fn = lpips__.LPIPS(net='vgg')
    device = torch.device("cuda")
    loss_fn = loss_fn.to(device)
    loss = loss_fn.forward(output, target)
    return loss

def lpips_l1_simple(output, target, imgs, loss_fn):
    loss1 = l1_loss(output, target)
    lpips_loss = lpips(output, target)
    loss2 = 0.
    for l in lpips_loss:
        loss2 += torch.sum(l)
    loss = 2 * loss1 + 0.1 * loss2

    return loss


def lpips_l1(multi_output, multi_targets, multi_imgs, loss_fn):

    output = multi_output[-1]
    target = multi_targets[-1]
    imgs = multi_imgs[-1]

    loss_lpips_pos = loss_fn.forward(output, target)
    loss_lpips_neg = loss_fn.forward(output, torch.sum(imgs, dim=1).unsqueeze(1))

    miu = 0.1
    loss_lpips = 0.
    slgma2 = [1.0, 1.0, 1.0, 1.0, 1.0]

    for i in range(len(loss_lpips_pos)):
        tmp = 0.
        for j in range(len(loss_lpips_pos[i])):
            tmp += slgma2[i] * (torch.sum(loss_lpips_pos[i][j]) / torch.sum(loss_lpips_neg[i][j]))
        tmp = tmp / output.shape[0]
        loss_lpips += tmp

    return 2 * l1_loss(output, target) + miu * loss_lpips


def lpips_l1_multi(multi_output, targets, imgs, loss_fn):

    loss_lpips_pos = []
    loss_lpips_neg = []
    for i in range(len(multi_output)):
        loss_lpips_pos.append(loss_fn.forward(multi_output[i], targets[i]))
        loss_lpips_neg.append(loss_fn.forward(multi_output[i], torch.sum(imgs[i], dim=1).unsqueeze(1)))

    a = 0.1
    loss_lpips = 0.
    total_loss_lpips = 0.
    total_loss_l1 = 0.
    miu2 = [1/64, 1/32, 1/8, 1/2, 1]
    slgma2 = [1.0, 1.0, 1.0, 1.0, 1.0]

    for j in range(len(loss_lpips_pos)):
        for i in range(len(loss_lpips_pos[0])):
            loss_lpips += slgma2[i] * (torch.sum(loss_lpips_pos[j][i]) / torch.sum(loss_lpips_neg[j][i]))
        total_loss_lpips += miu2[j] * loss_lpips
        total_loss_l1 += miu2[j] * l1_loss(multi_output[j], targets[j])

    print(2 * total_loss_l1 + a * total_loss_lpips)
    return 2 * total_loss_l1 + a * total_loss_lpips

# changed
def lpips_l1_multi_new(multi_output, targets, imgs, loss_fn):

    loss_lpips_pos = []
    loss_lpips_neg = []
    for i in range(len(multi_output)):
        loss_lpips_pos.append(loss_fn.forward(multi_output[i], targets[i]))
        loss_lpips_neg.append(loss_fn.forward(multi_output[i], torch.sum(imgs[i], dim=1).unsqueeze(1)))

    b = 2
    a = 0.1
    loss_lpips = 0.
    total_loss_lpips = 0.
    total_loss_l1 = 0.
    miu2 = [1/64, 1/32, 1/8, 1/2, 1]
    slgma7 = [1.0, 1.0, 1.0, 1.0, 1.0]

    for j in range(len(loss_lpips_pos)):
        for i in range(len(loss_lpips_pos[0])):
            tmp = 0.
            for s in range(len(loss_lpips_pos[j][i])):
                 tmp += slgma7[i] * (loss_lpips_pos[j][i][s] / loss_lpips_neg[j][i][s])
            tmp = tmp / len(loss_lpips_pos[j][i])
            loss_lpips += tmp
        total_loss_lpips += miu2[j] * loss_lpips
        total_loss_l1 += miu2[j] * l1_loss(multi_output[j], targets[j])

    return b * total_loss_l1 + a * total_loss_lpips


def lpips_l1_multi_wo_FP(multi_output, targets, imgs, loss_fn):

    loss_lpips_pos = []
    loss_lpips_neg = []
    for i in range(len(multi_output)):
        loss_lpips_pos.append(loss_fn.forward(multi_output[i], targets[4]))
        loss_lpips_neg.append(loss_fn.forward(multi_output[i], torch.sum(imgs[4], dim=1).unsqueeze(1)))

    a = 0.1
    loss_lpips = 0.
    total_loss_lpips = 0.
    total_loss_l1 = 0.
    miu2 = [1 / 64, 1 / 32, 1 / 8, 1 / 2, 1]
    slgma2 = [1.0, 1.0, 1.0, 1.0, 1.0]

    for j in range(len(loss_lpips_pos)):
        for i in range(len(loss_lpips_pos[0])):
            loss_lpips += slgma2[i] * (torch.sum(loss_lpips_pos[j][i]) / torch.sum(loss_lpips_neg[j][i]))
        total_loss_lpips += miu2[j] * loss_lpips
        total_loss_l1 += miu2[j] * l1_loss(multi_output[j], targets[4])

    return 2 * total_loss_l1 + a * total_loss_lpips


def lpips_l1_multi_wo_MCD(multi_output, targets, imgs, loss_fn):

    total_loss_l1 = 0.
    miu2 = [1 / 64, 1 / 32, 1 / 8, 1 / 2, 1]
    for i in range(len(multi_output)):
        total_loss_l1 += miu2[i] * l1_loss(multi_output[i], targets[i])

    return total_loss_l1