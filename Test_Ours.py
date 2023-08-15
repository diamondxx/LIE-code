import os
import argparse
from tqdm import tqdm
import loss.loss as module_loss
import loss.metric as module_metric
from torch.utils.data import DataLoader, ConcatDataset
import loss.lpips.lpips as lpips__
from dataset.RealLowlightDataset import RealLowlightDataset_test
from model import *
import shutil
from utils.util import *
import cv2
from torchvision.utils import make_grid, save_image
from PIL import Image
import torch


def main(config):
    display_path = config.save
    if os.path.exists(display_path):
        shutil.rmtree(display_path)
    os.makedirs(display_path)


    # build model architecture, then print to console
    model = pyramidTransformer(input_c_frame=3, input_c_event=5, output_c=3)
    print('Loading checkpoint: {} ...'.format(config.resume))
    checkpoint = torch.load(config.resume)
    state_dict = checkpoint['state_dict']

    device = torch.device('cuda:0')
    device_ids = [0, 1, 2, 3]

    if len(device_ids) > 1:
        model = torch.nn.DataParallel(model, device_ids=device_ids)
    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()

    # get function handles of loss and metrics
    loss_fn = lpips__.LPIPS(net='vgg')
    loss_fn = loss_fn.to(device)
    criterion = getattr(module_loss, 'lpips_l1_multi')  # return a function method
    metrics = [getattr(module_metric, met) for met in ['ssim', 'psnr']]

    total_loss = 0.
    total_psnr = 0.
    total_ssim = 0.
    total_lpips = 0.
    num = 0

    txt_path = config.save + 'result.txt'
    file = open(txt_path, 'w+', encoding='utf-8')

    scene_folder = np.sort(os.listdir(config.data))
    for f in scene_folder:
        print("Current test scene is: {}".format(f))
        file.write('Current test scene is: {}'.format(f))
        file.write('\n')

        scene_path = os.path.join(config.data, f)
        save_path = os.path.join(config.save, f)
        save_path_total = os.path.join(save_path, 'total')
        save_path_event = os.path.join(save_path, 'event')
        save_path_image = os.path.join(save_path, 'image')
        save_path_pred = os.path.join(save_path, 'pred')
        save_path_gt = os.path.join(save_path, 'gt')
        if not os.path.exists(save_path):
            os.makedirs(save_path)
            os.makedirs(save_path_total)
            os.makedirs(save_path_event)
            os.makedirs(save_path_image)
            os.makedirs(save_path_pred)
            os.makedirs(save_path_gt)

        # setup dataset instances
        test_dataset = RealLowlightDataset_test(scene_path)
        test_data_loader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False, num_workers=4)

        tloss = 0.
        total_metrics = torch.zeros(len(metrics))
        metric_lpips = 0.
        n_samples = 0
        tmp_psnr = 0.
        tmp_ssim = 0.

        with torch.no_grad():
            for batch_idx, (data_orig, data, event, gt, images, gts) in enumerate(tqdm(test_data_loader)):
                data_orig, data, event, gt = data_orig.to(device), data.to(device), event.to(device), gt.to(device)
                for i in range(len(images)):
                    images[i], gts[i] = images[i].to(device), gts[i].to(device)

                milti_output, output = model(data, event)

                # # image filter
                # output[:, :1, :, :] = unsharp_mask_filter(output[:, :1, :, :])
                # output[:, 1:2, :, :] = unsharp_mask_filter(output[:, 1:2, :, :])
                # output[:, 2:3, :, :] = unsharp_mask_filter(output[:, 2:3, :, :])
                # output = image_filter(output).unsqueeze(dim=0)

                # computing loss, metrics on test set
                # loss = criterion(output, gt, data, loss_fn)
                loss = criterion(milti_output, gts, images, loss_fn)
                batch_size = data.shape[0]
                n_samples += batch_size
                tloss += loss.item() * batch_size


                for j, metric in enumerate(metrics):
                    if j == 0:
                        tmp_ssim = metric(output, gt)
                    if j == 1:
                        tmp_psnr = metric(output, gt)
                    total_metrics[j] += metric(output, gt) * batch_size

                tmp_lpips = sum(loss_fn.forward(output, gt)).item()
                metric_lpips += tmp_lpips * batch_size

                # print and write each image
                print('\nImage{:04d}: ssim = {:.4f}, psnr = {:.4f}, lpips = {:.4f}'
                      .format(batch_idx + 1, tmp_ssim, tmp_psnr, tmp_lpips))
                file.write('Image{}: ssim = {:.4f}, psnr = {:.4f},  lpips = {:.4f}'
                           .format(batch_idx + 1, tmp_ssim, tmp_psnr, tmp_lpips))
                file.write('\n')

                batch_size, _, width, height = gt.shape
                events_tmp = torch.zeros((batch_size, 3, width, height))

                for j in range(batch_size):
                    events_ = make_event_preview(event[j].unsqueeze(0))
                    events_ = np.transpose(events_, (2, 0, 1))
                    events_tmp[j] = torch.from_numpy(events_)
                events_tmp = events_tmp.to(device)

                # save output image
                res_tensor = make_grid(make_image(
                    data_orig, events_tmp, output, gt).cpu(), nrow=4, normalize=False)
                save_image(res_tensor, save_path_total + '/{:04d}_psnr_{:.4f}_ssim_{:.4f}_lpips_{:.4f}.png'
                           .format(batch_idx+1, tmp_psnr, tmp_ssim, tmp_lpips))

                save_image(data_orig, save_path_image + '/{:04d}.png'.format(batch_idx+1))
                save_image(events_tmp, save_path_event + '/{:04d}.png'.format(batch_idx+1))
                save_image(output, save_path_pred + '/{:04d}.png'.format(batch_idx+1))
                save_image(gt, save_path_gt + '/{:04d}.png'.format(batch_idx+1))


        log = {'loss': tloss / n_samples}
        log.update({
            met.__name__: total_metrics[s].item() / n_samples for s, met in enumerate(metrics)
        })
        metric_lpips /= n_samples

        # write current test result
        file.write('current num is {}'.format(num))
        file.write('\n')
        print("num is {}".format(num))

        file.write('loss: {:.4f}'.format(log['loss']))
        file.write('\n')
        print('loss: {:.4f}'.format(log['loss']))

        file.write('psnr: {:.4f}'.format(log['psnr']))
        file.write('\n')
        print('psnr: {:.4f}'.format(log['psnr']))

        file.write('ssim: {:.4f}'.format(log['ssim']))
        file.write('\n')
        print("ssim: {:04f}".format(log['ssim']))

        file.write('lpips: {:.4f}'.format(metric_lpips))
        file.write('\n')
        print("lpips: {:04f}".format(metric_lpips))

        total_loss += log['loss']
        total_psnr += log['psnr']
        total_ssim += log['ssim']
        total_lpips += metric_lpips
        num += 1

    mean_loss = total_loss / num
    mean_psnr = total_psnr / num
    mean_ssim = total_ssim / num
    mean_lpips = total_lpips / num

    # write final test result
    file.write('\nFinal result:')
    file.write('\n')
    print("\nFinal result:")

    file.write('loss: {:.4f}'.format(mean_loss))
    file.write('\n')
    print("loss: {:.4f}".format(mean_loss))

    file.write('psnr: {:.4f}'.format(mean_psnr))
    file.write('\n')
    print("psnr: {:.4f}".format(mean_psnr))

    file.write('ssim: {:.4f}'.format(mean_ssim))
    file.write('\n')
    print("ssim: {:.4f}".format(mean_ssim))

    file.write('lpips: {:.4f}'.format(mean_lpips))
    file.write('\n')
    print("lpips: {:.4f}".format(mean_lpips))

    file.close()


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='LIE')
    parser.add_argument('-r', '--resume', default='./pretrained/model_indoor.pth', type=str)

    parser.add_argument('-f', '--data', default='./data/LIEDataset/orig_indoor_test', type=str)

    parser.add_argument('-s', '--save', default='./result/display_indoor/', type=str)
    parser.add_argument('-b', '--batch_size', default=1, type=str)
    config = parser.parse_args()


    main(config)

