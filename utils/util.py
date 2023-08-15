import json
import torch
import pandas as pd
from pathlib import Path
from itertools import repeat
from collections import OrderedDict
import numpy as np
from torchvision.utils import make_grid, save_image


def ensure_dir(dirname):
    dirname = Path(dirname)
    if not dirname.is_dir():
        dirname.mkdir(parents=True, exist_ok=False)

def read_json(fname):
    fname = Path(fname)
    with fname.open('rt') as handle:
        return json.load(handle, object_hook=OrderedDict)

def write_json(content, fname):
    fname = Path(fname)
    with fname.open('wt') as handle:
        json.dump(content, handle, indent=4, sort_keys=False)

def inf_loop(data_loader):
    ''' wrapper function for endless data loader. '''
    for loader in repeat(data_loader):
        yield from loader


def quick_norm(img):
    for i in range(img.shape[0]):
        img[i] = (img[i] - torch.min(img[i])) / (torch.max(img[i]) - torch.min(img[i]) + 1e-5)
    return img

def de_quick_norm(img):
    for i in range(img.shape[0]):
        img[i] = img[i] * (torch.max(img[i]) - torch.min(img[i]) + 1e-5) + torch.min(img[i])
    return img


def make_event_preview(events, mode='red-blue', num_bins_to_show=-1):

    # events: [1 x C x H x W] event tensor
    # mode: 'red-blue' or 'grayscale'
    # num_bins_to_show: number of bins of the voxel grid to show. -1 means show all bins.
    assert(mode in ['red-blue', 'grayscale'])
    if num_bins_to_show < 0:
        sum_events = torch.sum(events[0, :, :, :], dim=0).detach().cpu().numpy()
    else:
        sum_events = torch.sum(events[0, -num_bins_to_show:, :, :], dim=0).detach().cpu().numpy()

    if mode == 'red-blue':
        # Red-blue mode
        # positive events: blue, negative events: red
        event_preview = np.zeros((sum_events.shape[0], sum_events.shape[1], 3), dtype=np.uint8)
        b = event_preview[:, :, 0]
        r = event_preview[:, :, 2]
        b[sum_events > 0] = 255
        r[sum_events < 0] = 255
    else:
        # Grayscale mode
        # normalize event image to [0, 255] for display
        m, M = -10.0, 10.0
        event_preview = np.clip((255.0 * (sum_events - m) / (M - m)).astype(np.uint8), 0, 255)

    # print("event_preview.shape is {}".format(event_preview.shape))

    return event_preview


def prepare_device(n_gpu_use):
    """
    setup GPU device if available. get gpu device indices which are used for DataParallel
    """
    n_gpu = torch.cuda.device_count()
    if n_gpu_use > 0 and n_gpu == 0:
        print("Warning: There\'s no GPU available on this machine,"
              "training will be performed on CPU.")
        n_gpu_use = 0
    if n_gpu_use > n_gpu:
        print(f"Warning: The number of GPU\'s configured to use is {n_gpu_use}, but only {n_gpu} are "
              "available on this machine.")
        n_gpu_use = n_gpu
    device = torch.device('cuda:0' if n_gpu_use > 0 else 'cpu')
    list_ids = list(range(n_gpu_use))
    return device, list_ids

class MetricTracker:
    def __init__(self, *keys, writer=None):
        self.writer = writer
        self._data = pd.DataFrame(index=keys, columns=['total', 'counts', 'average'])
        self.reset()

    def reset(self):
        for col in self._data.columns:
            self._data[col].values[:] = 0

    def update(self, key, value, n=1):
        if self.writer is not None:
            self.writer.add_scalar(key, value)
        # print("1 key is {}".format(key))
        # print("2 value is {}".format(value))
        self._data.total[key] += value * n
        self._data.counts[key] += n
        self._data.average[key] = self._data.total[key] / self._data.counts[key]

    def avg(self, key):
        return self._data.average[key]

    def result(self):
        return dict(self._data.average)



def make_image(data, event, output, target):
    if event is not None:
        events = event / 255.0
    images = data
    if images.shape[1] == 1:
        images = torch.cat((images, images, images), dim=1)
    if events.shape[1] == 1:
        events = torch.cat((events, events, events), dim=1)
    if output.shape[1] == 1:
        output = torch.cat((output, output, output), dim=1)
    if target.shape[1] == 1:
        target = torch.cat((target, target, target), dim=1)

    if events is not None:
        res_tensor = torch.cat((events[0].unsqueeze(0),
                                images[0].unsqueeze(0),
                                output[0].unsqueeze(0),
                                target[0].unsqueeze(0)), dim=0)
    else:
        res_tensor = torch.cat((images[0].unsqueeze(0),
                                output[0].unsqueeze(0),
                                target[0].unsqueeze(0)), dim=0)
    for i in range(len(target)):
        if i == 0:
            continue
        if events is not None:
            events_ = events[i].unsqueeze(0)
        images_ = images[i].unsqueeze(0)
        output_ = output[i].unsqueeze(0)
        target_ = target[i].unsqueeze(0)
        if events is not None:
            res_tensor_ = torch.cat((events_, images_, output_, target_), dim=0)
        else:
            res_tensor_ = torch.cat((images_, output_, target_), dim=0)
        res_tensor = torch.cat((res_tensor, res_tensor_), dim=0)
    return res_tensor


def make_image_three_cat(images, output, target):
    if images.shape[1] == 1:
        images = torch.cat((images, images, images), dim=1)
    if output.shape[1] == 1:
        output = torch.cat((output, output, output), dim=1)
    if target.shape[1] == 1:
        target = torch.cat((target, target, target), dim=1)

    res_tensor = torch.cat((images[0].unsqueeze(0),
                            output[0].unsqueeze(0),
                            target[0].unsqueeze(0)), dim=0)
    for i in range(len(target)):
        if i == 0:
            continue
        images_ = images[i].unsqueeze(0)
        output_ = output[i].unsqueeze(0)
        target_ = target[i].unsqueeze(0)

        res_tensor_ = torch.cat((images_, output_, target_), dim=0)
        res_tensor = torch.cat((res_tensor, res_tensor_), dim=0)
    return res_tensor


def make_image_two_cat(images, output):
    if images.shape[1] == 1:
        images = torch.cat((images, images, images), dim=1)
    if output.shape[1] == 1:
        output = torch.cat((output, output, output), dim=1)
    res_tensor = torch.cat((images[0].unsqueeze(0),
                            output[0].unsqueeze(0)), dim=0)
    for i in range(len(output)):
        if i == 0:
            continue
        images_ = images[i].unsqueeze(0)
        output_ = output[i].unsqueeze(0)

        res_tensor_ = torch.cat((images_, output_), dim=0)
        res_tensor = torch.cat((res_tensor, res_tensor_), dim=0)
    return res_tensor
