import numpy as np
from PIL import Image
import os
from utils.timers import Timer
from utils.events_to_voxelgrid import *
import glob
from utils.util import *
from torchvision.utils import make_grid
from logger import TensorboardWriter
from torchvision.transforms import Resize
import cv2


class RealLowlightDataset:
    def __init__(self, base_folder, config):

        self.path_to_voxels_file = os.path.join(base_folder, 'voxels')
        self.path_to_image_file = os.path.join(base_folder, 'image')
        self.path_to_GT_file = os.path.join(base_folder, 'gt')
        self.event_rgb_separate_input = config['event_rgb_separate_input']

        self.bin_num = config['bin_num']
        self.width = config['width']
        self.height = config['height']
        self.voxels_num = len(os.listdir(self.path_to_voxels_file))
        self.image_num = len(os.listdir(self.path_to_image_file))  # the same as gt
        self.crop = config['is_crop']
        self.crop_size = config['crop_size']

    def __len__(self):
        if self.voxels_num > self.image_num:
            return self.image_num
        else:
            return self.voxels_num

    def __getitem__(self, i):
        # read event voxels
        event_numpy = np.load(os.path.join(self.path_to_voxels_file, '{:04d}.npy'.format(i + 1)))
        event_tensor = torch.from_numpy(event_numpy).type(torch.float32)
        torch_resize = Resize([self.crop_size, self.crop_size])
        event_tensor = torch_resize(event_tensor)


        # event denoise 2 normalized
        # begin
        nonzero_ev = (event_tensor != 0)
        num_nonzeros = nonzero_ev.sum()
        if num_nonzeros > 0:
            # compute mean and stddev of the **nonzero** elements of the event tensor
            # we do not use PyTorch's default mean() and std() functions since it's faster
            # to compute it by hand than applying those funcs to a masked array
            mean = event_tensor.sum() / num_nonzeros
            stddev = torch.sqrt((event_tensor ** 2).sum() / num_nonzeros - mean ** 2)
            mask = nonzero_ev.float()
            event_tensor = mask * (event_tensor - mean) / stddev
        # end

        image = Image.open(os.path.join(self.path_to_image_file, '{:04d}.png'.format(i + 1)))  # H W 3
        gt = Image.open(os.path.join(self.path_to_GT_file, '{:04d}.png'.format(i + 1)))  # H W 3

        image = image.resize((self.crop_size, self.crop_size))
        gt = gt.resize((self.crop_size, self.crop_size))

        gts = []
        gts.append(gt.resize((int(self.crop_size / 16), int(self.crop_size / 16))))
        gts.append(gt.resize((int(self.crop_size / 8), int(self.crop_size / 8))))
        gts.append(gt.resize((int(self.crop_size / 4), int(self.crop_size / 4))))
        gts.append(gt.resize((int(self.crop_size / 2), int(self.crop_size / 2))))
        gts.append(gt)
        images = []
        images.append(image.resize((int(self.crop_size / 16), int(self.crop_size / 16))))
        images.append(image.resize((int(self.crop_size / 8), int(self.crop_size / 8))))
        images.append(image.resize((int(self.crop_size / 4), int(self.crop_size / 4))))
        images.append(image.resize((int(self.crop_size / 2), int(self.crop_size / 2))))
        images.append(image)

        image_tensors = []
        GT_tensors = []
        for i in range(len(gts)):
            GT_tensors.append(torch.from_numpy(np.transpose(np.array(gts[i]), (2, 0, 1))).type(
                torch.float32) / 255.0)  # 3 H W
            image_tensors.append(quick_norm(torch.from_numpy(np.transpose(np.array(images[i]), (2, 0, 1))).type(
                torch.float32)))  # 3 H W

        image_tensor = quick_norm(torch.from_numpy(np.transpose(np.array(image), (2, 0, 1))).type(torch.float32))  # 3 H W
        GT_tensor = torch.from_numpy(np.transpose(np.array(gt), (2, 0, 1))).type(torch.float32) / 255.0  # 3 H W

        if self.event_rgb_separate_input:
            return image_tensor, event_tensor, GT_tensor, image_tensors, GT_tensors   # channels: 3 5 3
        else:
            return torch.cat((image_tensor, event_tensor), dim=0), GT_tensor  # channels: 8 3


class RealLowlightDataset_test:
    def __init__(self, base_folder):

        self.path_to_image_file = os.path.join(base_folder, 'image')
        self.path_to_GT_file = os.path.join(base_folder, 'gt')
        self.zx = os.path.join(base_folder, 'voxels')
        self.voxels_num = len(os.listdir(self.path_to_voxels_file))
        self.image_num = len(os.listdir(self.path_to_image_file))  # the same as gt
        self.crop_size = 256

    def __len__(self):
        if self.voxels_num > self.image_num:
            return self.image_num
        else:
            return self.voxels_num

    def __getitem__(self, idx):
        image = Image.open(os.path.join(self.path_to_image_file, '{:04d}.png'.format(idx+1)))  # H W 3
        gt = Image.open(os.path.join(self.path_to_GT_file, '{:04d}.png'.format(idx+1)))  # H W 3
        image = image.resize((256, 256))
        gt = gt.resize((256, 256))

        gts = []
        gts.append(gt.resize((int(self.crop_size / 16), int(self.crop_size / 16))))
        gts.append(gt.resize((int(self.crop_size / 8), int(self.crop_size / 8))))
        gts.append(gt.resize((int(self.crop_size / 4), int(self.crop_size / 4))))
        gts.append(gt.resize((int(self.crop_size / 2), int(self.crop_size / 2))))
        gts.append(gt)
        images = []
        images.append(image.resize((int(self.crop_size / 16), int(self.crop_size / 16))))
        images.append(image.resize((int(self.crop_size / 8), int(self.crop_size / 8))))
        images.append(image.resize((int(self.crop_size / 4), int(self.crop_size / 4))))
        images.append(image.resize((int(self.crop_size / 2), int(self.crop_size / 2))))
        images.append(image)

        image_tensors = []
        GT_tensors = []
        for i in range(len(gts)):
            GT_tensors.append(torch.from_numpy(np.transpose(np.array(gts[i]), (2, 0, 1))).type(
                torch.float32) / 255.0)  # 3 H W
            image_tensors.append(quick_norm(torch.from_numpy(np.transpose(np.array(images[i]), (2, 0, 1))).type(
                torch.float32)))  # 3 H W

        image_tensor_orig = torch.from_numpy(np.transpose(np.array(image), (2, 0, 1))).type(torch.float32)
        image_tensor = quick_norm(image_tensor_orig)  # 3 H W
        GT_tensor = torch.from_numpy(np.transpose(np.array(gt), (2, 0, 1))).type(torch.float32) / 255.0  # 3 H W

        event_voxel = np.load(os.path.join(self.path_to_voxels_file, '{:04d}.npy'.format(idx+1)))
        event_tensor = torch.from_numpy(event_voxel).type(torch.float32)
        torch_resize = Resize([256, 256])
        event_tensor = torch_resize(event_tensor)
        
        # begin
        nonzero_ev = (event_tensor != 0)
        num_nonzeros = nonzero_ev.sum()
        if num_nonzeros > 0:
            # compute mean and stddev of the **nonzero** elements of the event tensor
            # we do not use PyTorch's default mean() and std() functions since it's faster
            # to compute it by hand than applying those funcs to a masked array
            mean = event_tensor.sum() / num_nonzeros
            stddev = torch.sqrt((event_tensor ** 2).sum() / num_nonzeros - mean ** 2)
            mask = nonzero_ev.float()
            event_tensor = mask * (event_tensor - mean) / stddev
        # end

        return image_tensor_orig, image_tensor, event_tensor, GT_tensor, image_tensors, GT_tensors

