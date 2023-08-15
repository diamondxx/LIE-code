import os
from PIL import Image
import numpy as np


def data_aug(input_path, save_path, current_scene_num):
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    scene_dir = np.sort(os.listdir(input_path))
    for f in scene_dir:
        print(f)

        image_path = os.path.join(input_path, f, 'image')
        gt_path = os.path.join(input_path, f, 'gt')
        voxel_path = os.path.join(input_path, f, 'voxels')

        # 90
        current_scene_num += 1
        image_save_path_90 = os.path.join(save_path, 'scene{:04d}'.format(current_scene_num), 'image')
        gt_save_path_90 = os.path.join(save_path, 'scene{:04d}'.format(current_scene_num), 'gt')
        voxel_save_path_90 = os.path.join(save_path, 'scene{:04d}'.format(current_scene_num), 'voxels')

        # 180
        current_scene_num += 1
        image_save_path_180 = os.path.join(save_path, 'scene{:04d}'.format(current_scene_num), 'image')
        gt_save_path_180 = os.path.join(save_path, 'scene{:04d}'.format(current_scene_num), 'gt')
        voxel_save_path_180 = os.path.join(save_path, 'scene{:04d}'.format(current_scene_num), 'voxels')

        # up and down
        current_scene_num += 1
        image_save_path_ud = os.path.join(save_path, 'scene{:04d}'.format(current_scene_num), 'image')
        gt_save_path_ud = os.path.join(save_path, 'scene{:04d}'.format(current_scene_num), 'gt')
        voxel_save_path_ud = os.path.join(save_path, 'scene{:04d}'.format(current_scene_num), 'voxels')


        if not os.path.exists(image_save_path_90):
            os.makedirs(image_save_path_90)
            os.makedirs(gt_save_path_90)
            os.makedirs(voxel_save_path_90)

            os.makedirs(image_save_path_180)
            os.makedirs(gt_save_path_180)
            os.makedirs(voxel_save_path_180)

            os.makedirs(image_save_path_ud)
            os.makedirs(gt_save_path_ud)
            os.makedirs(voxel_save_path_ud)

        # loop image
        for ff in np.sort(os.listdir(image_path)):
            file = os.path.join(image_path, ff)
            image = Image.open(file)
            image_90 = image.transpose(Image.ROTATE_90)
            image_180 = image.transpose(Image.ROTATE_180)
            image_ud = image.transpose(Image.FLIP_TOP_BOTTOM)

            # save per image
            image_90.save(os.path.join(image_save_path_90, ff))
            image_180.save(os.path.join(image_save_path_180, ff))
            image_ud.save(os.path.join(image_save_path_ud, ff))

        # loop gt
        for ff in np.sort(os.listdir(gt_path)):
            file = os.path.join(gt_path, ff)
            gt = Image.open(file)
            gt_90 = gt.transpose(Image.ROTATE_90)
            gt_180 = gt.transpose(Image.ROTATE_180)
            gt_ud = gt.transpose(Image.FLIP_TOP_BOTTOM)

            # save per gt
            gt_90.save(os.path.join(gt_save_path_90, ff))
            gt_180.save(os.path.join(gt_save_path_180, ff))
            gt_ud.save(os.path.join(gt_save_path_ud, ff))

        # loop event
        for ff in np.sort(os.listdir(voxel_path)):
            file = os.path.join(voxel_path, ff)
            event = np.load(file)

            event_90 = []
            event_180 = []
            event_ud = []
            # 分片旋转
            for i in range(event.shape[0]):
                event_90.append(np.rot90(event[i], 1))
                event_180.append(np.rot90(event[i], 2))  # 逆时针180度
                event_ud.append(np.flip(event[i], axis=0))  # 上下翻转

            event_90 = np.array(event_90)
            event_180 = np.array(event_180)
            event_ud = np.array(event_ud)
            np.save(os.path.join(voxel_save_path_90, ff), event_90)
            np.save(os.path.join(voxel_save_path_180, ff), event_180)
            np.save(os.path.join(voxel_save_path_ud, ff), event_ud)


if __name__ == '__main__':
    input_path = '../data/dataset_voxel_0717/train_outdoor'
    save_path = '../data/dataset_voxel_0717/aug_data'
    init_scene_num = 744
    data_aug(input_path, save_path, init_scene_num)

    # change scene num name
    # file = np.sort(os.listdir(input_path))
    # for f in file:
    #     scene_path = os.path.join(input_path, f)
    #     num = int(scene_path[-4:]) + 300
    #
    #     os.rename(scene_path, input_path + '/scene{:04d}'.format(num))

    print("process end!")