import numpy as np
import os
import torch

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

if __name__ == '__main__':
    path_to_voxels_file = 'F:\IEEE TMM\\revised file\scene0005\\voxels/0001.npy'
    save_path = 'F:\IEEE TMM\\revised file\scene0005'
    event_numpy = np.load(path_to_voxels_file)
    event_tensor = torch.from_numpy(event_numpy).type(torch.float32)
    print(event_tensor.shape)

    # np.save(event_numpy, save_path)

    # events_tmp[i] = torch.from_numpy(np.transpose(make_event_preview(event_tensor.unsqueeze(0)), (2, 0, 1)))