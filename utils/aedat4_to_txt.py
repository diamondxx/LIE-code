import aedat
import sys
import os
import numpy as np


def aedat2txt(aedat_path, event_folder, idx):
    """,
    {0: {'type': 'events', 'width': 346, 'height': 260},
    2: {'type': 'imus'},
    1: {'type': 'frame', 'width': 346, 'height': 260},
    3: {'type': 'triggers'}}
    """
    if aedat_path[-3:] == 'txt':
        return
    decoder = aedat.Decoder(aedat_path)
    output_path = event_folder
    f = open(output_path + '/{:04d}.txt'.format(idx), 'w', encoding='utf-8')
    width = str(decoder.id_to_stream()[0]['width'])
    height = str(decoder.id_to_stream()[0]['height'])
    f.writelines(width + ' ')
    f.writelines(height + '\n')

    # count = 0
    for packet in decoder:
        if 'events' in packet:
            num = len(packet['events'])
            for i in range(num):
                f.write(str(packet['events'][i][0]) + ' ')
                f.write(str(packet['events'][i][1]) + ' ')
                f.write(str(packet['events'][i][2]) + ' ')
                f.write(str(int(packet['events'][i][3])) + '\n')
            # count += num
    # print(count)
    decoder = None
    f.close()
    os.remove(aedat_path)



def main():
    # input_path = str(sys.argv[1])
    file_path = './data/dataset_voxel_0717'
    # file_path = '../data/test'
    sublist = os.listdir(file_path)
    for f1 in sublist:
        print(f1)
        if f1 == 'Photometric value.txt':
            continue
        scene_path = os.path.join(file_path, f1)
        file1 = os.listdir(scene_path)
        for f2 in file1:
            event_ptah = os.path.join(scene_path, f2, 'event')
            if not os.path.exists(event_ptah):
                continue
            aedat_file = np.sort(os.listdir(event_ptah))
            num = 1
            for f3 in aedat_file:
                aedat_path = os.path.join(event_ptah, f3)
                aedat2txt(aedat_path, event_ptah, num)
                # remove aedat file
                num += 1

    print("process end!")


if __name__ == "__main__":
    main()


