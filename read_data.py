import numpy as np
from matplotlib import pyplot as plt
import os


def show(file, save_file):
    a = dict(np.load(file))
    print(a['relation_structure'])
    img = a['image'].reshape((16, 160, 160))
    rel = [[_.decode('utf8') for _ in r] for r in a['relation_structure']]
    tgt = a['target']
    meta_tgt = a['meta_target']
    print(rel, tgt, meta_tgt)

    plt.figure(figsize=(10, 5))

    for i in range(8):
        plt.subplot(3, 6, i // 3 * 6 + i % 3 + 1)
        plt.axis('off')
        plt.imshow(img[i], vmin=0, vmax=255)
    for i in range(8, 16):
        plt.subplot(3, 6, (i - 8) // 3 * 6 + (i - 8) % 3 + 4)
        plt.axis('off')
        plt.imshow(img[i], vmin=0, vmax=255)
    plt.suptitle('{}, {}'.format(tgt, rel))
    plt.savefig(save_file, dpi=144)
    # plt.show()

    # print(list(img[:, 40]))
    # exit()


path = 'datasets/PGM/neutral/'
files = os.listdir(path)
# file = 'PGM_interpolation_test_1020.npz'

save_path = 'visualization/pgm/neutral/'
if not os.path.isdir(save_path):
    os.mkdir(save_path)

for f in files:
    if f.find('test') != -1:
        show(path + f, save_path + f.split('.')[0])

# f = 'PGM_neutral_train_1019213.npz'
# show(path + f, save_path + f.split('.')[0])

# it seems that in interpolation, color progression goes like 024, 246, 468.
# while in neutral, its like 123, 234, 345.
