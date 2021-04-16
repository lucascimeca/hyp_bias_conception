import webdataset as wds
import numpy as np
import tarfile
from PIL import Image


from itertools import islice
from utils.simple_io import *

# Load dataset in BW
data_path = "./../data/"
filename = 'dsprites.npz'
dataset_name = data_path + filename

if not file_exists(dataset_name):
    tar_filenames = get_filenames(folder=data_path)
    print(tar_filenames)
    tar_filepaths = [data_path + filename for filename in tar_filenames]
    for f_path in tar_filepaths:
        tar = tarfile.open(f_path)
        tar.extractall()
        tar.close()

dataset = np.load(dataset_name, allow_pickle=True)

color_idxs = [(1, 0, 0), (0, 1, 0), (0, 0, 1), (1, 1, 1)]  # R, G, B, Black
with wds.ShardWriter(data_path + "color_dsprites_%06d.tar",
                     maxsize=int(1e9),
                     maxcount=int(10000)) as sink:
    for i, color_idxs in enumerate(color_idxs):
        data = np.zeros(
            (dataset['imgs'].shape[0], 3, dataset['imgs'].shape[1], dataset['imgs'].shape[2]),
            dtype=np.int8)
        for ci in color_idxs:
            data[:, ci, :, :] = ci * dataset['imgs']

        latents_values = dataset['latents_values']
        latents_values[:, 0] = i + 1
        latents_classes = dataset['latents_classes']
        latents_classes[:, 0] = i

        data = data.transpose()
        # with wds.TarWriter(data_path + "color_dsprites_{}.tar".format(i)) as dst:
        for j in range(dataset['imgs'].shape[0]):
            sample = {
                "__key__": str(i)+"x"+str(j),
                "png": Image.fromarray((data[:, :, :, j] * 255).astype(np.uint8)),
                "latents": latents_values[j, :].astype(np.float64).tostring(),
                "classes": latents_classes[j, :].astype(np.int64).tostring(),
            }
            sink.write(sample)

    print("DONE!")
