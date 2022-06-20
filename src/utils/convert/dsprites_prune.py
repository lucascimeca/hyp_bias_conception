import h5py
import time
import torch
import numpy as np
# from nsml import DATASET_PATH
from os import path as pt
from pathlib import Path


DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Load dataset in BW
# data_path = "../../data/"

def folder_exists(folder_name):
    return pt.isdir(folder_name)

def folder_create(folder_name, exist_ok=False, parents=True):
    path = Path(folder_name)
    try:
        if exist_ok and folder_exists(folder_name):
            return True
        path.mkdir(parents=parents)
    except Exception as e:
        raise e
    return True

new_data_path = "../../../data/"
if not folder_exists(new_data_path):
    folder_create(new_data_path)

# data_path = "../../data/" if len(DATASET_PATH) == 0 else DATASET_PATH + '/train/'
data_path = "../../../data/"
filename = 'dsprites.npz'
dataset_name = data_path + filename

bw_dataset = np.load(dataset_name, allow_pickle=True)
images = np.array(bw_dataset['imgs'])
latents = bw_dataset['latents_values']
classes = bw_dataset['latents_classes']

new_dataset = {}
new_dataset['latents_sizes'] = np.array([1, 3, 6, 40, 32, 32])


images_reshaped = images.reshape(images.shape[0], -1) - .5

all_indeces = set(range(images.shape[0]))
indeces_to_remove = set()
indeces_to_keep = set()
i = 0
while i < images.shape[0]:
    t_0 = time.time()

    res = np.dot(images_reshaped[i], images_reshaped[i:].T)
    indeces = [indeces_to_remove.add(idx+i) for idx in range(1, len(res)) if res[idx] == res[0]]

    indeces_to_keep.add(i)

    i += 1
    while i in indeces_to_remove:
        i += 1

    print("{:.2f}%, time elapsed: {:.2f}".format(100*i/images.shape[0], time.time() - t_0)) #, flush=True, end="\r"
    # print(indeces_to_remove)


# all_indeces = set(range(images.shape[0]))
# indeces_to_remove = set()
# indeces_to_keep = set()
# i = 0
# while i < images.shape[0]:
#     t_0 = time.time()
#     for j in range(i+1, images.shape[0]):
#         if np.array_equal(images[j], images[i]):
#             indeces_to_remove.add(j)
#
#     indeces_to_keep.add(i)
#
#     i += 1
#     while i in indeces_to_remove:
#         i += 1
#
#     print("{:.2f}%, time elapsed: {:.2f}".format(100*i/images.shape[0], time.time() - t_0), flush=True, end="\r")

indeces_to_keep = sorted(list(indeces_to_keep))

new_dataset['imgs'] = bw_dataset['imgs'][indeces_to_keep]
new_dataset['latents_values'] = bw_dataset['latents_values'][indeces_to_keep]
new_dataset['latents_classes'] = bw_dataset['latents_classes'][indeces_to_keep]

hf = h5py.File(new_data_path + 'bw_pruned_dsprites.h5', 'w')
for key in new_dataset.keys():
    hf.create_dataset(key, data=new_dataset[key])
hf.close()


print("Images kept: {}".format(len(indeces_to_keep)))
print("Images removed: {}".format(len(indeces_to_remove)))