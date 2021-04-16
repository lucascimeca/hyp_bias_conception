import h5py
from nsml import HAS_DATASET, DATASET_PATH, GPU_NUM

# Load dataset in BW
data_path = "./../data/"
# data_path = DATASET_PATH + '/train/'
out_filename1='multi_color_dsprites.h5'
out_filename2='multi_color_dsprites_uncompressed.h5'

dataset_name = data_path + out_filename1
dataset = h5py.File(dataset_name, 'r')
# latents_values = dataset['latents_values']
# latents_classes = dataset['latents_classes']

hf_uncompressed = h5py.File(data_path + out_filename2, 'w')
for key in dataset.keys():
    hf_uncompressed.create_dataset(key, data=dataset[key])
hf_uncompressed.close()