import h5py

from utils.misc.simple_io import *

# Load dataset in BW
data_path = "../../data/"
# filename = 'dsprites.npz'
filename = 'bw_dsprites_pruned.h5'
dataset_name = data_path + filename

# bw_dataset = np.load(dataset_name, allow_pickle=True)
bw_dataset = h5py.File(dataset_name, 'r', libver='latest', swmr=True)

new_dataset = {}
new_dataset['latents_sizes'] = np.array([4, 3, 6, 40, 32, 32])

# reshape images -- to deal with RGB later on
new_dataset['imgs'] = np.array(bw_dataset['imgs']).reshape(
    (bw_dataset['imgs'].shape[0], bw_dataset['imgs'].shape[1], bw_dataset['imgs'].shape[2], 1)
)
new_dataset['imgs'] = np.concatenate((new_dataset['imgs'],
                                      new_dataset['imgs'],
                                      new_dataset['imgs']), axis=-1)  # 3 channels
new_dataset['imgs'] = np.concatenate((new_dataset['imgs'],
                                      new_dataset['imgs'],
                                      new_dataset['imgs'],
                                      new_dataset['imgs']), axis=0)  # 4 colors

# do stuff
color_idxs = [(1, 0, 0), (0, 1, 0), (0, 0, 1), (1, 1, 1)]
bw_data_length = bw_dataset['imgs'].shape[0]


new_dataset['latents_values'] = np.concatenate((bw_dataset['latents_values'],
                                                bw_dataset['latents_values'],
                                                bw_dataset['latents_values'],
                                                bw_dataset['latents_values']), axis=0)
new_dataset['latents_classes'] = np.concatenate((bw_dataset['latents_classes'],
                                                 bw_dataset['latents_classes'],
                                                 bw_dataset['latents_classes'],
                                                 bw_dataset['latents_classes']), axis=0)

for i in range(len(color_idxs)):
    for channel, activation in enumerate(color_idxs[i]):
        new_dataset['imgs'][bw_data_length*i:bw_data_length*(i+1), :, :, channel] *= (255 * activation)
    new_dataset['latents_values'][bw_data_length*i:bw_data_length*(i+1), 0] = (i+1)
    new_dataset['latents_classes'][bw_data_length*i:bw_data_length*(i+1), 0] = i


hf = h5py.File(data_path + 'color_dsprites_pruned.h5', 'w')
for key in new_dataset.keys():
    hf.create_dataset(key, data=new_dataset[key])
hf.close()