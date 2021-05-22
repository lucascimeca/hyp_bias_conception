import h5py

# matplotlib.use('TkAgg')

from utils.misc.simple_io import *

MAX_OBJ_NO = 4

# Load dataset in BW
data_path = "../../data/"
out_filename='multi_color_dsprites.h5'

bw_filename='bw_dsprites.hdf5'
bw_dataset_name = data_path + bw_filename
bw_dataset = h5py.File(bw_dataset_name, 'r')
bw_latents_values = bw_dataset['latents']['values']
bw_latents_classes = bw_dataset['latents']['classes']

bw_latents_sizes = np.array([1, 3, 6, 40, 32, 32])
color_latents_sizes = np.array([4, 3, 6, 40, 32, 32])

color_dataset = {}
color_dataset['imgs'] = np.array(bw_dataset['imgs'])[:, :, :, np.newaxis]

latents_bases = np.concatenate((color_latents_sizes[::-1].cumprod()[::-1][1:], np.array([1, ])))

print(" --- creating color dataset ---")
color_dataset['imgs'] = np.concatenate((color_dataset['imgs'],
                                        color_dataset['imgs'],
                                        color_dataset['imgs']), axis=-1).astype(np.uint8)  # 3 channels
print(" --- initializing color arrays ")
color_dataset['imgs'] = np.concatenate((color_dataset['imgs'],
                                        color_dataset['imgs'],
                                        color_dataset['imgs'],
                                        color_dataset['imgs']), axis=0)  # 4 colors


print(" --- initializing latents ")
color_dataset['latents_values'] = np.concatenate((bw_latents_values,
                                                  bw_latents_values,
                                                  bw_latents_values,
                                                  bw_latents_values), axis=0)
color_dataset['latents_classes'] = np.concatenate((bw_latents_classes,
                                                   bw_latents_classes,
                                                   bw_latents_classes,
                                                   bw_latents_classes), axis=0).astype(np.uint8)

color_idxs = [(1, 0, 0), (0, 1, 0), (0, 0, 1), (1, 1, 1)]
bw_data_length = bw_dataset['imgs'].shape[0]

print(" --- creating color codes ")
for i in range(len(color_idxs)):
    for channel, activation in enumerate(color_idxs[i]):
        color_dataset['imgs'][bw_data_length * i:bw_data_length * (i + 1), :, :, channel] *= (255 * activation)
    color_dataset['latents_values'][bw_data_length * i:bw_data_length * (i + 1), 0] = (i + 1)
    color_dataset['latents_classes'][bw_data_length * i:bw_data_length * (i + 1), 0] = i
    print(" --- created color code {}".format(color_idxs[i]))


# created dataset color
print(" --- done! ")


def latent_to_index(latents):
    return np.dot(latents, latents_bases).astype(int)


def insert_aug_img(img, latent_classes, no_of_objects=1):
    locations = [(latent_classes[-2], latent_classes[-1])]

    min_distance = 12 - no_of_objects
    while no_of_objects > 1:
        x_satisfied = False
        y_satisfied = False

        cnt = 0
        mdx = min_distance
        while not x_satisfied:
            cnt += 1
            x_new = np.random.randint(32)
            x_satisfied = np.all([np.abs(x_new - x) >= mdx for x, _ in locations])

            if cnt > 100:
                mdx -= 1
                cnt = 0

        cnt = 0
        mdy = min_distance
        while not y_satisfied:
            cnt += 1
            y_new = np.random.randint(32)
            y_satisfied = np.all([np.abs(y_new - y) >= mdy for _, y in locations])

            if cnt > 50:
                mdy -= 1
                cnt = 0

        locations += [(x_new, y_new)]
        latent_class = latent_classes
        latent_class[-2] = x_new
        latent_class[-1] = y_new

        img |= color_dataset['imgs'][latent_to_index(latent_class)]

        no_of_objects -= 1

    return img


bw_dataset_imgs = np.array(bw_dataset['imgs'])

print(" --- creating multi color dataset --- ")
data_length = color_dataset['imgs'].shape[0]

write_chunks_size = 10000

chunk_dataset = {}
latents_sizes = np.array([4, 4, 3, 6, 40, 32, 32])
chunk_dataset['imgs'] = np.zeros(((write_chunks_size,) + color_dataset['imgs'].shape[1:])).astype(np.uint8)
chunk_dataset['latents_values'] = np.zeros((write_chunks_size, len(latents_sizes))).astype(color_dataset['latents_values'].dtype)
chunk_dataset['latents_classes'] = np.zeros((write_chunks_size, len(latents_sizes))).astype(np.uint8)

print(" --- starting chunk writing ")
created_dataset = False
chunk_idx = 0
for j in range(MAX_OBJ_NO):
    for i in range(data_length):
        img_idx = j*data_length + i
        chunk_dataset['imgs'][chunk_idx, :] = insert_aug_img(
            img=color_dataset['imgs'][i].copy(),
            latent_classes=color_dataset['latents_classes'][i].copy(),
            no_of_objects=j+1)

        chunk_dataset['latents_values'][chunk_idx, 1:] = color_dataset['latents_values'][i, :]
        chunk_dataset['latents_classes'][chunk_idx, 1:] = color_dataset['latents_classes'][i, :]
        chunk_dataset['latents_values'][chunk_idx, 0] = j + 1
        chunk_dataset['latents_classes'][chunk_idx, 0] = j

        chunk_idx += 1

        if chunk_idx % write_chunks_size == 0:
            if not created_dataset:
                hf = h5py.File(data_path + out_filename, 'w')
                for key in chunk_dataset.keys():
                    hf.create_dataset(key, data=chunk_dataset[key], compression="gzip", maxshape=(None,) + chunk_dataset[key].shape[1:])
                hf.create_dataset('latents_sizes', data=latents_sizes, compression="gzip")
                hf.close()
                created_dataset = True
                print(" --- created first dataset chunk ---")
            else:
                with h5py.File(data_path + out_filename, 'a') as hf:
                    for key in chunk_dataset.keys():
                        hf[key].resize((hf[key].shape[0] + write_chunks_size), axis=0)
                        hf[key][-write_chunks_size:] = chunk_dataset[key]

            chunk_idx = 0
            print("--- {} out of {}, {:.2f}%".format(img_idx+1,
                                                 MAX_OBJ_NO * color_dataset['latents_classes'].shape[0],
                                                 100*img_idx/(MAX_OBJ_NO * color_dataset['latents_classes'].shape[0])))

with h5py.File(data_path + out_filename, 'a') as hf:
    for key in chunk_dataset.keys():
        hf[key].resize((hf[key].shape[0] + chunk_idx), axis=0)
        hf[key][-chunk_idx:] = chunk_dataset[key][:chunk_idx]
print("--- {} out of {}, {:.2f}%".format(img_idx+1,
                                         MAX_OBJ_NO * color_dataset['latents_classes'].shape[0],
                                         100*img_idx/(MAX_OBJ_NO * color_dataset['latents_classes'].shape[0])))

print(" --- DONE! ")