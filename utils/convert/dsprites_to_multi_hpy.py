import h5py

# matplotlib.use('TkAgg')

from utils.misc.simple_io import *

MAX_OBJ_NO = 4

# Load dataset in BW
data_path = "../../data/"
filename='bw_dsprites.hdf5'
dataset_name = data_path + filename
bw_dataset = h5py.File(dataset_name, 'r')
bw_latents_values = bw_dataset['latents']['values']
bw_latents_classes = bw_dataset['latents']['classes']

bw_latents_sizes = np.array([1, 3, 6, 40, 32, 32])

new_dataset = {}
new_dataset['latents_sizes'] = np.array([4, 1, 3, 6, 40, 32, 32])

latents_bases = np.concatenate((bw_latents_sizes[::-1].cumprod()[::-1][1:], np.array([1, ])))


def latent_to_index(latents):
    return np.dot(latents, latents_bases).astype(int)


def insert_aug_img(new_dataset, bw_imgs, img_idx, bw_idx, no_of_objects=1):
    locations = [(bw_latents_classes[bw_idx][-2], bw_latents_classes[bw_idx][-1])]

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

            if cnt > 50:
                mdx -= 1
                # print("Solving adjecency problem, mdx: {}, No of attenpts: {}".format(mdx, cnt))
                cnt = 0

        cnt = 0
        mdy = min_distance
        while not y_satisfied:
            cnt += 1
            y_new = np.random.randint(32)
            y_satisfied = np.all([np.abs(y_new - y) >= mdy for _, y in locations])

            if cnt > 50:
                mdy -= 1
                # print("Solving adjecency problem, mdy: {}, No of attenpts: {}".format(mdy, cnt))
                cnt = 0

        locations += [(x_new, y_new)]
        latent_class = bw_latents_classes[bw_idx]
        latent_class[-2] = x_new
        latent_class[-1] = y_new

        new_dataset['imgs'][img_idx] |= bw_imgs[latent_to_index(latent_class)]

        no_of_objects -= 1

    if img_idx % 1000 == 0:
        print("{} out of {}, {:.2f}%".format(img_idx,
                                             new_dataset['imgs'].shape[0],
                                             100*img_idx/new_dataset['imgs'].shape[0]))


bw_dataset_imgs = np.array(bw_dataset['imgs'])

# reshape images -- to deal with RGB later on
new_dataset['imgs'] = np.concatenate((np.array(bw_dataset['imgs']),
                                      np.array(bw_dataset['imgs']),
                                      np.array(bw_dataset['imgs']),
                                      np.array(bw_dataset['imgs'])), axis=0)  # 4 objects


bw_data_length = bw_dataset['imgs'].shape[0]


new_dataset['latents_values'] = np.zeros((bw_data_length*4, len(new_dataset['latents_sizes'])))
new_dataset['latents_classes'] = np.zeros((bw_data_length*4, len(new_dataset['latents_sizes'])))


cnt = 0
for j in range(MAX_OBJ_NO):
    for i in range(bw_data_length):
        img_idx = j*bw_data_length + i
        insert_aug_img(
            new_dataset=new_dataset,
            bw_imgs=bw_dataset_imgs,
            img_idx=img_idx,
            bw_idx=i,
            no_of_objects=j+1)

        new_dataset['latents_values'][img_idx, 1:] = bw_latents_values[i, :]
        new_dataset['latents_classes'][img_idx, 1:] = bw_latents_classes[i, :]
        new_dataset['latents_values'][img_idx, 0] = j+1
        new_dataset['latents_classes'][img_idx, 0] = j


hf = h5py.File(data_path + 'multi_bwdsprites_compressed.h5', 'w')
for key in new_dataset.keys():
    # hf.create_dataset(key, data=new_dataset[key])
    hf.create_dataset(key, data=new_dataset[key], compression="gzip")
hf.close()