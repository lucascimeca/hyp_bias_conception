from torchvision import transforms
import matplotlib.pyplot as plt
import numpy as np
import h5py
import time
import matplotlib
import pandas as pd
import re
import glob
import PIL


from itertools import islice
from utils.simple_io import *

MAX_OBJ_NO = 4

# Load dataset
data_path = "./../data/"

latents_ranges = {
    'identity': (0, 4),
    'age': (0, 12),
    'gender': (0, 2),
    'ethnicity': (0, 5),
}
idx_to_latent = {
    0: 'identity',
    1: 'age',
    2: 'gender',
    3: 'ethnicity',
}
latent_to_idx = {
    'identity': 0,
    'age': 1,
    'gender': 2,
    'ethnicity': 4,
}

new_dataset = {}
new_dataset['latents_sizes'] = np.array([0, 0, 0, 0])
# new_dataset['']

files = []
for img in glob.glob(data_path + "UTKFace/*.jpg"):
    files.append(img)

for img in glob.glob(data_path + "crop_part1/*.jpg"):
    files.append(img)

df_data = pd.DataFrame(files, columns=["name"])
df_data.name = df_data.name.apply(str)
df_data["identity"] = df_data.name.apply(lambda x: re.findall(r"\d{1,3}_\d_\d", x)[0])
df_data["identity"] = df_data.identity.apply(lambda x: re.sub("_", " ", x))
df_data["age"] = df_data.identity.apply(lambda x: int(x.split(" ")[0]))
df_data["gender"] = df_data.identity.apply(lambda x: int(x.split(" ")[1]))
df_data["ethnicity"] = df_data.identity.apply(lambda x: int(x.split(" ")[2]))

# checking distribution for ages
df_data.age.hist()
plt.legend(["data"])
plt.show()

# fixing the label
idx = df_data[df_data.gender == 3].index
df_data.loc[idx, "gender"] = 1  # 1 means woman

# accuracy is a good metric fore gender
(df_data.gender.value_counts()).plot.barh()
plt.title("data")
plt.show()

ethnicity_map = {0: "White", 1: "Black", 2: "Asian", 3: "Indian", 4: "Others"}
df_data.ethnicity.replace(ethnicity_map).value_counts().sort_index().plot.barh()
plt.title("data")
plt.show()

identities = df_data['identity'].unique()
identity_map = {identity:idx for idx, identity in enumerate(df_data['identity'].unique())}
# set latent sizes based on data
new_dataset['latents_sizes'][0] = len(df_data['identity'].unique())
new_dataset['latents_sizes'][1] = len(df_data['age'].unique())
new_dataset['latents_sizes'][2] = len(df_data['gender'].unique())
new_dataset['latents_sizes'][3] = len(df_data['ethnicity'].unique())

# get first image to inizialize data arrays
path = df_data['name'][0]
img = PIL.Image.open(path).convert('RGB')
img = np.array(img)

new_dataset['imgs'] = np.zeros(shape=((df_data.shape[0],) + img.shape), dtype='uint8')
new_dataset['latents_values'] = np.zeros(shape=(df_data.shape[0], len(new_dataset['latents_sizes'])), dtype='uint32')
new_dataset['latents_classes'] = np.zeros(shape=(df_data.shape[0], len(new_dataset['latents_sizes'])), dtype='uint32')

for i in range(len(df_data)):
    path = df_data['name'][i]
    img = PIL.Image.open(path).convert('RGB')
    new_dataset['imgs'][i, :, :, :] = np.array(img)

    new_dataset['latents_values'][i, 0] = identity_map[df_data['identity'][i]] + 1
    new_dataset['latents_values'][i, 1] = df_data['age'][i]//10 + 1
    new_dataset['latents_values'][i, 2] = df_data['gender'][i] + 1
    new_dataset['latents_values'][i, 3] = df_data['ethnicity'][i] + 1

    new_dataset['latents_classes'][i, 0] = identity_map[df_data['identity'][i]]
    new_dataset['latents_classes'][i, 1] = df_data['age'][i]//10
    new_dataset['latents_classes'][i, 2] = df_data['gender'][i]
    new_dataset['latents_classes'][i, 3] = df_data['ethnicity'][i]
    if i % 100 == 0:
        print("{} out of {}, {:.2f}%".format(i,
                                             new_dataset['imgs'].shape[0],
                                             100*i/new_dataset['imgs'].shape[0]))


hf = h5py.File(data_path + 'UTKFace.h5', 'w')
for key in new_dataset.keys():
    hf.create_dataset(key, data=new_dataset[key])
hf.close()