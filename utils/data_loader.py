"""
DSprites for combinatorial feature switches.
Author: Luca Scimeca
"""
import torch
import numpy as np
import webdataset as wds
import matplotlib.gridspec as gridspec
import tarfile
from skimage import io, transform
import time
import random
import h5pickle as h5py
from torch.utils.data import TensorDataset, DataLoader
from matplotlib import pyplot as plt
from utils.simple_io import *
from nsml import HAS_DATASET, DATASET_PATH, GPU_NUM
from torch.utils.data import IterableDataset
from torch.utils.data import Dataset


def split_data(dataset, train_split, valid_split):
    train_size = round(train_split * len(dataset))
    val_size = round(valid_split * len(dataset))
    test_size = len(dataset) - train_size - val_size
    return torch.utils.data.random_split(dataset, [train_size, val_size, test_size])


class Rescale(object):
    """Rescale the image in a sample to a given size.

    Args:
        output_size (tuple or int): Desired output size. If tuple, output is
            matched to output_size. If int, smaller of image edges is matched
            to output_size keeping aspect ratio the same.
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        self.output_size = output_size

    def __call__(self, sample):
        img, img_class = sample

        h, w = img.shape[:2]
        if isinstance(self.output_size, int):
            if h > w:
                new_h, new_w = self.output_size * h / w, self.output_size
            else:
                new_h, new_w = self.output_size, self.output_size * w / h
        else:
            new_h, new_w = self.output_size

        new_h, new_w = int(new_h), int(new_w)
        img = transform.resize(img, (new_h, new_w))

        return img, img_class


class FeatureDataset(Dataset):

    def __init__(self, dataset, latents, classes, indeces, no_of_feature_lvs, latent_to_idx, task_dict, feature,
                 resize=None):
        self.dataset = dataset
        self.latents = latents
        self.classes = classes
        self.indeces = sorted(indeces)
        self.no_of_feature_lvs = no_of_feature_lvs
        self.latent_to_idx = latent_to_idx
        self.task_dict = task_dict
        self.feature = feature
        self.resize = resize

        # labels
        self.labels = np.ones((self.classes.shape[0]))-1
        for lv in range(self.no_of_feature_lvs):
            same_cell_condition = self.classes[:, self.latent_to_idx[feature]] == self.task_dict[feature][lv]
            self.labels[same_cell_condition] = lv

    def __len__(self):
        return len(self.indeces)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        if self.resize is not None:
            tensor_x = torch.Tensor(transform.resize(
                self.dataset['imgs'][self.indeces[idx]],
                self.resize))
        else:
            tensor_x = torch.Tensor(self.dataset['imgs'][self.indeces[idx]])

        if len(self.dataset['imgs'].shape) == 3:
            tensor_x = tensor_x.unsqueeze(0).contiguous()
        else:
            tensor_x = tensor_x.permute((2, 0, 1)).contiguous()

        tensor_y = torch.Tensor([self.labels[self.indeces[idx]]]).to(dtype=torch.long)

        return tensor_x, tensor_y

    def get_original_index(self, idx):
        return self.indeces[idx]


class FeatureCombinationCreator:
    latents_ranges = {
        'color': (0, 1),
        'shape': (0, 3),
        'scale': (0.5, 1),
        'orientation': (0, 2 * np.pi),
        'x_position': (0, 1),
        'y_position': (0, 1),
    }
    idx_to_latent = {
        0: 'color',
        1: 'shape',
        2: 'scale',
        3: 'orientation',
        4: 'x_position',
        5: 'y_position',
    }
    latent_to_idx = {
        'color': 0,
        'shape': 1,
        'scale': 2,
        'orientation': 3,
        'x_position': 4,
        'y_position': 5,
    }

    latents_sizes = np.array([1,  3,  6, 40, 32, 32])
    torch_dataset = None

    samples = None
    latents = None
    diag_samples = None
    diag_labels = None
    offdiag_samples = None
    offdiag_labels = None
    task_dict = None
    feature_variants = None

    round_one_dataset = None
    round_two_datasets = None

    cmap = None

    def __init__(self, data_path="./data/", filename='bw_dsprites.hdf5'):
        # Load dataset
        self.path = data_path if len(DATASET_PATH) == 0 else DATASET_PATH + '/train/'
        self.dataset_name = self.path + filename

        if not file_exists(self.dataset_name):
            tar_filenames = get_filenames(folder=self.path)
            print(tar_filenames)
            tar_filepaths = [self.path + filename for filename in tar_filenames]
            for f_path in tar_filepaths:
                tar = tarfile.open(f_path)
                tar.extractall()
                tar.close()

        print("### Loading data from disk... ###")
        if filename.endswith("npz"):
            tmp = np.load(self.dataset_name, allow_pickle=True)
            self.dataset = {}
            self.dataset['imgs'] = tmp['imgs']
            self.dataset['latents_values'] = tmp['latents_values']
            self.dataset['latents_classes'] = tmp['latents_classes']
        else:
            self.dataset = h5py.File(self.dataset_name, 'r', libver='latest', swmr=True)

        if 'latents' in self.dataset.keys():
            self.latents_values = self.dataset['latents']['values']
            self.latents_classes = self.dataset['latents']['classes']
        else:
            self.latents_values = self.dataset['latents_values']
            self.latents_classes = self.dataset['latents_classes']

        print("Finished Loading")

        self.dataset_size = self.dataset['imgs'].shape[0]

        # Define number of values per latents and functions to convert to indices
        self.latents_bases = np.concatenate((self.latents_sizes[::-1].cumprod()[::-1][1:], np.array([1, ])))

    def _latent_to_index(self, latents):
        return np.dot(latents, self.latents_bases).astype(int)

    def _sample_latent(self, size=1, latent_factors={}):

        if size < self.dataset_size:
            unique_samples = np.ones((1, self.latents_sizes.size))*-1
            size += 1
            while unique_samples.shape[0] < size:

                running_size = size-unique_samples.shape[0]
                samples = np.zeros((running_size, self.latents_sizes.size))
                # create int-ranges for each latent
                for lat_i, lat_size in enumerate(self.latents_sizes):
                    if len(list(latent_factors.keys())) != 0:
                        latent_key = self.idx_to_latent[lat_i]
                        min_rg, max_rg = self.latents_ranges[latent_key]

                        # get min range
                        if latent_factors[latent_key][0] < min_rg:
                            latent_factors[latent_key][0] = min_rg
                        min_ratio = (latent_factors[latent_key][0] - min_rg)/(max_rg - min_rg)
                        min_int = round(np.floor(min_ratio*lat_size))

                        # get max range
                        if latent_factors[latent_key][1] > max_rg:
                            latent_factors[latent_key][1] = max_rg
                        max_ratio = (latent_factors[latent_key][1] - min_rg)/(max_rg - min_rg)
                        max_int = round(np.ceil(max_ratio*lat_size))

                        # generate random sample
                        samples[:, lat_i] = np.random.randint(min_int, max_int, size=running_size)
                    else:
                        samples[:, lat_i] = np.random.randint(lat_size, size=running_size)
                unique_samples = np.unique(
                    np.concatenate((unique_samples, samples), axis=0),
                    axis=0
                )
            return unique_samples[1:, :]
        else:
            return self.latents_classes

    def get_dataset(self, number_of_samples=300000, train_split=0.7, valid_split=0.2,
                    output_targets=('shape', 'orientation'), **latent_factors):
        # Sample latents randomly
        latent_classes_sampled = self._sample_latent(size=number_of_samples, latent_factors=latent_factors)

        # Select images
        indices_sampled = self._latent_to_index(latent_classes_sampled)

        out_idxes = [self.latent_to_idx[latent] for latent in output_targets]

        # Make tensors
        tensor_x = torch.Tensor(self.dataset['imgs'][indices_sampled])
        tensor_y = torch.Tensor(latent_classes_sampled[:, out_idxes]).to(dtype=torch.int64)

        self.torch_dataset = TensorDataset(tensor_x, tensor_y)

        return split_data(self.torch_dataset, train_split, valid_split)

    def get_dataset_fvar(self, number_of_samples='all', train_split=0.7, valid_split=0.2,
                         features_variants=('shape', 'scale'), resize=None, **latent_factors):
        """
        The function creates two round of dsprites datasets based on two varying features.
        The first round is based on the main diagonal, when varying the two features (the number of variations is
        decided based on which feature has the lowest number of levels). The second round is based off the offdiagonal,
        this depends on which task we wish to solve, i.e. if based on feature 1 or feature 2, thus two datasets will be
        contained within the second round datasets.

        :param tuple(int, int) resize: parameter to resize input image
        :param number_of_samples: (int) number of unique samples to pick from dsprites dataset. Can be set to "all"
                                        to get the full dataset
        :param train_split: (float) proportion of training data. Used for all returned datasets.
        :param valid_split: (float) proportion of validation data. Used for all returned datasets.
        :param features_variants: tuple(str, str) name of features to use for label matrix
        :param latent_factors: additional into. It allows to specify ranges to consider for each feature
                               (see _sample_latent).
        :return: (dict, dict) two dictionaries containing the datasets for round one and round two.
        """

        self.feature_variants = features_variants
        # ----- create combination of features for first and second training -----
        self.no_of_feature_lvs = min(
            self.latents_sizes[i] for i in [self.latent_to_idx[latent] for latent in self.feature_variants])
        self.task_dict = {feature: {} for feature in self.feature_variants}

        for feature in self.task_dict.keys():
            self.task_dict[feature] = [round(x) for x in
                                       np.linspace(0, self.latents_sizes[self.latent_to_idx[feature]] - 1,
                                                   self.no_of_feature_lvs)]

        # ----- Create train set, second train set and valid set ---
        all_latents = np.array(self.dataset['latents_classes'])

        # Find corresponding indeces
        all_indeces = np.array(list(range(all_latents.shape[0])))

        # first train set is the feature combination matrix diagonal (linearize booleans for faster conditioning)
        diag_condition = np.array([False] * self.dataset['imgs'].shape[0])  # initialize boolens for "OR" chain
        for lv in range(self.no_of_feature_lvs):
            same_cell_condition = np.array([True] * self.dataset['imgs'].shape[0])  # initialize boolens for "AND" chain
            for feature in self.feature_variants:
                same_cell_condition &= all_latents[:, self.latent_to_idx[feature]] == self.task_dict[feature][lv]  # AND
            diag_condition |= same_cell_condition  # OR

        # extract diagonal and off-diagonal dataset elements
        diag_indeces = all_indeces[diag_condition]

        offdiag_indeces = {}
        for feature in self.feature_variants:
            offdiag_condition = np.array([False] * self.dataset['imgs'].shape[0])
            for lv in range(self.no_of_feature_lvs):
                offdiag_condition |= all_latents[:, self.latent_to_idx[feature]] == self.task_dict[feature][lv]  # OR

            offdiag_indeces[feature] = all_indeces[np.logical_and(np.logical_not(diag_condition), offdiag_condition)]

        # sample if you want less samples
        if isinstance(number_of_samples, int):
            if len(diag_indeces) > number_of_samples:
                choice = np.random.choice(len(diag_indeces), size=number_of_samples)
                diag_indeces = diag_indeces[choice]
            elif len(diag_indeces) < number_of_samples:
                print(
                    "Warning: {} samples for the main diagonal could not be found in the dataset, "
                    "{} are being used instead.".format(
                        number_of_samples, len(diag_indeces)))
                number_of_samples = len(diag_indeces)

            for feature in self.feature_variants:
                if len(offdiag_indeces[feature]) > number_of_samples:
                    choice = np.random.choice(len(offdiag_indeces[feature]), size=number_of_samples)
                    offdiag_indeces[feature] = offdiag_indeces[feature][choice]
                elif len(offdiag_indeces[feature]) < number_of_samples:
                    print(
                        "Warning: {} samples for the off diagonal could not be found in the dataset, "
                        "{} are being used instead.".format(
                            number_of_samples, len(offdiag_indeces[feature])))

        train_size = round(train_split * diag_indeces.shape[0])
        val_size = round(valid_split * diag_indeces.shape[0])
        test_size = diag_indeces.shape[0] - train_size - val_size

        # ----------  create round one dataset ----------

        ro_train_idx, ro_valid_idx, ro_test_idx = torch.utils.data.random_split(
            diag_indeces, [train_size, val_size, test_size]
        )

        self.round_one_dataset = {
            'train': FeatureDataset(
                indeces=ro_train_idx,
                dataset=self.dataset,
                latents=self.latents_values,
                classes=self.latents_classes,
                resize=resize,
                no_of_feature_lvs=self.no_of_feature_lvs,
                latent_to_idx=self.latent_to_idx,
                task_dict=self.task_dict,
                feature=self.feature_variants[0],  # doesn't matter within diag indeces
            ),
            'valid': FeatureDataset(
                indeces=ro_valid_idx,
                dataset=self.dataset,
                latents=self.latents_values,
                classes=self.latents_classes,
                resize=resize,
                no_of_feature_lvs=self.no_of_feature_lvs,
                latent_to_idx=self.latent_to_idx,
                task_dict=self.task_dict,
                feature=self.feature_variants[0]  # doesn't matter within diag indeces
            ),
            'test': FeatureDataset(
                indeces=ro_test_idx,
                dataset=self.dataset,
                latents=self.latents_values,
                classes=self.latents_classes,
                resize=resize,
                no_of_feature_lvs=self.no_of_feature_lvs,
                latent_to_idx=self.latent_to_idx,
                task_dict=self.task_dict,
                feature=self.feature_variants[0]  # doesn't matter within diag indeces
            ),
        }
        print("Created Training Dataset")

        # ----------  create round two dataset ----------
        # train on off diagonal (train split), test on off diag, by task

        self.round_two_datasets = {}
        for fn, feature in enumerate(self.feature_variants):
            rt_train_idx, rt_valid_idx, rt_test_idx = torch.utils.data.random_split(
                offdiag_indeces[feature], [train_size, val_size, test_size]
            )
            self.round_two_datasets[feature] = {
                'train': FeatureDataset(
                    indeces=rt_train_idx,
                    dataset=self.dataset,
                    latents=self.latents_values,
                    classes=self.latents_classes,
                    resize=resize,
                    no_of_feature_lvs=self.no_of_feature_lvs,
                    latent_to_idx=self.latent_to_idx,
                    task_dict=self.task_dict,
                    feature=feature
                ),
                'valid': FeatureDataset(
                    indeces=rt_valid_idx,
                    dataset=self.dataset,
                    latents=self.latents_values,
                    classes=self.latents_classes,
                    resize=resize,
                    no_of_feature_lvs=self.no_of_feature_lvs,
                    latent_to_idx=self.latent_to_idx,
                    task_dict=self.task_dict,
                    feature=feature
                ),
                'test': FeatureDataset(
                    indeces=rt_test_idx,
                    dataset=self.dataset,
                    latents=self.latents_values,
                    classes=self.latents_classes,
                    resize=resize,
                    no_of_feature_lvs=self.no_of_feature_lvs,
                    latent_to_idx=self.latent_to_idx,
                    task_dict=self.task_dict,
                    feature=feature
                ),
            }

            print("Created Dataset for task '{}' ({}/{})".format(feature, fn+1, len(self.feature_variants)))

        return self.round_one_dataset, self.round_two_datasets

    def show_examples(self, images=None, title='', num_images=25):
        if images is None:
            images = self.samples
        ncols = round(np.ceil(num_images ** 0.5))
        nrows = round(np.ceil(num_images / ncols))
        plt.figure()
        plt.title(title)
        _, axes = plt.subplots(ncols, nrows, figsize=(nrows * 3, ncols * 3))
        axes = axes.flatten()

        for ax_i, ax in enumerate(axes):
            if ax_i < num_images:
                ax.imshow(images[ax_i].squeeze(), cmap=self.cmap, interpolation='nearest')
                ax.set_xticks([])
                ax.set_yticks([])
            else:
                ax.axis('off')
        plt.show()

    def show_density(self, images=None, title=''):
        if images is None:
            images = self.samples
        plt.figure()
        plt.title(title)
        _, ax = plt.subplots()
        ax.imshow(images.squeeze().mean(axis=0), interpolation='nearest', cmap=self.cmap)
        ax.grid('off')
        ax.set_xticks([])
        ax.set_yticks([])
        plt.show()

    def show_tasks(self, ):

        features = list(self.task_dict.keys())[:2]

        spacing = self.no_of_feature_lvs // 10 + 1
        imgs_num = len(list(range(0, self.no_of_feature_lvs, spacing)))
        for feature_num, feature in enumerate(features):

            fig, ax = plt.subplots(figsize=(imgs_num*2, imgs_num*2))
            fig.patch.set_visible(False)
            ax.axis('off')
            ax = fig.add_subplot(111, frameon=False)
            ax.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
            ax.set_title("TASK: {}".format(feature), pad=10)

            grid = gridspec.GridSpec(nrows=imgs_num, ncols=imgs_num, figure=fig)
            grid.update(wspace=.5, hspace=0.1)  # set the spacing between axes.
            idx = 0
            used_idxs = set()
            for i, i_label in enumerate(range(0, self.no_of_feature_lvs, spacing)):   # first feature
                for j, j_label in enumerate(range(0, self.no_of_feature_lvs, spacing)):  # second feature

                    label = i_label if feature_num == 0 else j_label
                    if i == j:
                        # main diagonal
                        dataset = self.round_one_dataset['train']
                    else:
                        # off diagonal
                        dataset = self.round_two_datasets[feature]['train']

                    img_label = dataset[idx][1]
                    count = 0
                    while count <= len(dataset) and (img_label != label or idx in used_idxs):
                        idx += 1
                        count += 1
                        try:
                            img_label = dataset[idx][1]
                        except:
                            idx = 0
                    if count >= len(dataset):
                        print("failed to find feature combination {} {}-{}".format(
                            feature, i_label, j_label
                        ))
                        sub_ax = fig.add_subplot(grid[i, j])
                        sub_ax.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)

                    else:
                        sub_ax = fig.add_subplot(grid[i, j], frameon=False)

                        used_idxs.add(idx)
                        sub_ax.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
                        img = dataset[idx][0]
                        if img.shape[0] > 1:
                            self.cmap = None
                        else:
                            self.cmap = "Greys_r"
                        if img.dtype == torch.float:
                            img = (img*255).to(torch.uint8)

                        sub_ax.set_title("label: {} ($C^{}_{}={}$)".format(
                                img_label.item(),
                                self.latent_to_idx[feature],
                                self.latents_classes[dataset.get_original_index(idx), self.latent_to_idx[feature]],
                                self.latents_values[dataset.get_original_index(idx), self.latent_to_idx[feature]]),
                            fontsize=10)
                        if img.shape[0] > 1:
                            plt.imshow(img.permute((1, 2, 0)).contiguous().squeeze().to(torch.uint8), cmap=self.cmap)
                        else:
                            plt.imshow(img.squeeze(), cmap=self.cmap)
                        # plt.show()
                        a = 1


            fig.savefig(
                "C:/Users/Luca/Downloads/{}.png".format(feature),
                bbox_inches="tight"
            )
            plt.show()
        return True


class BWDSpritesCreator(FeatureCombinationCreator):

    latents_sizes = np.array([1,  3,  6, 40, 32, 32])

    def __init__(self, data_path="./data/", filename='dsprites.npz'):
        super().__init__(data_path, filename)
        self.cmap = 'Greys_r'


class ColorDSpritesCreator(FeatureCombinationCreator):

    latents_sizes = np.array([4, 3, 6, 40, 32, 32])

    def __init__(self, data_path="./data/", filename='color_dsprites.h5'):
        super().__init__(data_path, filename)
        self.cmap = None


class MultiDSpritesCreator(FeatureCombinationCreator):

    latents_ranges = {
        'object_number': (0, 4),
        'color': (0, 1),
        'shape': (0, 3),
        'scale': (0.5, 1),
        'orientation': (0, 2 * np.pi),
        'x_position': (0, 1),
        'y_position': (0, 1),
    }
    idx_to_latent = {
        0: 'object_number',
        1: 'color',
        2: 'shape',
        3: 'scale',
        4: 'orientation',
        5: 'x_position',
        6: 'y_position',
    }
    latent_to_idx = {
        'object_number': 0,
        'color': 1,
        'shape': 2,
        'scale': 3,
        'orientation': 4,
        'x_position': 5,
        'y_position': 6,
    }

    latents_sizes = np.array([4, 1, 3, 6, 40, 32, 32])

    def __init__(self, data_path="./data/", filename='multi_bwdsprites.h5'):
        super().__init__(data_path, filename)
        self.cmap = None


class MultiColorDSpritesCreator(FeatureCombinationCreator):

    latents_ranges = {
        'object_number': (0, 4),
        'color': (0, 4),
        'shape': (0, 3),
        'scale': (0.5, 1),
        'orientation': (0, 2 * np.pi),
        'x_position': (0, 1),
        'y_position': (0, 1),
    }
    idx_to_latent = {
        0: 'object_number',
        1: 'color',
        2: 'shape',
        3: 'scale',
        4: 'orientation',
        5: 'x_position',
        6: 'y_position',
    }
    latent_to_idx = {
        'object_number': 0,
        'color': 1,
        'shape': 2,
        'scale': 3,
        'orientation': 4,
        'x_position': 5,
        'y_position': 6,
    }

    latents_sizes = np.array([4, 4, 3, 6, 40, 32, 32])

    def __init__(self, data_path="./data/", filename='UTKFace.h5'):
        super().__init__(data_path, filename)
        self.cmap = None


class UTKFaceCreator(FeatureCombinationCreator):

    def __init__(self, data_path="./data/", filename='multi_color_dsprites.h5'):
        super().__init__(data_path, filename)

        self.idx_to_latent = {
            0: 'identity',
            1: 'age',
            2: 'gender',
            3: 'ethnicity',
        }
        self.latent_to_idx = {
            'identity': 0,
            'age': 1,
            'gender': 2,
            'ethnicity': 3,
        }

        self.latents_ranges = {}
        for latent in self.latent_to_idx.keys():
            class_max = max(self.dataset['latents_classes'][:, self.latent_to_idx[latent]]) + 1
            self.latents_ranges[latent] = (0, class_max)
            self.latents_sizes[self.latent_to_idx[latent]] = class_max

        self.cmap = None