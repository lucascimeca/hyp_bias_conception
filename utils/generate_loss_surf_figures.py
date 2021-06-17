
import torch
import copy
from utils.surf.plot_2D import *
from utils.surf.h52vtp import *
from utils.misc.simple_io import *

TENSORBOARD_FOLDER = "./runs/"
DIRECTION_FILES_FOLDER = "../models/pretrained/direction_files/"
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

feature_to_color_dict = {
    "diagonal": '#e52592',
    "scale": '#425066',  "scale_augmentation": '#425066',
    "color": '#12b5cb',  "color_augmentation": '#12b5cb',
    "shape": '#9334e6',  "shape_augmentation": '#9334e6',
    "orientation": '#e8710a',  "orientation_augmentation": '#e8710a',
    "object_number": '#7cb342',  "object_number_augmentation": '#7cb342',
    "x_position": '#62733c',  "x_position_augmentation": '#62733c',
    "age": '#ad63a8',  "age_augmentation": '#ad63a8',
    "gender": '#ff0004',  "gender_augmentation": '#ff0004',
    "ethnicity": '#7a7a7a',  "ethnicity_augmentation": '#7a7a7a',
}


def plot_spherical_rs(folder, cutoff=None, log_scale=False, type='loss', color_dict=None, deviation=False):
    files = get_filenames(folder)

    files_sphere = [file for file in files if file.endswith("npz")]
    if len(files_sphere) != 0:

        fig = plt.figure(figsize=(7, 4))
        ax = fig.add_subplot(111)
        plt.title("{} by spherical r".format(type.capitalize()))
        ax.set_xlabel("r")
        ax.set_ylabel(type)

        for file in files_sphere:
            filepath = DIRECTION_FILES_FOLDER + file
            data = np.load(filepath)

            task_name = file.split('-')[0].split("_")[0]
            local_rs = data['local_rs']

            if cutoff is None:
                cutoff = local_rs.max() + 1

            if type == 'loss':
                ys = data['spherical_losses'][local_rs < cutoff]
            else:
                ys = data['spherical_accs'][local_rs < cutoff]

            ys_avg = ys.mean(axis=1)
            if deviation:
                ys_std = ys.std(axis=1)
            else:
                ys_std = np.sqrt(ys.std(axis=1))/ys.shape[1]

            ax.plot(local_rs[local_rs < cutoff], ys_avg, label=task_name,
                    color=color_dict[task_name])
            ax.fill_between(local_rs[local_rs < cutoff], ys_avg + ys_std, ys_avg - ys_std,
                            facecolor=color_dict[task_name], alpha=0.35)

        if log_scale:
            plt.yscale("log")

        plt.legend()
        plt.show()



if __name__ == '__main__':
    plot_spherical_rs(DIRECTION_FILES_FOLDER, cutoff=3, type='loss', color_dict=feature_to_color_dict, deviation=False)
    plot_spherical_rs(DIRECTION_FILES_FOLDER, cutoff=3, type='loss', color_dict=feature_to_color_dict, deviation=True)
    plot_spherical_rs(DIRECTION_FILES_FOLDER, cutoff=3, type='accuracy', color_dict=feature_to_color_dict)

    files = get_filenames(DIRECTION_FILES_FOLDER)
    files_plane = [file for file in files if 'data' in file and file.endswith("h5")]
    for file in files_plane:
        filepath = DIRECTION_FILES_FOLDER + file
        # plot_2d_contour(filepath, surf_name='train_loss', vmin=0.0, vmax=5., vlevel=0.05, show=False)
        # h5_to_vtp(filepath, surf_name='train_loss', log=True, zmax=10., interp=1000)
        plot_2d_contour(filepath, surf_name='train_acc', vmin=30.0, vmax=101., vlevel=5.0, show=False)