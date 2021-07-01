import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from utils.results_utils import *
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import os

TENSORBOARD_FOLDER = "./../runs/"
RESULTS_FOLDER = "./../results/generated/"
DIRECTION_FILES_FOLDER = "./models/pretrained/direction_files/"
DIRECTION_PRETRAINED_FOLDER = "./models/pretrained/"
MODE_CONNECTIVITY_FOLDER = "../models/pretrained/mode_files/"
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
NORMALIZE = False

# ['#e52592', '#425066', '#12b5cb', '#f9ab00', '#9334e6', '#7cb342', '#e8710a']
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


class LogNormalize(colors.Normalize):

    def __init__(self, vmin=None, vmax=None, clip=None, log_alpha=None):
        self.log_alpha = log_alpha
        colors.Normalize.__init__(self, vmin, vmax, clip)

    def __call__(self, value, clip=None):
        log_v = np.ma.log(value - self.vmin)
        log_v = np.ma.maximum(log_v, self.log_alpha)
        return 0.9 * (log_v - self.log_alpha) / (np.log(self.vmax - self.vmin) - self.log_alpha)


def plane(grid, values, vmax=None, log_alpha=-5, N=7, cmap='jet_r'):
    cmap = plt.get_cmap(cmap)
    if vmax is None:
        clipped = values.copy()
    else:
        clipped = np.minimum(values, vmax)
    log_gamma = (np.log(clipped.max() - clipped.min()) - log_alpha) / N
    levels = clipped.min() + np.exp(log_alpha + log_gamma * np.arange(N + 1))
    levels[0] = clipped.min()
    levels[-1] = clipped.max()
    levels = np.concatenate((levels, [1e10]))
    norm = LogNormalize(clipped.min() - 1e-8, clipped.max() + 1e-8, log_alpha=log_alpha)
    contour = plt.contour(grid[:, :, 0], grid[:, :, 1], values, cmap=cmap, norm=norm,
                          linewidths=2.5,
                          zorder=1,
                          levels=levels)
    contourf = plt.contourf(grid[:, :, 0], grid[:, :, 1], values, cmap=cmap, norm=norm,
                            levels=levels,
                            zorder=0,
                            alpha=0.55)
    colorbar = plt.colorbar(format='%.2g')
    labels = list(colorbar.ax.get_yticklabels())
    labels[-1].set_text(r'$>\,$' + labels[-2].get_text())
    colorbar.ax.set_yticklabels(labels)
    return contour, contourf, colorbar


def plot_curve(file, feature_from, feature_to, results_folder, vmax=None, log_alpha=-5.0, dataset_name='diagonal'):
    plt.figure(figsize=(12.4, 7))

    if'diag' in dataset_name:
        ys = file['tr_nll']
        feature = "Diagonal"
    elif 'to' in dataset_name:
        ys = file['te_to_nll']
        feature = f"{feature_to}"
    elif 'from' in dataset_name:
        ys = file['te_from_nll']
        feature = f"{feature_from}"

    contour, contourf, colorbar = plane(
        file['grid'],
        ys,
        vmax=vmax,
        log_alpha=log_alpha,
        N=7
    )

    bend_coordinates = file['bend_coordinates']
    curve_coordinates = file['curve_coordinates']

    plt.title("Connectivity {} to {} --  {} dataset".format(feature_from, feature_to, feature), fontsize=24)

    plt.scatter(bend_coordinates[[0, 2], 0][0], bend_coordinates[[0, 2], 1][0], marker='o', c='k', s=120, zorder=2)
    plt.text(bend_coordinates[[0, 2], 0][0], bend_coordinates[[0, 2], 1][0]-file['grid'][0, :, 1].max()/10, f"{feature_from}",
             fontsize=20, ha='center', backgroundcolor='w')

    plt.scatter(bend_coordinates[[0, 2], 0][1], bend_coordinates[[0, 2], 1][1], marker='o', c='k', s=120, zorder=2)
    plt.text(bend_coordinates[[0, 2], 0][1], bend_coordinates[[0, 2], 1][1]-file['grid'][0, :, 1].max()/10, f"{feature_to}",
             fontsize=20, ha='center', backgroundcolor='w')

    plt.scatter(bend_coordinates[1, 0], bend_coordinates[1, 1], marker='D', c='k', s=120, zorder=2)

    plt.plot(curve_coordinates[:, 0], curve_coordinates[:, 1], linewidth=4, c='k', label='$w(t)$', zorder=4)
    plt.plot(bend_coordinates[[0, 2], 0], bend_coordinates[[0, 2], 1], c='k', linestyle='--', dashes=(3, 4),
             linewidth=3, zorder=2)

    plt.margins(0.0)
    plt.yticks(fontsize=18)
    plt.xticks(fontsize=18)
    colorbar.ax.tick_params(labelsize=18)
    plt.savefig(os.path.join(results_folder, 'train_loss_plane_from-{}_to-{}_dataset-{}.pdf'.format(feature_from, feature_to, dataset_name)),
                format='pdf',
                bbox_inches='tight')
    plt.show()


def plot_curve_trail(curve_file, feature_from, feature_to, kpi='loss', results_folder=""):
    fig = plt.figure(figsize=(7, 4))
    ax = fig.add_subplot(111)
    ax.grid()
    ax.set_title("Path tracing", fontsize=18)

    if kpi == 'loss':
        ys_diag = curve_file['tr_loss']
        ys_from = curve_file['te_from_loss']
        ys_to = curve_file['te_to_loss']
        ax.set_ylabel('Loss', fontsize=14)
    else:
        ys_diag = curve_file['tr_err']
        ys_from = curve_file['te_from_err']
        ys_to = curve_file['te_to_err']
        ax.set_ylabel('Error (%)', fontsize=14)

    plt.plot(ys_diag,
             color=feature_to_color_dict["diagonal"],
             label="Diagonal",
             lw=3,
             alpha=.8)
    plt.plot(ys_from,
             color=feature_to_color_dict[feature_from],
             label=f"{feature_from}".capitalize(),
             lw=3,
             alpha=.8)
    plt.plot(ys_to,
             color=feature_to_color_dict[feature_to],
             label=f"{feature_to}".capitalize(),
             lw=3,
             alpha=.8)

    ax.legend(fontsize=12)
    ax.set_xlabel('Path Step', fontsize=14)
    plt.show()
    fig.savefig(os.path.join(results_folder, 'loss_trail_from-{}_to-{}--{}.pdf'.format(feature_from, feature_to, kpi)),
                format='pdf',
                bbox_inches='tight')


if __name__ == "__main__":

    if not folder_exists(RESULTS_FOLDER):
        folder_create(RESULTS_FOLDER)

    print("retrieving solutions weights...")
    mode_folders = get_filenames(MODE_CONNECTIVITY_FOLDER)
    for mode_folder in mode_folders:#

        feature_from = mode_folder.split("-")[0]
        feature_to = mode_folder.split("-")[-1]

        plane_file = np.load(os.path.join(MODE_CONNECTIVITY_FOLDER + mode_folder, 'plane.npz'))

        plot_curve(file=plane_file,
                   feature_from=feature_from,
                   feature_to=feature_to,
                   results_folder=RESULTS_FOLDER,
                   dataset_name="diagonal")

        plot_curve(file=plane_file,
                   feature_from=feature_from,
                   feature_to=feature_to,
                   results_folder=RESULTS_FOLDER,
                   dataset_name="from")

        plot_curve(file=plane_file,
                   feature_from=feature_from,
                   feature_to=feature_to,
                   results_folder=RESULTS_FOLDER,
                   dataset_name="to")

        curve_file = np.load(os.path.join(MODE_CONNECTIVITY_FOLDER + mode_folder, 'curve.npz'))

        plot_curve_trail(kpi="loss",
                         curve_file=curve_file,
                         feature_from=feature_from,
                         feature_to=feature_to,
                         results_folder=RESULTS_FOLDER)

        plot_curve_trail(kpi="err",
                         curve_file=curve_file,
                         feature_from=feature_from,
                         feature_to=feature_to,
                         results_folder=RESULTS_FOLDER)