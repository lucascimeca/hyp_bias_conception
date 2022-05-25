import numpy as np
import matplotlib.pyplot as plt
from utils.results_utils import *

TENSORBOARD_FOLDER = "./../runs/"
RESULTS_FOLDER = "./../results/generated/"

# ARCHITECTURE = "resnet"
# ARCHITECTURE = "fft"
ARCHITECTURE = "resnet"
PLOT_TRAIN_LOSS = False
PLOT_TRAIN_ACC = True
PLOT_VALID_ACC = True


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




def plot_shades(all_xs_data, all_ys_data, title='Loss', x_label='Epoch', y_label='Loss', smoothing_parameter=0.6,
                cap=None, type='acc', smoothing=True, ylim=None):
    fig = plt.figure(figsize=(5, 4))
    ax = fig.add_subplot(111)

    text_labels = []

    for i, key in enumerate(all_xs_data.keys()):

        # extract task name
        k_labels = key.split('/')[0].split('_task_')
        if len(k_labels) == 1:
            if len(k_labels) != 0 and 'augmentation' in k_labels[0]:
                task_name = k_labels[0]
            else:
                task_name = "diagonal"
        else:
            task_name = k_labels[1]

        # extract data
        ys = np.concatenate([arr.reshape(1, -1) for arr in all_ys_data[key]], axis=0)
        xs = all_xs_data[key][0]
        ys_avg = np.average(ys, axis=0)

        # smooth curves
        if smoothing:
            for idx in range(1, ys_avg.shape[0]):
                ys_avg[idx] = ys_avg[idx-1] * smoothing_parameter + (1 - smoothing_parameter) * ys_avg[idx]

        # prepare +- std ranges for figure
        ys_std = np.std(ys, axis=0)
        ys_max = ys_avg + ys_std
        ys_min = ys_avg - ys_std

        if cap is None:
            cap = np.max(xs) + 1
        ax.plot(xs[xs < cap],
                ys_avg[xs < cap],
                label="{} avg. {} $\\pm \\sigma$".format(task_name, type),
                color=feature_to_color_dict[task_name],
                marker='*',
                linewidth=2)
        ax.fill_between(xs[xs < cap],
                        ys_max[xs < cap],
                        ys_min[xs < cap],
                        facecolor=feature_to_color_dict[task_name],
                        alpha=0.35)

        if task_name != "":
            text_labels.append([xs[xs < cap][-10], ys_avg[xs < cap][-10]+2, task_name])

    text_labels = sorted(text_labels, key=lambda x: x[1], reverse=True)
    text_ys = [x[1] for x in text_labels]
    for i, (x, y, label) in enumerate(text_labels):
        while np.any([abs(y-lby) <= 10 for lby in text_ys[:i]]):
            y -= 10
        text_ys[i] = y
        plt.text(x, y, label,
                 fontsize=16,
                 color=feature_to_color_dict[label])

    plt.ylim((0, 105))
    ax.set_title(f"Radial accuracy from {feature.split('_')[0]} augmented minima")
    # ax.legend()
    ax.set_xlabel("Radial distance at initialization")
    ax.set_ylabel("Accuracy")
    ax.grid()

    plt.show()
    fig.savefig(os.path.join(RESULTS_FOLDER, '{}_radius_rerun_{}.pdf'.format(ARCHITECTURE, feature)),
                format='pdf',
                dpi=300,
                bbox_inches='tight')




if __name__ == "__main__":

    feature_augmentation_folders = get_filenames(TENSORBOARD_FOLDER)

    dim = len(feature_augmentation_folders[0])


    #### ACCURACY #####
    for feature in feature_augmentation_folders:


        sample_files = get_filenames(f"{TENSORBOARD_FOLDER}{feature}/")

        file_data = np.load(f"{TENSORBOARD_FOLDER}{feature}/{sample_files[0]}")

        all_ys_data = {}
        all_xs_data = {}
        all_labels = {}
        for file in sample_files:
            file_data = np.load(f"{TENSORBOARD_FOLDER}{feature}/{file}")
            keys = [key for key in file_data.keys() if "test" in key and "acc" in key]
            for key in keys:
                if key not in all_ys_data.keys():
                    all_ys_data[key] = [file_data[key]]
                    all_xs_data[key] = [file_data['radiuses']]
                    all_labels[key] = key.split("_")[-1]
                else:
                    all_ys_data[key].append(file_data[key])
                    all_xs_data[key].append(file_data['radiuses'])

        plot_shades(all_xs_data, all_ys_data, cap=None)

        # fig = plt.figure(figsize=(7, 4))
        # ax = fig.add_subplot(111)
        # for key in all_ys_data.keys():
        #     ys = np.concatenate([arr.reshape(-1, 1) for arr in all_ys_data[key]], axis=0)
        #     xs = np.concatenate([arr.reshape(-1, 1) for arr in all_xs_data[key]], axis=0)
        #
        #     plt.plot(xs, ys,
        #              label=all_labels[key],
        #              color=feature_to_color_dict[all_labels[key]])
        #
        #
        # ax.grid()
        # plt.show()
