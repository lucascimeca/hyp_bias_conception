import numpy as np
import matplotlib.pyplot as plt
from utils.results_utils import *

TENSORBOARD_FOLDER = "./../runs/"
RESULTS_FOLDER = "./../results/generated/"

# ARCHITECTURE = "resnet"
# ARCHITECTURE = "fft"
ARCHITECTURE = "vit"
PLOT_TRAIN_LOSS = False
PLOT_TRAIN_ACC = False
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


def plot_shades(data, title='Loss', x_label='Epoch', y_label='Loss', smoothing_parameter=0.6, cap=20, depth=0,
                type='acc', smoothing=True, ylim=None):
    fig = plt.figure(figsize=(5, 4))
    ax = fig.add_subplot(111)

    text_labels = []

    for i, key in enumerate(data.keys()):

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
        ys = data[key][:, :, -1]  # value
        xs = data[key][0, :, 1]   # step
        ys_avg = np.average(ys, axis=0)

        # smooth curves
        if smoothing:
            for idx in range(1, ys_avg.shape[0]):
                ys_avg[idx] = ys_avg[idx-1] * smoothing_parameter + (1 - smoothing_parameter) * ys_avg[idx]

        # prepare +- std ranges for figure
        ys_std = np.std(ys, axis=0)
        ys_max = ys_avg + ys_std
        ys_min = ys_avg - ys_std

        ax.plot(xs[xs < cap],
                ys_avg[xs < cap],
                label="{} avg. {} $\\pm \\sigma$".format(task_name, type),
                color=feature_to_color_dict[task_name])
        ax.fill_between(xs[xs < cap],
                        ys_max[xs < cap],
                        ys_min[xs < cap],
                        facecolor=feature_to_color_dict[task_name],
                        alpha=0.35)


        if task_name != "":
            text_labels.append([xs[xs < cap][-7], ys_avg[xs < cap][-7]+2, task_name])

    text_labels = sorted(text_labels, key=lambda x: x[1], reverse=True)
    text_ys = [x[1] for x in text_labels]
    for i, (x, y, label) in enumerate(text_labels):
        while np.any([abs(y-lby) <= 10 for lby in text_ys[:i]]):
            y -= 10
        text_ys[i] = y
        plt.text(x, y, label,
                 fontsize=16,
                 color=feature_to_color_dict[label],
                 horizontalalignment='center')

    # plt.autoscale(enable=True, axis='y')
    if ylim is None:
        axes = plt.gca()
        y_min, y_max = axes.get_ylim()
        plt.ylim([y_min, y_max+5])
    else:
        plt.ylim(ylim)

    # ax.set_title(title)
    ax.legend(loc='lower left')
    ax.set_xlabel(x_label, fontsize=12)
    ax.set_ylabel(y_label, fontsize=12)
    ax.grid()
    plt.show()

    if 'test' in title.lower():
        fig.savefig(os.path.join(RESULTS_FOLDER, '{}_depth_{}_training_comparison_{}.png'.format(ARCHITECTURE, depth, title.split(" ")[0])),
                    format='png',
                    # dpi=300,
                    bbox_inches='tight')


def plot_all(data, title='Loss', x_label='Epoch', y_label='Loss', smoothing_parameter=0.6,
                type='acc', smoothing=True, ylim=None):
    fig = plt.figure(figsize=(7, 4))
    ax = fig.add_subplot(111)
    for i, key in enumerate(data.keys()):
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
        ys = data[key][:, :, -1]  # value
        xs = data[key][:, :, 1]  # step
        # smooth curves
        if smoothing:
            for idx in range(1, ys.shape[1]):
                ys[:, idx] = ys[:, idx - 1] * smoothing_parameter + (1 - smoothing_parameter) * ys[:, idx]
        # prepare +- std ranges for figure
        for j in range(ys.shape[0]):
            if j == 0:
                label = "{} avg. {} $\\pm \\sigma$".format(task_name, type)
            else:
                label = None
            ax.plot(xs[j, ys[j, :] >= 0],
                    ys[j, ys[j, :] >= 0],
                    label=label,
                    color=feature_to_color_dict[task_name])
    ax.set_title(title)
    ax.legend()
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.grid()
    plt.show()


if __name__ == "__main__":

    if not folder_exists(RESULTS_FOLDER):
        folder_create(RESULTS_FOLDER)

    data_dict = convert_tb_data_depth(f"{TENSORBOARD_FOLDER}", apply_row_limits=False)

    for depth in data_dict.keys():
        for experiment_name in data_dict[depth].keys():
            data = data_dict[depth][experiment_name]

            # -------- PLOT TEST ACC --------
            test_acc_keys = [key for key in data.keys() if 'test' in key]
            if len(test_acc_keys) > 0:
                plot_data_dict = {}
                plot_labels_dict = {}
                for key in test_acc_keys:
                    plot_data_dict[key] = data[key]
                    labels = key.split('/')
                    plot_labels_dict[key] = labels[0] + '_' + labels[-1]

                plot_shades(data=plot_data_dict,
                            title='{} - Test Accuracy - depth={}'.format(experiment_name, depth),
                            x_label='Epoch',
                            y_label='Accuracy',
                            # smoothing_parameter=0.0,
                            type='acc',
                            depth=depth,
                            cap=160,
                            ylim=(-5, 110)
                            )
