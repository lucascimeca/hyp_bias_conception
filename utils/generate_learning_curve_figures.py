import numpy as np
import matplotlib.pyplot as plt
from utils.results_utils import *

TENSORBOARD_FOLDER = "./../runs/"

# ['#e52592', '#425066', '#12b5cb', '#f9ab00', '#9334e6', '#7cb342', '#e8710a']
feature_to_color_dict = {
    "diagonal": '#e52592',
    "scale": '#425066',
    "color": '#12b5cb',
    "shape": '#9334e6',
    "orientation": '#e8710a',
    "object_number": '#7cb342',
    "x_position": '#62733c',
    "age": '#ad63a8',
    "gender": '#ff0004',
    "ethnicity": '#7a7a7a'
}


def plot_shades(data,  data_label, title='Loss', x_label='Epoch', y_label='Loss', smoothing_parameter=0.6, cap=20,
                type='acc', smoothing=True, ylim=None):
    fig = plt.figure(figsize=(7, 4))
    ax = fig.add_subplot(111)

    for i, key in enumerate(data.keys()):

        # extract task name
        k_labels = key.split('/')[0].split('_task_')
        if len(k_labels) == 1:
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
            plt.text(xs[xs < cap][-5],
                     ys_avg[xs < cap][-5]+2,
                     task_name,
                     fontsize=14,
                     color=feature_to_color_dict[task_name],
                     horizontalalignment='center')

    # plt.autoscale(enable=True, axis='y')
    if ylim is None:
        axes = plt.gca()
        y_min, y_max = axes.get_ylim()
        plt.ylim([y_min, y_max+5])
    else:
        plt.ylim(ylim)

    ax.set_title(title)
    ax.legend()
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.grid()
    plt.show()


if __name__ == "__main__":
    PLOT_TRAIN_LOSS = False
    PLOT_TRAIN_ACC = False
    PLOT_VALID_ACC = True

    data_dict = convert_tb_data(f"{TENSORBOARD_FOLDER}")

    for experiment_name in data_dict.keys():
        data = data_dict[experiment_name]

        # ------------------- PLOT FIGURES ------------------

        if PLOT_TRAIN_LOSS:
            # -------- PLOT TRAIN LOSS --------
            train_loss_keys = [key for key in data.keys() if 'train_loss' in key]
            if len(train_loss_keys) > 0:

                loss_data = data[train_loss_keys[0]]
                plot_shades(data={'loss': loss_data},
                            data_label={'loss': 'training loss'},
                            title='{} - Train Loss'.format(experiment_name),
                            x_label='Epoch',
                            y_label='Loss',
                            type='loss')

        if PLOT_TRAIN_ACC:
            # -------- PLOT TRAIN ACC --------
            train_acc_keys = [key for key in data.keys() if 'train_acc' in key]
            if len(train_acc_keys) > 0:

                acc_data = data[train_acc_keys[0]]
                plot_shades(data={'acc': acc_data},
                            data_label={'acc': 'training accuracy'},
                            title='{} - Train Accuracy'.format(experiment_name),
                            x_label='Epoch',
                            y_label='Accuracy',
                            type='acc')

        if PLOT_VALID_ACC:
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
                            data_label=plot_labels_dict,
                            title='{} - Test Accuracy'.format(experiment_name),
                            x_label='Epoch',
                            y_label='Accuracy',
                            smoothing_parameter=0.0,
                            type='acc',
                            cap=30,
                            ylim=(-5, 110)
                            )
