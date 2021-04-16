import re
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.python.summary.summary_iterator import summary_iterator

TENSORBOARD_FOLDER = "./../runs/"

# ['#e52592', '#425066', '#12b5cb', '#f9ab00', '#9334e6', '#7cb342', '#e8710a']
feature_to_color_dict = {
    "diagonal": '#e52592',
    "scale": '#425066',
    "color": '#12b5cb',
    "shape": '#9334e6',
    "orientation": '#e8710a',
    "object_number": '#7cb342'
}


def convert_tfevent(filepath):
    return pd.DataFrame([
        parse_tfevent(e) for e in summary_iterator(filepath) if len(e.summary.value)
    ])


def parse_tfevent(tfevent):
    return dict(
        wall_time=tfevent.wall_time,
        name=tfevent.summary.value[0].tag,
        step=tfevent.step,
        value=float(tfevent.summary.value[0].simple_value),
    )


def convert_tb_data(root_dir):

    # find all log files, extract and put them in dict.
    data_dict = {}
    row_limits = {}
    for (root, _, filenames) in os.walk(root_dir):
        for filename in filenames:
            if "events.out.tfevents" not in filename:
                continue

            path_info = re.split("[(/)(\\\)]", root)

            run_name = path_info[-2]
            exp_name = path_info[-1]

            if run_name not in data_dict.keys():
                data_dict[run_name] = {}
                row_limits[run_name] = {}

            if exp_name not in data_dict[run_name].keys():
                data_dict[run_name][exp_name] = {}

            values_dict = {}
            file_full_path = os.path.join(root, filename)
            df = convert_tfevent(file_full_path)

            if df.shape[0] == 0:
                continue

            df = df.sort_values(by=['wall_time'])

            value_names = df['name'].unique()
            for value_name in value_names:
                if filename not in data_dict[run_name][exp_name].keys():
                    data_dict[run_name][exp_name][filename] = {}

                values = np.array(df.loc[df['name'] == value_name]['value']).astype(np.float64).reshape(-1, 1)
                steps = np.array(df.loc[df['name'] == value_name]['step']).astype(np.float64).reshape(-1, 1)
                times = np.array(df.loc[df['name'] == value_name]['wall_time']).astype(np.float64).reshape(-1, 1)

                values_dict[value_name] = np.concatenate((times, steps, values), axis=1)

                if value_name not in row_limits[run_name] or row_limits[run_name][value_name] > values.shape[0]:
                    row_limits[run_name][value_name] = values.shape[0]

            data_dict[run_name][exp_name][filename] = values_dict

    numpy_dict = {}
    for run_name in data_dict.keys():
        numpy_dict[run_name] = {}

        for exp_name in data_dict[run_name].keys():
            exp_name_base = exp_name + '/'

            for file in data_dict[run_name][exp_name].keys():
                data = data_dict[run_name][exp_name][file]
                for value_name in data.keys():
                    exp_key = exp_name_base + value_name

                    if exp_key not in numpy_dict[run_name].keys():
                        numpy_dict[run_name][exp_key] = data[value_name][:row_limits[run_name][value_name], :].reshape(
                            (1, row_limits[run_name][value_name], data[value_name].shape[-1]))
                    else:
                        numpy_dict[run_name][exp_key] = np.concatenate(
                            (numpy_dict[run_name][exp_key], data[value_name][:row_limits[run_name][value_name], :].reshape(
                                (1, row_limits[run_name][value_name], data[value_name].shape[-1]))),
                            axis=0)
    return numpy_dict


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
    data_dict = convert_tb_data(f"{TENSORBOARD_FOLDER}")

    for experiment_name in data_dict.keys():
        data = data_dict[experiment_name]

        # ------------------- PLOT FIGURES ------------------

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
