import re
import pandas as pd
import torch
from utils.misc.simple_io import *
from tensorflow.python.summary.summary_iterator import summary_iterator


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


def convert_tb_data(root_dir, apply_row_limits=True):

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

            files = sorted(data_dict[run_name][exp_name].keys())
            for file in files:
                data = data_dict[run_name][exp_name][file]
                for value_name in data.keys():
                    exp_key = exp_name_base + value_name

                    if exp_key not in numpy_dict[run_name].keys():
                        if apply_row_limits:
                            numpy_dict[run_name][exp_key] = data[value_name][:row_limits[run_name][value_name], :].reshape(
                                (1, row_limits[run_name][value_name], data[value_name].shape[-1]))
                        else:
                            numpy_dict[run_name][exp_key] = data[value_name].reshape(
                                (1, data[value_name].shape[-2], data[value_name].shape[-1]))

                    else:
                        if apply_row_limits:
                            numpy_dict[run_name][exp_key] = np.concatenate(
                                (numpy_dict[run_name][exp_key], data[value_name][:row_limits[run_name][value_name], :]
                                 .reshape((1, row_limits[run_name][value_name], data[value_name].shape[-1]))),
                                axis=0)
                        else:
                            if numpy_dict[run_name][exp_key].shape[1] < data[value_name].shape[0]:
                                # create a zero array to fill the gap of unseen test data
                                fill = np.ones((numpy_dict[run_name][exp_key].shape[0],
                                                     data[value_name].shape[0] - numpy_dict[run_name][exp_key].shape[1],
                                                     numpy_dict[run_name][exp_key].shape[2])) * -1
                                numpy_dict[run_name][exp_key] = np.concatenate(
                                    (numpy_dict[run_name][exp_key], fill), axis=1
                                )

                            elif numpy_dict[run_name][exp_key].shape[1] > data[value_name].shape[0]:
                                # create a zero array to fill the gap of unseen test data
                                fill = np.ones((numpy_dict[run_name][exp_key].shape[1] - data[value_name].shape[0],
                                                     data[value_name].shape[1])) * -1

                                data[value_name] = np.concatenate(
                                    (data[value_name], fill), axis=0
                                )

                            numpy_dict[run_name][exp_key] = np.concatenate(
                                (numpy_dict[run_name][exp_key], data[value_name]
                                 .reshape((1, data[value_name].shape[-2], data[value_name].shape[-1]))),
                                axis=0)
    return numpy_dict


def load_weight_data(root_dir):

    print("retrieving training & validation curves")
    acc_data = convert_tb_data(root_dir, apply_row_limits=False)

    filename_groups = []
    for (root, _, fs) in os.walk(root_dir):
        filenames = []
        for filename in fs:
            if "weights" not in filename:
                continue
            filenames += [(root, filename)]
        if len(filenames) > 0:
            filename_groups += [filenames]

    data_dict = {}
    label_overfit_dictionary = {}
    for filenames in filename_groups:
        sorted_filenames = sorted(filenames, key=lambda x: int(x[1].split('-')[2]))
        sample_no_idx = -1
        for root, filename in sorted_filenames:
            path_info = re.split("[(/)(\\\)]", root)
            run_name = path_info[-1]
            sample_no = filename.split('-')[-2]
            epoch = filename.split('-')[-1].split('.')[0]

            if run_name not in data_dict.keys():
                data_dict[run_name] = {}
                label_overfit_dictionary[run_name] = []

            if sample_no not in data_dict[run_name]:
                sample_no_idx += 1
                data_dict[run_name][sample_no] = []
                label_overfit_dictionary[run_name] += [[]]

            # get validation curves
            test_acc_keys = [key for key in acc_data[run_name].keys() if 'test' in key and 'round_two' in key]
            if len(test_acc_keys) == 0:
                test_acc_keys = [key for key in acc_data[run_name].keys() if 'test' in key]
            epoch_accs = []
            epoch_labels = []
            for key in test_acc_keys:
                epoch_accs += [acc_data[run_name][key][sample_no_idx, int(epoch), -1]]
                labels = key.split('/')
                if "augmentation" in run_name:
                    epoch_labels += [run_name]
                else:
                    epoch_labels += [labels[0].split('_')[-1]]

            file_full_path = os.path.join(root, filename)
            data_dict[run_name][sample_no] += [file_full_path]
            label_overfit_dictionary[run_name][-1] += [epoch_labels[np.argmax(epoch_accs)]]

    data = {}
    for exp in data_dict.keys():
        shape0 = len(list(data_dict[exp].keys()))
        shape1 = len(data_dict[exp][list(data_dict[exp].keys())[0]])
        weight_tmp = torch.load(data_dict[exp][list(data_dict[exp].keys())[0]][0])
        shape2 = torch.cat([weight_tmp[key].view(1, -1) for key in weight_tmp.keys()], dim=1).shape[1]

        data[exp] = torch.zeros((shape0, shape1, shape2))
        for i, sample_no in enumerate(data_dict[exp].keys()):
            for j in range(len(data_dict[exp][sample_no])):
                weight_tmp = torch.load(data_dict[exp][sample_no][j])
                data[exp][i, j, :] = torch.cat([weight_tmp[key].view(1, -1) for key in weight_tmp.keys()], dim=1)
                print("progress for exp {}: {:.2f}%".format(exp, (i*shape1 + j)*100/(shape0*shape1)),
                      flush=True,
                      end='\r')
    return data, label_overfit_dictionary
