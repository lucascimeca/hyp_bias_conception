import numpy as np
import matplotlib.pyplot as plt
from utils.results_utils import *

TENSORBOARD_FOLDER = "./../runs/"
RESULTS_FOLDER = "./../results/generated/"

# ARCHITECTURE = "resnet"
# ARCHITECTURE = "fft"
ARCHITECTURE = "vit"
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


if __name__ == "__main__":

    feature_augmentation_folders = get_filenames(TENSORBOARD_FOLDER)

    dim = len(feature_augmentation_folders[0])


    #### ACCURACY #####
    for feature in feature_augmentation_folders:
        file = get_filenames(f"{TENSORBOARD_FOLDER}{feature}/")[0]

        data = np.load(f"{TENSORBOARD_FOLDER}{feature}/{file}")

        fig = plt.figure(figsize=(7, 4))
        ax = fig.add_subplot(111)

        xs = data['augmentation_no']

        keys = [key for key in data.keys() if "test" in key and "acc" in key]
        for key in keys:
            ys = data[key]

            tested_feature = key.split("_")[-1]
            plt.plot(xs, ys,
                     label=tested_feature,
                     color=feature_to_color_dict[tested_feature])

        plt.ylim((0, 105))
        ax.set_title(f"Augmentation by {feature}")
        ax.legend()
        ax.set_xlabel("injection amount")
        ax.set_ylabel("Accuracy")
        ax.grid()
        plt.show()
