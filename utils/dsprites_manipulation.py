from utils.data_loader import *
import numpy as np
import matplotlib

# import matplotlib.pyplot as plt
matplotlib.use('TkAgg')


########### change all #################

# dsc = ColorDSpritesCreator(data_path='./../data/')
# dsc.get_dataset_fvar(
#     number_of_samples=10000,
#     features_variants=('color', 'shape'),
#     color=(0, 4),
#     shape=(0, 3),
#     scale=(0.5, 1),
#     orientation=(0, 2 * np.pi),
#     x_position=(0, 1),
#     y_position=(0, 1),
# )
# dsc.show_tasks()
#
########## change all #################
# dsc = BWDSpritesCreator(data_path='./../data/')
# dsc.get_dataset_fvar(
#     number_of_samples=10000,
#     features_variants=('shape', 'scale'),
#     color=(0, 1),
#     shape=(0, 3),
#     scale=(0.5, 1),
#     orientation=(0, 2 * np.pi),
#     x_position=(0, 1),
#     y_position=(0, 1),
# )
# dsc.show_tasks()

########## change all #################
# dsc = MultiDSpritesCreator(data_path='./../data/', filename='multi_bwdsprites.h5')
# dsc.get_dataset_fvar(
#     number_of_samples=10000,
#     features_variants=('object_number', 'scale'),
#     object_number=(0, 4),
#     color=(0, 1),
#     shape=(0, 3),
#     scale=(0.5, 1),
#     orientation=(0, 2 * np.pi),
#     x_position=(0, 1),
#     y_position=(0, 1),
# )
# dsc.show_tasks()


########## change all #################
dsc = MultiColorDSpritesCreator(data_path='./../data/', filename='multi_color_dsprites.h5')
dsc.get_dataset_fvar(
    number_of_samples=10000,
    features_variants=('object_number', 'color'),
    object_number=(0, 4),
    color=(0, 4),
    shape=(0, 3),
    scale=(0.5, 1),
    orientation=(0, 2 * np.pi),
    x_position=(0, 1),
    y_position=(0, 1),
)
dsc.show_tasks()

# ########### pos first quadrant #################
# dsc.get_dataset(
#     number_of_samples=5000,
#     output_targets=['x_position', 'y_position'],
#     color=(0, 1),
#     shape=(0, 3),
#     scale=(0.5, 1),
#     orientation=(0, 2 * np.pi),
#     x_position=(0, .5),
#     y_position=(0, .5),
# )
# dsc.show_examples(num_images=25)
# dsc.show_density()
#
# ########### pos third quadrant #################
# dsc.get_dataset(
#     number_of_samples=5000,
#     output_targets=['x_position', 'y_position'],
#     color=(0, 1),
#     shape=(0, 3),
#     scale=(0.5, 1),
#     orientation=(0, 2 * np.pi),
#     x_position=(.5, 1),
#     y_position=(.5, 1),
# )
# dsc.show_examples(num_images=25)
# dsc.show_density()
#
# ########### change orientation #################
# dsc.get_dataset(
#     number_of_samples=5000,
#     output_targets=['x_position', 'y_position'],
#     color=(0, 1),
#     shape=(2, 3),
#     scale=(0.999, 1),
#     orientation=(0, 2 * np.pi),
#     x_position=(.5, .5001),
#     y_position=(.5, .5001),
# )
# dsc.show_examples(num_images=25)
# dsc.show_density()
