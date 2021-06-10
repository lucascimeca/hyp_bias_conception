from utils.data_loader import *
import numpy as np
import matplotlib

# import matplotlib.pyplot as plt
# matplotlib.use('TkAgg')

######### regular DSprites #################
# dsc = BWDSpritesCreator(data_path='./../data/', filename='bw_dsprites_pruned.h5')
# dsc.get_dataset_fvar(
#     number_of_samples=10000,
#     features_variants=('orientation', 'x_position'),
#     color=(0, 1),
#     shape=(0, 3),
#     scale=(0.5, 1),
#     orientation=(0, 2 * np.pi),
#     x_position=(0, 1),
#     y_position=(0, 1),
# )
# dsc.show_tasks()

########### Color DSprites #################

dsc = ColorDSpritesCreator(data_path='./../data/', filename='color_dsprites_pruned.h5')
dsc.get_dataset_fvar(
    number_of_samples='all',
    features_variants=('shape', 'scale', 'orientation', 'color'),
    train_split=1.,
    valid_split=0.
)
dsc.show_tasks()

########## Multi DSprites (still black and white but multiple elements) #################
# dsc = MultiDSpritesCreator(data_path='./../data/', filename='multi_bwdsprites.h5')
# dsc.get_dataset_fvar(
#     number_of_samples='all',
#     features_variants=('object_number', 'scale'),
#     train_split=1.,
#     valid_split=0.
# )
# dsc.show_tasks()


########## Multi-ColorD Sprites -> both colored and with many images #################
# dsc = MultiColorDSpritesCreator(data_path='./../data/', filename='multi_color_dsprites.h5')
# dsc.get_dataset_fvar(
#     number_of_samples='all',
#     features_variants=('object_number', 'color'),
#     train_split=1.,
#     valid_split=0.
# )
# dsc.show_tasks()

######### UTK Face dataset #################
# Data
dsc = UTKFaceCreator(data_path='./../data/', filename='UTKFace.h5')
dsc.get_dataset_fvar(
    number_of_samples='all',
    features_variants=('ethnicity', 'age', 'gender'),
    resize=(64, 64),
    train_split=1.,
    valid_split=0.
)
dsc.show_tasks()
