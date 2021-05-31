
import torch
import copy
from utils.surf.plot_2D import *
from utils.surf.h52vtp import *
from utils.misc.simple_io import *

TENSORBOARD_FOLDER = "./runs/"
DIRECTION_FILES_FOLDER = "../models/pretrained/direction_files/"
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


files = get_filenames(DIRECTION_FILES_FOLDER)
files = [file for file in files if 'data' in file and file.endswith("h5")]

for file in files:
    filepath = DIRECTION_FILES_FOLDER + file
    plot_2d_contour(filepath, surf_name='train_loss', vmin=0.0, vmax=10., vlevel=0.05, show=False)
    h5_to_vtp(filepath, surf_name='train_loss', log=True, zmax=10, interp=1000)