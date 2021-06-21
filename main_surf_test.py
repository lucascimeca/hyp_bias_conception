from __future__ import print_function

import argparse
import random
import time

import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data as data
import torch.distributed as dist
import webbrowser
import zipfile
import sys
import traceback

import nsml
import copy
import models as models
import copy
import h5py
import socket
import os
import sys
import numpy as np
import torchvision
import utils.surf.mpi4pytorch as mpi

from utils.surf import dataloader
from utils.surf import evaluation
import utils.surf.projection as proj
from utils.surf import net_plotter
from utils.surf import plot_2D
from utils.surf import plot_1D
from utils.surf import scheduler
from models.convnet import ConvNet
from models.resnet import ResNet
from models.ffnet import FFNet
from timm.models.vision_transformer import VisionTransformer
from nsml import GPU_NUM
from nsml import PARALLEL_WORLD, PARALLEL_PORTS, MY_RANK
from utils import AverageMeter, accuracy
from utils.data_loader import *
from torch.utils.tensorboard import SummaryWriter
from tensorboard import program
from utils.misc.simple_io import *
from functools import partial

NSML = True
TENSORBOARD_FOLDER = "./runs/"
DIRECTION_FILES_FOLDER = "./models/pretrained/direction_files/"
DIRECTION_PRETRAINED_FOLDER = "./models/pretrained/"
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

global_step = 0


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def run_experiment(args, experiments=None, dsc=None, samples=10, scope=None):

    # A flag for distributed setup
    WORLD_SIZE = len(PARALLEL_WORLD)
    if WORLD_SIZE > 1:  # distributed setup
        master_addr = "tcp://{}:{}".format(PARALLEL_WORLD[0], PARALLEL_PORTS[0])
        print("Initiating distributed process group... My rank is {}".format(MY_RANK))
        dist.init_process_group(backend='gloo', init_method=master_addr, rank=MY_RANK, world_size=WORLD_SIZE)
        args.train_batch //= WORLD_SIZE

    #--------------------------------------------------------------------------
    # Check plotting resolution
    #--------------------------------------------------------------------------

    files = get_filenames(DIRECTION_PRETRAINED_FOLDER, file_ending='pth')
    for sample_file in files:
        global best_acc, global_step

        # Random seed
        if args.manualSeed is None:
            args.manualSeed = random.randint(1, 10000)
        random.seed(args.manualSeed)
        torch.manual_seed(args.manualSeed)
        if use_cuda:
            torch.cuda.manual_seed_all(args.manualSeed)


        fs = sample_file.split('-')
        arch = fs[0]
        exp_key = fs[1]
        sample_no = fs[2]
        feature_to_augment = exp_key.split("_")[0]

        #  -------------- DATA --------------------------
        resize = None
        if 'face' in args.dataset:
            resize = (64, 64)

        print('==> Preparing dataset dsprites ')
        round_one_dataset, round_two_datasets = dsc.get_dataset_fvar(
            number_of_samples=5000,
            features_variants=experiments[exp_key],
            resize=resize,
            train_split=1.,
            valid_split=0.,
        )

        num_classes = dsc.no_of_feature_lvs

        # create train dataset for main diagonal -- round one
        training_data = round_one_dataset['train']

        # augment with offdiagonal if necessary
        if 'augmentation' in exp_key:
            training_data.merge_new_dataset(round_two_datasets[feature_to_augment]['train'])  # main diag

        print("Data Length: {}".format(len(training_data)))

        # create train dataset for main diagonal -- round one
        trainloader = data.DataLoader(training_data, batch_size=args.train_batch,
                                      shuffle=True, num_workers=args.workers, worker_init_fn=seed_worker)

        args.xmin, args.xmax, args.xnum = -1., 1., 51
        args.ymin, args.ymax, args.ynum = -1., 1., 51

        #  -------------- MODEL --------------------------
        print("==> creating model '{}'".format(args.arch))
        if 'color' in args.dataset or 'face' in args.dataset:
            no_of_channels = 3
        else:
            no_of_channels = 1

        if 'resnet' in args.arch:
            model = ResNet(
                num_classes=num_classes,
                depth=args.depth,
                no_of_channels=no_of_channels)
        elif 'convnet' in args.arch:
            model = ConvNet(
                num_classes=num_classes,
                no_of_channels=no_of_channels)
        elif 'ffnet' in args.arch:
            model = FFNet(
                num_classes=num_classes,
                no_of_channels=no_of_channels)
        elif 'vit' in args.arch:
            model = VisionTransformer(
                img_size=64, patch_size=8, embed_dim=192, depth=12, num_heads=3, mlp_ratio=4, qkv_bias=True,
                norm_layer=partial(nn.LayerNorm, eps=1e-6), num_classes=num_classes,
                in_chans=no_of_channels)
        else:
            raise NotImplementedError()

        state_dict = torch.load(DIRECTION_PRETRAINED_FOLDER + sample_file)
        state_dict_rex = {key[7:]: state_dict[key] for key in state_dict.keys()}
        model.load_state_dict(state_dict_rex)

        w = net_plotter.get_weights(model)  # initial parameters
        s = copy.deepcopy(model.state_dict())  # deepcopy since state_dict are references

        model = nn.DataParallel(model).to(DEVICE)

        # --------------------------------------------------------------------------
        # Setup the direction file and the surface file
        # --------------------------------------------------------------------------
        if not folder_exists(DIRECTION_FILES_FOLDER):
            folder_create(DIRECTION_FILES_FOLDER)
        dir_file = DIRECTION_FILES_FOLDER + "/direction_{}_{}".format(exp_key, sample_no) # name the direction file

        if rank == 0:
            net_plotter.setup_direction2(dir_file, model)

        args.raw_data = True
        surf_file = net_plotter.name_surface_file2(args, dir_file)

        if rank == 0:
            net_plotter.setup_surface_file(args, surf_file, dir_file)

        # wait until master has setup the direction file and surface file
        mpi.barrier(comm)

        # load directions
        d = net_plotter.load_directions(dir_file)

        # calculate the consine similarity of the two directions
        if len(d) == 2 and rank == 0:
            similarity = proj.cal_angle(proj.nplist_to_tensor(d[0]), proj.nplist_to_tensor(d[1]))
            print('cosine similarity between x-axis and y-axis: %f' % similarity)


        cudnn.benchmark = True
        print('    Total params: %.2fM' % (sum(p.numel() for p in model.parameters()) / 1000000.0))

        nsml.bind(model=model)
        if args.pause:
            nsml.paused(scope=locals())


        # ---- LOGS folder ----
        exp_folder = "{}{}/".format(TENSORBOARD_FOLDER, exp_key)
        folder_create(exp_folder, exist_ok=True)
        logs_folder = exp_folder

        """
            Calculate the loss values and accuracies of modified models in parallel
            using MPI reduce.
        """

        f = h5py.File(surf_file, 'r+' if rank == 0 else 'r')
        losses, accuracies = [], []
        xcoordinates = f['xcoordinates'][:]
        ycoordinates = f['ycoordinates'][:] if 'ycoordinates' in f.keys() else None

        if 'train_loss' not in f.keys():
            shape = xcoordinates.shape if ycoordinates is None else (len(xcoordinates), len(ycoordinates))
            losses = -np.ones(shape=shape)
            accuracies = -np.ones(shape=shape)
            if rank == 0:
                f['train_loss'] = losses
                f['train_acc'] = accuracies
        else:
            losses = f['train_loss'][:]
            accuracies = f['train_acc'][:]

        # Generate a list of indices of 'losses' that need to be filled in.
        # The coordinates of each unfilled index (with respect to the direction vectors
        # stored in 'd') are stored in 'coords'.
        inds, coords, inds_nums = scheduler.get_job_indices(losses, xcoordinates, ycoordinates, comm)

        print('Computing %d values for rank %d' % (len(inds), rank))
        start_time = time.time()
        total_sync = 0.0

        # optimizer = optim.Adam(model.parameters())
        criterion = nn.CrossEntropyLoss()

        # Loop over all uncalculated loss values
        for count, ind in enumerate(inds):
            # Get the coordinates of the loss value being calculated
            coord = coords[count]

            # Load the weights corresponding to those coordinates into the net
            net_plotter.set_weights(model.module if torch.cuda.device_count() > 1 else model, w, d, coord)

            # Record the time to compute the loss value
            loss_start = time.time()

            loss, acc = test(trainloader, model, criterion, save=False)

            loss_compute_time = time.time() - loss_start

            # Record the result in the local array
            losses.ravel()[ind] = loss
            accuracies.ravel()[ind] = acc

            # Send updated plot data to the master node
            syc_start = time.time()
            losses = mpi.reduce_max(comm, losses)
            accuracies = mpi.reduce_max(comm, accuracies)
            syc_time = time.time() - syc_start
            total_sync += syc_time

            # Only the master node writes to the file - this avoids write conflicts
            if rank == 0:
                f['train_loss'][:] = losses
                f['train_acc'][:] = accuracies
                f.flush()

            print('Evaluating rank %d  %d/%d  (%.1f%%)  coord=%s \t%s= %.3f \t%s=%.2f \ttime=%.2f \tsync=%.2f' % (
                rank, count, len(inds), 100.0 * count / len(inds), str(coord), 'train_loss', loss,
                'train_acc', acc, loss_compute_time, syc_time))

        # This is only needed to make MPI run smoothly. If this process has less work than
        # the rank0 process, then we need to keep calling reduce so the rank0 process doesn't block
        for i in range(max(inds_nums) - len(inds)):
            losses = mpi.reduce_max(comm, losses)
            accuracies = mpi.reduce_max(comm, accuracies)

        total_time = time.time() - start_time
        print('Rank %d done!  Total time: %.2f Sync: %.2f' % (rank, total_time, total_sync))

        f.close()


def test(testloader, model, criterion, save=False, folder=''):
    global best_acc, global_step
    losses = AverageMeter()
    accuracies = AverageMeter()
    # switch to evaluate mode
    model.train()
    if save:
        outputs_to_save = []
        targets_to_save = []
        indeces_to_save = []

    for batch_idx, (indeces, inputs, targets) in enumerate(testloader):
        inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)

        # compute output
        model.to(DEVICE)
        outputs = model(inputs)

        if save:
            outputs_to_save += [outputs.clone().detach().cpu()]
            targets_to_save += [targets.clone().detach().cpu()]
            indeces_to_save += [indeces.clone().detach().cpu()]

        loss = criterion(outputs, targets.squeeze())
        # measure accuracy and record loss
        acc, = accuracy(outputs.data, targets.data)
        losses.update(loss.data.item(), inputs.size(0))
        accuracies.update(acc.item(), inputs.size(0))

    if save:
        outputs_to_save = torch.cat(outputs_to_save, dim=0).numpy()
        targets_to_save = torch.cat(targets_to_save, dim=0).numpy()
        indeces_to_save = torch.cat(indeces_to_save, dim=0).numpy()

        np.savez_compressed(folder + 'predictions',
                            outputs=outputs_to_save,
                            targets=targets_to_save,
                            indeces=indeces_to_save)

    return losses.avg, accuracies.avg


def zipfolder(foldername, target_dir):
    zipobj = zipfile.ZipFile(foldername + '.zip', 'w', zipfile.ZIP_DEFLATED)
    rootlen = len(target_dir)
    for base, dirs, files in os.walk(target_dir):
        for file in files:
            fn = os.path.join(base, file)
            zipobj.write(fn, fn[rootlen:])


if __name__ == '__main__':

    model_names = sorted(
        name for name in models.__dict__
        if name.islower() and not name.startswith('__') and callable(models.__dict__[name])
    )
    parser = argparse.ArgumentParser(description='PyTorch Feature Combination Mode')
    # Datasets
    parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                        help='number of data loading workers (default: 4)')
    parser.add_argument('--dataset', default='color', type=str, help='bw, color, multi and multicolor supported')
    # Optimization options
    parser.add_argument('--epochs', default=25, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('--train-batch', default=256, type=int, metavar='N',
                        help='train batchsize')
    parser.add_argument('--test-batch', default=256, type=int, metavar='N',
                        help='test batchsize')
    parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                        metavar='LR', help='initial learning rate')
    parser.add_argument('--drop', '--dropout', default=0, type=float,
                        metavar='Dropout', help='Dropout ratio')
    parser.add_argument('--schedule', type=int, nargs='+', default=[500],
                        help='Decrease learning rate at these epochs.')
    parser.add_argument('--gamma', type=float, default=0.1, help='LR is multiplied by gamma on schedule.')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                        help='momentum')
    parser.add_argument('--weight-decay', '--wd', default=5e-4, type=float,
                        metavar='W', help='weight decay (default: 1e-4)')
    # Architecture (resnet, ffnet, vit, convnet)
    parser.add_argument('--arch', '-a', metavar='ARCH', default='vit',
                        choices=model_names,
                        help='model architecture: ' +
                             ' | '.join(model_names) +
                             ' (default: resnet20)')
    parser.add_argument('--depth', type=int, default=20, help='Model depth.')
    # Miscs
    parser.add_argument('--manualSeed', default=123, type=int, help='manual seed')
    # nsml
    parser.add_argument('--pause', default=0, type=int)
    parser.add_argument('--mode', default='train', type=str)

    args = parser.parse_args()
    state = {k: v for k, v in args._get_kwargs()}

    # Use CUDA
    use_cuda = int(GPU_NUM) != 0

    comm, rank, nproc = None, 0, 1

    # in case of multiple GPUs per node, set the GPU to use for each rank
    # if args.cuda:
    if not torch.cuda.is_available():
        raise Exception('User selected cuda option, but cuda is not available on this machine')
    gpu_count = torch.cuda.device_count()
    torch.cuda.set_device(rank % gpu_count)
    print('Rank %d use GPU %d of %d GPUs on %s' %
          (rank, torch.cuda.current_device(), gpu_count, socket.gethostname()))

    # experiments
    if 'color' in args.dataset:
        # experiments
        experiments = {
            "shape_augmentation": ('shape', 'scale', 'orientation', 'color'),
            "scale_augmentation": ('shape', 'scale', 'orientation', 'color'),
            "orientation_augmentation": ('shape', 'scale', 'orientation', 'color'),
            "color_augmentation": ('shape', 'scale', 'orientation', 'color'),
            "diagonal": ('shape', 'scale', 'orientation', 'color'),
        }
        dsc = ColorDSpritesCreator(
            data_path='./data/',
            filename="color_dsprites_pruned.h5"
        )
    elif 'face' in args.dataset:
        experiments = {
            "age_augmentation": ('age', 'gender', 'ethnicity'),
            "gender_augmentation": ('age', 'gender', 'ethnicity'),
            "ethnicity_augmentation": ('age', 'gender', 'ethnicity'),
            "diagonal": ('age', 'gender', 'ethnicity')
        }
        dsc = UTKFaceCreator(
            data_path='./data/',
            filename='UTKFace.h5'
        )
    else:
        raise NotImplementedError(args.dataset)

    try:
        run_experiment(args,
                       experiments=experiments,
                       dsc=dsc,
                       samples=100,
                       scope=locals())
        print("done")

    except Exception as e:
        print("Error: {}".format(e))
        raise e

    finally:
        print("saving zip...")
        zipfolder("runs", DIRECTION_FILES_FOLDER)
        traceback.print_exc()
        sys.exit()