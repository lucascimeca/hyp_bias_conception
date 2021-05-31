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


def run_experiment(args, experiments=None, dsc=None, scope=None):

    # A flag for distributed setup
    WORLD_SIZE = len(PARALLEL_WORLD)
    if WORLD_SIZE > 1:  # distributed setup
        master_addr = "tcp://{}:{}".format(PARALLEL_WORLD[0], PARALLEL_PORTS[0])
        print("Initiating distributed process group... My rank is {}".format(MY_RANK))
        dist.init_process_group(backend='gloo', init_method=master_addr, rank=MY_RANK, world_size=WORLD_SIZE)
        args.train_batch //= WORLD_SIZE

    if not folder_exists(DIRECTION_FILES_FOLDER):
        folder_create(DIRECTION_FILES_FOLDER)

    #--------------------------------------------------------------------------
    # Check plotting resolution
    #--------------------------------------------------------------------------
    files = get_filenames(DIRECTION_PRETRAINED_FOLDER, file_ending='pth')

    tot_exp_no = len(files) * args.r_levels * args.samples_no
    cnt = 0

    for sample_file in files:
        global best_acc, global_step

        fs = sample_file.split('-')
        arch = fs[0]
        exp_key = fs[1]
        sample_no = fs[2]
        feature_to_augment = exp_key.split("_")[0]

        # Data
        resize = None
        if 'face' in args.dataset:
            resize = (64, 64)

        print('==> Preparing dataset dsprites ')
        round_one_dataset, round_two_datasets = dsc.get_dataset_fvar(
            number_of_samples=10000,
            features_variants=experiments[exp_key],
            resize=resize,
            train_split=0.5,
            valid_split=0.3,
        )

        num_classes = dsc.no_of_feature_lvs

        # Model
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
        model.load_state_dict(state_dict_rex)# deepcopy since state_dict are references

        model = nn.DataParallel(model).to(DEVICE)

        cudnn.benchmark = True
        print('    Total params: %.2fM' % (sum(p.numel() for p in model.parameters()) / 1000000.0))

        nsml.bind(model=model)
        if args.pause:
            nsml.paused(scope=locals())

        if 'augmentation' in exp_key:

            training_data = copy.deepcopy(round_two_datasets[feature_to_augment]['train'])  # off diag
            training_data.add_indeces(round_one_dataset['train'].get_indeces())       # main diag

            test_data = copy.deepcopy(round_two_datasets[feature_to_augment]['valid'])  # off diag
            test_data.add_indeces(round_one_dataset['valid'].get_indeces())       # main diag

            # create train dataset for main diagonal -- round one
            trainloader = data.DataLoader(training_data, batch_size=args.train_batch,
                                          shuffle=True, num_workers=args.workers, worker_init_fn=seed_worker)

        else:

            # create train dataset for main diagonal -- round one
            trainloader = data.DataLoader(round_one_dataset['train'], batch_size=args.train_batch,
                                          shuffle=True, num_workers=args.workers, worker_init_fn=seed_worker)

            # create train all validation datasets
            testloaders = {"round_two_task_{}".format(feature): data.DataLoader(round_two_datasets[feature]['valid'],
                                                                                batch_size=args.test_batch,
                                                                                shuffle=True,
                                                                                num_workers=args.workers,
                                                                                worker_init_fn=seed_worker)
                           for feature in round_two_datasets.keys()}

        criterion = nn.CrossEntropyLoss()

        base_loss, base_acc = test(trainloader, model, criterion, save=False)

        print('\n\n {} -----  BASE LOSS={:.3f}, ACCURACY={:.2f}\n\n'.format(exp_key, base_loss, base_acc))
        spherical_losses = []
        spherical_accs = []
        local_rs = []

        weights = copy.deepcopy(net_plotter.get_weights(model))  # initial parameters

        step = args.max_r/args.r_levels
        for r in torch.arange(step, args.max_r+step, step):

            changes = []
            for p in model.parameters():
                random_params = torch.randn((args.samples_no,) + p.shape)
                changes.append(random_params)

            random_params = torch.cat([change.view(change.shape[0], -1) for change in changes], dim=1)
            norm = torch.norm(random_params, dim=0)
            random_params /= norm
            random_params *= r

            losses = []
            accs = []
            for si in range(args.samples_no):
                start_time = time.time()

                w_idx = 0
                for (p, w) in zip(model.parameters(), weights):
                    n_of_weights = torch.prod(torch.tensor(p.shape))
                    p.data = w + random_params[si, w_idx:w_idx+n_of_weights].view(p.shape)
                    w_idx = n_of_weights

                loss, acc = test(trainloader, model, criterion, save=False)
                losses.append(loss-base_loss)
                accs.append(acc)

                cnt += 1

                print('{:.3f}% -- r={:.2f}, Loss Change={:.2f}, Acc={:.2f}... time elapsed:{:.2f} '
                      .format(cnt/tot_exp_no, r, loss-base_loss, acc, time.time() - start_time))


            spherical_losses.append(losses)
            spherical_accs.append(accs)
            local_rs.append(r)

            print('\nr={}, Avg. Loss Change={:.3f}, Avg. Acc={:.2f}\n'.format(
                r, np.mean(spherical_losses[-1]), np.mean(spherical_accs[-1])))

        np.savez_compressed(DIRECTION_FILES_FOLDER + '{}-sphere_loss'.format(exp_key),
                            spherical_losses=spherical_losses,
                            spherical_accs=spherical_accs,
                            local_rs=local_rs)


def test(testloader, model, criterion, save=False, folder=''):
    global best_acc, global_step
    losses = AverageMeter()
    accuracies = AverageMeter()
    # switch to evaluate mode
    model.eval()
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
    parser.add_argument('--arch', '-a', metavar='ARCH', default='resnet',
                        choices=model_names,
                        help='model architecture: ' +
                             ' | '.join(model_names) +
                             ' (default: resnet20)')
    parser.add_argument('--depth', type=int, default=20, help='Model depth.')
    parser.add_argument('--max_r', default=1.5, type=int, help='max radiuses')
    parser.add_argument('--r_levels', default=100, type=int, help='max radius')
    parser.add_argument('--samples_no', default=100, type=int, help='number of samples per radius')
    # nsml
    parser.add_argument('--pause', default=0, type=int)
    parser.add_argument('--mode', default='train', type=str)

    args = parser.parse_args()
    state = {k: v for k, v in args._get_kwargs()}

    # Use CUDA
    use_cuda = int(GPU_NUM) != 0

    # Random seed
    args.manualSeed = 123
    random.seed(args.manualSeed)
    torch.manual_seed(args.manualSeed)
    if use_cuda:
        torch.cuda.manual_seed_all(args.manualSeed)

    if use_cuda:
        torch.cuda.manual_seed_all(args.manualSeed)

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
            filename="color_dsprites.h5"
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