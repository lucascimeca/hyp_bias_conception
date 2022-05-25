from __future__ import print_function

import argparse
import random
import time

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
import itertools

# import nsml
from models.convnet import ConvNet
from models.resnet import ResNet
from models.ffnet import FFNet
from timm.models.vision_transformer import VisionTransformer
# from nsml import GPU_NUM
# from nsml import PARALLEL_WORLD, PARALLEL_PORTS, MY_RANK
from utils import AverageMeter, accuracy
from utils.data_loader import *
from torch.utils.tensorboard import SummaryWriter
from tensorboard import program
from utils.misc.simple_io import *
from functools import partial
from utils.args import *

TENSORBOARD_FOLDER = "./runs/"

best_accuracies = {}  # best test accuracy
global_step = 0


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def run_experiment(args, experiments=None, dsc=None, samples=10, scope=None):


    web_browser_opened = False
    folder_create(TENSORBOARD_FOLDER, exist_ok=True)

    for exp_key in experiments.keys():
        sample_no = 0

        random.seed(args.manualSeed)
        np.random.seed(args.manualSeed)
        torch.manual_seed(args.manualSeed)
        torch.cuda.manual_seed(args.manualSeed)
        torch.cuda.manual_seed_all(args.manualSeed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

        while sample_no < samples:
            global global_step

            # Data
            resize = None
            if 'face' in args.dataset:
                resize = (64, 64)

            print('==> Preparing dataset dsprites ')
            round_one_dataset, round_two_datasets = dsc.get_dataset_fvar(
                number_of_samples=5000,
                features_variants=experiments[exp_key],
                resize=resize,
                train_split=0.7,
                valid_split=0.3,
            )

            num_classes = dsc.no_of_feature_lvs

            # create train dataset for main diagonal -- round one
            round_one_trainloader = data.DataLoader(round_one_dataset['train'], batch_size=args.train_batch,
                                                    shuffle=True, num_workers=args.workers, worker_init_fn=seed_worker)

            # create dataset for off diagonal, depending on task -- round two
            # round_two_trainloaders = {feature: data.DataLoader(round_two_datasets[feature]['train'],
            #                                                    batch_size=args.train_batch, shuffle=True,
            #                                                    num_workers=args.workers, worker_init_fn=seed_worker)
            #                           for feature in round_two_datasets.keys()}

            # create train all validation datasets
            testloaders = {"round_two_task_{}".format(feature): data.DataLoader(round_two_datasets[feature]['valid'],
                                                                                batch_size=args.test_batch, shuffle=True,
                                                                                num_workers=args.workers,
                                                                                worker_init_fn=seed_worker)
                           for feature in round_two_datasets.keys()}

            testloaders['round_one'] = data.DataLoader(round_one_dataset['valid'], batch_size=args.test_batch,
                                                       shuffle=True, num_workers=args.workers,
                                                       worker_init_fn=seed_worker)

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
                optimizer = optim.Adam(model.parameters())
            elif 'convnet' in args.arch:
                model = ConvNet(
                    num_classes=num_classes,
                    no_of_channels=no_of_channels)
                optimizer = optim.Adam(model.parameters(),
                                       weight_decay=1e-3)
            elif 'ffnet' in args.arch:
                model = FFNet(
                    num_classes=num_classes,
                    no_of_channels=no_of_channels)
                optimizer = optim.Adam(model.parameters())
            elif 'vit' in args.arch:
                model = VisionTransformer(
                    img_size=64, patch_size=8, embed_dim=192, depth=12, num_heads=6, mlp_ratio=4, qkv_bias=True,
                    norm_layer=partial(nn.LayerNorm, eps=1e-6), num_classes=num_classes, in_chans=no_of_channels)
                optimizer = optim.SGD(model.parameters(),
                                      lr=5e-3,
                                      momentum=0.9,
                                      weight_decay=1e-4)
            else:
                raise NotImplementedError()

            model = nn.DataParallel(model).cuda()

            cudnn.benchmark = True
            print('    Total params: %.2fM' % (sum(p.numel() for p in model.parameters()) / 1000000.0))

            criterion = nn.CrossEntropyLoss()

            # optimizer = optim.Adam(model.parameters())


            # ---- LOGS folder ----
            exp_folder = "{}{}/".format(TENSORBOARD_FOLDER, exp_key)
            folder_create(exp_folder, exist_ok=True)
            logs_folder = exp_folder

            # --- TensorBoard ----
            # initialize Tensor Board
            task_writers = {}
            for task in testloaders.keys():
                folder = "{}/{}/".format(logs_folder, task)
                if folder_exists(folder):
                    log_files = get_filenames(folder)
                    if len(log_files) != 0:
                        if sample_no == 0:
                            print("Found data for run {}, Skipping {} samples.".format(
                                experiments[exp_key],
                                len(log_files))
                            )
                        sample_no = len(log_files)
                        if sample_no >= samples:
                            break

                task_writers[task] = SummaryWriter(log_dir=folder)

            if sample_no >= samples:
                break

            print("\n########### Experiment Set: {}, Sample: {} ###########\n".format(experiments[exp_key], sample_no))

            # open tensorboard online
            tb = program.TensorBoard()
            tb.configure(argv=[None,
                               '--host', 'localhost',
                               '--reload_interval', '15',
                               '--port', '8080',
                               '--logdir', TENSORBOARD_FOLDER])
            url = tb.launch()
            if not web_browser_opened:
                webbrowser.open(url + '#timeseries')
                web_browser_opened = True

            # Train and val round one
            start_time = time.time()
            for epoch in range(args.epochs):

                adjust_learning_rate(optimizer, epoch, args)
                print('\nEpoch: [%d | %d]' % (epoch + 1, args.epochs))
                losses = AverageMeter()
                accuracies = AverageMeter()
                for batch_idx, (_, inputs, targets) in enumerate(round_one_trainloader):
                    model.train()
                    global_step += 1
                    inputs, targets = inputs.cuda(), targets.cuda(non_blocking=True)
                    inputs, targets = torch.autograd.Variable(inputs), torch.autograd.Variable(targets)

                    # compute output
                    outputs = model(inputs)
                    loss = criterion(outputs, targets.squeeze())
                    # measure accuracy and record loss
                    acc, = accuracy(outputs.data, targets.data)
                    losses.update(loss.data.item(), inputs.size(0))
                    accuracies.update(acc.item(), inputs.size(0))

                    # compute gradient and do SGD step
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                    # update tensorboard if not in NSML
                    # if not NSML:
                    task_writers['round_one'].add_scalar('round_one/train_loss', loss, epoch)
                    task_writers['round_one'].add_scalar('round_one/train_acc', acc, epoch)

                print('Epoch {}: train_loss={}, train_acc={}'.
                      format(epoch, losses.avg, accuracies.avg))

                print("TESTING ACCURACY")
                update = False

                test_reports_loss = {}
                test_reports_acc = {}
                for task in testloaders.keys():
                    folder = "{}{}/".format(logs_folder, task)
                    test_loss, test_acc = test(testloaders[task], model, criterion,
                                               save=epoch==args.epochs-1,
                                               folder=folder)
                    test_reports_loss['test__loss_'+task] = test_loss
                    test_reports_acc['test__acc_'+task] = test_acc
                    if task not in best_accuracies.keys() or test_acc > best_accuracies[task]:
                        best_accuracies[task] = test_acc
                        # if it is the best, save
                        update = True

                    print("{} {}, BEST {}".format(task, test_acc, best_accuracies[task]))

                    # update tensorboard if not in NSML
                    # if not NSML:
                    task_writers[task].add_scalar('round_one/test', test_acc, epoch)
                    task_writers[task].flush()



def test(testloader, model, criterion, save=False, folder=''):
    global global_step
    losses = AverageMeter()
    accuracies = AverageMeter()
    # switch to evaluate mode
    model.train()
    if save:
        outputs_to_save = []
        targets_to_save = []
        indeces_to_save = []

    for batch_idx, (indeces, inputs, targets) in enumerate(testloader):
        # if use_cuda:
        inputs, targets = inputs.cuda(), targets.cuda()
        inputs, targets = torch.autograd.Variable(inputs, volatile=True), torch.autograd.Variable(targets)

        # compute output
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


def adjust_learning_rate(optimizer, epoch, args):
    # learning rate scheduler
    global state
    if epoch in args.schedule:
        state['lr'] *= args.gamma
        for param_group in optimizer.param_groups:
            param_group['lr'] = state['lr']


def zipfolder(foldername, target_dir):
    zipobj = zipfile.ZipFile(foldername + '.zip', 'w', zipfile.ZIP_DEFLATED)
    rootlen = len(target_dir)
    for base, dirs, files in os.walk(target_dir):
        for file in files:
            fn = os.path.join(base, file)
            zipobj.write(fn, fn[rootlen:])


if __name__ == '__main__':

    args = fetch_args()
    args.dataset = 'bw'
    args.epochs = 120
    args.arch = 'resnet20'

    state = {k: v for k, v in args._get_kwargs()}

    if 'bw' in args.dataset:
        combinations = itertools.combinations(['shape', 'scale', 'orientation', 'x_position'],
                                              3)
        experiments = {
            "shape_scale_orientation": ('shape', 'scale', 'orientation'),
            "scale_orientation": ('scale', 'orientation'),
            "shape_scale": ('shape', 'scale'),
            "shape_orientation": ('shape', 'orientation'),
        }
        dsc = BWDSpritesCreator(
            data_path="../data/",
            filename="dsprites.hdf5"
        )
    elif 'multicolor' in args.dataset:
        # experiments
        combinations = itertools.combinations(['shape', 'scale', 'orientation', 'object_number', 'color', 'x_position'],
                                              5)
        experiments = {}
        for comb in combinations:
            key = "_".join([str_elem.replace('_', '-') for str_elem in comb])
            experiments[key] = comb
        dsc = MultiColorDSpritesCreator(
            data_path='./data/',
            filename="multi_color_dsprites.h5"
        )
    elif 'color' in args.dataset:
        experiments = {}
        experiments['shape_scale_orientation'] = ('shape', 'scale', 'orientation')
        experiments['shape_scale_orientation_color'] = ('shape', 'scale', 'orientation', 'color')
        dsc = ColorDSpritesCreator(
            data_path='./data/',
            filename="color_dsprites_pruned.h5"
        )
    elif 'multi' in args.dataset:
        # experiments
        experiments = {
            "shape_scale_orientation_object-number": ('shape', 'scale', 'orientation', 'object_number'),
            "shape_scale_object-number": ('shape', 'scale', 'object_number'),
            "shape_orientation_object-number": ('scale', 'orientation', 'object_number'),
            "scale_object-number": ('scale', 'object_number'),
        }
        dsc = MultiDSpritesCreator(
            data_path='./data/',
            filename="multi_bwdsprites.h5"
        )
    elif 'face' in args.dataset:
        experiments = {}
        experiments['age_gender'] = ('age', 'gender')
        experiments['age_gender_ethnicity'] = ('age', 'gender', 'ethnicity')
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
                       samples=10,
                       scope=locals())
        print("done")

    except Exception as e:
          print("Error: {}".format(e))
          raise e

    finally:
        print("saving zip...")
        zipfolder("runs", TENSORBOARD_FOLDER)
        traceback.print_exc()
        sys.exit()