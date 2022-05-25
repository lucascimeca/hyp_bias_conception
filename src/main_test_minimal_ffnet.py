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
import copy

import nsml
import models as models
from models.convnet import ConvNet
from models.resnet import ResNet
from models.ffnet import FFNet, FFNetTest
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
        dist.init_process_group(backend='gloo', init_method=master_addr,
                                rank=MY_RANK, world_size=WORLD_SIZE)
        args.train_batch //= WORLD_SIZE

    web_browser_opened = False
    folder_create(TENSORBOARD_FOLDER, exist_ok=True)

    # Data
    resize = None
    if 'face' in args.dataset:
        resize = (64, 64)

    print('==> Preparing dataset dsprites ')
    round_one_dataset, round_two_datasets = dsc.get_dataset_fvar(
        number_of_samples=5000,
        features_variants=experiments,
        resize=resize,
        train_split=0.7,
        valid_split=0.3,
    )

    num_classes = dsc.no_of_feature_lvs

    # while sample_no < samples:
    global global_step

    for depth in range(0, 15, 2):

        for exp_key in sorted(experiments):

            random.seed(args.manualSeed)
            np.random.seed(args.manualSeed)
            torch.manual_seed(args.manualSeed)
            torch.cuda.manual_seed(args.manualSeed)
            torch.cuda.manual_seed_all(args.manualSeed)
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True

            # create train dataset for main diagonal -- round one
            training_data = copy.deepcopy(round_one_dataset['train'])

            # augment with offdiagonal if necessary
            training_data.merge_new_dataset(round_two_datasets[exp_key]['train'])  # main diag

            print("Data Length: {}".format(len(training_data)))

            # create train dataset for main diagonal -- round one
            trainloader = data.DataLoader(training_data,
                                          batch_size=args.train_batch,
                                          shuffle=True,
                                          num_workers=args.workers,
                                          worker_init_fn=seed_worker)

            testloaders = {"round_two_task_{}".format(feature): data.DataLoader(round_two_datasets[feature]['valid'],
                                                                                batch_size=args.test_batch, shuffle=True,
                                                                                num_workers=args.workers,
                                                                                worker_init_fn=seed_worker)
                           for feature in round_two_datasets.keys()}
            testloaders['round_one'] = data.DataLoader(round_one_dataset['train'], batch_size=args.test_batch,
                                                       shuffle=True, num_workers=args.workers,
                                                       worker_init_fn=seed_worker)

            #  -------------- MODEL --------------------------
            print("==> creating model '{}'".format(args.arch))
            if 'color' in args.dataset or 'face' in args.dataset:
                no_of_channels = 3
            else:
                no_of_channels = 1

            model = FFNetTest(
                num_classes=num_classes,
                no_of_channels=no_of_channels,
                depth=depth)
            optimizer = optim.Adam(model.parameters())

            model = nn.DataParallel(model).cuda()

            cudnn.benchmark = True
            print('    Total params: %.2fM' % (sum(p.numel() for p in model.parameters()) / 1000000.0))

            criterion = nn.CrossEntropyLoss()

            optimizer = optim.Adam(model.parameters())

            nsml.bind(model=model)
            if args.pause:
                nsml.paused(scope=locals())

            # ---- LOGS folder ----
            exp_folder = "{}/{}/depth_{}/".format(TENSORBOARD_FOLDER, exp_key, depth)
            folder_create(exp_folder, exist_ok=True)
            logs_folder = exp_folder

            # --- TensorBoard ----
            # initialize Tensor Board
            task_writers = {}
            for task in testloaders.keys():
                folder = "{}/{}/".format(logs_folder, task)
                task_writers[task] = SummaryWriter(log_dir=folder)

            print("\n########### Experiment Set: {}, Depth: {} ###########\n".format(experiments, depth))

            if not NSML:
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
            patience = 0
            start_time = time.time()
            for epoch in range(args.epochs):

                adjust_learning_rate(optimizer, epoch, args)
                print('\nEpoch: [%d | %d]' % (epoch + 1, args.epochs))
                losses = AverageMeter()
                accuracies = AverageMeter()
                for batch_idx, (_, inputs, targets) in enumerate(trainloader):
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
                    # report result
                    nsml.report(
                        epoch=epoch,
                        step=time.time() - start_time,
                        scope=dict(scope, **locals()),
                        train__top1_acc=acc.item(),
                        train__loss=loss.data.item()
                    )

                    # update tensorboard if not in NSML
                    task_writers['round_one'].add_scalar('round_one/train_loss', loss, epoch)
                    task_writers['round_one'].add_scalar('round_one/train_acc', acc, epoch)

                print('Epoch {}: train_loss={}, train_acc={}'.
                      format(epoch, losses.avg, accuracies.avg))

                print("TESTING ACCURACY")
                update = False

                test_reports_loss = {}
                test_reports_acc = {}
                best_accuracies = {}  # best test accuracy

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

                nsml.report(
                    summary=True,
                    epoch=epoch,
                    step=time.time() - start_time,
                    **test_reports_acc,
                    **test_reports_loss
                )

                # if any task improved then save
                if update:
                    nsml.save(global_step)
                    patience = 0
                else:
                    patience += 1

                if patience >= 20 and epoch > 50:
                    break


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

    model_names = sorted(
        name for name in models.__dict__
        if name.islower() and not name.startswith('__') and callable(models.__dict__[name])
    )
    parser = argparse.ArgumentParser(description='PyTorch Feature Combination Mode')
    # Datasets
    parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                        help='number of data loading workers (default: 4)')
    parser.add_argument('--dataset', default='face', type=str, help='bw, color, multi and multicolor supported')
    # Optimization options
    parser.add_argument('--epochs', default=120, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('--train-batch', default=500, type=int, metavar='N',
                        help='train batchsize')
    parser.add_argument('--test-batch', default=500, type=int, metavar='N',
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
    parser.add_argument('--arch', '-a', metavar='ARCH', default='ffnet',
                        choices=model_names,
                        help='model architecture: ' +
                             ' | '.join(model_names) +
                             ' (default: resnet20)')
    parser.add_argument('--depth', type=int, default=20, help='Model depth.')
    parser.add_argument('--growthRate', type=int, default=12, help='Growth rate for DenseNet.')
    parser.add_argument('--compressionRate', type=int, default=2, help='Compression Rate (theta) for DenseNet.')
    # Miscs
    parser.add_argument('--manualSeed', default=123, type=int, help='manual seed')
    # Random Erasing
    parser.add_argument('--prob', default=0, type=float, help='Random Erasing probability')
    parser.add_argument('--sh', default=0.4, type=float, help='max erasing area')
    parser.add_argument('--r1', default=0.3, type=float, help='aspect of erasing area')
    # nsml
    parser.add_argument('--pause', default=0, type=int)
    parser.add_argument('--iteration', default=0, type=str)
    parser.add_argument('--mode', default='train', type=str)
    parser.add_argument('--log-interval', default=100, type=int)

    args = parser.parse_args()
    state = {k: v for k, v in args._get_kwargs()}

    # Use CUDA
    use_cuda = int(GPU_NUM) != 0

    experiments = ('shape', 'scale', 'orientation', 'color')
    dsc = ColorDSpritesCreator(
        data_path='./data/',
        filename="color_dsprites_pruned.h5"
    )

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