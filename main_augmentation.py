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

import nsml
import copy
import models as models
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


global_step = 0


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


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
    for exp_key in experiments.keys():
        augmentation = exp_key.split('_')[0]
        sample_no = 0
        while sample_no < samples:

            global best_acc, global_step

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

            # for augmentation in experiments[exp_key]:

            # create data augmentations (by adding the appropriate indeces)

            if 'augmentation' in exp_key:

                training_data = copy.deepcopy(round_two_datasets[augmentation]['train'])  # off diag
                training_data.add_indeces(round_one_dataset['train'].get_indeces())  # main diag

                test_data = copy.deepcopy(round_two_datasets[augmentation]['valid'])  # off diag
                test_data.add_indeces(round_one_dataset['valid'].get_indeces())  # main diag

                # create train dataset for main diagonal -- round one
                trainloader = data.DataLoader(training_data, batch_size=args.train_batch,
                                              shuffle=True, num_workers=args.workers, worker_init_fn=seed_worker)

                # create train all validation datasets
                testloaders = {
                    "round_one": data.DataLoader(round_one_dataset['valid'], batch_size=args.test_batch,
                                                 shuffle=True, num_workers=args.workers, worker_init_fn=seed_worker),
                    "{}_augmentation".format(augmentation): data.DataLoader(test_data,
                                                                            batch_size=args.test_batch,
                                                                            shuffle=True,
                                                                            num_workers=args.workers,
                                                                            worker_init_fn=seed_worker)
                }

            else:

                # create train dataset for main diagonal -- round one
                trainloader = data.DataLoader(round_one_dataset['train'], batch_size=args.train_batch,
                                              shuffle=True, num_workers=args.workers, worker_init_fn=seed_worker)

                # create train all validation datasets
                testloaders = {
                    "round_two_task_{}".format(feature): data.DataLoader(round_two_datasets[feature]['valid'],
                                                                         batch_size=args.test_batch,
                                                                         shuffle=True,
                                                                         num_workers=args.workers,
                                                                         worker_init_fn=seed_worker)
                    for feature in round_two_datasets.keys()}
                testloaders['round_one'] = data.DataLoader(round_one_dataset['valid'], batch_size=args.test_batch,
                                                           shuffle=True, num_workers=args.workers,
                                                           worker_init_fn=seed_worker)

            # (re)initialize Model
            print("==> creating model '{}'".format(args.arch))
            if 'color' in args.dataset or 'face' in args.dataset:
                no_of_channels = 3
            else:
                no_of_channels = 1

            if args.arch.endswith('resnet'):
                model = ResNet(
                    num_classes=num_classes,
                    depth=args.depth,
                    no_of_channels=no_of_channels)
                # optimizer = optim.SGD(model.parameters(),
                #                           lr=args.lr,
                #                           momentum=args.momentum,
                #                           weight_decay=args.weight_decay)
                # optimizer = optim.RMSprop(model.parameters())
                optimizer = optim.Adadelta(model.parameters())
                # optimizer = optim.Adam(model.parameters())
            elif args.arch.endswith('convnet'):
                model = ConvNet(
                    num_classes=num_classes,
                    no_of_channels=no_of_channels)
                optimizer = optim.Adam(model.parameters())
            elif args.arch.endswith('ffnet'):
                model = FFNet(
                    num_classes=num_classes,
                    no_of_channels=no_of_channels)
                optimizer = optim.Adam(model.parameters())
            elif args.arch.endswith('vit'):
                model = VisionTransformer(
                    img_size=64, patch_size=8, embed_dim=192, depth=12, num_heads=6, mlp_ratio=4, qkv_bias=True,
                    norm_layer=partial(nn.LayerNorm, eps=1e-6), num_classes=num_classes, in_chans=no_of_channels)
                # optimizer = optim.AdamW(model.parameters(), lr=3e-2, weight_decay=1e-4)
                optimizer = optim.SGD(model.parameters(),
                                      lr=3e-2,
                                      momentum=0.9,
                                      weight_decay=1e-4)
            else:
                raise NotImplementedError()

            model = nn.DataParallel(model).cuda()

            cudnn.benchmark = True
            print('    Total params: %.2fM' % (sum(p.numel() for p in model.parameters()) / 1000000.0))

            criterion = nn.CrossEntropyLoss()

            state = {k: v for k, v in args._get_kwargs()}

            nsml.bind(model=model)
            if args.pause:
                nsml.paused(scope=locals())

            # create train dataset for main diagonal -- round one

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

                task_writers[task] = SummaryWriter(log_dir=folder)

            print("\n########### Experiment Set: {} of {}, Sample: {} ###########\n".format(exp_key, experiments[exp_key], sample_no))

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
            best_accuracies = {}  # best test accuracy
            best_loss = None
            best_acc = None
            start_time = time.time()
            patience_cnt = 0
            weight_states_to_save = None
            zero_loss = False
            for epoch in range(args.epochs):

                # adjust_learning_rate(optimizer, epoch, args)
                print('\nEpoch: [%d | %d] LR: %f' % (epoch + 1, args.epochs, state['lr']))
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

                    task_writers['round_one'].add_scalar('round_one/train_loss', loss, epoch)
                    task_writers['round_one'].add_scalar('round_one/train_acc', acc, epoch)

                print('Epoch {}: train_loss={}, train_acc={}'.
                      format(epoch, losses.avg, accuracies.avg))

                # print("TESTING ACCURACY")

                # test_reports_loss = {}
                # test_reports_acc = {}
                # for task in testloaders.keys():
                #     folder = "{}{}/".format(logs_folder, task)
                #     test_loss, test_acc = test(testloaders[task], model, criterion,
                #                                save=epoch == args.epochs-1,
                #                                folder=folder)
                #     test_reports_loss['test__loss_'+task] = test_loss
                #     test_reports_acc['test__acc_'+task] = test_acc
                #     if task not in best_accuracies.keys() or test_acc > best_accuracies[task]:
                #         best_accuracies[task] = test_acc
                #
                #         if test_loss == 0:
                #             zero_loss = True

                    # print("{} {}, BEST {}".format(task, test_acc, best_accuracies[task]))

                    # task_writers[task].add_scalar('round_one/test', test_acc, epoch)
                    # task_writers[task].flush()

                # nsml.report(
                #     summary=True,
                #     epoch=epoch,
                #     step=time.time() - start_time,
                #     **test_reports_acc,
                #     **test_reports_loss
                # )

                # if any task improved then save
                if (best_loss is None or losses.avg < best_loss) \
                        and losses.avg < 0.05 and accuracies.avg == 100.0:
                    print("---- potential local minima at epoch: {}".format(epoch))
                    best_loss = losses.avg
                    best_acc = accuracies.avg
                    patience_cnt = 0
                    weight_states_to_save = copy.deepcopy(model.state_dict())

                elif best_loss is not None and losses.avg >= best_loss:
                        patience_cnt += 1

                if patience_cnt >= args.patience:
                    if epoch == args.epochs-1:
                        best_loss = losses.avg
                        best_acc = accuracies.avg
                    torch.save(model.state_dict(), '{}weights-{}-{}-{}---loss_{:.4f}-acc_{}.pth'.format(
                        logs_folder,
                        exp_key,
                        sample_no,
                        epoch,
                        best_loss,
                        int(best_acc)
                    ))
                    break

            sample_no += 1

            # --- here do round two...
            # for round_no, feature in round_two_datasets.keys():
            #     pass


if __name__ == '__main__':

    model_names = sorted(
        name for name in models.__dict__
        if name.islower() and not name.startswith('__') and callable(models.__dict__[name])
    )
    parser = argparse.ArgumentParser(description='PyTorch Feature Combination Mode')
    # Datasets
    parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                        help='number of data loading workers (default: 4)')
    parser.add_argument('--dataset', default='face', type=str, help='bw, color, multi, multicolor and face supported')
    # Optimization options
    parser.add_argument('--epochs', default=2000, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('--train-batch', default=256, type=int, metavar='N',
                        help='train batchsize')
    parser.add_argument('--test-batch', default=256, type=int, metavar='N',
                        help='test batchsize')
    parser.add_argument('--lr', '--learning-rate', default=1e-3, type=float,
                        metavar='LR', help='initial learning rate')
    parser.add_argument('--drop', '--dropout', default=0, type=float,
                        metavar='Dropout', help='Dropout ratio')
    parser.add_argument('--schedule', type=int, nargs='+', default=[40, 80, 120],
                        help='Decrease learning rate at these epochs.')
    parser.add_argument('--gamma', type=float, default=0.1, help='LR is multiplied by gamma on schedule.')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                        help='momentum')
    parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                        metavar='W', help='weight decay (default: 1e-4)')
    # Architecture (resnet, ffnet, vit, convnet)
    parser.add_argument('--arch', '-a', metavar='ARCH', default='vit',
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
    parser.add_argument('--patience', '--early-stopping-patience', default=10, type=float,
                        metavar='Patience', help='Early Stopping Dropout Patience')
    parser.add_argument('--ESWTS', '--early-stopping-weights-to-save', default=10, type=float,
                        metavar='ESWTS', help='Early Stopping Weights to Save')

    args = parser.parse_args()

    # Use CUDA
    use_cuda = int(GPU_NUM) != 0

    # Random seed
    if args.manualSeed is None:
        args.manualSeed = random.randint(1, 10000)
    random.seed(args.manualSeed)
    torch.manual_seed(args.manualSeed)
    if use_cuda:
        torch.cuda.manual_seed_all(args.manualSeed)

    if 'bw' in args.dataset:
        experiments = {
            "shape_scale_orientation": ('shape', 'scale', 'orientation'),
        }
        dsc = BWDSpritesCreator(
            data_path="./data/",
            filename="dsprites.npz"
        )
    elif 'multicolor' in args.dataset:
        # experiments
        experiments = {
            "shape_scale_orientation_color": ('shape', 'scale', 'orientation', 'color', 'object_number'),
        }
        dsc = MultiColorDSpritesCreator(
            data_path='./data/',
            filename="multi_color_dsprites.h5"
        )
    elif 'color' in args.dataset:
        # experiments
        experiments = {
            "shape_augmentation": ('shape', 'scale', 'orientation', 'color'),
            "scale_augmentation": ('shape', 'scale', 'orientation', 'color'),
            "orientation_augmentation": ('shape', 'scale', 'orientation', 'color'),
            "color_augmentation": ('shape', 'scale', 'orientation', 'color'),
            "diagonal": ('shape', 'scale', 'orientation', 'color'),
            # "shape_orientation": ('shape', 'orientation'),
        }
        dsc = ColorDSpritesCreator(
            data_path='./data/',
            filename="color_dsprites.h5"
        )
    elif 'multi' in args.dataset:
        # experiments
        experiments = {
            "shape_scale_orientation_object-number": ('shape', 'scale', 'orientation', 'object_number'),
        }
        dsc = MultiDSpritesCreator(
            data_path='./data/',
            filename="multi_bwdsprites.h5"
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
                       samples=4,
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
