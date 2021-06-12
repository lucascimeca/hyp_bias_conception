from __future__ import print_function

import argparse
import random
import time

import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.parallel
import torch.utils.data as data
import torch.distributed as dist
import torch.optim as optim
import zipfile
import traceback
import tabulate

import nsml
import models as models
import copy
import sys
import utils.mode_connectivity.curves as curves
import utils.mode_connectivity.utils as curve_utils

from utils.surf import net_plotter
from models.convnet import ConvNet
from models.resnet import ResNet, ResnetWithCurve
from models.ffnet import FFNet
from timm.models.vision_transformer import VisionTransformer
from nsml import GPU_NUM
from nsml import PARALLEL_WORLD, PARALLEL_PORTS, MY_RANK
from utils import AverageMeter, accuracy
from utils.data_loader import *
from utils.misc.simple_io import *
from functools import partial

NSML = True
TENSORBOARD_FOLDER = "./runs/"
DIRECTION_FILES_FOLDER = "./models/pretrained/direction_files/"
DIRECTION_PRETRAINED_FOLDER = "./models/pretrained/"
MODE_CONNECTIVITY_FOLDER = "./models/pretrained/mode_files/"
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

    for sample_from in files:

        fs_from = sample_from.split('-')
        exp_key_from = fs_from[1]
        sample_no_from = fs_from[2]
        feature_from = exp_key_from.split("_")[0]

        samples_to = [sample for sample in files if exp_key not in sample]

        for sample_to in samples_to:

            fs_to = sample_from.split('-')
            exp_key_to = fs_to[1]
            sample_no_to = fs_to[2]
            feature_to = exp_key_to.split("_")[0]


            #  -------------- DATA --------------------------
            resize = None
            if 'face' in args.dataset:
                resize = (64, 64)

            print('==> Preparing dataset dsprites ')
            round_one_dataset, round_two_datasets = dsc.get_dataset_fvar(
                number_of_samples=5000,
                features_variants=experiments[exp_key_from],
                resize=resize,
                train_split=1.,
                valid_split=0.,
            )

            num_classes = dsc.no_of_feature_lvs

            # create train dataset for main diagonal -- round one
            training_data = round_one_dataset['train']
            training_data_to = copy.deepcopy(training_data)
            training_data_from = copy.deepcopy(training_data)

            training_data_to.merge_new_dataset(round_two_datasets[feature_to]['train'])
            training_data_from.merge_new_dataset(round_two_datasets[feature_from]['train'])

            print("Data diag: {}".format(len(training_data)))
            print("Data offdiag {}: {}".format(feature_from, len(training_data_to)))
            print("Data offdiag {}: {}".format(feature_to, len(training_data_from)))

            # create train dataset for main diagonal -- round one
            trainloader_diag = data.DataLoader(training_data, batch_size=args.train_batch,
                                          shuffle=True, num_workers=args.workers, worker_init_fn=seed_worker)
            trainloader_from = data.DataLoader(training_data_to, batch_size=args.train_batch,
                                          shuffle=True, num_workers=args.workers, worker_init_fn=seed_worker)
            trainloader_to = data.DataLoader(training_data_from, batch_size=args.train_batch,
                                          shuffle=True, num_workers=args.workers, worker_init_fn=seed_worker)


            # Model
            print("==> creating model '{}'".format(args.arch))
            if 'color' in args.dataset or 'face' in args.dataset:
                no_of_channels = 3
            else:
                no_of_channels = 1

            if 'resnet' in args.arch:
                architecture = ResnetWithCurve
                base_model = ResNet(
                    num_classes=num_classes,
                    depth=args.depth,
                    no_of_channels=no_of_channels)
                architecture.base = base_model
                optimizer = optim.Adadelta(base_model.parameters())
            elif 'convnet' in args.arch:
                base_model = ConvNet(
                    num_classes=num_classes,
                    no_of_channels=no_of_channels)
                optimizer = optim.Adadelta(base_model.parameters())
            elif 'ffnet' in args.arch:
                base_model = FFNet(
                    num_classes=num_classes,
                    no_of_channels=no_of_channels)
            elif 'vit' in args.arch:
                base_model = VisionTransformer(
                    img_size=64, patch_size=8, embed_dim=192, depth=12, num_heads=3, mlp_ratio=4, qkv_bias=True,
                    norm_layer=partial(nn.LayerNorm, eps=1e-6), num_classes=num_classes,
                    in_chans=no_of_channels)
                optimizer = optim.SGD(model.parameters(),
                                      lr=5e-3,
                                      momentum=0.9,
                                      weight_decay=1e-4)
            else:
                raise NotImplementedError()

            cudnn.benchmark = True
            print('    Total params: %.2fM' % (sum(p.numel() for p in model.parameters()) / 1000000.0))

            nsml.bind(model=base_model)

            criterion = nn.CrossEntropyLoss()


            os.makedirs(MODE_CONNECTIVITY_FOLDER, exist_ok=True)
            with open(os.path.join(MODE_CONNECTIVITY_FOLDER, 'command.sh'), 'w') as f:
                f.write(' '.join(sys.argv))
                f.write('\n')

            torch.backends.cudnn.benchmark = True
            torch.manual_seed(args.seed)
            torch.cuda.manual_seed(args.seed)


            curve = curves.Bezier
            model = curves.CurveNet(
                num_classes,
                curve,
                architecture.curve,
                args.num_bends,
                args.fix_start,
                args.fix_end,
                architecture_kwargs=base_model.kwargs,
            )


            # k are the number of bends
            for smpl, k in [(sample_from, 0), (sample_to, 2)]:

                state_dict = torch.load(DIRECTION_PRETRAINED_FOLDER + smpl)
                state_dict_rex = {key[7:]: state_dict[key] for key in state_dict.keys()}
                base_model.load_state_dict(state_dict_rex)

                print('Loading {} as point {}' % (smpl, k))
                model.import_base_parameters(base_model, k)
            if args.init_linear:
                print('Linear initialization.')
                model.init_linear()

            model = nn.DataParallel(model).to(DEVICE)
            model.cuda()

            def learning_rate_schedule(base_lr, epoch, total_epochs):
                alpha = epoch / total_epochs
                if alpha <= 0.5:
                    factor = 1.0
                elif alpha <= 0.9:
                    factor = 1.0 - (alpha - 0.5) / 0.4 * 0.99
                else:
                    factor = 0.01
                return factor * base_lr

            regularizer = curves.l2_regularizer(args.wd)

            if 'resnet' in args.arch:
                optimizer = optim.Adadelta(model.parameters())
            elif 'vit' in args.arch:
                optimizer = optim.SGD(model.parameters(),
                                      lr=5e-3,
                                      momentum=0.9,
                                      weight_decay=1e-4)


            start_epoch = 1

            columns = ['ep', 'lr', 'tr_loss', 'tr_acc',
                       'te_nll_{}'.format(feature_from), 'te_acc_{}'.format(feature_from),
                       'te_nll_{}'.format(feature_to), 'te_acc_{}'.format(feature_to), 'time']

            curve_utils.save_checkpoint(
                MODE_CONNECTIVITY_FOLDER,
                start_epoch - 1,
                model_state=model.state_dict(),
                optimizer_state=optimizer.state_dict()
            )

            has_bn = curve_utils.check_bn(model)
            test_res_from = {'loss': None, 'accuracy': None, 'nll': None}
            test_res_to = {'loss': None, 'accuracy': None, 'nll': None}
            for epoch in range(start_epoch, 200 + 1):
                time_ep = time.time()

                lr = learning_rate_schedule(args.lr, epoch, args.epochs)
                curve_utils.adjust_learning_rate(optimizer, lr)

                train_res = curve_utils.train(trainloader_diag, model, optimizer, criterion, regularizer)
                if not has_bn:
                    test_res_from = curve_utils.test(training_data_from, model, criterion, regularizer)
                    test_res_to = curve_utils.test(training_data_to, model, criterion, regularizer)

                if epoch % 50 == 0:
                    curve_utils.save_checkpoint(
                        MODE_CONNECTIVITY_FOLDER,
                        epoch,
                        model_state=model.state_dict(),
                        optimizer_state=optimizer.state_dict()
                    )

                time_ep = time.time() - time_ep
                values = [epoch, lr,
                          train_res['loss'], train_res['accuracy'],
                          test_res_from['nll'], test_res_from['accuracy'],
                          test_res_to['nll'], test_res_to['accuracy'], time_ep]

                table = tabulate.tabulate([values], columns, tablefmt='simple', floatfmt='9.4f')
                if epoch % 40 == 1 or epoch == start_epoch:
                    table = table.split('\n')
                    table = '\n'.join([table[1]] + table)
                else:
                    table = table.split('\n')[2]
                print(table)

            if args.epochs % args.save_freq != 0:
                curves.save_checkpoint(
                    args.dir,
                    args.epochs,
                    model_state=model.state_dict(),
                    optimizer_state=optimizer.state_dict()
                )














            base_loss, base_acc = test(trainloader, model, criterion, save=False)
            print('\n\n {} -----  BASE LOSS={:.3f}, ACCURACY={:.2f}\n\n'.format(exp_key, base_loss, base_acc))
            print(sample_file)

            # continue

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
                          .format(100*cnt/tot_exp_no, r, loss-base_loss, acc, time.time() - start_time))

                spherical_losses.append(losses)
                spherical_accs.append(accs)
                local_rs.append(r)

                print('\nr={}, Avg. Loss Change={:.3f}, Avg. Acc={:.2f}\n'.format(
                    r, np.mean(spherical_losses[-1]), np.mean(spherical_accs[-1])))

            np.savez_compressed(DIRECTION_FILES_FOLDER + '{}_{}-sphere_loss'.format(exp_key, sample_no),
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
    parser.add_argument('--dataset', default='face', type=str, help='bw, color, multi and multicolor supported')
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
    parser.add_argument('--max_r', default=1., type=int, help='max radiuses')
    parser.add_argument('--r_levels', default=50, type=int, help='max radius')
    parser.add_argument('--samples_no', default=300, type=int, help='number of samples per radius')
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