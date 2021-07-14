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
SEED = 12345
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


def stats(values, dl):
    min = np.min(values)
    max = np.max(values)
    avg = np.mean(values)
    int = np.sum(0.5 * (values[:-1] + values[1:]) * dl[1:]) / np.sum(dl[1:])
    return min, max, avg, int


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
        inputs, targets = inputs.cuda(non_blocking=True), targets.cuda(non_blocking=True)

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

    return {'loss': losses.avg, 'accuracy': accuracies.avg}


def zipfolder(foldername, target_dir):
    zipobj = zipfile.ZipFile(foldername + '.zip', 'w', zipfile.ZIP_DEFLATED)
    rootlen = len(target_dir)
    for base, dirs, files in os.walk(target_dir):
        for file in files:
            fn = os.path.join(base, file)
            zipobj.write(fn, fn[rootlen:])


def create_datasets(dsc, experiments, exp_key_from, exp_key_to, feature_from, feature_to):
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

    # create train dataset for main diagonal -- round one
    training_data = copy.deepcopy(round_one_dataset['train'])

    training_data_to = copy.deepcopy(training_data)
    if 'augmentation' in exp_key_to:
        training_data_to.merge_new_dataset(round_two_datasets[feature_to]['train'])

    training_data_from = copy.deepcopy(training_data)
    if 'augmentation' in exp_key_from:
        training_data_from.merge_new_dataset(round_two_datasets[feature_from]['train'])

    print("Data diag: {}".format(len(training_data)))
    print("Data offdiag {}: {}".format(feature_from, len(training_data_to)))
    print("Data offdiag {}: {}".format(feature_to, len(training_data_from)))

    # create train dataset for main diagonal -- round one
    trainloader_diag = data.DataLoader(training_data, batch_size=args.train_batch,
                                       shuffle=True, num_workers=args.workers, worker_init_fn=seed_worker)
    trainloader_from = data.DataLoader(training_data_from, batch_size=args.train_batch,
                                       shuffle=True, num_workers=args.workers, worker_init_fn=seed_worker)
    trainloader_to = data.DataLoader(training_data_to, batch_size=args.train_batch,
                                     shuffle=True, num_workers=args.workers, worker_init_fn=seed_worker)
    return trainloader_diag, trainloader_from, trainloader_to


def get_xy(point, origin, vector_x, vector_y):
    return np.array([np.dot(point - origin, vector_x), np.dot(point - origin, vector_y)])


def test_path(model, criterion, trainloader_diag, trainloader_from, trainloader_to, feature_from, feature_to, exp_folder):
    T = 61
    ts = np.linspace(0.0, 1.0, T)
    tr_loss = np.zeros(T)
    tr_acc = np.zeros(T)
    te_to_loss = np.zeros(T)
    te_to_acc = np.zeros(T)
    te_from_loss = np.zeros(T)
    te_from_acc = np.zeros(T)
    tr_err = np.zeros(T)
    te_to_err = np.zeros(T)
    te_from_err = np.zeros(T)
    dl = np.zeros(T)

    previous_weights = None

    columns = ['t', 'Train loss', 'Train error (%)',
               'Test loss {}'.format(feature_from), 'Test error {} (%)'.format(feature_from),
               'Test loss {}'.format(feature_to), 'Test error {} (%)'.format(feature_to)]

    regularizer = curve_utils.l2_regularizer(args.weight_decay)

    t = torch.FloatTensor([0.0]).cuda()
    for i, t_value in enumerate(ts):
        t.data.fill_(t_value)
        weights = model.weights(t)
        if previous_weights is not None:
            dl[i] = np.sqrt(np.sum(np.square(weights - previous_weights)))
        previous_weights = weights.copy()

        curve_utils.update_bn(trainloader_diag, model, t=t)
        tr_res = curve_utils.test(trainloader_diag, model, criterion, regularizer, t=t)
        te_to_res = curve_utils.test(trainloader_to, model, criterion, regularizer, t=t)
        te_from_res = curve_utils.test(trainloader_from, model, criterion, regularizer, t=t)
        tr_loss[i] = tr_res['nll']
        tr_acc[i] = tr_res['accuracy']
        tr_err[i] = 100.0 - tr_acc[i]
        te_to_loss[i] = te_to_res['nll']
        te_from_loss[i] = te_from_res['nll']
        te_to_acc[i] = te_to_res['accuracy']
        te_from_acc[i] = te_from_res['accuracy']
        te_to_err[i] = 100.0 - te_to_acc[i]
        te_from_err[i] = 100.0 - te_from_acc[i]

        values = [t, tr_loss[i], tr_err[i], te_from_loss[i], te_from_err[i], te_to_loss[i], te_to_err[i]]
        table = tabulate.tabulate([values], columns, tablefmt='simple', floatfmt='10.4f')
        if i % 40 == 0:
            table = table.split('\n')
            table = '\n'.join([table[1]] + table)
        else:
            table = table.split('\n')[2]
        print(table)

    tr_loss_min, tr_loss_max, tr_loss_avg, tr_loss_int = stats(tr_loss, dl)
    tr_err_min, tr_err_max, tr_err_avg, tr_err_int = stats(tr_err, dl)

    te_from_loss_min, te_from_loss_max, te_from_loss_avg, te_from_loss_int = stats(te_from_loss, dl)
    te_from_err_min, te_from_err_max, te_from_err_avg, te_from_err_int = stats(te_from_err, dl)

    te_to_loss_min, te_to_loss_max, te_to_loss_avg, te_to_loss_int = stats(te_to_loss, dl)
    te_to_err_min, te_to_err_max, te_to_err_avg, te_to_err_int = stats(te_to_err, dl)

    print('Length: %.2f' % np.sum(dl))
    print(tabulate.tabulate([
        ['train loss', tr_loss[0], tr_loss[-1], tr_loss_min, tr_loss_max, tr_loss_avg, tr_loss_int],
        ['train error (%)', tr_err[0], tr_err[-1], tr_err_min, tr_err_max, tr_err_avg, tr_err_int],
        ['test from loss', te_from_err[0], te_from_loss[-1], te_from_loss_min, te_from_loss_max,
         te_from_loss_avg, te_from_loss_int],
        ['test to loss', te_to_loss[0], te_to_loss[-1], te_to_loss_min, te_to_loss_max, te_to_loss_avg,
         te_to_loss_int],
        ['test error (%) {}'.format(feature_from), te_from_err[0], te_from_err[-1], te_from_err_min, te_from_err_max, te_from_err_avg,
         te_from_err_int],
        ['test error (%) {}'.format(feature_to), te_to_err[0], te_to_err[-1], te_to_err_min, te_to_err_max, te_to_err_avg,
         te_to_err_int],
    ], [
        '', 'start', 'end', 'min', 'max', 'avg', 'int'
    ], tablefmt='simple', floatfmt='10.4f'))

    np.savez(
        os.path.join(exp_folder, 'curve.npz'),
        ts=ts,
        dl=dl,
        tr_loss=tr_loss,
        tr_loss_min=tr_loss_min,
        tr_loss_max=tr_loss_max,
        tr_loss_avg=tr_loss_avg,
        tr_loss_int=tr_loss_int,
        tr_acc=tr_acc,
        tr_err=tr_err,
        tr_err_min=tr_err_min,
        tr_err_max=tr_err_max,
        tr_err_avg=tr_err_avg,
        tr_err_int=tr_err_int,
        te_from_loss=te_from_loss,
        te_from_loss_min=te_from_loss_min,
        te_from_loss_max=te_from_loss_max,
        te_from_loss_avg=te_from_loss_avg,
        te_from_loss_int=te_from_loss_int,
        te_to_loss=te_to_loss,
        te_to_loss_min=te_to_loss_min,
        te_to_loss_max=te_to_loss_max,
        te_to_loss_avg=te_to_loss_avg,
        te_to_loss_int=te_to_loss_int,
        te_from_acc=te_from_acc,
        te_from_err=te_from_err,
        te_from_err_min=te_from_err_min,
        te_from_err_max=te_from_err_max,
        te_from_err_avg=te_from_err_avg,
        te_from_err_int=te_from_err_int,
        te_to_acc=te_to_acc,
        te_to_err=te_to_err,
        te_to_err_min=te_to_err_min,
        te_to_err_max=te_to_err_max,
        te_to_err_avg=te_to_err_avg,
        te_to_err_int=te_to_err_int,
    )


def create_plane(model, base_model, criterion, regularizer, trainloader_diag, trainloader_from, trainloader_to, feature_from, feature_to, exp_folder):
    w = list()
    curve_parameters = list(model.net.parameters())
    for i in range(args.num_bends):
        w.append(np.concatenate([
            p.data.cpu().numpy().ravel() for p in curve_parameters[i::args.num_bends]
        ]))

    print('Weight space dimensionality: %d' % w[0].shape[0])

    u = w[2] - w[0]
    dx = np.linalg.norm(u)
    u /= dx

    v = w[1] - w[0]
    v -= np.dot(u, v) * u
    dy = np.linalg.norm(v)
    v /= dy

    bend_coordinates = np.stack(get_xy(p, w[0], u, v) for p in w)

    ts = np.linspace(0.0, 1.0, 61)
    curve_coordinates = []
    for t in np.linspace(0.0, 1.0, 61):
        weights = model.weights(torch.Tensor([t]).cuda())
        curve_coordinates.append(get_xy(weights, w[0], u, v))
    curve_coordinates = np.stack(curve_coordinates)

    G = args.grid_points
    alphas = np.linspace(0.0 - args.margin_left, 1.0 + args.margin_right, G)
    betas = np.linspace(0.0 - args.margin_bottom, 1.0 + args.margin_top, G)

    tr_loss = np.zeros((G, G))
    tr_nll = np.zeros((G, G))
    tr_acc = np.zeros((G, G))
    tr_err = np.zeros((G, G))

    te_from_loss = np.zeros((G, G))
    te_from_nll = np.zeros((G, G))
    te_from_acc = np.zeros((G, G))
    te_from_err = np.zeros((G, G))

    te_to_loss = np.zeros((G, G))
    te_to_nll = np.zeros((G, G))
    te_to_acc = np.zeros((G, G))
    te_to_err = np.zeros((G, G))

    grid = np.zeros((G, G, 2))
    columns = ['X', 'Y', 'Train loss', 'Train nll', 'Train error (%)',
               'Test loss {}'.format(feature_from), 'Test error (%) {}'.format(feature_from),
               'Test loss {}'.format(feature_to), 'Test error (%) {}'.format(feature_to)]

    for i, alpha in enumerate(alphas):
        for j, beta in enumerate(betas):
            p = w[0] + alpha * dx * u + beta * dy * v

            offset = 0
            for parameter in base_model.parameters():
                size = np.prod(parameter.size())
                value = p[offset:offset + size].reshape(parameter.size())
                parameter.data.copy_(torch.from_numpy(value))
                offset += size

            curve_utils.update_bn(trainloader_diag, base_model)

            tr_res = curve_utils.test(trainloader_diag, base_model, criterion, regularizer)
            te_from_res = curve_utils.test(trainloader_from, base_model, criterion, regularizer)
            te_to_res = curve_utils.test(trainloader_to, base_model, criterion, regularizer)

            tr_loss_v, tr_nll_v, tr_acc_v = tr_res['loss'], tr_res['nll'], tr_res['accuracy']
            te_from_loss_v, te_from_nll_v, te_from_acc_v = te_from_res['loss'], te_from_res['nll'], te_from_res['accuracy']
            te_to_loss_v, te_to_nll_v, te_to_acc_v = te_to_res['loss'], te_to_res['nll'], te_to_res['accuracy']

            c = get_xy(p, w[0], u, v)
            grid[i, j] = [alpha * dx, beta * dy]

            tr_loss[i, j] = tr_loss_v
            tr_nll[i, j] = tr_nll_v
            tr_acc[i, j] = tr_acc_v
            tr_err[i, j] = 100.0 - tr_acc[i, j]

            te_from_loss[i, j] = te_from_loss_v
            te_from_nll[i, j] = te_from_nll_v
            te_from_acc[i, j] = te_from_acc_v
            te_from_err[i, j] = 100.0 - te_from_acc[i, j]

            te_to_loss[i, j] = te_to_loss_v
            te_to_nll[i, j] = te_to_nll_v
            te_to_acc[i, j] = te_to_acc_v
            te_to_err[i, j] = 100.0 - te_to_acc[i, j]

            values = [
                grid[i, j, 0], grid[i, j, 1], tr_loss[i, j], tr_nll[i, j], tr_err[i, j],
                te_from_nll[i, j], te_from_err[i, j], te_to_nll[i, j], te_to_err[i, j]
            ]
            table = tabulate.tabulate([values], columns, tablefmt='simple', floatfmt='10.4f')
            if j == 0:
                table = table.split('\n')
                table = '\n'.join([table[1]] + table)
            else:
                table = table.split('\n')[2]
            print(table)

    np.savez(
        os.path.join(exp_folder, 'plane.npz'),
        ts=ts,
        bend_coordinates=bend_coordinates,
        curve_coordinates=curve_coordinates,
        alphas=alphas,
        betas=betas,
        grid=grid,
        tr_loss=tr_loss,
        tr_acc=tr_acc,
        tr_nll=tr_nll,
        tr_err=tr_err,
        te_from_loss=te_from_loss,
        te_from_acc=te_from_acc,
        te_from_nll=te_from_nll,
        te_from_err=te_from_err,
        te_to_loss=te_to_loss,
        te_to_acc=te_to_acc,
        te_to_nll=te_to_nll,
        te_to_err=te_to_err,
    )


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
    files = sorted(get_filenames(DIRECTION_PRETRAINED_FOLDER, file_ending='pth'))

    folder_create(MODE_CONNECTIVITY_FOLDER, exist_ok=True)

    for i, sample_from in enumerate(files[:-1]):

        fs_from = sample_from.split('-')
        exp_key_from = fs_from[1]
        feature_from = exp_key_from.split("_")[0]

        for sample_to in files[i+1:]:

            random.seed(args.manualSeed)
            np.random.seed(args.manualSeed)
            torch.manual_seed(args.manualSeed)
            torch.cuda.manual_seed(args.manualSeed)
            torch.cuda.manual_seed_all(args.manualSeed)
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True

            fs_to = sample_to.split('-')
            exp_key_to = fs_to[1]
            feature_to = exp_key_to.split("_")[0]

            #  -------------- DATA --------------------------

            trainloader_diag, trainloader_from, trainloader_to = create_datasets(
                dsc=dsc,
                experiments=experiments,
                exp_key_from=exp_key_from,
                exp_key_to=exp_key_to,
                feature_from=feature_from,
                feature_to=feature_to
            )
            num_classes = dsc.no_of_feature_lvs

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
                architecture_kwargs = {
                    'depth': args.depth,
                    'no_of_channels': no_of_channels
                }
            elif 'vit' in args.arch:
                architecture = ResnetWithCurve
                base_model = VisionTransformer(
                    img_size=64, patch_size=8, embed_dim=192, depth=12, num_heads=3, mlp_ratio=4, qkv_bias=True,
                    norm_layer=partial(nn.LayerNorm, eps=1e-6), num_classes=num_classes,
                    in_chans=no_of_channels)
                architecture.base = base_model
                architecture_kwargs = {
                    'img_size': 64,
                    'patch_size':  8,
                    'embed_dim':  192,
                    'depth':  12,
                    'num_heads':  3,
                    'mlp_ratio':  4,
                    'qkv_bias':  True,
                    'norm_layer':  partial(nn.LayerNorm, eps=1e-6),
                    'num_classes':  num_classes,
                    'in_chans':  no_of_channels
                }
            else:
                raise NotImplementedError()

            cudnn.benchmark = True
            print('    Total params: %.2fM' % (sum(p.numel() for p in base_model.parameters()) / 1000000.0))

            nsml.bind(model=base_model)

            criterion = nn.CrossEntropyLoss()

            exp_folder = MODE_CONNECTIVITY_FOLDER + "{}-to-{}".format(feature_from, feature_to)
            folder_create(exp_folder, exist_ok=True)
            with open(os.path.join(exp_folder, 'command.sh'), 'w') as f:
                f.write(' '.join(sys.argv))
                f.write('\n')

            torch.backends.cudnn.benchmark = True
            torch.manual_seed(SEED)
            torch.cuda.manual_seed(SEED)

            args.num_bends = 3
            args.fix_start = True
            args.fix_end = True
            curve = curves.Bezier
            model = curves.CurveNet(
                num_classes,
                curve,
                architecture.curve,
                args.num_bends,
                args.fix_start,
                args.fix_end,
                architecture_kwargs=architecture_kwargs,
            )

            # k are the number of bends
            for smpl, k in [(sample_from, 0), (sample_to, 2)]:

                state_dict = torch.load(DIRECTION_PRETRAINED_FOLDER + smpl)
                state_dict_rex = {key[7:]: state_dict[key] for key in state_dict.keys()}
                base_model.load_state_dict(state_dict_rex)

                print('Loading {} as point {}'.format(smpl, k))
                model.import_base_parameters(base_model, k)
            if args.init_linear:
                print('Linear initialization.')
                model.init_linear()

            # model = nn.DataParallel(model).to(DEVICE)
            base_model.cuda()
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

            regularizer = curve_utils.l2_regularizer(args.weight_decay)

            if 'resnet' in args.arch:
                optimizer = optim.Adadelta(model.parameters())
            elif 'vit' in args.arch:
                optimizer = optim.SGD(model.parameters(),
                                      lr=5e-3,
                                      momentum=0.9,
                                      weight_decay=1e-4)
            else:
                raise NotImplementedError

            start_epoch = 1

            columns = ['ep', 'lr', 'tr_loss', 'tr_acc',
                       'te_loss_{}'.format(feature_from), 'te_acc_{}'.format(feature_from),
                       'te_loss_{}'.format(feature_to), 'te_acc_{}'.format(feature_to), 'time']

            curve_utils.save_checkpoint(
                exp_folder,
                start_epoch - 1,
                model_state=model.state_dict(),
                optimizer_state=optimizer.state_dict()
            )

            test_res_from = {'loss': None, 'accuracy': None, 'nll': None}
            test_res_to = {'loss': None, 'accuracy': None, 'nll': None}
            for epoch in range(start_epoch, args.epochs + 1):
                time_ep = time.time()

                lr = learning_rate_schedule(args.lr, epoch, args.epochs)
                curve_utils.adjust_learning_rate(optimizer, lr)

                train_res = curve_utils.train(trainloader_diag, model, optimizer, criterion, regularizer)
                if epoch % 5 == 0:
                    test_res_from = test(trainloader_from, model, criterion, save=False)
                    test_res_to = test(trainloader_to, model, criterion, save=False)

                if epoch % 50 == 0:
                    curve_utils.save_checkpoint(
                        exp_folder,
                        epoch,
                        model_state=model.state_dict(),
                        optimizer_state=optimizer.state_dict()
                    )

                time_ep = time.time() - time_ep
                values = [epoch, lr,
                          train_res['loss'], train_res['accuracy'],
                          test_res_from['loss'], test_res_from['accuracy'],
                          test_res_to['loss'], test_res_to['accuracy'], time_ep]

                table = tabulate.tabulate([values], columns, tablefmt='simple', floatfmt='9.4f')
                if epoch % 40 == 1 or epoch == start_epoch:
                    table = table.split('\n')
                    table = '\n'.join([table[1]] + table)
                else:
                    table = table.split('\n')[2]
                print(table)

            if args.epochs % args.save_freq != 0:
                curve_utils.save_checkpoint(
                    exp_folder,
                    args.epochs,
                    model_state=model.state_dict(),
                    optimizer_state=optimizer.state_dict()
                )

            # ---------- test trained path ------------------------
            test_path(
                model=model,
                criterion=criterion,
                trainloader_diag=trainloader_diag,
                trainloader_from=trainloader_from,
                trainloader_to=trainloader_to,
                feature_from=feature_from,
                feature_to=feature_to,
                exp_folder=exp_folder
            )

            # ---------- create plane for plotting of path ------------------------
            create_plane(
                model=model,
                base_model=base_model,
                criterion=criterion,
                regularizer=regularizer,
                trainloader_diag=trainloader_diag,
                trainloader_from=trainloader_from,
                trainloader_to=trainloader_to,
                feature_from=feature_from,
                feature_to=feature_to,
                exp_folder=exp_folder
            )



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
    parser.add_argument('--epochs', default=200, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('--train-batch', default=256, type=int, metavar='N',
                        help='train batchsize')
    parser.add_argument('--test-batch', default=256, type=int, metavar='N',
                        help='test batchsize')
    parser.add_argument('--lr', '--learning-rate', default=0.01, type=float,
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
    parser.add_argument('--manualSeed', default=123, type=int, help='manual seed')
    # nsml
    parser.add_argument('--pause', default=0, type=int)
    parser.add_argument('--mode', default='train', type=str)
    # for mode connectivity stuff

    parser.set_defaults(init_linear=True)
    parser.add_argument('--init_linear_off', dest='init_linear', action='store_false',
                        help='turns off linear initialization of intermediate points (default: on)')
    parser.add_argument('--save_freq', type=int, default=100, metavar='N',
                        help='save frequency (default: 50)')
    parser.add_argument('--grid_points', type=int, default=21, metavar='N',
                        help='number of points in the grid (default: 21)')
    parser.add_argument('--margin_left', type=float, default=0.2, metavar='M',
                        help='left margin (default: 0.2)')
    parser.add_argument('--margin_right', type=float, default=0.2, metavar='M',
                        help='right margin (default: 0.2)')
    parser.add_argument('--margin_bottom', type=float, default=0.5, metavar='M',
                        help='bottom margin (default: 0.)')
    parser.add_argument('--margin_top', type=float, default=0.5, metavar='M',
                        help='top margin (default: 0.2)')

    args = parser.parse_args()
    state = {k: v for k, v in args._get_kwargs()}

    # Use CUDA
    use_cuda = int(GPU_NUM) != 0

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
        zipfolder("runs", MODE_CONNECTIVITY_FOLDER)
        traceback.print_exc()
        sys.exit()