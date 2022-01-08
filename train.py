from __future__ import print_function
import datetime
import os
import time
import sys
import numpy as np
import torch
import torch.utils.data
from torch.utils.data.dataloader import default_collate
from torch import nn
import torch.nn.functional as F
import torchvision
from torchvision import transforms

import utils

from datasets.moving_mnist import TrainDataset, TestDataset
import models.moving_mnist as models

def train_one_epoch(model, criterion, optimizer, lr_scheduler, data_loader, device, epoch, print_freq):

    model.train()

    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value}'))
    metric_logger.add_meter('clips/s', utils.SmoothedValue(window_size=10, fmt='{value:.3f}'))

    header = 'Epoch: [{}]'.format(epoch)
    for video in metric_logger.log_every(data_loader, print_freq, header):
        start_time = time.time()
        video = video.to(device)
        output = model(video)
        batch_size = video.shape[0]
        length = video.shape[1]
        target = video[:,length//2:,:,:]
        loss = criterion(output, target)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        metric_logger.update(loss=loss.item(), lr=optimizer.param_groups[0]["lr"])
        metric_logger.meters['clips/s'].update(batch_size / (time.time() - start_time))
        lr_scheduler.step()
        sys.stdout.flush()

def evaluate(model, criterion, data_loader, device):
    model.eval()
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'
    total_cd = 0.0
    n = 0.0
    with torch.no_grad():
        for video in metric_logger.log_every(data_loader, 100, header):
            video = video.to(device, non_blocking=True)
            output = model(video)
            batch_size = video.shape[0]
            length = video.shape[1]
            target = video[:,length//2:,:,:]
            loss = criterion(output, target)

            metric_logger.update(loss=loss.item())
            total_cd += loss.item() * batch_size
            n += batch_size
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()

    total_cd /= n

    print(' * chamfer distance: %f'%total_cd)

    return total_cd


def main(args):

    if args.output_dir:
        utils.mkdir(args.output_dir)

    print(args)
    print("torch version: ", torch.__version__)
    print("torchvision version: ", torchvision.__version__)

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    device = torch.device('cuda')

    # Data loading code
    print("Loading data")

    st = time.time()

    dataset = TrainDataset(
                root=args.data_path,
                seq_length=args.clip_len,
                num_digits=args.num_digits,
                image_size=args.image_size,
                step_length=args.step_length)

    dataset_test = TestDataset(root=args.test_path)

    print("Creating data loaders")

    data_loader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.workers, pin_memory=True)

    data_loader_test = torch.utils.data.DataLoader(dataset_test, batch_size=args.batch_size, num_workers=args.workers, pin_memory=True, drop_last=False)

    print("Creating model")
    Model = getattr(models, args.model)
    model = Model(radius=args.radius, num_samples=args.num_samples, subsampling=args.subsampling)

    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
    model.to(device)

    lr = args.lr
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=args.momentum, weight_decay=args.weight_decay)

    # convert scheduler to be per iteration, not per epoch, for warmup that lasts
    # between different epochs
    warmup_iters = args.lr_warmup_epochs * len(data_loader)
    lr_milestones = [len(data_loader) * m for m in args.lr_milestones]
    lr_scheduler = utils.WarmupMultiStepLR(optimizer, milestones=lr_milestones, gamma=args.lr_gamma, warmup_iters=warmup_iters, warmup_factor=1e-5)

    model_without_ddp = model

    if args.resume:
        checkpoint = torch.load(args.resume, map_location='cpu')
        model_without_ddp.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
        args.start_epoch = checkpoint['epoch'] + 1

    print("Start training")
    start_time = time.time()
    acc = 0
    for epoch in range(args.start_epoch, args.epochs):
        train_one_epoch(model, utils.chamfer_distance, optimizer, lr_scheduler, data_loader, device, epoch, args.print_freq)

        acc = max(acc, evaluate(model, utils.chamfer_distance, data_loader_test, device=device))
        '''
        if args.output_dir:
            checkpoint = {
                'model': model_without_ddp.state_dict(),
                'optimizer': optimizer.state_dict(),
                'lr_scheduler': lr_scheduler.state_dict(),
                'epoch': epoch,
                'args': args}
            utils.save_on_master(
                checkpoint,
                os.path.join(output_dir, 'model_{}.pth'.format(epoch)))
            utils.save_on_master(
                checkpoint,
                os.path.join(output_dir, 'checkpoint.pth'))
        '''

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))
    print('Accuracy {}'.format(acc))


def parse_args():
    import argparse
    parser = argparse.ArgumentParser(description='Moving MNIST Prediction')

    parser.add_argument('--data-path', default='data/mnist', type=str, help='dataset')
    parser.add_argument('--test-path', default='data/test-1mnist-64-128point-20step.npy', type=str, help='dataset')
    parser.add_argument('--seed', default=0, type=int, help='random seed')
    parser.add_argument('--model', default='PointRNN', type=str, help='model')
    # input
    parser.add_argument('--clip-len', default=20, type=int, help='number of frames per clip')
    parser.add_argument('--num-digits', default=1, type=int, help='number of moving digits')
    parser.add_argument('--image-size', default=64, type=int, help='digit bouncing range')
    parser.add_argument('--step-length', default=0.1, type=float, help='bouncing step size')
    # Model
    parser.add_argument('--radius', default=4.0, type=float, help='radius for ball query')
    parser.add_argument('--num-samples', default=4, type=int, help='number of neighbors for the ball query')
    parser.add_argument('--subsampling', default=2, type=int, help='spatial subsampling rate')
    # training
    parser.add_argument('-b', '--batch-size', default=32, type=int)
    parser.add_argument('--epochs', default=5000000, type=int, metavar='N', help='number of total epochs to run')
    parser.add_argument('-j', '--workers', default=10, type=int, metavar='N', help='number of data loading workers (default: 16)')
    parser.add_argument('--lr', default=0.01, type=float, help='initial learning rate')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M', help='momentum')
    parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float, metavar='W', help='weight decay (default: 1e-4)', dest='weight_decay')
    parser.add_argument('--lr-milestones', nargs='+', default=[500, 1000], type=int, help='decrease lr on milestones')
    parser.add_argument('--lr-gamma', default=0.1, type=float, help='decrease lr by a factor of lr-gamma')
    parser.add_argument('--lr-warmup-epochs', default=10, type=int, help='number of warmup epochs')
    # output
    parser.add_argument('--print-freq', default=10, type=int, help='print frequency')
    parser.add_argument('--output-dir', default='', type=str, help='path where to save')
    # resume
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument('--start-epoch', default=0, type=int, metavar='N', help='start epoch')

    args = parser.parse_args()

    return args

if __name__ == "__main__":
    args = parse_args()
    main(args)
