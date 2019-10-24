from __future__ import print_function

import warnings
warnings.filterwarnings("ignore")

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms

import os
import argparse
import csv
import sys

from models import *
from utils import *
import utils

import numpy as np
import random


parser = argparse.ArgumentParser(description='PyTorch Imagenet Training')
parser.add_argument('--arch', default='resnet50', type=str, help='model name, supportted: resnet50, resnet101, resnet152')
parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
parser.add_argument('--sess', default='resnet', type=str, help='session name')
parser.add_argument('--seed', default=0, type=int, help='random seed')
parser.add_argument('--weight_decay', default=1e-4, type=float, help='weight decay (default=1e-4)')
parser.add_argument('--batchsize', default=256, type=int, help='batch size')
parser.add_argument('--n_epoch', default=200, type=int, help='total number of epochs')
parser.add_argument('--lr', default=0.1, type=float, help='base learning rate (default=0.1)')

parser.add_argument('--data_dir', type=str, help='path to the folder contains train/validation.zip')

# For resnet with bn, tau is fixed scalar. For resnet without bn, all blocks share the same tau initialized as argument.
parser.add_argument('--tau', default=1.0, type=float, help='value of scaling factor tau')

args = parser.parse_args()

torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)
np.random.seed(args.seed)
random.seed(args.seed)
torch.backends.cudnn.deterministic = True

use_cuda = torch.cuda.is_available()
best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch
batch_size = args.batchsize

# Model
if args.resume:
    # Load checkpoint.
    print('==> Resuming from checkpoint..')
    assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
    checkpoint_file = './checkpoint/' + args.arch + '_' + args.sess + '_' + 'tau%.5f_'%args.tau + str(args.seed) + '.ckpt'
    checkpoint = torch.load(checkpoint_file)
    net = checkpoint['net']
    best_acc = checkpoint['acc']
    start_epoch = checkpoint['epoch'] + 1
    torch.set_rng_state(checkpoint['rng_state'])
else:
    print("=> creating model '{}' for ImageNet".format(args.arch))
    net = eval(args.arch+'(%f)'%args.tau)


# Load entire dataset into memory to reduce hard disk access during training. This might take 10+ minutes.
print('Loading data from zip file')
traindir = os.path.join(args.data_dir, 'train.zip')
validdir = os.path.join(args.data_dir, 'validation.zip')
print('Loading data into memory, this migh take 10+ minutes')
sys.stdout.flush()
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
train_data = utils.InMemoryZipDataset(traindir, transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ]), num_workers=16)
valid_data = utils.InMemoryZipDataset(validdir, transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ]), num_workers=16)
print('Found {} in training data'.format(len(train_data)))
print('Found {} in validation data'.format(len(valid_data)))
sys.stdout.flush()
train_sampler = None
trainloader = torch.utils.data.DataLoader(train_data, batch_size=args.batchsize, shuffle=True, pin_memory=True, num_workers=4)
testloader = torch.utils.data.DataLoader(valid_data, batch_size=args.batchsize, shuffle=False, pin_memory=True, num_workers=4)


result_folder = './results/'
if not os.path.exists(result_folder):
    os.makedirs(result_folder)

logname = result_folder + args.arch + '_' + args.sess + '_' + str(args.seed) + '.csv'

if use_cuda:
    net.cuda()
    net = torch.nn.DataParallel(net)
    print('Using', torch.cuda.device_count(), 'GPUs.')
    cudnn.benchmark = True
    print('Using CUDA..')
    
loss_func = nn.CrossEntropyLoss()
parameters_bias = [p[1] for p in net.named_parameters() if ('bias1' in p[0] or 'bias2' in p[0] or 'bias3' in p[0])]
parameters_tau = [p[1] for p in net.named_parameters() if 'tau' in p[0]]
# When use bn, parmeters_bias and parameters_tau are empty.
parameters_others = [p[1] for p in net.named_parameters() if not ('bias1' in p[0] or 'bias2' in p[0] or 'bias3' in p[0] or 'tau' in p[0])]
optimizer = optim.SGD(
        [{'params': parameters_bias, 'lr': args.lr/10., 'initial_lr': args.lr/10.}, 
        {'params': parameters_tau, 'lr': args.lr/10., 'initial_lr':  args.lr/10.}, 
        {'params': parameters_others, 'initial_lr':  args.lr}], 
        lr=args.lr, 
        momentum=0.9, 
        weight_decay=args.weight_decay)

# Training
def train(epoch):
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda()

        optimizer.zero_grad()
        outputs = net(inputs)
        loss = loss_func(outputs, targets)
        loss.backward()
        # Take the average gradient of all blocks, do nothing if using bn
        try:
            net.module.scale_tau_grad()
        except:
            pass
        optimizer.step()

        train_loss += loss.item()
        prec1, prec5 = accuracy(outputs.data, targets.data, topk=(1, 5))
        acc1 = prec1.item()
        acc5 = prec5.item()

        progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc1: %.3f | Acc5: %.3f '
            % (train_loss/(batch_idx+1), acc1, acc5))

    return (train_loss/batch_idx, acc1, acc5)

def test(epoch):
    global best_acc
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    acc1 = []
    acc5 = []
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            if use_cuda:
                inputs, targets = inputs.cuda(), targets.cuda()
            outputs = net(inputs)
            loss = loss_func(outputs, targets)

            test_loss += loss.item()
            prec1, prec5 = accuracy(outputs.data, targets.data, topk=(1, 5))
            acc1.append(prec1.item())
            acc5.append(prec5.item())

        acc1 = np.mean(acc1)
        acc5 = np.mean(acc5)
        print('\n\nTest loss: %.3f, top1 accuracy: %.3f, top5 accuracy: %.3f'%(test_loss/(batch_idx+1), acc1, acc5))
        #progress_bar(batch_idx, len(testloader), 'Test loss: %.3f | Acc1: %.3f | Acc5: %.3f %%'
        #        % (test_loss/(batch_idx+1), acc1, acc5))

        # Save checkpoint.
        if acc1 > best_acc:
            best_acc = acc1
            checkpoint(acc1, epoch)

    return (test_loss/batch_idx, acc1, acc5)

def checkpoint(acc, epoch):
    # Save checkpoint.
    print('Saving..')
    state = {
        'net': net,
        'acc': acc,
        'epoch': epoch,
        'rng_state': torch.get_rng_state()
    }
    if not os.path.isdir('checkpoint'):
        os.mkdir('checkpoint')
    torch.save(state, './checkpoint/' + args.arch + '_' + args.sess + '_' + 'tau%.5f_'%args.tau + str(args.seed) + '.ckpt')

def adjust_learning_rate(optimizer, epoch):
    """decrease the learning rate at 60 and 120 epoch"""
    if(epoch<60):
        decay = 1.
    elif(epoch<120):
        decay = 10.
    else:
        decay = 100.
    for param_group in optimizer.param_groups:
        param_group['lr'] = param_group['initial_lr'] / decay
    return args.lr / decay

if not os.path.exists(logname):
    with open(logname, 'w') as logfile:
        logwriter = csv.writer(logfile, delimiter=',')
        logwriter.writerow(['epoch', 'lr', 'train loss', 'train acc', 'test loss', 'test acc1', 'test acc5'])


for epoch in range(start_epoch, args.n_epoch):
    lr = adjust_learning_rate(optimizer, epoch)
    train_loss, train_acc1, train_acc5 = train(epoch)
    test_loss, test_acc1, test_acc5 = test(epoch)
    with open(logname, 'a') as logfile:
        logwriter = csv.writer(logfile, delimiter=',')
        logwriter.writerow([epoch, lr, train_loss, train_acc1, test_loss, test_acc1, test_acc5])
