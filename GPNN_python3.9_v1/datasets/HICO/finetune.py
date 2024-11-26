"""
Created on Feb 25, 2018

@author: Siyuan Qi

Description of the file.

"""

import os
import shutil
import argparse
import time

import numpy as np
import torch
import torchvision

import metadata
import hico_config
import roi_feature_model


def main(args):
    best_prec1 = 0.0
    args.distributed = args.world_size > 1
    if args.distributed:
        torch.distributed.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                             world_size=args.world_size)

    # Create model
    if args.feature_type == 'vgg':
        model = roi_feature_model.Vgg16(num_classes=len(metadata.action_classes))
    elif args.feature_type == 'resnet':
        model = roi_feature_model.Resnet152(num_classes=len(metadata.action_classes))
    elif args.feature_type == 'densenet':
        model = roi_feature_model.Densenet(num_classes=len(metadata.action_classes))
    input_imsize = (224, 224)

    if not args.distributed:
        if args.feature_type.startswith('alexnet') or args.feature_type.startswith('vgg'):
            model.features = torch.nn.DataParallel(model.features)
            model.cuda()
        else:
            model = torch.nn.DataParallel(model).cuda()
    else:
        model.cuda()
        model = torch.nn.parallel.DistributedDataParallel(model)

    # Define loss function (criterion) and optimizer
    criterion = torch.nn.CrossEntropyLoss().cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # Optionally resume from a checkpoint
    if args.resume:
        checkpoint_path = os.path.join(args.resume, 'model_best.pth')
        if os.path.isfile(checkpoint_path):
            print(f"=> loading checkpoint '{checkpoint_path}'")
            checkpoint = torch.load(checkpoint_path)
            args.start_epoch = checkpoint['epoch']
            best_prec1 = checkpoint['best_prec1']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print(f"=> loaded checkpoint '{checkpoint_path}' (epoch {checkpoint['epoch']})")
        else:
            print(f"=> no checkpoint found at '{checkpoint_path}'")

    torch.backends.cudnn.benchmark = True

    # Data loading code
    normalize = torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                 std=[0.229, 0.224, 0.225])
    transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        normalize,
    ])
    train_dataset = roi_feature_model.HICO(args.data, input_imsize, transform, 'train')
    test_dataset = roi_feature_model.HICO(args.data, input_imsize, transform, 'test')

    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=args.batch_size, shuffle=True,
                                               num_workers=args.workers, pin_memory=True)
    test_loader = torch.utils.data.DataLoader(test_dataset,
                                              batch_size=args.batch_size, shuffle=False,
                                              num_workers=args.workers, pin_memory=False)

    for epoch in range(args.start_epoch, args.epochs):
        adjust_learning_rate(optimizer, epoch)

        # Train for one epoch
        train(train_loader, model, criterion, optimizer, epoch)

        if epoch == 0 or epoch >= 5:
            # Evaluate on test set
            prec1 = validate(test_loader, model, criterion)

            # Save the best checkpoint
            is_best = prec1 > best_prec1
            best_prec1 = max(prec1, best_prec1)
            print(f'Best precision: {best_prec1:.03f}')
            save_checkpoint({
                'epoch': epoch + 1,
                'arch': args.feature_type,
                'state_dict': model.state_dict(),
                'best_prec1': best_prec1,
                'optimizer': optimizer.state_dict(),
            }, is_best, args.resume)

    test_prec = validate(test_loader, model, criterion, test=True)
    print(f'Testing precision: {test_prec:.04f}')


def train(train_loader, model, criterion, optimizer, epoch):
    # Updated for compatibility with Python 3.9
    model.train()
    end = time.time()

    for i, (input, target) in enumerate(train_loader):
        target = target.cuda(non_blocking=True)
        input_var = torch.autograd.Variable(input)
        target_var = torch.autograd.Variable(target)

        # Compute output
        _, output = model(input_var)
        loss = criterion(output, target_var.squeeze(1))

        # Update optimizer
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if i % args.print_freq == 0:
            print(f'Epoch: [{epoch}][{i}/{len(train_loader)}]\tLoss {loss.item():.4f}')


def validate(val_loader, model, criterion, test=False):
    model.eval()
    end = time.time()

    correct = 0
    total = 0
    with torch.no_grad():  # Avoid computing gradients during evaluation
        for i, (input, target) in enumerate(val_loader):
            target = target.cuda(non_blocking=True)
            input_var = torch.autograd.Variable(input)
            target_var = torch.autograd.Variable(target)

            # Compute output
            _, output = model(input_var)
            loss = criterion(output, target_var.squeeze(1))

            _, predicted = torch.max(output, 1)
            total += target.size(0)
            correct += (predicted == target.squeeze(1)).sum().item()

            if i % args.print_freq == 0:
                print(f'Validation: [{i}/{len(val_loader)}]\tLoss {loss.item():.4f}')

    accuracy = correct / total
    return accuracy


def save_checkpoint(state, is_best, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    checkpoint_path = os.path.join(save_dir, 'checkpoint.pth')
    torch.save(state, checkpoint_path)
    if is_best:
        shutil.copyfile(checkpoint_path, os.path.join(save_dir, 'model_best.pth'))


def adjust_learning_rate(optimizer, epoch):
    lr = args.lr * (0.8 ** (epoch // 2))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def parse_arguments():
    paths = hico_config.Paths()
    feature_type = 'resnet'

    parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
    parser.add_argument('--feature-type', default=feature_type, help='feature_type')
    parser.add_argument('--data', metavar='DIR', default=paths.data_root, help='path to dataset')
    parser.add_argument('-j', '--workers', default=1, type=int, metavar='N',
                        help='number of data loading workers')
    parser.add_argument('--epochs', default=10, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('--batch-size', default=32, type=int,
                        metavar='N', help='mini-batch size')
    parser.add_argument('--lr', '--learning-rate', default=1e-4, type=float,
                        metavar='LR', help='initial learning rate')
    parser.add_argument('--print-freq', '-p', default=10, type=int,
                        metavar='N', help='print frequency')
    parser.add_argument('--resume', default='./checkpoints', type=str, metavar='PATH',
                        help='path to latest checkpoint')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_arguments()
    main(args)
