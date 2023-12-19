import argparse
import datetime
import math
import os
import shutil
import time
import warnings
import gc
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
# import torch.nn.parallel
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from Sketchy import SketchyDataset
from pathlib import Path
from timm.models.layers import trunc_normal_

import torch.nn.functional as F

from torch.autograd import Variable

import models.convnextv2 as convnextv2
import models.convnextv2_t as convnextv2_t
from SoftCrossEntropy import SoftCrossEntropy

warnings.filterwarnings("ignore")


def get_args_parser():
    parser = argparse.ArgumentParser('SBKA fine-tuning', add_help=False)

    # Model parameters
    parser.add_argument('--model', default='convnextv2_tiny', type=str, metavar='MODEL',
                        help='Name of model to train')
    parser.add_argument('--drop_path', type=float, default=0, metavar='PCT',
                        help='Drop path rate (default: 0.1)')

    parser.add_argument('--input_size', default=224, type=int,
                        help='image input size')
    parser.add_argument('--layer_decay_type', type=str, choices=['single', 'group'], default='single',
                        help="""Layer decay strategies. The single strategy assigns a distinct decaying value for each layer,
                                whereas the group strategy assigns the same decaying value for three consecutive layers""")

    parser.add_argument('--clip_grad', type=float, default=None, metavar='NORM',
                        help='Clip gradient norm (default: None, no clipping)')

    parser.add_argument('--finetune', default='../pretrained_model/convnextv2_tiny_1k_224_ema.pt',
                        help='finetune from checkpoint')

    parser.add_argument('--head_init_scale', default=1.0, type=float,
                        help='classifier head initial scale, typically adjusted in fine-tuning')
    # Dataset parameters
    parser.add_argument('--nb_classes', default=100, type=int,
                        help='number of the classification types')

    parser.add_argument('--eval', action='store_true', help='Perform evaluation only')
    parser.add_argument('--device', default='cuda', help='device to use for training / testing')

    parser.add_argument('--zero_version', metavar='VERSION', default='zeroshot1', type=str,
                        help='zeroshot version for training and testing (default: zeroshot1)')

    parser.add_argument('--batch_size', default=210, type=int, metavar='N',
                        help='number of samples per batch')
    parser.add_argument('--savedir', '-s', metavar='DIR',
                        default='../checkpoint/sketchy/',
                        help='path to save dir')

    parser.add_argument('--epochs', default=20, type=int, metavar='N',
                        help='number of total epochs to run')

    parser.add_argument('--lr', '--learning-rate', default=0.0001, type=float,
                        metavar='LR', help='initial learning rate (default: 0.01)')
    parser.add_argument('--weight_decay', '--wd', default=5e-4, type=float,
                        metavar='W', help='weight decay (default: 5e-4)')
    parser.add_argument('--print-freq', '-p', default=50, type=int,
                        metavar='N', help='print frequency (default: 50)')
    parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                        help='evaluate model on validation set')
    parser.add_argument('-f', '--freeze_features', dest='freeze_features', action='store_true',
                        help='freeze features of the base network')

    parser.add_argument('--model_prefix', default='', type=str)


    return parser


def train(train_loader, train_loader_ext, model, criterion, criterion_kd, optimizer, model_t, criterion_kd_1, optimizer_1, epoch):
    batch_time = AverageMeter()
    losses = AverageMeter()
    losses_kld = AverageMeter()
    losses_kd_1 = AverageMeter()
    losses_kd_2 = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    end = time.time()
    for i, ((input, target, cid_mask), (input_ext, target_ext, cid_mask_ext)) in enumerate(
            zip(train_loader, train_loader_ext)):
        input_all = torch.cat([input, input_ext], dim=0)
        tag_zeros = torch.zeros(input.size()[0], 1)
        tag_ones = torch.ones(input_ext.size()[0], 1)
        tag_all = torch.cat([tag_zeros, tag_ones], dim=0)

        target_all = torch.cat([target, target_ext], dim=0)
        cid_mask_all = torch.cat([cid_mask, cid_mask_ext], dim=0)

        shuffle_idx = np.arange(input_all.size()[0])
        np.random.shuffle(shuffle_idx)
        input_all = input_all[shuffle_idx]
        tag_all = tag_all[shuffle_idx]
        target_all = target_all[shuffle_idx]
        cid_mask_all = cid_mask_all[shuffle_idx]

        input_all = input_all.cuda()
        tag_all = tag_all.cuda()
        target_all = target_all.type(torch.LongTensor).view(-1, ).cuda()
        cid_mask_all = cid_mask_all.float().cuda()


        output, output_kd, output_feat = model(input_all)
        loss = criterion(output, target_all)

        loss_kld = torch.Tensor([0]).cuda()
        zn = Variable(torch.cuda.FloatTensor(np.random.normal(0, 1, (len(input_all), output_feat.size(-1)))))
        loss_kld = F.kl_div(F.log_softmax(output_feat, dim=-1), F.softmax(zn, dim=-1), reduction='batchmean')

        if epoch <= 9:
            model_t.eval()
            with torch.no_grad():
                output_t = model_t(input_all)

            loss_kd_1 = criterion_kd(output_kd, output_t, tag_all, cid_mask_all * 0.3)

            acc1, acc5 = accuracy(output, target_all, topk=(1, 5))
            losses.update(loss.item(), input_all.size(0))
            losses_kd_1.update(loss_kd_1.item(), input_ext.size(0))
            losses_kld.update(loss_kld.item(), input_ext.size(0))

            top1.update(acc1[0], input_all.size(0))
            top5.update(acc5[0], input_all.size(0))

            optimizer.zero_grad()


            loss_total = loss + loss_kd_1 + 0.1 * loss_kld
            loss_total.backward()
            optimizer.step()

            batch_time.update(time.time() - end)
            end = time.time()
        else:
            model_t.train()
            output_t = model_t(input_all)
            loss_kd_1 = criterion_kd(output_kd, output_t, tag_all, cid_mask_all * 0.3)
            loss_kd_2 = criterion_kd_1(output_t, output_kd, tag_all, cid_mask_all * 0.3)

            acc1, acc5 = accuracy(output, target_all, topk=(1, 5))
            losses.update(loss.item(), input_all.size(0))

            losses_kld.update(loss_kld.item(), input_ext.size(0))

            losses_kd_1.update(loss_kd_1.item(), input_ext.size(0))
            losses_kd_2.update(loss_kd_2.item(), input_ext.size(0))
            top1.update(acc1[0], input_all.size(0))
            top5.update(acc5[0], input_all.size(0))

            optimizer.zero_grad()
            loss_total = loss + loss_kd_1 + 0.1 * loss_kld
            loss_total.backward(retain_graph=True)

            optimizer_1.zero_grad()
            loss_kd_2.backward()
            optimizer.step()
            optimizer_1.step()
        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0 or i == len(train_loader) - 1:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.2f} ({batch_time.avg:.2f})\t'
                  'Loss {loss.val:.3f} {loss_kld.val:.3f} {losses_kd_1.val:.3f} {losses_kd_2.val:.3f} ({loss.avg:.3f} '
                  '{loss_kld.avg:.3f} {losses_kd_1.avg:.3f} {losses_kd_2.avg:.3f})\t '
                  'Acc@1 {top1.val:.2f} ({top1.avg:.2f})'.format(
                epoch, i, len(train_loader), batch_time=batch_time,
                loss=losses, loss_kld=losses_kld, losses_kd_1=losses_kd_1, losses_kd_2=losses_kd_2, top1=top1))
    return loss_total.cpu().detach().numpy()



def validate(val_loader, model, criterion, criterion_kd, model_t):
    batch_time = AverageMeter()
    losses = AverageMeter()
    losses_kd = AverageMeter()

    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    model.eval()
    model_t.eval()
    end = time.time()
    for i, (input, target) in enumerate(val_loader):
        input = torch.autograd.Variable(input, requires_grad=False).cuda()
        target = target.type(torch.LongTensor).view(-1, )
        target = torch.autograd.Variable(target).cuda()

        # compute output
        with torch.no_grad():

            output_t = model_t(input)
            output, output_kd, _ = model(input)

        loss = criterion(output, target)
        loss_kd = criterion_kd(output_kd, output_t)

        # measure accuracy and record loss
        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        losses.update(loss.item(), input.size(0))
        losses_kd.update(loss_kd.item(), input.size(0))

        top1.update(acc1[0], input.size(0))
        top5.update(acc5[0], input.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0 or i == len(val_loader) - 1:
            print('Test: [{0}/{1}]\t'
                  'Time {batch_time.val:.2f} ({batch_time.avg:.2f})\t'
                  'Loss {loss.val:.3f} {loss_kd.val:.3f} ({loss.avg:.3f} {loss_kd.avg:.3f})\t'
                  'Acc@1 {top1.val:.2f} ({top1.avg:.2f})\t'
                  'Acc@5 {top5.val:.2f} ({top5.avg:.2f})'.format(
                i, len(val_loader), batch_time=batch_time, loss=losses, loss_kd=losses_kd,
                top1=top1, top5=top5))

    print(' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'
          .format(top1=top1, top5=top5))

    return top1.avg.cpu().detach().numpy()


def save_checkpoint(state, wp_epoch, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if state['epoch'] > wp_epoch and is_best:
        print(state['epoch'], "yes, that is probably the best model")
        filepath = '/'.join(filename.split('/')[0:-1])
        shutil.copyfile(filename, os.path.join(filepath, 'model_best.pth.tar'))

class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def adjust_learning_rate(optimizer, epoch):
    lr = args.lr * math.pow(0.001, float(epoch) / args.epochs)

    print('epoch: {}, lr: {}'.format(epoch, lr))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""

    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].contiguous().view(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


gc.collect()
torch.cuda.empty_cache()
args = get_args_parser()
args = args.parse_args()
print(args)

device = torch.device(args.device)


if args.zero_version == "zeroshot1":
    print("train on Sketchy Ext...")
    nb_classes = 100
    args.savedir = os.path.join(args.savedir, 'zeroshot1/')
else:
    print("train on Sketchy Ext Split...")
    nb_classes = 104
    args.savedir = os.path.join(args.savedir, 'zeroshot2/')
warmup_epoch = 9
# -------------------------*********************----------------------
model = convnextv2.__dict__[args.model](
    num_classes=1000,
    drop_path_rate=args.drop_path,
    head_init_scale=args.head_init_scale,
    linear_num=nb_classes,
)

model_t = convnextv2_t.__dict__['convnextv2_tiny'](num_classes=1000, drop_path_rate=args.drop_path, head_init_scale=0.)


if args.finetune and not args.eval:
    device_ids = [0, 1]

    state_dict = model.state_dict()
    print("model_dict:", state_dict.keys())

    checkpoint = torch.load(args.finetune, map_location='cpu')
    print("Load pre-trained checkpoint from: %s" % args.finetune)        # here here here
    checkpoint_model = checkpoint['model']

    print("Load pre-trained checkpoint(teacher model) from: %s" % args.finetune)
    model_t.load_state_dict(checkpoint_model, strict=True)


    trash_vars = [k for k in checkpoint_model.keys() if k not in state_dict.keys()]
    print('trashed vars from resume dict:')
    print(trash_vars)

    model.load_state_dict(checkpoint_model, strict=False)

    # manually initialize fc layer
    trunc_normal_(model.linear.weight, std=2e-5)
    torch.nn.init.constant_(model.linear.bias, 0.)

    model = nn.DataParallel(model, device_ids=device_ids)
    model_t = nn.DataParallel(model_t, device_ids=device_ids)
    model.to(device)
    model_t.to(device)

    for name, value in model_t.named_parameters():
        if name != 'module.head.weight' and name != 'module.head.bias':
            value.requires_grad = False
    for name, child in model_t.named_children():
        for param in child.parameters():
            print(param.requires_grad)

# print(model)

criterion_train = nn.CrossEntropyLoss().cuda()
criterion_train_kd = SoftCrossEntropy().cuda()
criterion_train_kd_1 = SoftCrossEntropy().cuda()
criterion_test = nn.CrossEntropyLoss().cuda()
#
optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
optimizer_1 = torch.optim.Adam(model_t.parameters(), lr=args.lr, weight_decay=args.weight_decay)


cudnn.benchmark = True
immean = [0.485, 0.456, 0.406]  # RGB channel mean for imagenet
imstd = [0.229, 0.224, 0.225]

transformations = transforms.Compose([transforms.ToPILImage(),  # transforms.Compose是pytorch的图像预处理包,一般用Compose把多个步骤整合到一起
                                      transforms.Resize([224, 224]),
                                      transforms.ToTensor(),
                                      transforms.Normalize(immean, imstd)])

sketchy_train = SketchyDataset(split='train', zero_version=args.zero_version, \
                               transform=transformations, aug=True, cid_mask=True)
train_loader = DataLoader(dataset=sketchy_train, batch_size=args.batch_size // 3, shuffle=True, num_workers=2)

sketchy_train_ext = SketchyDataset(split='train', version='all_photo', zero_version=args.zero_version, \
                                   transform=transformations, aug=True, cid_mask=True)
train_loader_ext = DataLoader(dataset=sketchy_train_ext, \
                              batch_size=args.batch_size // 3 * 2, shuffle=True, num_workers=2)

sketchy_val = SketchyDataset(split='val', zero_version=args.zero_version, transform=transformations, aug=False)
val_loader = DataLoader(dataset=sketchy_val, batch_size=args.batch_size, shuffle=False, num_workers=2)

print(str(datetime.datetime.now()) + ' data loaded.')

if args.evaluate:
    acc1 = validate(val_loader, model, criterion_test)

if not os.path.exists(args.savedir):
    os.makedirs(args.savedir)


best_acc1 = 0
for epoch in range(args.epochs):
    model.train()

    adjust_learning_rate(optimizer, epoch)
    adjust_learning_rate(optimizer_1, epoch)

    train(train_loader, train_loader_ext, model, criterion_train, criterion_train_kd, optimizer,
                       model_t, criterion_train_kd_1, optimizer_1, epoch)
    acc1 = validate(val_loader, model, criterion_test, criterion_train_kd, model_t)
    is_best = acc1 > best_acc1
    best_acc1 = max(acc1, best_acc1)


    save_checkpoint({
        'epoch': epoch,
        'state_dict': model.state_dict(),
        'best_acc1': best_acc1,
        'optimizer': optimizer.state_dict(),
        'optimizer_1': optimizer_1.state_dict(),
    }, warmup_epoch, is_best, filename=os.path.join(args.savedir, 'checkpoint.pth.tar'))

