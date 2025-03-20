# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import argparse
from logging import getLogger
import pickle
import os
import math
import numpy as np
import torch

from .logger import create_logger, PD_Stats

import torch.distributed as dist
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F

FALSY_STRINGS = {"off", "false", "0"}
TRUTHY_STRINGS = {"on", "true", "1"}


logger = getLogger()


def bool_flag(s):
    """
    Parse boolean arguments from the command line.
    """
    if s.lower() in FALSY_STRINGS:
        return False
    elif s.lower() in TRUTHY_STRINGS:
        return True
    else:
        raise argparse.ArgumentTypeError("invalid value for a boolean flag")


def restart_from_checkpoint(ckp_paths, run_variables=None, **kwargs):
    """
    Re-start from checkpoint
    """
    # look for a checkpoint in exp repository
    if isinstance(ckp_paths, list):
        for ckp_path in ckp_paths:
            if os.path.isfile(ckp_path):
                break
    else:
        ckp_path = ckp_paths

    if not os.path.isfile(ckp_path):
        return

    logger.info("Found checkpoint at {}".format(ckp_path))

    # open checkpoint file
    checkpoint = torch.load(
        ckp_path, map_location="cuda:" + str(torch.distributed.get_rank() % torch.cuda.device_count())
    )

    # key is what to look for in the checkpoint file
    # value is the object to load
    # example: {'state_dict': model}
    for key, value in kwargs.items():
        if key in checkpoint and value is not None:
            try:
                msg = value.load_state_dict(checkpoint[key], strict=False)
                print(msg)
            except TypeError:
                msg = value.load_state_dict(checkpoint[key])
            logger.info("=> loaded {} from checkpoint '{}'".format(key, ckp_path))
        else:
            logger.warning(
                "=> failed to load {} from checkpoint '{}'".format(key, ckp_path)
            )

    # re load variable important for the run
    if run_variables is not None:
        for var_name in run_variables:
            if var_name in checkpoint:
                run_variables[var_name] = checkpoint[var_name]


def fix_random_seeds(seed=31):
    """
    Fix random seeds.
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)


class AverageMeter(object):
    """computes and stores the average and current value"""

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


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res

class MeanClusteringError(nn.Module):
    """
    Mean Absolute Error
    """

    def __init__(self, num_classes, tau=1):
        super(MeanClusteringError, self).__init__()
        self.register_buffer('embedding', torch.eye(num_classes))
        self.tau = tau

    def to_onehot(self, target):
        return self.embedding[target]

    def forward(self, input, target, threshold=1):
        pred = F.softmax(input / self.tau, dim=1)
        q = self.to_onehot(target).detach()
        p = ((1. - q) * pred).sum(1) / pred.sum(1)
        return (p.log()).mean()

class MAELoss(nn.Module):
    """
    MAE: Mean Absolute Error
    2017 AAAI | Robust Loss Functions under Label Noise for Deep Neural Networks
    Ref: https://github.com/HanxunH/Active-Passive-Losses/blob/master/loss.py
    """
    def __init__(self, num_classes=2):
        super(MAELoss, self).__init__()
        self.num_classes = num_classes

    def forward(self, preds, labels):
        pred = F.softmax(preds, dim=1)
        pred = torch.clamp(pred, min=1e-7, max=1.0)
        label_oh = F.one_hot(labels.long(), self.num_classes).float()
        loss = 1. - torch.sum(label_oh * pred, dim=1)
        return loss.mean()
    
class GCELoss(nn.Module):
    """
    GCE: Generalized Cross Entropy
    2018 NeurIPS | Generalized cross entropy loss for training deep neural networks with noisy labels
    Ref: https://github.com/AlanChou/Truncated-Loss/blob/master/TruncatedLoss.py
    """
    def __init__(self, q=0.7):
        super(GCELoss, self).__init__()
        self.q = q

    def forward(self, preds, labels):
        pred = F.softmax(preds, dim=1)
        pred = torch.clamp(pred, min=1e-7, max=1.0)
        Yg = torch.gather(pred, 1, torch.unsqueeze(labels, 1).long())
        Lq = ((1-(Yg**self.q))/self.q)
        return torch.mean(Lq)
class GECELoss(nn.Module):


    def __init__(self, q=0.7):
        super(GECELoss, self).__init__()
        self.q = q

    def forward(self, preds, labels):
        pred = F.softmax(preds, dim=1)
        pred = torch.clamp(pred, min=1e-7, max=1.0)
        Yg = torch.gather(pred, 1, torch.unsqueeze(labels, 1).long())
        Lq = ((1-(Yg**self.q))/math.log(1 + self.q))
        return torch.mean(Lq)
class FocalLoss(torch.nn.Module):
    '''
        https://github.com/clcarwin/focal_loss_pytorch/blob/master/focalloss.py
    '''

    def __init__(self, gamma=0.5, alpha=None):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        if isinstance(alpha, (float, int)):
            self.alpha = torch.Tensor([alpha, 1-alpha])
        if isinstance(alpha, list):
            self.alpha = torch.Tensor(alpha)
        #self.size_average = size_average

    def forward(self, input, target):
        if input.dim() > 2:
            input = input.view(input.size(0), input.size(1), -1)  # N,C,H,W => N,C,H*W
            input = input.transpose(1, 2)                         # N,C,H*W => N,H*W,C
            input = input.contiguous().view(-1, input.size(2))    # N,H*W,C => N*H*W,C
        target = target.view(-1, 1)

        logpt = F.log_softmax(input, dim=1)
        logpt = logpt.gather(1, target)
        logpt = logpt.view(-1)
        pt = torch.autograd.Variable(logpt.data.exp())

        if self.alpha is not None:
            if self.alpha.type() != input.data.type():
                self.alpha = self.alpha.type_as(input.data)
            at = self.alpha.gather(0, target.data.view(-1))
            logpt = logpt * torch.autograd.Variable(at)

        loss = -1 * (1-pt)**self.gamma * logpt
        return loss.mean()
    
class NLNL(torch.nn.Module):
    def __init__(self, train_loader, num_classes=10, ln_neg=1):
        super(NLNL, self).__init__()
        self.num_classes = num_classes
        self.ln_neg = ln_neg
        weight = torch.FloatTensor(num_classes).zero_() + 1.
        if not hasattr(train_loader.dataset[0], 'targets'):
            weight = [1] * num_classes
            weight = torch.FloatTensor(weight)
        else:
            for i in range(num_classes):
                weight[i] = (torch.from_numpy(np.array(train_loader.dataset.targets[0])) == i).sum()
            weight = 1 / (weight / weight.max())
        self.weight = weight.cuda()
        self.criterion = torch.nn.CrossEntropyLoss(weight=self.weight)
        self.criterion_nll = torch.nn.NLLLoss()

    def forward(self, pred, labels):
        # 产生补充标签，因为随机数从 1 开始选取，因此不可能为原类别标签
        labels_neg = (labels.unsqueeze(-1).repeat(1, self.ln_neg) + torch.LongTensor(len(labels), self.ln_neg).cuda().random_(1, self.num_classes)) % self.num_classes
        labels_neg = torch.autograd.Variable(labels_neg)

        assert labels_neg.max() <= self.num_classes-1
        assert labels_neg.min() >= 0
        assert (labels_neg != labels.unsqueeze(-1).repeat(1, self.ln_neg)).sum() == len(labels)*self.ln_neg

        s_neg = torch.log(torch.clamp(1. - F.softmax(pred, 1), min=1e-5, max=1.)) #产生负损失
        s_neg *= self.weight[labels].unsqueeze(-1).expand(s_neg.size()).cuda()
        loss_neg = self.criterion_nll(s_neg.repeat(self.ln_neg, 1), labels_neg.t().contiguous().view(-1))
        return loss_neg

class RCELoss(nn.Module):
    def __init__(self, num_classes=10, scale=1.0):
        super(RCELoss, self).__init__()
        self.num_classes = num_classes
        self.scale = scale

    def forward(self, pred, labels):
        pred = F.softmax(pred, dim=1)
        pred = torch.clamp(pred, min=1e-7, max=1.0)
        label_one_hot = F.one_hot(labels, self.num_classes).float().to(pred.device)
        label_one_hot = torch.clamp(label_one_hot, min=1e-4, max=1.0)
        loss = (-1 * torch.sum(pred * torch.log(label_one_hot), dim=1))
        return self.scale * loss.mean()
    
class SCELoss(nn.Module):
    def __init__(self, num_classes=10, a=0.5, b=0.5):
        super(SCELoss, self).__init__()
        self.num_classes = num_classes
        self.a = a
        self.b = b
        self.cross_entropy = nn.CrossEntropyLoss()

    def forward(self, pred, labels):
        ce = self.cross_entropy(pred, labels)
        # RCE
        pred = F.softmax(pred, dim=1)
        pred = torch.clamp(pred, min=1e-7, max=1.0)
        label_one_hot = F.one_hot(labels, self.num_classes).float().to(pred.device)
        label_one_hot = torch.clamp(label_one_hot, min=1e-4, max=1.0)
        rce = (-1 * torch.sum(pred * torch.log(label_one_hot), dim=1))

        loss = self.a * ce + self.b * rce.mean()
        return loss
    
class NCELoss(nn.Module):
    def __init__(self, num_classes, scale=1.0):
        super(NCELoss, self).__init__()
        self.num_classes = num_classes
        self.scale = scale

    def forward(self, pred, labels):
        pred = F.log_softmax(pred, dim=1)
        label_one_hot = F.one_hot(labels, self.num_classes).float().to(pred.device)
        loss = -1 * torch.sum(label_one_hot * pred, dim=1) / (-pred.sum(dim=1))
        return self.scale * loss.mean()
    
class NMAE(nn.Module):
    def __init__(self, num_classes=10, scale=1.0):
        super(NMAE, self).__init__()
        self.num_classes = num_classes
        self.scale = scale

    def forward(self, pred, labels):
        pred = F.softmax(pred, dim=1)
        label_one_hot = F.one_hot(labels, self.num_classes).float().to(pred.device)
        norm = 1 / (self.num_classes - 1)
        loss = 1. - torch.sum(label_one_hot * pred, dim=1)
        return self.scale * norm * loss.mean()
    
class NRCELoss(nn.Module):
    def __init__(self, num_classes, scale=1.0):
        super(NRCELoss, self).__init__()
        self.num_classes = num_classes
        self.scale = scale

    def forward(self, pred, labels):
        pred = F.softmax(pred, dim=1)
        pred = torch.clamp(pred, min=1e-7, max=1.0)
        label_one_hot = F.one_hot(labels, self.num_classes).float().to(pred.device)
        label_one_hot = torch.clamp(label_one_hot, min=1e-4, max=1.0)
        norm = 1 / 4 * (self.num_classes - 1)
        rce = (-1 * torch.sum(pred * torch.log(label_one_hot), dim=1))
        return self.scale * norm * rce.mean()
    
class NormalizedFocalLoss(torch.nn.Module):
    def __init__(self, gamma=0.5, num_classes=10, alpha=None, scale=1.0):
        super(NormalizedFocalLoss, self).__init__()
        self.gamma = gamma
        self.size_average = 512
        self.num_classes = num_classes
        self.scale = scale

    def forward(self, input, target):
        target = target.view(-1, 1)
        logpt = F.log_softmax(input, dim=1)
        normalizor = torch.sum(-1 * (1 - logpt.data.exp()) ** self.gamma * logpt, dim=1)
        logpt = logpt.gather(1, target)
        logpt = logpt.view(-1)
        pt = torch.autograd.Variable(logpt.data.exp())
        loss = -1 * (1-pt)**self.gamma * logpt
        loss = self.scale * loss / normalizor

        return loss.mean()
    
class NGCELoss(nn.Module):
    def __init__(self, num_classes=10, q=0.7, scale=1.0):
        super(NGCELoss, self).__init__()
        self.num_classes = num_classes
        self.q = q
        self.scale = scale

    def forward(self, pred, labels):
        pred = F.softmax(pred, dim=1)
        pred = torch.clamp(pred, min=1e-7, max=1.0)
        label_one_hot = F.one_hot(labels, self.num_classes).float().to(pred.device)
        numerators = 1. - torch.pow(torch.sum(label_one_hot * pred, dim=1), self.q)
        denominators = self.num_classes - pred.pow(self.q).sum(dim=1)
        loss = numerators / denominators
        return self.scale * loss.mean()