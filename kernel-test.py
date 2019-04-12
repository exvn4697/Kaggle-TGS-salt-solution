import torch
import pandas as pd
import torch.nn as nn
import numpy as np
import torchvision.transforms as transforms

from transformations import (
    ToTensor, Normalize, MasksAdder,
    AffineAugmenter, CropAugmenter, ClassAdder
)

import gc
import os
import cv2
from torch.utils.data import Dataset, DataLoader

def load_image(path, mask=False, to256=False):
    """
    Load image from a given path and pad it on the sides, so that eash side is divisible by 32 (newtwork requirement)

    if pad = True:
        returns image as numpy.array, tuple with padding in pixels as(x_min_pad, y_min_pad, x_max_pad, y_max_pad)
    else:
        returns image as numpy.array
    """
    img = cv2.imread(str(path))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    height, width, _ = img.shape

    if to256:
        height *= 2
        width *= 2

    # Padding in needed for UNet models because they need image size to be divisible by 32
    if height % 32 == 0:
        y_min_pad = 0
        y_max_pad = 0
    else:
        y_pad = 32 - height % 32
        y_min_pad = int(y_pad / 2)
        y_max_pad = y_pad - y_min_pad

    if width % 32 == 0:
        x_min_pad = 0
        x_max_pad = 0
    else:
        x_pad = 32 - width % 32
        x_min_pad = int(x_pad / 2)
        x_max_pad = x_pad - x_min_pad

    if to256:
        if mask:
            img = cv2.resize(img, (202, 202), interpolation=cv2.INTER_NEAREST)
        else:
            img = cv2.resize(img, (202, 202), interpolation=cv2.INTER_LINEAR)

    img = cv2.copyMakeBorder(img, y_min_pad, y_max_pad, x_min_pad, x_max_pad, cv2.BORDER_REFLECT_101)

    if mask:
        img = img[:, :, 0:1] // 255
    else:
        img = img / 255.0

    return img.astype(np.float32)


def load_train_data(train_images_path, train_masks_path, clean_small_masks=True, to256=False):
    file_paths = os.listdir(train_images_path)

    images = [load_image(train_images_path + file_path, to256=to256) for file_path in file_paths]
    masks = [load_image(train_masks_path + file_path, mask=True, to256=to256) for file_path in file_paths]

    filtered_images = images
    filtered_masks = masks

    images = np.array(filtered_images)
    masks = np.array(filtered_masks)

    return images, masks


def load_test_data(test_images_path, load_images=False, to256=False):
    file_paths = os.listdir(test_images_path)
    if load_images:
        images = np.array([load_image(test_images_path + file_path, to256=to256) for file_path in file_paths])
        return file_paths, images
    else:
        return file_paths, None


class SaltDataset(Dataset):
    def __init__(self, x, y=None, transform=None, predict=False):
        self.x = x
        self.y = y
        self.predict = predict
        self.transform = transform

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, idx):
        if not self.predict:
            x_ = self.x[idx]
            y_ = self.y[idx]
            result = {'x': x_, 'y': y_}
        else:
            x_ = self.x[idx]
            result = {'x': x_}

        if self.transform:
            result = self.transform(result)

        return result


def build_data_loader(x, y, transform, batch_size, shuffle, num_workers, predict):
    dataset = SaltDataset(x, y, transform=transform, predict=predict)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
    return dataloader


def trim_masks(masks, height, width):

    if masks.shape[1:3] == (256, 256):
        new_masks = []
        for i in range(masks.shape[0]):
            new_mask = masks[i][26: 256 - 26, 26: 256 - 26]
            new_mask = cv2.resize(new_mask, (101, 101), interpolation=cv2.INTER_NEAREST)
            new_mask = new_mask.reshape(101, 101, 1)
            new_masks.append(new_mask)
        masks = np.array(new_masks)
        gc.collect()
    else:
        if height % 32 == 0:
            y_min_pad = 0
            y_max_pad = 0
        else:
            y_pad = 32 - height % 32
            y_min_pad = int(y_pad / 2)
            y_max_pad = y_pad - y_min_pad

        if width % 32 == 0:
            x_min_pad = 0
            x_max_pad = 0
        else:
            x_pad = 32 - width % 32
            x_min_pad = int(x_pad / 2)
            x_max_pad = x_pad - x_min_pad

        masks = masks[:, y_min_pad: 128 - y_max_pad, x_min_pad: 128 - x_max_pad]

    return masks


def load_model(model, checkpoint_path):
    state = torch.load(checkpoint_path, map_location='cpu')
    model.load_state_dict(state['state_dict'])
    return model


def average_weigths(target, sources):
    sources_named_parameters = [dict(source.named_parameters()) for source in sources]
    frac = 1 / len(sources)

    for name, param in target.named_parameters():
        source_params = [frac * source_named_parameters[name].data for source_named_parameters
                         in sources_named_parameters]

        param.data.copy_(sum(source_params))

    return target


def make_if_not_exist(dir_path):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
        
from torch import nn
from torch.nn import functional as F
from itertools import chain
import pretrainedmodels

# import torch.nn as nn
import math
import torch.utils.model_zoo as model_zoo
import sys
import pdb
from torch.autograd import Variable
import functools

torch_ver = torch.__version__[:3]

# if torch_ver == '0.4':
#     BASE_DIR = os.path.dirname(os.path.abspath(__file__))
#     sys.path.append(os.path.join(BASE_DIR, '../inplace_abn'))
#     from bn import InPlaceABNSync
#     BatchNorm2d = functools.partial(InPlaceABNSync, activation='none')
#
# elif torch_ver == '0.3':
#     BASE_DIR = os.path.dirname(os.path.abspath(__file__))
#     sys.path.append(os.path.join(BASE_DIR, '../inplace_abn_03'))
#     from modules import InPlaceABNSync
#     BatchNorm2d = functools.partial(InPlaceABNSync, activation='none')

from .base_oc_block import BaseOC_Context_Module


class ASP_OC_Module(nn.Module):
    def __init__(self, features, out_features=512, dilations=(12, 24, 36)):
        super(ASP_OC_Module, self).__init__()
        self.context = nn.Sequential(nn.Conv2d(features, out_features, kernel_size=3, padding=1, dilation=1, bias=True),
                                   nn.BatchNorm2d(out_features),
                                   BaseOC_Context_Module(in_channels=out_features, out_channels=out_features, key_channels=out_features//2, value_channels=out_features, 
                                    dropout=0, sizes=([2])))
        self.conv2 = nn.Sequential(nn.Conv2d(features, out_features, kernel_size=1, padding=0, dilation=1, bias=False),
                                   nn.BatchNorm2d(out_features))
        self.conv3 = nn.Sequential(nn.Conv2d(features, out_features, kernel_size=3, padding=dilations[0], dilation=dilations[0], bias=False),
                                   nn.BatchNorm2d(out_features))
        self.conv4 = nn.Sequential(nn.Conv2d(features, out_features, kernel_size=3, padding=dilations[1], dilation=dilations[1], bias=False),
                                   nn.BatchNorm2d(out_features))
        self.conv5 = nn.Sequential(nn.Conv2d(features, out_features, kernel_size=3, padding=dilations[2], dilation=dilations[2], bias=False),
                                   nn.BatchNorm2d(out_features))

        self.conv_bn_dropout = nn.Sequential(
            nn.Conv2d(out_features * 5, out_features, kernel_size=1, padding=0, dilation=1, bias=False),
            nn.BatchNorm2d(out_features),
            nn.Dropout2d(0.1)
            )

    def _cat_each(self, feat1, feat2, feat3, feat4, feat5):
        assert(len(feat1)==len(feat2))
        z = []
        for i in range(len(feat1)):
            z.append(torch.cat((feat1[i], feat2[i], feat3[i], feat4[i], feat5[i]), 1))
        return z

    def forward(self, x):
        if isinstance(x, Variable):
            _, _, h, w = x.size()
        elif isinstance(x, tuple) or isinstance(x, list):
            _, _, h, w = x[0].size()
        else:
            raise RuntimeError('unknown input type')

        feat1 = self.context(x)
        feat2 = self.conv2(x)
        feat3 = self.conv3(x)
        feat4 = self.conv4(x)
        feat5 = self.conv5(x)

        if isinstance(x, Variable):
            out = torch.cat((feat1, feat2, feat3, feat4, feat5), 1)
        elif isinstance(x, tuple) or isinstance(x, list):
            out = self._cat_each(feat1, feat2, feat3, feat4, feat5)
        else:
            raise RuntimeError('unknown input type')

        output = self.conv_bn_dropout(out)
        return output

MODEL_NAME = 'se_resnext50_32x4d'


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1)


class ConvRelu(nn.Module):
    def __init__(self, in_, out, stride):
        super().__init__()
        self.conv = conv3x3(in_, out, stride)
        self.activation = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.activation(x)
        return x


class DecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, dilations=(1, 3, 5), batch_norm=False):
        super().__init__()
        self.conv_0 = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=3, stride=2,
                                         padding=1, output_padding=1)

        self.batch_norm = batch_norm

        self.blocks = nn.Sequential(
            *[nn.Sequential(
                nn.Conv2d(out_channels, out_channels, kernel_size=3, dilation=dilation, padding=dilation),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_channels, out_channels, kernel_size=3, dilation=dilation, padding=dilation),
                nn.ReLU(inplace=True)
            ) for dilation in dilations]
        )

    def forward(self, x):
        conv0 = self.conv_0(x)
        conv0 = F.relu(conv0, inplace=True)
        blocks = [self.blocks[index](conv0) for index in range(len(self.blocks))]
        return torch.cat(blocks, dim=1)


class SawSeenNet(nn.Module):
    def __init__(self, base_channels, pretrained=False, frozen=True):
        super(SawSeenNet, self).__init__()

        self.base_channels = base_channels
        self.pretrained = pretrained
        self.frozen = frozen
        self.training = False

        self.pool = nn.MaxPool2d(2, 2)

        self.probability = 0.2
        self.probability_class = 0.4

        self.dropout = F.dropout2d

        self.encoder = pretrainedmodels.__dict__[MODEL_NAME](num_classes=1000, pretrained='imagenet')

        if self.frozen:
            for p in self.encoder.parameters():
                p.data.requires_grad_(requires_grad=False)

        self.init_conv = self.encoder.layer0.conv1
        self.bn1 = self.encoder.layer0.bn1
        self.relu = self.encoder.layer0.relu1
        self.maxpool = self.encoder.layer0.pool

        self.enc_0 = self.encoder.layer1

        self.enc_1 = self.encoder.layer2

        self.enc_2 = self.encoder.layer3

        self.enc_3 = self.encoder.layer4

        self.middle_conv = ConvRelu(self.base_channels * 32, self.base_channels * 8, stride=2)
        self.avg_pooling = nn.AdaptiveAvgPool2d((1, 1))
        self.class_conv_compressor = nn.Conv2d(512, 64, kernel_size=1)
        self.class_conv = nn.Conv2d(64, 1, kernel_size=1)

        self.dec_3 = DecoderBlock(self.base_channels * 8, self.base_channels * 4, dilations=(1,))
        self.dec_3_pred = nn.Conv2d(self.base_channels * 4, 1, kernel_size=3, padding=1)

        self.dec_2 = DecoderBlock(2304, self.base_channels * 2, dilations=(3,))
        self.dec_2_pred = nn.Conv2d(self.base_channels * 2, 1, kernel_size=3, padding=1)

        self.dec_1 = DecoderBlock(1152, self.base_channels * 1, dilations=(3,))
        self.dec_1_pred = nn.Conv2d(self.base_channels * 1, 1, kernel_size=3, padding=1)

        self.dec_0 = DecoderBlock(960, self.base_channels // 2, dilations=(5,))
        self.dec_0_pred = nn.Conv2d(self.base_channels // 2, 1, kernel_size=3, padding=1)

        self.dec_final_0 = DecoderBlock(480, self.base_channels // 2, dilations=(5,))

        self.context = asp_oc_block.ASP_OC_Module(128, 48)

        self.final = nn.Conv2d(48, 1, kernel_size=5, padding=2)

        self._init_weights()

    def forward(self, x):
        init_conv = self.init_conv(x)
        init_conv = self.bn1(init_conv)
        init_conv = self.relu(init_conv)

        enc_0 = self.enc_0(init_conv)

        enc_1 = self.enc_1(enc_0)

        enc_2 = self.enc_2(enc_1)

        enc_3 = self.enc_3(enc_2)

        middle_conv = self.middle_conv(enc_3)

        middle_pooling = self.avg_pooling(middle_conv)
        middle_pooling = F.relu(self.class_conv_compressor(middle_pooling), inplace=True)
        class_empty_pred = self.class_conv(
            self.dropout(middle_pooling,
                         p=self.probability_class, training=self.training)
        ).view(-1, 1)

        dec_3 = self.dec_3(middle_conv)
        dec_3_pred = self.dec_3_pred(
            self.dropout(dec_3, p=self.probability, training=self.training)
        )

        dec_3_cat = torch.cat([
            dec_3,
            enc_3
        ], 1)

        dec_2 = self.dec_2(dec_3_cat)
        dec_2_pred = self.dec_2_pred(
            self.dropout(dec_2, p=self.probability, training=self.training)
        )

        dec_2_cat = torch.cat([
            dec_2,
            enc_2
        ], 1)

        dec_1 = self.dec_1(dec_2_cat)
        dec_1_pred = self.dec_1_pred(
            self.dropout(dec_1, p=self.probability, training=self.training)
        )

        dec_1_cat = torch.cat([
            dec_1,
            F.interpolate(dec_2, scale_factor=2, mode='nearest'),
            F.interpolate(dec_3, scale_factor=4, mode='nearest'),
            enc_1
        ], 1)

        dec_0 = self.dec_0(dec_1_cat)
        dec_0_pred = self.dec_0_pred(
            self.dropout(dec_0, p=self.probability, training=self.training)
        )

        dec_0_cat = torch.cat([
            dec_0,
            F.interpolate(dec_1, scale_factor=2, mode='nearest'),
            F.interpolate(dec_2, scale_factor=4, mode='nearest'),
            enc_0
        ], 1)

        dec_final_0 = self.dec_final_0(dec_0_cat)

        hyper_column = torch.cat([
            self.dropout(dec_final_0,
                         p=self.probability, training=self.training),

            self.dropout(F.interpolate(dec_0, scale_factor=2, mode='nearest'),
                         p=self.probability, training=self.training),

            self.dropout(F.interpolate(dec_1, scale_factor=4, mode='nearest'),
                         p=self.probability, training=self.training),
        ], dim=1)

        oc = self.context(hyper_column)
        final = self.final(oc)

        return final, class_empty_pred, dec_0_pred, dec_1_pred, dec_2_pred, dec_3_pred

    def set_training(self, flag):
        if flag:
            self.train()
        else:
            self.eval()

        self.training = flag

    def _init_weights(self):
        pretrained_modules = self.encoder.modules()

        not_pretrained_modules = [
            self.middle_conv,
            self.class_conv,
            self.dec_3,
            self.dec_3_pred,
            self.dec_2,
            self.dec_2_pred,
            self.dec_1,
            self.dec_1_pred,
            self.dec_0,
            self.dec_0_pred,
            self.dec_final_0,
            self.final,
        ]

        not_pretrained_modules = chain(*[module.modules() for module in not_pretrained_modules])

        if not self.pretrained:
            self._init_modules(pretrained_modules)

        self._init_modules(not_pretrained_modules)

    @staticmethod
    def _init_modules(modules):
        for m in modules:
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight.data, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

from torch import nn
from __future__ import print_function, division

from torch.autograd import Variable
try:
    from itertools import  ifilterfalse
except ImportError: # py3k
    from itertools import  filterfalse


def lovasz_grad(gt_sorted):
    """
    Computes gradient of the Lovasz extension w.r.t sorted errors
    See Alg. 1 in paper
    """
    p = len(gt_sorted)
    gts = gt_sorted.sum()
    intersection = gts - gt_sorted.float().cumsum(0)
    union = gts + (1 - gt_sorted).float().cumsum(0)
    jaccard = 1. - intersection / union
    if p > 1:  # cover 1-pixel case
        jaccard[1:p] = jaccard[1:p] - jaccard[0:-1]
    return jaccard


def iou_binary(preds, labels, EMPTY=1., ignore=None, per_image=True):
    """
    IoU for foreground class
    binary: 1 foreground, 0 background
    """
    if not per_image:
        preds, labels = (preds,), (labels,)
    ious = []
    for pred, label in zip(preds, labels):
        intersection = ((label == 1) & (pred == 1)).sum()
        union = ((label == 1) | ((pred == 1) & (label != ignore))).sum()
        if not union:
            iou = EMPTY
        else:
            iou = float(intersection) / union
        ious.append(iou)
    iou = mean(ious)    # mean accross images if per_image
    return 100 * iou


def iou(preds, labels, C, EMPTY=1., ignore=None, per_image=False):
    """
    Array of IoU for each (non ignored) class
    """
    if not per_image:
        preds, labels = (preds,), (labels,)
    ious = []
    for pred, label in zip(preds, labels):
        iou = []
        for i in range(C):
            if i != ignore: # The ignored label is sometimes among predicted classes (ENet - CityScapes)
                intersection = ((label == i) & (pred == i)).sum()
                union = ((label == i) | ((pred == i) & (label != ignore))).sum()
                if not union:
                    iou.append(EMPTY)
                else:
                    iou.append(float(intersection) / union)
        ious.append(iou)
    ious = map(mean, zip(*ious)) # mean accross images if per_image
    return 100 * np.array(ious)


# --------------------------- BINARY LOSSES ---------------------------


def lovasz_hinge(logits, labels, per_image=True, ignore=None):
    """
    Binary Lovasz hinge loss
      logits: [B, H, W] Variable, logits at each pixel (between -\infty and +\infty)
      labels: [B, H, W] Tensor, binary ground truth masks (0 or 1)
      per_image: compute the loss per image instead of per batch
      ignore: void class id
    """
    if per_image:
        loss = mean(lovasz_hinge_flat(*flatten_binary_scores(log.unsqueeze(0), lab.unsqueeze(0), ignore))
                    for log, lab in zip(logits, labels))
    else:
        loss = lovasz_hinge_flat(*flatten_binary_scores(logits, labels, ignore))
    return loss


def weighted_lovasz_hinge(logits, labels, weights, per_image=True, ignore=None):
    """
    Binary Lovasz hinge loss
      logits: [B, H, W] Variable, logits at each pixel (between -\infty and +\infty)
      labels: [B, H, W] Tensor, binary ground truth masks (0 or 1)
      per_image: compute the loss per image instead of per batch
      ignore: void class id
    """
    if per_image:
        loss = mean(lovasz_hinge_flat(*flatten_binary_scores(log.unsqueeze(0), lab.unsqueeze(0), ignore)) * weight
                    for log, lab, weight in zip(logits, labels, weights))
    else:
        loss = lovasz_hinge_flat(*flatten_binary_scores(logits, labels, ignore))
    return loss


def lovasz_hinge_flat(logits, labels):
    """
    Binary Lovasz hinge loss
      logits: [P] Variable, logits at each prediction (between -\infty and +\infty)
      labels: [P] Tensor, binary ground truth labels (0 or 1)
      ignore: label to ignore
    """
    if len(labels) == 0:
        # only void pixels, the gradients should be 0
        return logits.sum() * 0.
    signs = 2. * labels.float() - 1.
    errors = (1. - logits * Variable(signs))
    errors_sorted, perm = torch.sort(errors, dim=0, descending=True)
    perm = perm.data
    gt_sorted = labels[perm]
    grad = lovasz_grad(gt_sorted)

    # loss = torch.dot(F.relu(errors_sorted), Variable(grad))

    loss = torch.dot(F.elu(errors_sorted) + 1, Variable(grad))

    return loss


def flatten_binary_scores(scores, labels, ignore=None):
    """
    Flattens predictions in the batch (binary case)
    Remove labels equal to 'ignore'
    """
    scores = scores.view(-1)
    labels = labels.view(-1)
    if ignore is None:
        return scores, labels
    valid = (labels != ignore)
    vscores = scores[valid]
    vlabels = labels[valid]
    return vscores, vlabels


class StableBCELoss(torch.nn.modules.Module):
    def __init__(self):
         super(StableBCELoss, self).__init__()

    def forward(self, input, target):
         neg_abs = - input.abs()
         loss = input.clamp(min=0) - input * target + (1 + neg_abs.exp()).log()
         return loss.mean()


def binary_xloss(logits, labels, ignore=None):
    """
    Binary Cross entropy loss
      logits: [B, H, W] Variable, logits at each pixel (between -\infty and +\infty)
      labels: [B, H, W] Tensor, binary ground truth masks (0 or 1)
      ignore: void class id
    """
    logits, labels = flatten_binary_scores(logits, labels, ignore)
    loss = StableBCELoss()(logits, Variable(labels.float()))
    return loss


# --------------------------- MULTICLASS LOSSES ---------------------------


def lovasz_softmax(probas, labels, only_present=False, per_image=False, ignore=None):
    """
    Multi-class Lovasz-Softmax loss
      probas: [B, C, H, W] Variable, class probabilities at each prediction (between 0 and 1)
      labels: [B, H, W] Tensor, ground truth labels (between 0 and C - 1)
      only_present: average only on classes present in ground truth
      per_image: compute the loss per image instead of per batch
      ignore: void class labels
    """
    if per_image:
        loss = mean(lovasz_softmax_flat(*flatten_probas(prob.unsqueeze(0), lab.unsqueeze(0), ignore), only_present=only_present)
                          for prob, lab in zip(probas, labels))
    else:
        loss = lovasz_softmax_flat(*flatten_probas(probas, labels, ignore), only_present=only_present)
    return loss


def lovasz_softmax_flat(probas, labels, only_present=False):
    """
    Multi-class Lovasz-Softmax loss
      probas: [P, C] Variable, class probabilities at each prediction (between 0 and 1)
      labels: [P] Tensor, ground truth labels (between 0 and C - 1)
      only_present: average only on classes present in ground truth
    """
    C = probas.size(1)
    losses = []
    for c in range(C):
        fg = (labels == c).float() # foreground for class c
        if only_present and fg.sum() == 0:
            continue
        errors = (Variable(fg) - probas[:, c]).abs()
        errors_sorted, perm = torch.sort(errors, 0, descending=True)
        perm = perm.data
        fg_sorted = fg[perm]
        losses.append(torch.dot(errors_sorted, Variable(lovasz_grad(fg_sorted))))
    return mean(losses)


def flatten_probas(probas, labels, ignore=None):
    """
    Flattens predictions in the batch
    """
    B, C, H, W = probas.size()
    probas = probas.permute(0, 2, 3, 1).contiguous().view(-1, C)  # B * H * W, C = P, C
    labels = labels.view(-1)
    if ignore is None:
        return probas, labels
    valid = (labels != ignore)
    vprobas = probas[valid.nonzero().squeeze()]
    vlabels = labels[valid]
    return vprobas, vlabels


def xloss(logits, labels, ignore=None):
    """
    Cross entropy loss
    """
    return F.cross_entropy(logits, Variable(labels), ignore_index=255)


# --------------------------- HELPER FUNCTIONS ---------------------------

def mean(l, ignore_nan=False, empty=0):
    """
    nanmean compatible with generators.
    """
    l = iter(l)
    if ignore_nan:
        l = ifilterfalse(np.isnan, l)
    try:
        n = 1
        acc = next(l)
    except StopIteration:
        if empty == 'raise':
            raise ValueError('Empty mean')
        return empty
    for n, v in enumerate(l, 2):
        acc += v
    if n == 1:
        return acc
    return acc / n


class LovaszLoss(nn.Module):
    def __init__(self):
        super(LovaszLoss, self).__init__()

    def forward(self, input, target):
        return lovasz_hinge(input, target, per_image=True)


class LossAggregator(nn.Module):
    def __init__(self, losses, weights):
        super(LossAggregator, self).__init__()
        self.losses = losses
        self.weights = weights

    def forward(self, input, target):
        losses = [loss(input, target) * weight for loss, weight in zip(self.losses, self.weights)]
        return sum(losses)

    def set_weights(self, new_weights):
        self.weights = new_weights
        
from attrdict import AttrDict
import abc

SMOOTH = 1e-6


class Metric(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def update(self, input, target):
        pass

    @abc.abstractmethod
    def compute(self):
        pass


class BinaryAccuracy(Metric):
    def __init__(self):
        self._num_correct = 0
        self._num_examples = 0

    def update(self, input, target):
        target = target > 0.5
        target = target.float()

        correct = torch.eq(torch.round(input).type(target.type()), target).view(-1)
        self._num_correct += torch.sum(correct).item()
        self._num_examples += correct.size()[0]

    def compute(self):
        return self._num_correct / self._num_examples


class DiceCoefficient(Metric):
    def __init__(self):
        self._dice_coeffs = 0
        self._num_examples = 0

    def update(self, input, target, threshold=0.5):
        input = input > threshold
        input = input.float()

        target = target > 0.5
        target = target.float()

        iflat = input.view(-1)
        tflat = target.view(-1)
        intersection = (iflat * tflat).sum()

        self._dice_coeffs += ((2. * intersection.item() + SMOOTH)
                              / (iflat.sum().item() + tflat.sum().item() + SMOOTH))

        self._num_examples += 1

    def compute(self):
        return self._dice_coeffs / self._num_examples


class CompMetric(Metric):
    def __init__(self):
        self.reset()

    def update(self, input, target, threshold=0.5):
        input = input > threshold
        target = target > threshold
        input = input.cpu().numpy()
        target = target.cpu().numpy()

        self._comp_metrics += calc_metric(input, target).item()

        self._num_examples += input.shape[0]

    def compute(self):
        return self._comp_metrics / self._num_examples

    def reset(self):
        self._comp_metrics = 0
        self._num_examples = 0


def calc_iou(actual, pred):
    intersection = np.count_nonzero(actual * pred)
    union = np.count_nonzero(actual) + np.count_nonzero(pred) - intersection
    iou_result = intersection / union if union != 0 else 0.
    return iou_result


def calc_ious(actuals, preds):
    ious_ = np.array([calc_iou(a, p) for a, p in zip(actuals, preds)])
    return ious_


def calc_precisions(thresholds, ious):
    thresholds = np.reshape(thresholds, (1, -1))
    ious = np.reshape(ious, (-1, 1))
    ps = ious > thresholds
    mps = ps.mean(axis=1)
    return mps


def indiv_scores(masks, preds):
    masks[masks > 0] = 1
    preds[preds > 0] = 1
    ious = calc_ious(masks, preds)
    thresholds = [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]
    precisions = calc_precisions(thresholds, ious)
    emptyMasks = np.count_nonzero(masks.reshape((len(masks), -1)), axis=1) == 0
    emptyPreds = np.count_nonzero(preds.reshape((len(preds), -1)), axis=1) == 0
    adjust = (emptyMasks == emptyPreds).astype(np.float)
    precisions[emptyMasks] = adjust[emptyMasks]

    return precisions


def calc_metric(preds, masks):
    return np.sum(indiv_scores(masks, preds))


# PyTroch version


def iou_pytorch(outputs: torch.Tensor, labels: torch.Tensor):
    # You can comment out this line if you are passing tensors of equal shape
    # But if you are passing output from UNet or something it will most probably
    # be with the BATCH x 1 x H x W shape
    outputs = outputs.squeeze(1)  # BATCH x 1 x H x W => BATCH x H x W

    intersection = (outputs & labels).sum((1, 2)).float()  # Will be zero if Truth=0 or Prediction=0
    union = (outputs | labels).sum((1, 2)).float()  # Will be zzero if both are 0

    iou = (intersection + SMOOTH) / (union + SMOOTH)  # We smooth our devision to avoid 0/0

    thresholded = torch.clamp(20 * (iou - 0.5), 0, 10).ceil() / 10  # This is equal to comparing with thresolds

    return thresholded.sum()  # Or thresholded.mean() if you are interested in average across the batch


# Numpy version
# Well, it's the same function, so I'm going to omit the comments

def iou_numpy(outputs: np.array, labels: np.array):
    outputs = outputs.squeeze(1)

    intersection = (outputs & labels).sum((1, 2))
    union = (outputs | labels).sum((1, 2))

    iou = (intersection + SMOOTH) / (union + SMOOTH)

    thresholded = np.ceil(np.clip(20 * (iou - 0.5), 0, 10)) / 10

    return thresholded  # Or thresholded.mean()
    
from sklearn.model_selection import KFold
import time
from tensorboardX import SummaryWriter
import logging

logger = logging.getLogger(__name__)


class Logger(object):
    def __init__(self, log_dir, vanilla_logger=logger, skip=False):
        self.writer = SummaryWriter(log_dir)
        self.info = vanilla_logger.info
        self.debug = vanilla_logger.debug
        self.warning = vanilla_logger.warning
        self.skip = skip

    def scalar_summary(self, tag, value, step):
        print(tag.ljust(20, ' '), '{:.4f}'.format(value), step)

        if self.skip:
            return
        self.writer.add_scalar(tag, value, step)

    def histo_summary(self, tag, values, step):
        if self.skip:
            return
        self.writer.add_histogram(tag, values, step, bins='tensorflow')

class CosineWithRestarts(torch.optim.lr_scheduler._LRScheduler):  # pylint: disable=protected-access
    """
    Cosine annealing with restarts.
    This is decribed in the paper https://arxiv.org/abs/1608.03983.
    Parameters
    ----------
    optimizer : ``torch.optim.Optimizer``
    t_max : ``int``
        The maximum number of iterations within the first cycle.
    eta_min : ``float``, optional (default=0)
        The minimum learning rate.
    last_epoch : ``int``, optional (default=-1)
        The index of the last epoch. This is used when restarting.
    factor : ``float``, optional (default=1)
        The factor by which the cycle length (``T_max``) increases after each restart.
    """

    def __init__(self,
                 optimizer: torch.optim.Optimizer,
                 t_max: int,
                 eta_min: float = 0.,
                 last_epoch: int = -1,
                 factor: float = 1.) -> None:
        assert t_max > 0
        assert eta_min >= 0
        if t_max == 1 and factor == 1:
            print("Cosine annealing scheduler will have no effect on the learning "
                           "rate since T_max = 1 and factor = 1.")
        self.t_max = t_max
        self.eta_min = eta_min
        self.factor = factor
        self._last_restart = 0
        self._cycle_counter = 0
        self._cycle_factor = 1.
        self._updated_cycle_len = t_max
        self._initialized = False
        super(CosineWithRestarts, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        """Get updated learning rate."""
        # HACK: We need to check if this is the first time ``self.get_lr()`` was called,
        # since ``torch.optim.lr_scheduler._LRScheduler`` will call ``self.get_lr()``
        # when first initialized, but the learning rate should remain unchanged
        # for the first epoch.
        if not self._initialized:
            self._initialized = True
            return self.base_lrs

        step = self.last_epoch + 1
        self._cycle_counter = step - self._last_restart

        lrs = [
                self.eta_min + ((lr - self.eta_min) / 2) * (
                        np.cos(
                                np.pi *
                                (self._cycle_counter % self._updated_cycle_len) /
                                self._updated_cycle_len
                        ) + 1
                )
                for lr in self.base_lrs
        ]

        if self._cycle_counter % self._updated_cycle_len == 0:
            # Adjust the cycle length.
            self._cycle_factor *= self.factor
            self._cycle_counter = 0
            self._updated_cycle_len = int(self._cycle_factor * self.t_max)
            self._last_restart = step

        return lrs


class Trainer:
    def __init__(self, config, model, loss, loss_weights,
                 metrics, data_loaders):
        self.config = config
        self.cuda_index = self.config.cuda_index
        self.model = model

        self.model.cuda(self.cuda_index)

        self.criterion = loss
        self.small_criterion = nn.BCEWithLogitsLoss()
        self.loss_weights = loss_weights
        self.metrics = metrics
        self.data_loaders = data_loaders
        self.data_loader_train = data_loaders.train
        self.data_loader_val = data_loaders.val
        self.data_loader_test = data_loaders.test
        self.model_pattern = None

        self.optimizer = None
        self.scheduler = None

        self.class_loss = nn.BCEWithLogitsLoss()

        self._init_optimizer(self.config.lr)

        self.logger = Logger(self.config.logs_dir)
        self.train_iteration = 0
        self.epoch_iteration = 0
        self.encoder_frozen = True
        self.best_val_loss = +np.inf
        self.best_val_metric = 0
        self.best_model_path = None

    def _init_optimizer(self, init_lr):
        self.optimizer = torch.optim.SGD(self.model.parameters(),
                                         lr=init_lr,
                                         momentum=self.config.momentum,
                                         weight_decay=0.0001)

        self.scheduler = CosineWithRestarts(self.optimizer,
                                            t_max=self.config.cycle_length,
                                            eta_min=self.config.min_lr)

    def _train_per_epoch(self):
        losses = []

        self.model.set_training(True)
        if self.loss_weights is not None and self.epoch_iteration < len(self.loss_weights):
            self.criterion.set_weights(self.loss_weights[self.epoch_iteration])

        metrics = {metric_name: self.metrics[metric_name]() for metric_name in self.metrics}
        class_metric = BinaryAccuracy()

        for sample_batch in self.data_loader_train:
            x, y, y_class, y_64, y_32, y_16, y_8 = \
                sample_batch['x'], sample_batch['y'], sample_batch['y_class'],\
                sample_batch['y_64'], sample_batch['y_32'], sample_batch['y_16'], sample_batch['y_8']

            y_pred, y_pred_class, y_pred_64, y_pred_32, y_pred_16, y_pred_8 = self.model(x.cuda(self.cuda_index))

            self.optimizer.zero_grad()

            loss = self.calc_loss(y, y_class, y_64, y_32, y_16, y_8,
                                  y_pred, y_pred_class, y_pred_64, y_pred_32, y_pred_16, y_pred_8)

            losses.append(loss.item())
            loss.backward()
            self.optimizer.step()

            y_pred = torch.sigmoid(y_pred)
            y_pred_class = torch.sigmoid(y_pred_class)

            for metric_name in metrics:
                metrics[metric_name].update(y_pred, y.cuda(self.cuda_index))

            class_metric.update(y_pred_class, y_class.cuda(self.cuda_index))

            self.train_iteration += 1

        train_loss = sum(losses) / len(losses)
        self.logger.scalar_summary('train_loss', train_loss, self.epoch_iteration)

        metrics['class_binary_accuracy'] = class_metric

        for metric_name in metrics:
            metrics[metric_name] = metrics[metric_name].compute()
            self.logger.scalar_summary('tr_{}'.format(metric_name),
                                       metrics[metric_name], self.epoch_iteration)

    def _validate(self, val_type='val'):
        losses = []

        self.model.set_training(False)

        if val_type == 'val':
            dataloader = self.data_loader_val
        elif val_type == 'test':
            dataloader = self.data_loader_test

        metrics = {metric_name: self.metrics[metric_name]() for metric_name in self.metrics}
        class_metric = BinaryAccuracy()

        with torch.no_grad():
            for sample_batch in dataloader:
                x, y, y_class, y_64, y_32, y_16, y_8 = \
                    sample_batch['x'], sample_batch['y'], sample_batch['y_class'], \
                    sample_batch['y_64'], sample_batch['y_32'], sample_batch['y_16'], sample_batch['y_8']

                y_pred, y_pred_class, y_pred_64, y_pred_32, y_pred_16, y_pred_8 = self.model(x.cuda(self.cuda_index))

                loss = self.calc_loss(y, y_class, y_64, y_32, y_16, y_8,
                                      y_pred, y_pred_class, y_pred_64, y_pred_32, y_pred_16, y_pred_8)

                losses.append(loss.item())

                y_pred = torch.sigmoid(y_pred)
                y_pred_class = torch.sigmoid(y_pred_class)

                for metric_name in metrics:
                    metrics[metric_name].update(y_pred, y.cuda(self.cuda_index))

                class_metric.update(y_pred_class, y_class.cuda(self.cuda_index))

                self.train_iteration += 1

        val_loss = sum(losses) / len(losses)

        self.logger.scalar_summary('{}_loss'.format(val_type), val_loss, self.epoch_iteration)

        metrics['class_binary_accuracy'] = class_metric

        for metric_name in metrics:
            metrics[metric_name] = metrics[metric_name].compute()
            self.logger.scalar_summary('{}_{}'.format(val_type, metric_name),
                                       metrics[metric_name], self.epoch_iteration)

        val_metric = metrics[self.config.val_metric_criterion]

        if val_metric > self.best_val_metric:
            self.best_val_metric = val_metric
            self.best_model_path = self.model_pattern.format(self.epoch_iteration, val_metric)
            self.save_checkpoint(self.best_model_path)

        return val_metric

    def calc_loss(self,
                  y, y_class, y_64, y_32, y_16, y_8,
                  y_pred, y_pred_class, y_pred_64, y_pred_32, y_pred_16, y_pred_8):

        y_64_cuda = y_64.cuda(self.cuda_index)
        y_64_cuda_class = torch.sum(y_64_cuda, dim=(1, 2, 3)) > 0

        y_32_cuda = y_32.cuda(self.cuda_index)
        y_32_cuda_class = torch.sum(y_32_cuda, dim=(1, 2, 3)) > 0

        y_16_cuda = y_16.cuda(self.cuda_index)
        y_16_cuda_class = torch.sum(y_16_cuda, dim=(1, 2, 3)) > 0

        y_8_cuda = y_8.cuda(self.cuda_index)
        y_8_cuda_class = torch.sum(y_8_cuda, dim=(1, 2, 3)) > 0

        loss = self.criterion(y_pred, y.cuda(self.cuda_index))
        loss += self.config.masks_weight * self.criterion(y_pred_64[y_64_cuda_class], y_64_cuda[y_64_cuda_class])
        loss += self.config.masks_weight * self.criterion(y_pred_32[y_32_cuda_class], y_32_cuda[y_32_cuda_class])
        loss += self.config.masks_weight * self.criterion(y_pred_16[y_16_cuda_class], y_16_cuda[y_16_cuda_class])
        loss += self.config.masks_weight * self.criterion(y_pred_8[y_8_cuda_class], y_8_cuda[[y_8_cuda_class]])
        loss += self.config.class_weight * self.class_loss(y_pred_class, y_class.cuda(self.cuda_index))

        return loss

    def save_checkpoint(self, checkpoint_path, optimizer=False):
        if optimizer:
            state = {'state_dict': self.model.state_dict(),
                     'optimizer': self.optimizer.state_dict(),
                     'scheduler': self.scheduler.state_dict()}
        else:
            state = {'state_dict': self.model.state_dict()}

        torch.save(state, checkpoint_path)

    def load_checkpoint(self, checkpoint_path, optimizer=False):
        state = torch.load(checkpoint_path, map_location='cpu')
        self.model.load_state_dict(state['state_dict'])

        if optimizer:
            self.optimizer.load_state_dict(state['optimizer'])
            self.scheduler.load_state_dict(state['scheduler'])

    def train(self, num_epochs, model_pattern):
        self.model_pattern = model_pattern
        for i in range(num_epochs):
            init_time = time.time()

            cur_lr = self.scheduler.get_lr()[0]
            print('cur_lr: {:.4f}'.format(cur_lr))
            if cur_lr == self.config.lr and self.epoch_iteration > (self.config.bce_epochs + self.config.intermediate_epochs):
                print('reinit best val metric')
                self.best_val_metric = 0

            self._train_per_epoch()
            self._validate('val')

            if self.epoch_iteration == (self.config.bce_epochs + self.config.intermediate_epochs):
                print('reinit optimizer')
                self._init_optimizer(self.config.tune_lr)

            if self.epoch_iteration > (self.config.bce_epochs + self.config.intermediate_epochs):
                self.scheduler.step()

            self.epoch_iteration += 1
            print('elapsed time per epoch: {:.1f} s'.format(time.time() - init_time))
            print()

        self.load_checkpoint(self.best_model_path)
        self._validate('test')

        return self.model


IMG_SIZE_ORIGIN = 101
IMG_SIZE_TARGET = 12
IMG_INPUT_CHANNELS = 1

INPUT_PATH = './input/'
TRAIN_PATH = INPUT_PATH + 'train/'
TEST_PATH = INPUT_PATH + 'test/'

TRAIN_IMAGES_PATH = TRAIN_PATH + 'images/'
TRAIN_MASKS_PATH = TRAIN_PATH + 'masks/'
TEST_IMAGES_PATH = TEST_PATH + 'images/'

EXP_NAME = 'final_model'
FOLDS_FILE_PATH = './' + EXP_NAME + '_fold_{}.npy'
CUDA_ID = 1

LR = 0.01
TUNE_LR = 0.01
MIN_LR = 0.0001
MOMENTUM = 0.9
CYCLE_LENGTH = 64
CYCLES = 4

BATCH_SIZE = 16
RANDOM_SEED = 42
FOLDS = 5
THRESHOLD = 0.5
BCE_EPOCHS = 32
INTERMEDIATE_EPOCHS = 2
PRETRAINED_COOLDOWN = 2
DROPOUT_COOLDOWN = 2
NUM_EPOCHS = BCE_EPOCHS + INTERMEDIATE_EPOCHS + CYCLES * CYCLE_LENGTH

CLASS_WEIGHT = 0.05
MASKS_WEIGHT = 0.12
PROB = 0.5
PROB_CLASS = 0.8

VAL_METRIC_CRITERION = 'comp_metric'
MODEL_FILE_DIR = './saved_models_' + str(EXP_NAME)
LOGS_DIR = './logs/logs_' + str(EXP_NAME)

make_if_not_exist(MODEL_FILE_DIR)
make_if_not_exist(LOGS_DIR)

MODEL_FILE_PATH = MODEL_FILE_DIR + '/model_{}_{:.4f}'


def predict(config, model, data_loader, thresholding=True, threshold=THRESHOLD, tta=True):
    model.set_training(False)

    y_preds = []
    with torch.no_grad():
        for sample_batch in data_loader:
            x = sample_batch['x']
            y_pred, *preds = model(x.cuda(config.cuda_index))
            y_pred = torch.sigmoid(y_pred)

            if tta:
                x_flipped = x.flip(3)
                y_pred_flipped, *preds = model(x_flipped.cuda(config.cuda_index))
                y_pred_flipped = torch.sigmoid(y_pred_flipped)
                y_pred_flipped = y_pred_flipped.flip(3)

                y_pred += y_pred_flipped
                y_pred /= 2

            if thresholding:
                y_pred = y_pred > threshold

            y_pred = y_pred.cpu().numpy().transpose((0, 2, 3, 1))
            y_preds.append(y_pred)

    y_preds = np.concatenate(y_preds, axis=0)
    return y_preds


def k_fold():
    images, masks = load_train_data(TRAIN_IMAGES_PATH, TRAIN_MASKS_PATH)
    test_file_paths, test_images = load_test_data(TEST_IMAGES_PATH, load_images=True, to256=False)

    train_transformer = transforms.Compose([CropAugmenter(), AffineAugmenter(), MasksAdder(), ToTensor(),
                                            Normalize(), ClassAdder()])

    eval_transformer = transforms.Compose([MasksAdder(), ToTensor(), Normalize(), ClassAdder()])

    predict_transformer = transforms.Compose([ToTensor(predict=True), Normalize(predict=True)])

    test_images_loader = build_data_loader(test_images, None, predict_transformer, batch_size=BATCH_SIZE,
                                           shuffle=False, num_workers=4, predict=True)

    k_fold = KFold(n_splits=FOLDS, random_state=RANDOM_SEED, shuffle=True)

    test_masks_folds = []

    config = AttrDict({
        'cuda_index': CUDA_ID,
        'momentum': MOMENTUM,
        'lr': LR,
        'tune_lr': TUNE_LR,
        'min_lr': MIN_LR,
        'bce_epochs': BCE_EPOCHS,
        'intermediate_epochs': INTERMEDIATE_EPOCHS,
        'cycle_length': CYCLE_LENGTH,
        'logs_dir': LOGS_DIR,
        'masks_weight': MASKS_WEIGHT,
        'class_weight': CLASS_WEIGHT,
        'val_metric_criterion': 'comp_metric'
    })

    for index, (train_index, valid_index) in list(enumerate(k_fold.split(images))):
        print('fold_{}\n'.format(index))

        x_train_fold, x_valid = images[train_index], images[valid_index]
        y_train_fold, y_valid = masks[train_index], masks[valid_index]

        train_data_loader = build_data_loader(x_train_fold, y_train_fold, train_transformer, batch_size=BATCH_SIZE,
                                              shuffle=True, num_workers=4, predict=False)
        val_data_loader = build_data_loader(x_valid, y_valid, eval_transformer, batch_size=BATCH_SIZE,
                                            shuffle=False, num_workers=4, predict=False)
        test_data_loader = build_data_loader(x_valid, y_valid, eval_transformer, batch_size=BATCH_SIZE,
                                             shuffle=False, num_workers=4, predict=False)

        data_loaders = AttrDict({
            'train': train_data_loader,
            'val': val_data_loader,
            'test': test_data_loader
        })

        zers = np.zeros(BCE_EPOCHS)
        zers += 0.1
        lovasz_ratios = np.linspace(0.1, 0.9, INTERMEDIATE_EPOCHS)
        lovasz_ratios = np.hstack((zers, lovasz_ratios))
        bce_ratios = 1.0 - lovasz_ratios
        loss_weights = [(bce_ratio, lovasz_ratio) for bce_ratio, lovasz_ratio in zip(bce_ratios, lovasz_ratios)]

        loss = LossAggregator((nn.BCEWithLogitsLoss(), LovaszLoss()), weights=[0.9, 0.1])

        metrics = {'binary_accuracy': BinaryAccuracy,
                   'dice_coefficient': DiceCoefficient,
                   'comp_metric': CompMetric}

        segmentor = SawSeenNet(base_channels=64, pretrained=True, frozen=False).cuda(config.cuda_index)

        trainer = Trainer(config=config, model=segmentor, loss=loss, loss_weights=loss_weights,
                          metrics=metrics, data_loaders=data_loaders)

        segmentor = trainer.train(num_epochs=NUM_EPOCHS, model_pattern=MODEL_FILE_PATH + '_{}_fold.pth'.format(index))

        test_masks = predict(config, segmentor, test_images_loader, thresholding=False)
        test_masks = trim_masks(test_masks, height=IMG_SIZE_ORIGIN, width=IMG_SIZE_ORIGIN)

        test_masks_folds.append(test_masks)

        np.save(FOLDS_FILE_PATH.format(index), test_masks)

    result_masks = np.zeros_like(test_masks_folds[0])

    for test_masks in test_masks_folds:
        result_masks += test_masks

    result_masks = result_masks.astype(dtype=np.float32)
    result_masks /= FOLDS
    result_masks = result_masks > THRESHOLD

    return test_file_paths, result_masks


if __name__ == '__main__':
    test_file_paths, result_masks = k_fold()

    def rle_encoding(x):
        dots = np.where(x.T.flatten() == 1)[0]
        run_lengths = []
        prev = -2
        for b in dots:
            if (b > prev + 1):
                run_lengths.extend((b + 1, 0))
            run_lengths[-1] += 1
            prev = b
        return run_lengths

    all_masks = []

    for p_mask in list(result_masks):
        p_mask = rle_encoding(p_mask)
        all_masks.append(' '.join(map(str, p_mask)))

    test_ids = [test_file_path.split('.')[0] for test_file_path in test_file_paths]

    submit = pd.DataFrame([test_ids, all_masks]).T
    submit.columns = ['id', 'rle_mask']
    submit.to_csv('submit_exp_{}_cuda_{}.csv'.format(EXP_NAME, CUDA_ID), index=False)
