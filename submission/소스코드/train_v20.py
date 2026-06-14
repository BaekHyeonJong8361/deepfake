"""
train_v20.py
============
v20: F3NetLite Fusion (v11 architecture) trained on DF40 + SNS-Domain Augmentation
==================================================================================

Phase 2: Fusion Specialist for DF40 (The "짬뽕" Model)
  - 목적: RGB와 주파수(Frequency)를 동시에 활용하여 최신 딥페이크(DF40) 대응 + SNS 오탐 방지
  - 모델: F3NetLite (v8/v11/v16 아키텍처)
  - 데이터: splits_v11_v2 (DF40 fake + FF++/DFDC real)
  - Augmentation: SNS-Domain Aug (v15/v16/v18에서 검증됨)

NOTE: 삭제된 원본을 train_v20.cpython-39.pyc 에서 복원.
      - F3NetLiteV20 / FAD / SEBlockFusion / get_v20_train_transforms /
        make_binary_balanced_sampler / evaluate_v20 등 재사용 심볼은 복원됨.
      - RandomBilateralFilter.__call__ 와 RandomDoubleJPEG.__call__ 두 메서드는
        .pyc 디컴파일이 truncated 되어, 클래스 속성/표준 idiom 기반으로 재구성함
        (byte-exact 아님 — 아래 "[복원]" 주석 표시).
      - train() 메인 루프는 복원 불가하므로 stub 처리 (splits_v11_v2 데이터도 삭제됨).
"""
import os
import io
import math
import random
import json
import numpy as np
import pandas as pd
import cv2
from PIL import Image, ImageFilter
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, WeightedRandomSampler
import torch_dct as dct
import timm
from sklearn.metrics import accuracy_score, f1_score, balanced_accuracy_score
from dataset import FastDeepfakeDataset, RandomJPEGCompression
import torchvision.transforms as transforms
from train_v6 import (
    BinaryFocalLoss, seed_worker, safe_auc, warmup_cosine,
    get_eval_transforms, RandomResizeRescale, RandomGaussianNoise,
)
from train_v9 import FocalLossWithLabelSmoothing, mixup_batch, mixup_loss


class RandomStrongGaussianBlur(object):

    def __init__(self, p=1, sigma_min=0.8, sigma_max=2):
        self.p = p
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max

    def __call__(self, img):
        if random.random() > self.p:
            return img
        return img.filter(ImageFilter.GaussianBlur(radius=random.uniform(self.sigma_min, self.sigma_max)))


class RandomBilateralFilter(object):

    def __init__(self, p=1, d_choices=(5, 7), sigma_color_range=(30, 60), sigma_space_range=(30, 60)):
        self.p = p
        self.d_choices = d_choices
        self.sigma_color_range = sigma_color_range
        self.sigma_space_range = sigma_space_range

    def __call__(self, img):
        if random.random() > self.p:
            return img
        d = random.choice(self.d_choices)
        # [복원] 아래 본문은 .pyc truncated 로 클래스 속성 기반 재구성 (byte-exact 아님)
        sigma_color = random.uniform(*self.sigma_color_range)
        sigma_space = random.uniform(*self.sigma_space_range)
        img_cv = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        img_cv = cv2.bilateralFilter(img_cv, d, sigma_color, sigma_space)
        return Image.fromarray(cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB))


class RandomMedianBlur(object):

    def __init__(self, p=1, k_choices=(3, 5)):
        self.p = p
        self.k_choices = k_choices

    def __call__(self, img):
        if random.random() > self.p:
            return img
        img_cv = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        img_cv = cv2.medianBlur(img_cv, random.choice(self.k_choices))
        return Image.fromarray(cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB))


class OneOfSmoothing(object):

    def __init__(self, transforms, p=1):
        self.transforms = transforms
        self.p = p

    def __call__(self, img):
        if random.random() > self.p:
            return img
        return random.choice(self.transforms)(img)


class RandomDoubleJPEG(object):

    def __init__(self, q1_range=(50, 90), q2_range=(50, 90), p=1):
        self.q1_range = q1_range
        self.q2_range = q2_range
        self.p = p

    def __call__(self, img):
        if random.random() > self.p:
            return img
        # [복원] 아래 본문은 .pyc truncated 로 클래스 속성 기반 재구성 (byte-exact 아님)
        q1 = random.randint(*self.q1_range)
        q2 = random.randint(*self.q2_range)
        buf = io.BytesIO()
        img.save(buf, format='JPEG', quality=q1)
        buf.seek(0)
        img = Image.open(buf).convert('RGB')
        buf2 = io.BytesIO()
        img.save(buf2, format='JPEG', quality=q2)
        buf2.seek(0)
        return Image.open(buf2).convert('RGB')


def get_v20_train_transforms(img_size=299):
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.05),
        RandomResizeRescale(target_size=img_size, min_scale=0.5, max_scale=1.5, p=0.5),
        OneOfSmoothing([
            RandomStrongGaussianBlur(p=1, sigma_min=0.8, sigma_max=2),
            RandomBilateralFilter(p=1),
            RandomMedianBlur(p=1),
        ], p=0.5),
        RandomGaussianNoise(p=0.3, sigma_max=8),
        RandomDoubleJPEG(p=0.8),
        RandomJPEGCompression(min_quality=40, max_quality=95),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


class FAD(nn.Module):

    def __init__(self, img_size, num_bands):
        super().__init__()
        self.num_bands = num_bands
        i = torch.arange(img_size).view(-1, 1)
        j = torch.arange(img_size).view(1, -1)
        radius = (i + j).float()
        max_r = radius.max()
        filters = torch.zeros(num_bands, img_size, img_size)
        thresholds = [0, max_r / 16, max_r / 8, max_r + 1]
        for k in range(num_bands):
            filters[k] = ((radius >= thresholds[k]) & (radius < thresholds[k + 1])).float()
        self.register_buffer('base_filters', filters)
        self.add_on_filters = nn.Parameter(torch.randn(num_bands, img_size, img_size) * 0.1)

    def forward(self, x):
        x_dct = dct.dct_2d(x, norm='ortho')
        outputs = []
        for k in range(self.num_bands):
            f_k = self.base_filters[k] + torch.sigmoid(self.add_on_filters[k])
            x_filtered = x_dct * f_k.unsqueeze(0).unsqueeze(0)
            outputs.append(dct.idct_2d(x_filtered, norm='ortho'))
        return torch.cat(outputs, dim=1)


class SEBlockFusion(nn.Module):

    def __init__(self, rgb_dim, freq_dim, reduction=16):
        super().__init__()
        total = rgb_dim + freq_dim
        self.bn_rgb = nn.BatchNorm1d(rgb_dim)
        self.bn_freq = nn.BatchNorm1d(freq_dim)
        self.fc_squeeze = nn.Linear(total, total // reduction)
        self.fc_excite = nn.Linear(total // reduction, total)
        self.classifier = nn.Linear(total, 1)

    def forward(self, rgb_feat, freq_feat):
        rgb_feat = self.bn_rgb(rgb_feat)
        freq_feat = self.bn_freq(freq_feat)
        combined = torch.cat([rgb_feat, freq_feat], dim=1)
        gate = torch.sigmoid(self.fc_excite(F.relu(self.fc_squeeze(combined))))
        return self.classifier(combined * gate)


class F3NetLiteV20(nn.Module):

    def __init__(self, img_size, num_bands, pretrained):
        super().__init__()
        self.fad = FAD(img_size=img_size, num_bands=num_bands)
        self.rgb_branch = timm.create_model('xception', pretrained=pretrained, num_classes=0, global_pool='avg')
        self.freq_branch = timm.create_model('xception', pretrained=pretrained, num_classes=0, global_pool='avg', in_chans=3 * num_bands)
        self.fusion = SEBlockFusion(rgb_dim=2048, freq_dim=2048)

    def forward(self, x):
        freq_input = self.fad(x)
        rgb_feat = self.rgb_branch(x)
        freq_feat = self.freq_branch(freq_input)
        return self.fusion(rgb_feat, freq_feat).squeeze(1)


def make_binary_balanced_sampler(csv_path):
    df = pd.read_csv(csv_path)
    labels = df['label'].astype(int).values
    n_pos = (labels == 1).sum()
    n_neg = (labels == 0).sum()
    weights = np.where(labels == 1, 1 / n_pos, 1 / n_neg)
    weights = torch.as_tensor(weights, dtype=torch.double)
    num_samples = int(2 * min(n_pos, n_neg) * 2)
    return WeightedRandomSampler(weights=weights, num_samples=num_samples, replacement=True)


def evaluate_v20(model, loader, criterion_eval, device, name=''):
    model.eval()
    total_loss, ys, ps, hs = 0, [], [], []
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            outputs = model(images)
            loss = criterion_eval(outputs, labels)
            probs = torch.sigmoid(outputs)
            preds = (probs >= 0.5).float()
            total_loss += loss.item() * images.size(0)
            ys.extend(labels.cpu().numpy().tolist())
            ps.extend(probs.cpu().numpy().tolist())
            hs.extend(preds.cpu().numpy().tolist())
    return {
        'name': name,
        'loss': total_loss / len(loader.dataset),
        'acc': accuracy_score(ys, hs),
        'auc': safe_auc(ys, ps),
    }


def train():
    # 원본 train() 루프는 .pyc 디컴파일이 truncated 되어 복원 불가. splits_v11_v2 데이터도 삭제됨.
    raise NotImplementedError(
        "train_v20.train() 본문은 .pyc 바이트코드에서 복원 불가 (truncated decompilation). "
        "F3NetLiteV20 아키텍처/transforms 등 재사용 심볼은 복원됨."
    )


if __name__ == '__main__':
    train()
