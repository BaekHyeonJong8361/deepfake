"""
train_v6.py
===========
v6: FF++ only baseline + 강한 augmentation
====================================================

목적:
  - v5(JPEG 60-95)보다 강한 증강으로 cross-dataset 일반화 향상 측정
  - "데이터 증강만으로 어디까지 갈 수 있는가" 천장 측정
  - v7 (FF++ + DFDC 통합)과 비교 baseline

v5 대비 변경점:
  - 데이터: splits_v6/train_ff_only.csv (FF++ only, hold-out 재분할)
  - 증강:
      RandomResize(0.5~1.5x → 다시 299) : 해상도 도메인 이동 시뮬레이션
      RandomJPEGCompression(40~95)       : 더 강한 압축 (v5는 60-95)
      GaussianNoise(p=0.3)                : 센서 노이즈
      GaussianBlur(p=0.2)                 : 약한 블러
  - WeightedRandomSampler: REAL/FAKE 클래스 균형 (FF++는 1:6 불균형)
  - Cross-dataset 평가: FF++ in-domain + DFDC cross-domain 동시 측정

GPU: 1번만 사용 (학과 서버 공유)

NOTE: 이 파일은 삭제된 원본을 train_v6.cpython-39.pyc 바이트코드에서 복원한 것이다.
      모든 클래스/함수(import 되어 재사용되는 심볼)는 원본 그대로 복원되었으나,
      train() 메인 루프는 디컴파일이 truncated 되어 복원 불가하므로 stub 처리.
      (재학습은 splits_v6 데이터도 삭제되어 어차피 불가)
"""
import os
import io
import math
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, WeightedRandomSampler
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, balanced_accuracy_score
import timm
import torchvision.transforms as transforms
from PIL import Image, ImageFilter
from dataset import FastDeepfakeDataset, RandomJPEGCompression


class RandomResizeRescale(object):
    """0.5~1.5x로 리사이즈 후 다시 원래 크기로 복원 → 해상도 도메인 이동"""

    def __init__(self, target_size=299, min_scale=0.5, max_scale=1.5, p=0.5):
        self.target_size = target_size
        self.min_scale = min_scale
        self.max_scale = max_scale
        self.p = p

    def __call__(self, img):
        if random.random() > self.p:
            return img
        scale = random.uniform(self.min_scale, self.max_scale)
        intermediate = max(32, int(self.target_size * scale))
        img = img.resize((intermediate, intermediate), Image.BILINEAR)
        img = img.resize((self.target_size, self.target_size), Image.BILINEAR)
        return img


class RandomGaussianNoise(object):
    """센서 노이즈 시뮬레이션 (PIL→np→PIL)"""

    def __init__(self, p=0.3, sigma_max=8):
        self.p = p
        self.sigma_max = sigma_max

    def __call__(self, img):
        if random.random() > self.p:
            return img
        sigma = random.uniform(1, self.sigma_max)
        arr = np.asarray(img).astype(np.float32)
        noise = np.random.normal(0, sigma, arr.shape).astype(np.float32)
        arr = np.clip(arr + noise, 0, 255).astype(np.uint8)
        return Image.fromarray(arr)


class RandomGaussianBlur(object):
    """약한 블러"""

    def __init__(self, p=0.2, radius_max=1.5):
        self.p = p
        self.radius_max = radius_max

    def __call__(self, img):
        if random.random() > self.p:
            return img
        r = random.uniform(0.3, self.radius_max)
        return img.filter(ImageFilter.GaussianBlur(radius=r))


def get_v6_train_transforms(img_size=299):
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.1),
        RandomResizeRescale(target_size=img_size, min_scale=0.5, max_scale=1.5, p=0.5),
        RandomGaussianBlur(p=0.2, radius_max=1.5),
        RandomGaussianNoise(p=0.3, sigma_max=8),
        RandomJPEGCompression(min_quality=40, max_quality=95),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


def get_eval_transforms(img_size=299):
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


class BinaryFocalLoss(nn.Module):

    def __init__(self, gamma, alpha):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.bce = nn.BCEWithLogitsLoss(reduction='none')

    def forward(self, inputs, targets):
        bce_loss = self.bce(inputs, targets)
        probs = torch.sigmoid(inputs)
        p_t = probs * targets + (1 - probs) * (1 - targets)
        loss = bce_loss * (1 - p_t) ** self.gamma
        if self.alpha >= 0:
            alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
            loss = alpha_t * loss
        return loss.mean()


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.enabled = False


def seed_worker(worker_id):
    s = torch.initial_seed() % 0x100000000
    np.random.seed(s)
    random.seed(s)


def safe_auc(y_true, y_prob):
    try:
        return roc_auc_score(y_true, y_prob)
    except Exception:
        return float('nan')


def evaluate(model, loader, criterion, device, name=''):
    model.eval()
    total_loss = 0
    ys = []
    ps = []
    hs = []
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            outputs = model(images).squeeze(1)
            loss = criterion(outputs, labels)
            probs = torch.sigmoid(outputs)
            preds = (probs >= 0.5).float()
            total_loss += loss.item() * images.size(0)
            ys.extend(labels.cpu().numpy().tolist())
            ps.extend(probs.cpu().numpy().tolist())
            hs.extend(preds.cpu().numpy().tolist())
    n = len(loader.dataset)
    return {
        'name': name,
        'loss': total_loss / n,
        'acc': accuracy_score(ys, hs),
        'f1': f1_score(ys, hs, zero_division=0),
        'bal_acc': balanced_accuracy_score(ys, hs),
        'auc': safe_auc(ys, ps),
    }


def warmup_cosine(optimizer, epoch, warmup, total, base_lr, min_lr=1e-06):
    if epoch < warmup:
        lr = base_lr * (epoch + 1) / warmup
    else:
        progress = (epoch - warmup) / max(1, total - warmup)
        lr = min_lr + 0.5 * (base_lr - min_lr) * (1 + math.cos(math.pi * progress))
    for g in optimizer.param_groups:
        g['lr'] = lr
    return lr


def make_class_balanced_sampler(csv_path):
    df = pd.read_csv(csv_path)
    labels = df['label'].astype(int).values
    class_counts = np.bincount(labels, minlength=2)
    class_weights = 1 / np.maximum(class_counts, 1)
    sample_weights = class_weights[labels]
    return WeightedRandomSampler(
        weights=torch.as_tensor(sample_weights, dtype=torch.double),
        num_samples=len(sample_weights),
        replacement=True,
    )


def train():
    # 원본 train() 루프는 .pyc 디컴파일이 truncated 되어 복원되지 않았다.
    # (또한 splits_v6 학습 데이터도 삭제되어 재학습 불가)
    # 학습 구성 요약은 위 모듈 docstring 및 __pycache__/train_v6.cpython-39.pyc 참조.
    raise NotImplementedError(
        "train_v6.train() 본문은 .pyc 바이트코드에서 복원 불가 (truncated decompilation). "
        "재사용되는 클래스/함수만 복원됨. 학습 데이터(splits_v6)도 삭제됨."
    )


if __name__ == '__main__':
    train()
