"""
train_v9.py
===========
v9: F3NetLite + Strong Regularization (MixUp + Label Smoothing + Dropout)
========================================================================

v8 대비 변경점 (변수 통제 - regularization만 변경):
  - 모델: F3NetLite (v8과 동일, 단 classifier head에 Dropout 0.3)
  - Loss: Focal Loss → Focal + Label Smoothing 0.05 (calibration)
  - Augmentation: v8 동일 + MixUp(α=0.2) per batch
  - 데이터/optimizer/sampler/lr/wd: 모두 v8과 동일
  - 환경: 시스템 python → venv (torch 2.8 + cuDNN 9, cudnn ON 가능)

근거: v8 분석에서 epoch 7부터 train/val divergence 명확
  - Train Loss 0.0093 → 0.0048 (계속 하락)
  - Val Loss 0.0195 → 0.0263 (상승, +35%)
  - AUC 0.983대 정체 → ceiling 도달
  → regularization 강화로 v8 ceiling 0.005~0.010 돌파 시도

목적:
  - v8 vs v9 비교 → "regularization 강화의 effect" 통제 측정
  - DFDC ceiling(0.988) 근처에서 추가 향상 가능 여부
  - FF AUC 0.974 → 0.980+ 가능성 검증

NOTE: 삭제된 원본을 train_v9.cpython-39.pyc 에서 복원. FocalLossWithLabelSmoothing/
      mixup_batch/mixup_loss/F3NetLiteV9 등 재사용 심볼은 복원되었고, train() 메인
      루프는 truncated 되어 stub 처리.
"""
import os
import math
import random
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torch_dct as dct
import timm
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, balanced_accuracy_score
from dataset import FastDeepfakeDataset
from train_v6 import (
    BinaryFocalLoss, seed_worker, safe_auc, warmup_cosine,
    get_v6_train_transforms, get_eval_transforms,
)
from train_v7 import make_4way_balanced_sampler
from train_v8 import FAD, SEBlockFusion


class SEBlockFusionWithDropout(nn.Module):
    """
    v8의 SEBlockFusion에 classifier head Dropout 0.3 추가.
    feature backbone에는 dropout 추가 X (학습된 표현 보호).
    """

    def __init__(self, rgb_dim, freq_dim, reduction, dropout):
        super().__init__()
        total = rgb_dim + freq_dim
        self.bn_rgb = nn.BatchNorm1d(rgb_dim)
        self.bn_freq = nn.BatchNorm1d(freq_dim)
        self.fc_squeeze = nn.Linear(total, total // reduction)
        self.fc_excite = nn.Linear(total // reduction, total)
        self.dropout = nn.Dropout(p=dropout)
        self.classifier = nn.Linear(total, 1)

    def forward(self, rgb_feat, freq_feat):
        rgb_feat = self.bn_rgb(rgb_feat)
        freq_feat = self.bn_freq(freq_feat)
        combined = torch.cat([rgb_feat, freq_feat], dim=1)
        gate = F.relu(self.fc_squeeze(combined))
        gate = torch.sigmoid(self.fc_excite(gate))
        weighted = combined * gate
        weighted = self.dropout(weighted)
        return self.classifier(weighted)


class F3NetLiteV9(nn.Module):
    """v8 F3NetLite와 동일 구조, classifier head에만 Dropout 0.3 추가"""

    def __init__(self, img_size, num_bands, pretrained, dropout):
        super().__init__()
        self.rgb_branch = timm.create_model('xception', pretrained=pretrained, num_classes=0, global_pool='avg')
        self.fad = FAD(img_size=img_size, num_bands=num_bands)
        self.freq_branch = timm.create_model('xception', pretrained=pretrained, num_classes=0, global_pool='avg', in_chans=3 * num_bands)
        self.fusion = SEBlockFusionWithDropout(rgb_dim=2048, freq_dim=2048, reduction=16, dropout=dropout)

    def forward(self, x):
        rgb_feat = self.rgb_branch(x)
        freq_input = self.fad(x)
        freq_feat = self.freq_branch(freq_input)
        logit = self.fusion(rgb_feat, freq_feat).squeeze(1)
        return logit


class FocalLossWithLabelSmoothing(nn.Module):
    """
    Focal Loss(γ=2, α=0.25)에 label smoothing 추가.
    - 0/1 hard label → (smoothing/2, 1-smoothing/2)로 부드럽게
    - confidence 폭주(val loss 폭증) 방지
    """

    def __init__(self, gamma, alpha, smoothing):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.smoothing = smoothing

    def forward(self, logits, targets):
        targets_smooth = targets * (1 - self.smoothing) + 0.5 * self.smoothing
        bce = F.binary_cross_entropy_with_logits(logits, targets_smooth, reduction='none')
        probs = torch.sigmoid(logits)
        pt = torch.where(targets >= 0.5, probs, 1 - probs)
        alpha_t = torch.where(
            targets >= 0.5,
            torch.full_like(probs, self.alpha),
            torch.full_like(probs, 1 - self.alpha),
        )
        focal_weight = alpha_t * (1 - pt) ** self.gamma
        return (focal_weight * bce).mean()


def mixup_batch(images, labels, alpha=0.2):
    """
    images: (B, C, H, W), labels: (B,) float
    return: mixed_images, labels_a, labels_b, lam
    """
    if alpha <= 0:
        return (images, labels, labels, 1)
    lam = np.random.beta(alpha, alpha)
    lam = max(lam, 1 - lam)
    batch_size = images.size(0)
    index = torch.randperm(batch_size, device=images.device)
    mixed_images = lam * images + (1 - lam) * images[index]
    labels_a = labels
    labels_b = labels[index]
    return (mixed_images, labels_a, labels_b, lam)


def mixup_loss(criterion, logits, labels_a, labels_b, lam):
    return lam * criterion(logits, labels_a) + (1 - lam) * criterion(logits, labels_b)


def set_seed_v9(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True


def evaluate_v9(model, loader, criterion_eval, device, name=''):
    """평가 시에는 mixup/smoothing 없이 pure focal loss (또는 BCE) 사용"""
    model.eval()
    total_loss = 0
    ys = []
    ps = []
    hs = []
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
    n = len(loader.dataset)
    return {
        'name': name,
        'loss': total_loss / n,
        'acc': accuracy_score(ys, hs),
        'f1': f1_score(ys, hs, zero_division=0),
        'bal_acc': balanced_accuracy_score(ys, hs),
        'auc': safe_auc(ys, ps),
    }


def train():
    # 원본 train() 루프는 .pyc 디컴파일이 truncated 되어 복원 불가. splits_v6 데이터도 삭제됨.
    raise NotImplementedError(
        "train_v9.train() 본문은 .pyc 바이트코드에서 복원 불가 (truncated decompilation)."
    )


if __name__ == '__main__':
    train()
