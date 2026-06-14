"""
train_v8.py
===========
v8: F3-Net Lite (FAD-only dual-branch + SE-block late fusion)
====================================================

v7 대비 변경점 (변수 통제 - 모델만 변경):
  - 모델: Xception (단일) → F3NetLite (RGB Xception + FAD-Frequency Xception + SE Fusion)
  - 데이터/aug/loss/optimizer/sampler: v7과 100% 동일
  - 배치 크기만 64 → 16 (모델 2배로 메모리 한계)

근거: notion_v8_report 분석
  - FAD-only로 main contribution (LFS는 단일 GPU 부담)
  - MixBlock 대신 SE-block late fusion (재현성·안정성)
  - Frequency-aware decomposition으로 generative artifact 직접 포착

목적:
  - v7 vs v8 비교 → "frequency dual-branch 도입의 effect" 통제 측정
  - cross-dataset 일반화 향상 검증 (DFDC 외 추가 데이터셋은 별건)

NOTE: 삭제된 원본을 train_v8.cpython-39.pyc 에서 복원. FAD/SEBlockFusion/F3NetLite 등
      재사용 심볼은 복원되었고, train() 메인 루프는 truncated 되어 stub 처리.
"""
import os
import math
import random
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
    BinaryFocalLoss, seed_worker, safe_auc, evaluate, warmup_cosine,
    get_v6_train_transforms, get_eval_transforms,
)
from train_v7 import make_4way_balanced_sampler


class FAD(nn.Module):
    """
    이미지를 주파수 band별로 분해 후 IDCT로 spatial 복원.
    - base_filters: 고정 binary band mask (low/mid/high)
    - add_on_filters: 학습 가능한 보정 (sigmoid로 [0,1] 제한)
    - 출력: 3 bands × 3 channels = 9 channel tensor
    """

    def __init__(self, img_size, num_bands):
        super().__init__()
        self.num_bands = num_bands
        self.register_buffer('base_filters', self._make_base_filters(img_size, num_bands))
        self.add_on_filters = nn.Parameter(torch.randn(num_bands, img_size, img_size) * 0.1)

    @staticmethod
    def _make_base_filters(size, n_bands):
        """
        DCT (0,0)이 DC라는 점에 기반해 (i+j) 거리로 band 분할.
        thresholds: 저주파에 더 많은 해상도를 주는 비대칭 분할.
        """
        i = torch.arange(size).view(-1, 1)
        j = torch.arange(size).view(1, -1)
        radius = (i + j).float()
        max_r = radius.max()
        filters = torch.zeros(n_bands, size, size)
        thresholds = [0, max_r / 16, max_r / 8, max_r + 1]
        for k in range(n_bands):
            mask = (radius >= thresholds[k]) & (radius < thresholds[k + 1])
            filters[k] = mask.float()
        return filters

    def forward(self, x):
        """
        x: (B, 3, H, W) - 정규화된 RGB
        return: (B, 9, H, W) - 3 bands concatenated
        """
        x_dct = dct.dct_2d(x, norm='ortho')
        outputs = []
        for k in range(self.num_bands):
            f_k = self.base_filters[k] + torch.sigmoid(self.add_on_filters[k])
            x_filtered = x_dct * f_k.unsqueeze(0).unsqueeze(0)
            y_k = dct.idct_2d(x_filtered, norm='ortho')
            outputs.append(y_k)
        return torch.cat(outputs, dim=1)


class SEBlockFusion(nn.Module):
    """
    RGB feature와 Frequency feature를 channel attention으로 융합.
    - BatchNorm1d로 두 branch 분포 정규화 (frequency branch dead 방지)
    - Squeeze-Excitation gating
    - Final FC → logit
    """

    def __init__(self, rgb_dim, freq_dim, reduction):
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
        gate = F.relu(self.fc_squeeze(combined))
        gate = torch.sigmoid(self.fc_excite(gate))
        weighted = combined * gate
        return self.classifier(weighted)


class F3NetLite(nn.Module):
    """
    RGB branch (Xception, in_chans=3) + FAD-Frequency branch (Xception, in_chans=9)
    + SE-block late fusion → binary logit
    """

    def __init__(self, img_size, num_bands, pretrained):
        super().__init__()
        self.rgb_branch = timm.create_model('xception', pretrained=pretrained, num_classes=0, global_pool='avg')
        self.fad = FAD(img_size=img_size, num_bands=num_bands)
        self.freq_branch = timm.create_model('xception', pretrained=pretrained, num_classes=0, global_pool='avg', in_chans=3 * num_bands)
        self.fusion = SEBlockFusion(rgb_dim=2048, freq_dim=2048, reduction=16)

    def forward(self, x):
        rgb_feat = self.rgb_branch(x)
        freq_input = self.fad(x)
        freq_feat = self.freq_branch(freq_input)
        logit = self.fusion(rgb_feat, freq_feat).squeeze(1)
        return logit


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.enabled = False


def _evaluate_v8(model, loader, criterion, device, name=''):
    """
    train_v6.evaluate가 model(images).squeeze(1)을 호출하지만
    F3NetLite는 이미 (B,) shape logit 반환. 따라서 별도 평가 함수.
    """
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


def train():
    # 원본 train() 루프는 .pyc 디컴파일이 truncated 되어 복원 불가. splits_v6 데이터도 삭제됨.
    raise NotImplementedError(
        "train_v8.train() 본문은 .pyc 바이트코드에서 복원 불가 (truncated decompilation)."
    )


if __name__ == '__main__':
    train()
