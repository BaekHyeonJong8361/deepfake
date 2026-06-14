"""
train_v17.py
============
v17: Stage 1 FreqOnlyXception with SNS-Domain Augmentation
==========================================================

목적:
  - Stage 1의 마지막 퍼즐인 v10(FreqOnlyXception)에 SNS-augmentation을 적용.
  - 주파수 전용 모델이 '스튜디오 보정'이나 '인스타 필터'의 주파수 변형을
    최신 딥페이크의 생성 흔적으로 오인하는 False Positive 방지.
  - v15(Xception), v16(F3NetLite)과 함께 완벽한 Stage 1 앙상블 완성.

변수 통제 (GEMINI.md §5):
  - v10 대비 변경: **augmentation transform 만 변경** (get_v17_train_transforms)
  - 동일 유지: 모델(FreqOnlyXception), 데이터(FF++/DFDC combined), Sampler(4-way),
                Loss(Focal+LabelSmoothing), optimizer(AdamW), lr=1e-4, wd=1e-4,
                bs=16, epochs=30, mixup_alpha=0.2, dropout=0.3

환경 (GEMINI.md §2):
  - venv 활성화 필수 (torch 2.8 + cuDNN 91002)
  - cuDNN ON (enabled=True, benchmark=True)
"""

import os
import io
import random
import json
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms

import cv2
from PIL import Image, ImageFilter
import torch_dct as dct
import timm

from dataset import FastDeepfakeDataset

from train_v6 import (
    BinaryFocalLoss,
    seed_worker,
    warmup_cosine,
    get_eval_transforms,
    RandomResizeRescale,
    RandomGaussianNoise,
)
from train_v7 import make_4way_balanced_sampler
from train_v8 import FAD
from train_v9 import (
    FocalLossWithLabelSmoothing,
    mixup_batch,
    mixup_loss,
    evaluate_v9 as evaluate_model,
)
from train_freq_only_v10 import FreqOnlyXception

# =========================================================
# v15/v16 공통 SNS Augmentation 컴포넌트
# =========================================================
class RandomStrongGaussianBlur(object):
    def __init__(self, p=1.0, sigma_min=0.8, sigma_max=2.0):
        self.p, self.sigma_min, self.sigma_max = p, sigma_min, sigma_max
    def __call__(self, img):
        if random.random() > self.p: return img
        return img.filter(ImageFilter.GaussianBlur(radius=random.uniform(self.sigma_min, self.sigma_max)))

class RandomBilateralFilter(object):
    def __init__(self, p=1.0, d_choices=(5, 7), sigma_color_range=(30, 60), sigma_space_range=(30, 60)):
        self.p, self.d_choices, self.sigma_color_range, self.sigma_space_range = p, d_choices, sigma_color_range, sigma_space_range
    def __call__(self, img):
        if random.random() > self.p: return img
        d = random.choice(self.d_choices)
        sc = random.uniform(*self.sigma_color_range)
        ss = random.uniform(*self.sigma_space_range)
        img_cv = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        img_cv = cv2.bilateralFilter(img_cv, d, sc, ss)
        return Image.fromarray(cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB))

class RandomMedianBlur(object):
    def __init__(self, p=1.0, k_choices=(3, 5)):
        self.p, self.k_choices = p, k_choices
    def __call__(self, img):
        if random.random() > self.p: return img
        img_cv = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        img_cv = cv2.medianBlur(img_cv, random.choice(self.k_choices))
        return Image.fromarray(cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB))

class OneOfSmoothing(object):
    def __init__(self, transforms, p=1.0):
        self.transforms, self.p = transforms, p
    def __call__(self, img):
        if random.random() > self.p: return img
        return random.choice(self.transforms)(img)

class RandomDoubleJPEG(object):
    def __init__(self, q1_range=(50, 90), q2_range=(50, 90), p=1.0):
        self.q1_range, self.q2_range, self.p = q1_range, q2_range, p
    def __call__(self, img):
        if random.random() > self.p: return img
        q1, q2 = random.randint(*self.q1_range), random.randint(*self.q2_range)
        out = io.BytesIO()
        img.save(out, format='JPEG', quality=q1)
        img = Image.open(out)
        out2 = io.BytesIO()
        img.save(out2, format='JPEG', quality=q2)
        return Image.open(out2)

def get_v17_train_transforms(img_size=299):
    """v15/v16과 동일한 SNS-aware augmentation 레시피."""
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.05),
        RandomResizeRescale(target_size=img_size, min_scale=0.5, max_scale=1.5, p=0.5),
        OneOfSmoothing([
            RandomStrongGaussianBlur(p=1.0, sigma_min=0.8, sigma_max=2.0),
            RandomBilateralFilter(p=1.0),
            RandomMedianBlur(p=1.0),
        ], p=0.5),
        RandomGaussianNoise(p=0.3, sigma_max=8.0),
        RandomDoubleJPEG(p=0.8),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

def set_seed_v17(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True

def train():
    set_seed_v17(42)

    device = torch.device("cuda:1")
    pin_memory = True

    print(f"[v17] Frequency-Only Xception + SNS-aug | device={device}")
    print(f"      torch={torch.__version__} | cudnn enabled={torch.backends.cudnn.enabled} | cudnn ver={torch.backends.cudnn.version()}")

    batch_size = 16
    num_epochs = 30
    base_lr = 1e-4
    weight_decay = 1e-4
    patience = 6
    warmup_epochs = 2
    num_workers = min(8, os.cpu_count() or 4)

    mixup_alpha = 0.2
    label_smoothing = 0.05
    dropout = 0.3

    save_dir = "saved_models"
    os.makedirs(save_dir, exist_ok=True)
    best_path = os.path.join(save_dir, "freq_only_best_v17_sns_aug.pth")

    splits_dir = "splits_v6"
    train_csv = os.path.join(splits_dir, "train_combined.csv")
    val_csv = os.path.join(splits_dir, "val_combined.csv")

    print("📁 데이터 로딩...")
    train_ds = FastDeepfakeDataset(train_csv, transform=get_v17_train_transforms(), label_dtype=torch.float32)
    val_ds = FastDeepfakeDataset(val_csv, transform=get_eval_transforms(), label_dtype=torch.float32)

    print(f"  train: {len(train_ds):,} | val: {len(val_ds):,}")

    print("\n🎯 4-way balanced sampler 구성:")
    sampler = make_4way_balanced_sampler(train_csv)

    common_loader = dict(
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        worker_init_fn=seed_worker,
        persistent_workers=(num_workers > 0),
    )
    train_loader = DataLoader(train_ds, sampler=sampler, **common_loader)
    val_loader = DataLoader(val_ds, shuffle=False, **common_loader)

    print("\n🤖 FreqOnlyXception 로딩...")
    model = FreqOnlyXception(img_size=299, num_bands=3, pretrained=True, dropout=dropout).to(device)

    criterion_train = FocalLossWithLabelSmoothing(gamma=2.0, alpha=0.25, smoothing=label_smoothing)
    criterion_eval = BinaryFocalLoss(gamma=2.0, alpha=0.25)
    optimizer = optim.AdamW(model.parameters(), lr=base_lr, weight_decay=weight_decay)

    best_score = -1.0
    early_stop = 0

    print(f"\n{'='*60}")
    print(f"[v17] lr={base_lr}, wd={weight_decay}, bs={batch_size}, epochs={num_epochs}")
    print(f"[v17] AUG: SNS Simulation (Bilateral/Gaussian/Median OneOf + Double-JPEG + ColorJitter)")
    print(f"[v17] 데이터: FF++ + DFDC combined (Stage 1 마지막 퍼즐)")
    print(f"{'='*60}\n")

    for epoch in range(num_epochs):
        model.train()
        loss_sum = 0.0
        cur_lr = warmup_cosine(optimizer, epoch, warmup_epochs, num_epochs, base_lr)

        for images, labels in train_loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            mixed_images, labels_a, labels_b, lam = mixup_batch(images, labels, alpha=mixup_alpha)

            optimizer.zero_grad(set_to_none=True)
            outputs = model(mixed_images)
            loss = mixup_loss(criterion_train, outputs, labels_a, labels_b, lam)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            loss_sum += loss.item() * images.size(0)

        avg_train = loss_sum / len(train_loader.dataset)
        val_m = evaluate_model(model, val_loader, criterion_eval, device, "VAL")

        print(f"[Epoch {epoch+1}/{num_epochs}] lr={cur_lr:.2e} | TrainLoss: {avg_train:.4f} | ValAUC: {val_m['auc']:.4f}")

        score = val_m["auc"]
        if score > best_score:
            print(f"  ⭐ best 갱신! ({best_score:.4f} → {score:.4f})")
            best_score = score
            early_stop = 0
            torch.save({
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                "best_val_auc": best_score,
                "val_metrics": val_m,
            }, best_path)
        else:
            early_stop += 1

        if early_stop >= patience:
            print(f"  🛑 {patience}회 연속 개선 없음 → 조기 종료")
            break

    print(f"\n✅ v17 학습 완료. 베스트 모델: {best_path}\n{'='*60}")

if __name__ == "__main__":
    train()
