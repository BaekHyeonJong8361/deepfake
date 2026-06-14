"""
train_v15.py
============
v15: Stage 1 Lead (Xception RGB) with SNS-Domain Augmentation
=============================================================
목적: Stage 1의 v7(Xception RGB) False Positive 개선 (SNS 이미지 오탐 방지).
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
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import transforms

import cv2
from PIL import Image, ImageFilter
import timm

from dataset import FastDeepfakeDataset

# v7/v6 모듈 재사용
from train_v6 import (
    BinaryFocalLoss,
    seed_worker,
    safe_auc,
    evaluate,
    warmup_cosine,
    get_eval_transforms,
    RandomResizeRescale,
    RandomGaussianNoise,
)
from train_v7 import make_4way_balanced_sampler

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

def get_v15_train_transforms(img_size=299):
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

def set_seed_v15(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True

def train():
    set_seed_v15(42)
    device = torch.device("cuda:1")

    print(f"[v15] Xception RGB + SNS-aug | device={device}")
    print(f"  torch={torch.__version__} | cudnn={torch.backends.cudnn.version()} | enabled={torch.backends.cudnn.enabled}")

    batch_size, num_epochs, base_lr = 64, 30, 1e-4
    weight_decay, patience, warmup_epochs, num_workers = 1e-4, 6, 2, 8

    save_dir = "saved_models"
    os.makedirs(save_dir, exist_ok=True)
    best_path = os.path.join(save_dir, "xception_best_v15_sns_aug.pth")

    splits_dir = "splits_v6"
    train_csv = os.path.join(splits_dir, "train_combined.csv")
    val_csv = os.path.join(splits_dir, "val_combined.csv")

    train_ds = FastDeepfakeDataset(train_csv, transform=get_v15_train_transforms(), label_dtype=torch.float32)
    val_ds   = FastDeepfakeDataset(val_csv,   transform=get_eval_transforms(),  label_dtype=torch.float32)
    
    print(f"  train: {len(train_ds):,} | val: {len(val_ds):,}")
    sampler = make_4way_balanced_sampler(train_csv)

    common_loader = dict(batch_size=batch_size, num_workers=num_workers, pin_memory=True, worker_init_fn=seed_worker, persistent_workers=True)
    train_loader = DataLoader(train_ds, sampler=sampler, **common_loader)
    val_loader   = DataLoader(val_ds,   shuffle=False,   **common_loader)

    model = timm.create_model("xception", pretrained=True, num_classes=1).to(device)
    criterion = BinaryFocalLoss(gamma=2.0, alpha=0.25)
    optimizer = optim.AdamW(model.parameters(), lr=base_lr, weight_decay=weight_decay)

    best_score, early_stop = -1.0, 0
    for epoch in range(num_epochs):
        model.train()
        loss_sum = 0.0
        cur_lr = warmup_cosine(optimizer, epoch, warmup_epochs, num_epochs, base_lr)
        for images, labels in train_loader:
            images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(images).squeeze(1), labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            loss_sum += loss.item() * images.size(0)

        val_m = evaluate(model, val_loader, criterion, device, "VAL")
        print(f"[Epoch {epoch+1}/{num_epochs}] lr={cur_lr:.2e} | TrainLoss: {loss_sum/len(train_ds):.4f} | ValAUC: {val_m['auc']:.4f}")

        if val_m["auc"] > best_score:
            print(f"  ⭐ best 갱신! ({best_score:.4f} → {val_m['auc']:.4f})")
            best_score = val_m["auc"]
            early_stop = 0
            torch.save({"epoch": epoch + 1, "model_state_dict": model.state_dict(), "best_val_auc": best_score}, best_path)
        else:
            early_stop += 1
            if early_stop >= patience: break

    print(f"\n✅ v15 완료. 베스트: {best_path}")

if __name__ == "__main__":
    train()
