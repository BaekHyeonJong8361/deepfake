"""
train_v16.py
============
v16: Stage 1 F3NetLite (RGB + Freq) with SNS-Domain Augmentation
================================================================
"""

import os
import io
import random
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

# 기존 유틸리티 재사용
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

def get_v16_train_transforms(img_size=299):
    """v15와 동일한 부드러운 SNS-aware augmentation 레시피."""
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

# =========================================================
# F3Net Components (v8 기반)
# =========================================================
class FAD(nn.Module):
    def __init__(self, img_size=299, num_bands=3):
        super().__init__()
        self.num_bands = num_bands
        self.register_buffer("base_filters", self._make_base_filters(img_size, num_bands))
        self.add_on_filters = nn.Parameter(torch.randn(num_bands, img_size, img_size) * 0.1)

    @staticmethod
    def _make_base_filters(size, n_bands):
        i = torch.arange(size).view(-1, 1)
        j = torch.arange(size).view(1, -1)
        radius = (i + j).float()
        max_r = radius.max()
        filters = torch.zeros(n_bands, size, size)
        thresholds = [0.0, max_r / 16, max_r / 8, max_r + 1]
        for k in range(n_bands):
            filters[k] = ((radius >= thresholds[k]) & (radius < thresholds[k + 1])).float()
        return filters

    def forward(self, x):
        x_dct = dct.dct_2d(x, norm="ortho")
        outputs = []
        for k in range(self.num_bands):
            f_k = self.base_filters[k] + torch.sigmoid(self.add_on_filters[k])
            x_filtered = x_dct * f_k.unsqueeze(0).unsqueeze(0)
            outputs.append(dct.idct_2d(x_filtered, norm="ortho"))
        return torch.cat(outputs, dim=1)

class SEBlockFusion(nn.Module):
    def __init__(self, rgb_dim=2048, freq_dim=2048, reduction=16):
        super().__init__()
        total = rgb_dim + freq_dim
        self.bn_rgb = nn.BatchNorm1d(rgb_dim)
        self.bn_freq = nn.BatchNorm1d(freq_dim)
        self.fc_squeeze = nn.Linear(total, total // reduction)
        self.fc_excite = nn.Linear(total // reduction, total)
        self.classifier = nn.Linear(total, 1)

    def forward(self, rgb_feat, freq_feat):
        rgb_feat, freq_feat = self.bn_rgb(rgb_feat), self.bn_freq(freq_feat)
        combined = torch.cat([rgb_feat, freq_feat], dim=1)
        gate = torch.sigmoid(self.fc_excite(F.relu(self.fc_squeeze(combined))))
        return self.classifier(combined * gate)

class F3NetLite(nn.Module):
    def __init__(self, img_size=299, num_bands=3, pretrained=True):
        super().__init__()
        self.fad = FAD(img_size=img_size, num_bands=num_bands)
        self.rgb_branch = timm.create_model("xception", pretrained=pretrained, num_classes=0, global_pool="avg", in_chans=3)
        self.freq_branch = timm.create_model("xception", pretrained=pretrained, num_classes=0, global_pool="avg", in_chans=3 * num_bands)
        self.fusion = SEBlockFusion(rgb_dim=2048, freq_dim=2048)

    def forward(self, x):
        freq_input = self.fad(x)
        rgb_feat = self.rgb_branch(x)
        freq_feat = self.freq_branch(freq_input)
        return self.fusion(rgb_feat, freq_feat)

# =========================================================
# Training Setup
# =========================================================
def set_seed_v16(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True

def train():
    set_seed_v16(42)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    print(f"[v16] F3NetLite (RGB+Freq) + SNS-aug | device={device}")
    print(f"  torch={torch.__version__} | cudnn={torch.backends.cudnn.version()} | enabled={torch.backends.cudnn.enabled}")

    batch_size, num_epochs, base_lr = 16, 30, 1e-4
    weight_decay, patience, warmup_epochs, num_workers = 1e-4, 6, 2, 8

    save_dir = "saved_models"
    os.makedirs(save_dir, exist_ok=True)
    best_path = os.path.join(save_dir, "f3netlite_best_v16_sns_aug.pth")
    last_path = os.path.join(save_dir, "f3netlite_last_v16_sns_aug.pth")

    splits_dir = "splits_v6"
    train_csv, val_csv = os.path.join(splits_dir, "train_combined.csv"), os.path.join(splits_dir, "val_combined.csv")

    print("📁 데이터 로딩...")
    train_ds = FastDeepfakeDataset(train_csv, transform=get_v16_train_transforms(), label_dtype=torch.float32)
    val_ds   = FastDeepfakeDataset(val_csv,   transform=get_eval_transforms(),  label_dtype=torch.float32)
    
    print(f"  train: {len(train_ds):,} | val: {len(val_ds):,}")
    sampler = make_4way_balanced_sampler(train_csv)

    common_loader = dict(batch_size=batch_size, num_workers=num_workers, pin_memory=True, worker_init_fn=seed_worker, persistent_workers=True)
    train_loader = DataLoader(train_ds, sampler=sampler, **common_loader)
    val_loader   = DataLoader(val_ds,   shuffle=False,   **common_loader)

    print("\n🤖 F3NetLite 로딩...")
    model = F3NetLite(img_size=299, num_bands=3, pretrained=True).to(device)

    criterion = BinaryFocalLoss(gamma=2.0, alpha=0.25)
    optimizer = optim.AdamW(model.parameters(), lr=base_lr, weight_decay=weight_decay)

    best_score, early_stop = -1.0, 0
    start_epoch = 0

    if os.path.exists(last_path):
        print(f"🔄 Resume: {last_path} 발견 → 이어서 학습")
        ckpt = torch.load(last_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        start_epoch = ckpt["epoch"]
        best_score = ckpt.get("best_score", -1.0)
        early_stop = ckpt.get("early_stop", 0)
    elif os.path.exists(best_path):
        print(f"🔄 Resume: {best_path} 발견 (last 없음) → 이어서 학습")
        ckpt = torch.load(best_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        start_epoch = ckpt["epoch"]
        best_score = ckpt.get("best_val_auc", -1.0)

    print(f"\n{'='*60}\n[v16] Training started from epoch {start_epoch+1}.\n{'='*60}\n")

    for epoch in range(start_epoch, num_epochs):
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
            best_score, early_stop = val_m["auc"], 0
            torch.save({
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "best_val_auc": best_score
            }, best_path)
        else:
            early_stop += 1

        torch.save({
            "epoch": epoch + 1,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "best_score": best_score,
            "early_stop": early_stop
        }, last_path)

        if early_stop >= patience:
            print(f"  🛑 {patience}회 연속 개선 없음 → 조기 종료")
            break

    print(f"\n✅ v16 학습 완료. 베스트 모델: {best_path}")

if __name__ == "__main__":
    train()
