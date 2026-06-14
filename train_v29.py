"""
train_v29.py
=============
v29: v20 과 완전 동일 구조·학습 방식, 데이터만 늘림 (변수 1개만 변경).

목적:
  - CLAUDE.md §2 변수 통제 원칙 준수
  - v27 (freeze + fifi) 의 효과가 "freeze 덕인지 데이터 덕인지" 분리 측정
  - 이 v29 vs v27 비교로 "freeze 단독 효과" 역산 가능

v20 대비 유일한 차이:
  - train_csv: splits_v11_v2/train_v11.csv → splits_v27/train_v27.csv (fifi 7798 추가)
  - 나머지 모두 v20 그대로:
    - F3NetLiteV20 (Xception RGB + Xception Freq + SE Fusion)
    - ImageNet pretrained from scratch (v20 ckpt 사용 X)
    - get_v20_train_transforms (mixup 포함)
    - make_binary_balanced_sampler (oversample 1x, 균등)
    - LR 1e-4, AdamW, warmup_cosine, 30 epochs, patience 6
    - FocalLossWithLabelSmoothing + BinaryFocalLoss

NOTE: v27 과 다르게 freeze 없음, oversample 균등 → "데이터만" 효과 측정.
"""

import os, json, random
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from dataset import FastDeepfakeDataset
from train_v6 import (BinaryFocalLoss, seed_worker, safe_auc,
                      warmup_cosine, get_eval_transforms)
from train_v9 import FocalLossWithLabelSmoothing
from train_v20 import (F3NetLiteV20, get_v20_train_transforms,
                       make_binary_balanced_sampler, evaluate_v20,
                       mixup_batch, mixup_loss)


TRAIN_CSV = '/home/t26106/deepfake/splits_v27/train_v27.csv'  # ← 유일한 변경점
VAL_CSV   = '/home/t26106/deepfake/splits_v27/val_v27.csv'

SAVE_PATH = '/home/t26106/deepfake/saved_models/f3netlite_best_v29_v20arch_more_data.pth'
RESULTS   = '/home/t26106/deepfake/saved_models/v29_results.json'


def train():
    torch.backends.cudnn.enabled = False
    random.seed(42); np.random.seed(42); torch.manual_seed(42)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'[v29] v20 architecture | more data (fifi included) | from ImageNet pretrained | device={device}')

    train_ds = FastDeepfakeDataset(TRAIN_CSV, transform=get_v20_train_transforms(), label_dtype=torch.float32)
    val_ds   = FastDeepfakeDataset(VAL_CSV,   transform=get_eval_transforms(),      label_dtype=torch.float32)

    sampler = make_binary_balanced_sampler(TRAIN_CSV)
    common = dict(batch_size=16, num_workers=8, pin_memory=True,
                  worker_init_fn=seed_worker, persistent_workers=True)
    train_loader = DataLoader(train_ds, sampler=sampler, **common)
    val_loader   = DataLoader(val_ds,   shuffle=False, **common)

    # v20 과 동일: ImageNet pretrained from scratch
    model = F3NetLiteV20(img_size=299, num_bands=3, pretrained=True).to(device)

    criterion_train = FocalLossWithLabelSmoothing(gamma=2.0, alpha=0.25, smoothing=0.05)
    criterion_eval  = BinaryFocalLoss(gamma=2.0, alpha=0.25)
    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)

    num_epochs, patience = 30, 6
    best_score, early_stop = -1.0, 0
    history = []

    for epoch in range(num_epochs):
        model.train(); loss_sum = 0.0
        cur_lr = warmup_cosine(optimizer, epoch, 2, num_epochs, 1e-4)

        for images, labels in train_loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            mixed_images, labels_a, labels_b, lam = mixup_batch(images, labels, alpha=0.2)
            optimizer.zero_grad(set_to_none=True)
            outputs = model(mixed_images)
            loss = mixup_loss(criterion_train, outputs, labels_a, labels_b, lam)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            loss_sum += loss.item() * images.size(0)

        train_loss = loss_sum / len(train_ds)
        val_m = evaluate_v20(model, val_loader, criterion_eval, device, 'VAL')

        print(f'[v29 Ep {epoch+1:2d}/{num_epochs}] lr={cur_lr:.1e} | TrainLoss={train_loss:.4f} | '
              f'ValAUC={val_m["auc"]:.4f} | ValAcc={val_m["acc"]:.4f}')

        history.append({'epoch': epoch+1, 'val_auc': val_m['auc'], 'val_acc': val_m['acc']})

        if val_m['auc'] > best_score:
            best_score = val_m['auc']; early_stop = 0
            torch.save({'epoch': epoch+1, 'model_state_dict': model.state_dict(),
                        'best_val_auc': best_score}, SAVE_PATH)
            print(f'  ✅ best saved (val_auc={best_score:.4f})')
        else:
            early_stop += 1
            if early_stop >= patience:
                print(f'  ⛔ Early stop')
                break

    with open(RESULTS, 'w') as f:
        json.dump({'best_val_auc': best_score, 'history': history,
                   'config': {'arch': 'F3NetLiteV20',
                              'train_csv': TRAIN_CSV,
                              'pretrained': 'imagenet (from scratch)',
                              'note': 'v20 과 동일, train_csv 만 변경 (fifi 7798 추가)'}},
                  f, indent=2, ensure_ascii=False)

    print(f'\n✅ v29 완료. best val AUC={best_score:.4f}')


if __name__ == '__main__':
    train()
