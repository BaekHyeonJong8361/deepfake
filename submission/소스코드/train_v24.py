"""
train_v24.py
============
v24: v20 fine-tune + SNS real 도메인 데이터 추가
=================================================

목적: SNS 뷰티 필터 영상 real FP 감소 (현재 v20 OOD AUC 0.7714 → 개선 목표)

변경점 (v20 대비):
  1. 학습 데이터: splits_v24 (v20 데이터 + insta SNS real 53개 영상)
  2. v20 체크포인트에서 fine-tune (low LR)
  3. WeightedSampler에서 SNS real 5x 오버샘플링
  4. SNS-aware best 저장 (sns_val_only.csv 정확도 기준)

아키텍처/aug 동일 (변수 통제):
  - F3NetLiteV20 (RGB Xception + Freq Xception + SE Fusion)
  - get_v20_train_transforms (변경 없음)
"""

import os
import json
import random
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, WeightedRandomSampler

from sklearn.metrics import accuracy_score, roc_auc_score

from dataset import FastDeepfakeDataset
from train_v6 import BinaryFocalLoss, seed_worker, safe_auc, warmup_cosine, get_eval_transforms
from train_v9 import FocalLossWithLabelSmoothing
from train_v20 import F3NetLiteV20, get_v20_train_transforms


# v24 데이터 경로
TRAIN_CSV  = '/home/t26106/deepfake/splits_v24/train_v24.csv'
VAL_CSV    = '/home/t26106/deepfake/splits_v24/val_v24.csv'
SNS_VAL_CSV = '/home/t26106/deepfake/splits_v24/sns_val_only.csv'

# v20 체크포인트 (fine-tune 출발점)
V20_CKPT   = '/home/t26106/deepfake/saved_models/f3netlite_best_v20_df40_sns_aug.pth'
SAVE_PATH  = '/home/t26106/deepfake/saved_models/f3netlite_best_v24_sns_real_finetune.pth'
RESULTS    = '/home/t26106/deepfake/saved_models/v24_results.json'


def make_v24_sampler(csv_path, sns_oversample=5.0):
    """
    SNS real에 5x 가중치, real/fake 균형 유지.

    구조:
      - fake          : 1 / n_fake
      - non-SNS real  : 1 / n_non_sns_real
      - SNS real      : sns_oversample / n_sns_real
    """
    df = pd.read_csv(csv_path)
    n_total = len(df)
    labels = df['label'].astype(int).values
    is_sns = (df['dataset'].astype(str).str.lower() == 'sns').values

    n_fake = (labels == 1).sum()
    n_non_sns_real = ((labels == 0) & (~is_sns)).sum()
    n_sns_real = ((labels == 0) & is_sns).sum()

    weights = np.zeros(n_total, dtype=np.float64)
    weights[(labels == 1)] = 1.0 / max(n_fake, 1)
    weights[(labels == 0) & (~is_sns)] = 1.0 / max(n_non_sns_real, 1)
    weights[(labels == 0) & is_sns] = sns_oversample / max(n_sns_real, 1)

    num_samples = int(2 * min(n_fake, n_non_sns_real + n_sns_real) * 2)
    print(f'[Sampler] fake={n_fake}, non-SNS-real={n_non_sns_real}, SNS-real={n_sns_real} (oversample {sns_oversample}x)')
    print(f'[Sampler] num_samples per epoch: {num_samples}')
    return WeightedRandomSampler(
        weights=torch.as_tensor(weights, dtype=torch.double),
        num_samples=num_samples,
        replacement=True,
    )


def evaluate(model, loader, criterion, device, name=''):
    model.eval()
    total_loss, ys, ps, hs = 0.0, [], [], []
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
    return {
        'name': name,
        'loss': total_loss / max(len(loader.dataset), 1),
        'acc': accuracy_score(ys, hs) if ys else 0.0,
        'auc': safe_auc(ys, ps) if ys and len(set(ys)) >= 2 else 0.0,
        'real_acc': accuracy_score(
            [y for y in ys if y == 0],
            [h for y, h in zip(ys, hs) if y == 0]
        ) if any(y == 0 for y in ys) else 0.0,
    }


def train():
    # cuDNN 워크어라운드 (CLAUDE.md 항목 4)
    torch.backends.cudnn.enabled = False
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'[v24] Fine-tune from v20 | device={device}')

    # 데이터셋
    train_ds = FastDeepfakeDataset(TRAIN_CSV, transform=get_v20_train_transforms(), label_dtype=torch.float32)
    val_ds   = FastDeepfakeDataset(VAL_CSV,   transform=get_eval_transforms(),      label_dtype=torch.float32)
    sns_val_ds = FastDeepfakeDataset(SNS_VAL_CSV, transform=get_eval_transforms(),  label_dtype=torch.float32)

    sampler = make_v24_sampler(TRAIN_CSV, sns_oversample=5.0)

    common = dict(batch_size=16, num_workers=8, pin_memory=True, worker_init_fn=seed_worker, persistent_workers=True)
    train_loader = DataLoader(train_ds, sampler=sampler, **common)
    val_loader   = DataLoader(val_ds, shuffle=False, **common)
    sns_val_loader = DataLoader(sns_val_ds, shuffle=False, **common)

    # 모델 + v20 체크포인트 로딩
    model = F3NetLiteV20(img_size=299, num_bands=3, pretrained=False).to(device)
    if os.path.exists(V20_CKPT):
        ckpt = torch.load(V20_CKPT, map_location=device, weights_only=False)
        state = ckpt.get('model_state_dict', ckpt)
        model.load_state_dict(state)
        print(f'[v24] v20 체크포인트 로딩 완료 (val AUC: {ckpt.get("best_val_auc", "?")})')
    else:
        print(f'[v24] WARNING: v20 ckpt not found at {V20_CKPT} → pretrained ImageNet으로 시작')

    # 손실
    criterion_train = FocalLossWithLabelSmoothing(gamma=2.0, alpha=0.25, smoothing=0.05)
    criterion_eval  = BinaryFocalLoss(gamma=2.0, alpha=0.25)

    # Fine-tune: LR 10x 낮춤
    optimizer = optim.AdamW(model.parameters(), lr=1e-5, weight_decay=1e-4)

    num_epochs, patience = 12, 5
    best_score, early_stop = -1.0, 0
    history = []

    for epoch in range(num_epochs):
        model.train()
        loss_sum = 0.0
        cur_lr = warmup_cosine(optimizer, epoch, 1, num_epochs, 1e-5)

        for images, labels in train_loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            outputs = model(images)
            loss = criterion_train(outputs, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            loss_sum += loss.item() * images.size(0)

        train_loss = loss_sum / len(train_ds)
        val_m  = evaluate(model, val_loader, criterion_eval, device, 'VAL')
        sns_m  = evaluate(model, sns_val_loader, criterion_eval, device, 'SNS_VAL')

        # SNS-aware composite score: SNS real accuracy (높을수록 좋음) + 전체 AUC
        composite = 0.6 * sns_m['real_acc'] + 0.4 * val_m['auc']

        print(f'[v24 Ep {epoch+1:2d}/{num_epochs}] lr={cur_lr:.1e} | TrainLoss={train_loss:.4f} | '
              f'ValAUC={val_m["auc"]:.4f} | SNS_RealAcc={sns_m["real_acc"]:.4f} | Composite={composite:.4f}')

        history.append({
            'epoch': epoch + 1,
            'lr': cur_lr,
            'train_loss': train_loss,
            'val_auc': val_m['auc'],
            'val_real_acc': val_m['real_acc'],
            'sns_real_acc': sns_m['real_acc'],
            'composite': composite,
        })

        if composite > best_score:
            best_score = composite
            early_stop = 0
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'best_composite': best_score,
                'val_auc': val_m['auc'],
                'sns_real_acc': sns_m['real_acc'],
            }, SAVE_PATH)
            print(f'  ✅ best saved (composite={best_score:.4f})')
        else:
            early_stop += 1
            if early_stop >= patience:
                print(f'  ⛔ Early stop (patience={patience})')
                break

    # 결과 저장
    with open(RESULTS, 'w') as f:
        json.dump({
            'best_composite': best_score,
            'history': history,
            'config': {
                'base_ckpt': V20_CKPT,
                'lr': 1e-5,
                'sns_oversample': 5.0,
                'num_epochs': num_epochs,
                'patience': patience,
            },
        }, f, indent=2, ensure_ascii=False)

    print(f'\n✅ v24 완료. best composite={best_score:.4f}')
    print(f'   체크포인트: {SAVE_PATH}')
    print(f'   결과: {RESULTS}')


if __name__ == '__main__':
    train()
