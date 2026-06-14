"""
train_v25.py
============
v25: v24와 동일하지만 SNS 오버샘플링 5x → 2x로 완화

목적: v24의 fake score 압축 문제 해결
  - v24: SNS_real 5x oversample → fake confidence 압축됨 (0.001~0.006)
  - v25: SNS_real 2x oversample → fake 학습 비중 회복 기대

변경점 (v24 대비 1개):
  - sns_oversample: 5.0 → 2.0
나머지 모두 동일:
  - v20 체크포인트에서 fine-tune
  - LR 1e-5
  - F3NetLiteV20 아키텍처
  - get_v20_train_transforms
"""

import os
import json
import random
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from dataset import FastDeepfakeDataset
from train_v6 import BinaryFocalLoss, seed_worker, safe_auc, warmup_cosine, get_eval_transforms
from train_v9 import FocalLossWithLabelSmoothing
from train_v20 import F3NetLiteV20, get_v20_train_transforms
from train_v24 import make_v24_sampler, evaluate  # 재사용


TRAIN_CSV  = '/home/t26106/deepfake/splits_v24/train_v24.csv'
VAL_CSV    = '/home/t26106/deepfake/splits_v24/val_v24.csv'
SNS_VAL_CSV = '/home/t26106/deepfake/splits_v24/sns_val_only.csv'

V20_CKPT   = '/home/t26106/deepfake/saved_models/f3netlite_best_v20_df40_sns_aug.pth'
SAVE_PATH  = '/home/t26106/deepfake/saved_models/f3netlite_best_v25_sns_oversample2x.pth'
RESULTS    = '/home/t26106/deepfake/saved_models/v25_results.json'

SNS_OVERSAMPLE = 2.0  # ← v24의 5.0에서 변경 (유일한 차이점)


def train():
    torch.backends.cudnn.enabled = False
    random.seed(42); np.random.seed(42); torch.manual_seed(42)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'[v25] Fine-tune from v20 | SNS oversample={SNS_OVERSAMPLE}x | device={device}')

    train_ds = FastDeepfakeDataset(TRAIN_CSV, transform=get_v20_train_transforms(), label_dtype=torch.float32)
    val_ds   = FastDeepfakeDataset(VAL_CSV,   transform=get_eval_transforms(),      label_dtype=torch.float32)
    sns_val_ds = FastDeepfakeDataset(SNS_VAL_CSV, transform=get_eval_transforms(),  label_dtype=torch.float32)

    sampler = make_v24_sampler(TRAIN_CSV, sns_oversample=SNS_OVERSAMPLE)

    common = dict(batch_size=16, num_workers=8, pin_memory=True, worker_init_fn=seed_worker, persistent_workers=True)
    train_loader = DataLoader(train_ds, sampler=sampler, **common)
    val_loader   = DataLoader(val_ds, shuffle=False, **common)
    sns_val_loader = DataLoader(sns_val_ds, shuffle=False, **common)

    model = F3NetLiteV20(img_size=299, num_bands=3, pretrained=False).to(device)
    if os.path.exists(V20_CKPT):
        ckpt = torch.load(V20_CKPT, map_location=device, weights_only=False)
        state = ckpt.get('model_state_dict', ckpt)
        model.load_state_dict(state)
        print(f'[v25] v20 체크포인트 로딩 완료')

    criterion_train = FocalLossWithLabelSmoothing(gamma=2.0, alpha=0.25, smoothing=0.05)
    criterion_eval  = BinaryFocalLoss(gamma=2.0, alpha=0.25)
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
        composite = 0.6 * sns_m['real_acc'] + 0.4 * val_m['auc']

        print(f'[v25 Ep {epoch+1:2d}/{num_epochs}] lr={cur_lr:.1e} | TrainLoss={train_loss:.4f} | '
              f'ValAUC={val_m["auc"]:.4f} | SNS_RealAcc={sns_m["real_acc"]:.4f} | Composite={composite:.4f}')

        history.append({
            'epoch': epoch + 1,
            'val_auc': val_m['auc'],
            'sns_real_acc': sns_m['real_acc'],
            'composite': composite,
        })

        if composite > best_score:
            best_score = composite; early_stop = 0
            torch.save({'epoch': epoch+1, 'model_state_dict': model.state_dict(),
                        'best_composite': best_score, 'val_auc': val_m['auc'],
                        'sns_real_acc': sns_m['real_acc']}, SAVE_PATH)
            print(f'  ✅ best saved (composite={best_score:.4f})')
        else:
            early_stop += 1
            if early_stop >= patience:
                print(f'  ⛔ Early stop')
                break

    with open(RESULTS, 'w') as f:
        json.dump({'best_composite': best_score, 'history': history,
                   'config': {'base': V20_CKPT, 'lr': 1e-5, 'sns_oversample': SNS_OVERSAMPLE}},
                  f, indent=2, ensure_ascii=False)

    print(f'\n✅ v25 완료. best composite={best_score:.4f}')


if __name__ == '__main__':
    train()
