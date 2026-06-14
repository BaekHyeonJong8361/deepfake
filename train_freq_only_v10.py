"""
train_freq_only_v10.py
======================
v10: Frequency-Only Xception (단일 branch, FAD 기반)
====================================================

목적 (Plan B - Triple Ensemble의 한 축):
  - 주파수 정보만 사용하는 독립 모델 학습
  - 최종 앙상블 = v7 (RGB-only) + v10 (Freq-only) + v8/v9 (Joint RGB+Freq)
  - 사용자 초기 아이디어: "RGB 모델 + Freq 모델 동시 평가 → 더블체크" 구현

v8/v9 대비 변경점 (변수 통제):
  - 모델: dual-branch (RGB+Freq) → single-branch (Freq only)
  - 입력: RGB 이미지 + FAD(9ch) → FAD(9ch) 만
  - Fusion: SE-block fusion 제거 (단일 branch라 불필요)
  - Regularization: v9와 동일 (MixUp α=0.2 + LabelSmoothing 0.05 + Dropout 0.3)
  - 데이터/optimizer/sampler/lr/wd/aug: v8/v9와 동일

비고:
  - Independent ensemble을 위해 RGB 정보를 보지 않는 게 핵심
  - FAD에서 base_filters는 고정, add_on_filters만 학습
  - timm xception(in_chans=9): 첫 conv 가중치 평균/복제로 초기화
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

# 변수 통제 보장을 위해 v6/v7/v8/v9에서 재사용
from train_v6 import (
    BinaryFocalLoss,
    seed_worker,
    safe_auc,
    warmup_cosine,
    get_v6_train_transforms,
    get_eval_transforms,
)
from train_v7 import make_4way_balanced_sampler
from train_v8 import FAD                       # 동일 FAD 모듈 (재사용)
from train_v9 import (
    FocalLossWithLabelSmoothing,
    mixup_batch,
    mixup_loss,
    evaluate_v9 as evaluate_model,            # 평가 루틴은 동일
)


# =========================================================
# v10: Frequency-Only Xception
# =========================================================
class FreqOnlyXception(nn.Module):
    """
    FAD → Xception(in_chans=9) → BN+Dropout+Linear → logit

    Args:
        img_size: 입력 이미지 크기 (default 299)
        num_bands: FAD band 수 (default 3 = low/mid/high)
        pretrained: ImageNet pretrained 사용
        dropout: classifier head dropout (v9와 동일 0.3)
    """
    def __init__(self, img_size=299, num_bands=3, pretrained=True, dropout=0.3):
        super().__init__()
        self.fad = FAD(img_size=img_size, num_bands=num_bands)
        self.backbone = timm.create_model(
            "xception",
            pretrained=pretrained,
            num_classes=0,
            global_pool="avg",
            in_chans=3 * num_bands,
        )
        self.bn = nn.BatchNorm1d(2048)
        self.dropout = nn.Dropout(p=dropout)
        self.classifier = nn.Linear(2048, 1)

    def forward(self, x):
        # x: (B, 3, H, W) - 정규화된 RGB (FAD 입력으로만 사용)
        freq_input = self.fad(x)              # (B, 9, H, W)
        feat = self.backbone(freq_input)      # (B, 2048)
        feat = self.bn(feat)
        feat = self.dropout(feat)
        logit = self.classifier(feat).squeeze(1)
        return logit


# =========================================================
# 유틸
# =========================================================
def set_seed_v10(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # venv (torch 2.8 + cuDNN 9) → cuDNN ON 가능
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True


# =========================================================
# Main
# =========================================================
def train():
    set_seed_v10(42)

    if torch.cuda.is_available() and torch.cuda.device_count() > 1:
        device = torch.device("cuda:1")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pin_memory = device.type == "cuda"

    print(f"[v10] Frequency-Only Xception | device={device}")
    print(f"      torch={torch.__version__} | cudnn enabled={torch.backends.cudnn.enabled} | cudnn ver={torch.backends.cudnn.version()}")

    # 하이퍼파라미터 (v8/v9와 동일)
    batch_size = 16
    num_epochs = 30
    base_lr = 1e-4
    weight_decay = 1e-4
    patience = 6
    warmup_epochs = 2
    num_workers = min(8, os.cpu_count() or 4)

    # v9와 동일한 regularization 스택
    mixup_alpha = 0.2
    label_smoothing = 0.05
    dropout = 0.3

    save_dir = "saved_models"
    os.makedirs(save_dir, exist_ok=True)
    best_path = os.path.join(save_dir, "freq_only_best_v10.pth")
    last_path = os.path.join(save_dir, "freq_only_last_v10.pth")

    splits_dir = "splits_v6"
    train_csv = os.path.join(splits_dir, "train_combined.csv")
    val_csv = os.path.join(splits_dir, "val_combined.csv")
    test_ff_csv = os.path.join(splits_dir, "test_ff_only.csv")
    test_dfdc_csv = os.path.join(splits_dir, "test_dfdc_only.csv")

    print("📁 데이터 로딩...")
    train_ds = FastDeepfakeDataset(train_csv, transform=get_v6_train_transforms(), label_dtype=torch.float32)
    val_ds = FastDeepfakeDataset(val_csv, transform=get_eval_transforms(), label_dtype=torch.float32)
    test_ff_ds = FastDeepfakeDataset(test_ff_csv, transform=get_eval_transforms(), label_dtype=torch.float32)
    test_dfdc_ds = FastDeepfakeDataset(test_dfdc_csv, transform=get_eval_transforms(), label_dtype=torch.float32)

    print(f"  train: {len(train_ds):,} | val: {len(val_ds):,} | test_ff: {len(test_ff_ds):,} | test_dfdc: {len(test_dfdc_ds):,}")

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
    test_ff_loader = DataLoader(test_ff_ds, shuffle=False, **common_loader)
    test_dfdc_loader = DataLoader(test_dfdc_ds, shuffle=False, **common_loader)

    print("\n🤖 FreqOnlyXception 로딩 (FAD + Xception in_chans=9, dropout=0.3)...")
    model = FreqOnlyXception(img_size=299, num_bands=3, pretrained=True, dropout=dropout).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  파라미터: {n_params/1e6:.2f}M (참고: v8 F3NetLite ≈ 41M)")

    criterion_train = FocalLossWithLabelSmoothing(gamma=2.0, alpha=0.25, smoothing=label_smoothing)
    criterion_eval = BinaryFocalLoss(gamma=2.0, alpha=0.25)
    optimizer = optim.AdamW(model.parameters(), lr=base_lr, weight_decay=weight_decay)

    best_score = -1.0
    early_stop = 0
    start_epoch = 0

    # Resume
    if os.path.exists(last_path):
        print(f"\n🔄 resume: {last_path} 발견 → 이어서 학습")
        # torch 2.6+ weights_only 기본값 True 이슈 회피 (자체 학습 산출물이라 안전)
        ckpt = torch.load(last_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        start_epoch = ckpt["epoch"]
        best_score = ckpt.get("best_score", -1.0)
        early_stop = ckpt.get("early_stop", 0)
        print(f"  복원 완료: start_epoch={start_epoch+1} | best_score={best_score:.4f} | early_stop={early_stop}")
    else:
        print(f"\n🆕 새 학습 시작 (resume 체크포인트 없음)")

    print(f"\n{'='*60}")
    print(f"[v10] lr={base_lr}, wd={weight_decay}, bs={batch_size}, epochs={num_epochs}")
    print(f"[v10] Loss=Focal(γ=2.0,α=0.25) + LabelSmoothing={label_smoothing}")
    print(f"[v10] MixUp α={mixup_alpha} | Classifier Dropout={dropout}")
    print(f"[v10] Aug=v6 set (Flip+CJ+RandomResize+Blur+Noise+JPEG)")
    print(f"[v10] Sampler=4-way (FF/DFDC × REAL/FAKE)")
    print(f"[v10] 모델: FAD → Xception(in_chans=9) → BN+Dropout+Linear")
    print(f"[v10] 변수 통제: 데이터/aug/sampler/optimizer/reg 모두 v9와 동일, 모델만 freq-only")
    print(f"[v10] 환경: venv (torch {torch.__version__} + cuDNN {torch.backends.cudnn.version()})")
    print(f"{'='*60}\n")

    for epoch in range(start_epoch, num_epochs):
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

        print(f"[Epoch {epoch+1}/{num_epochs}] lr={cur_lr:.2e}")
        print(f"  Train Loss: {avg_train:.4f}")
        print(f"  Val   Loss: {val_m['loss']:.4f} | Acc: {val_m['acc']*100:.2f}% | F1: {val_m['f1']:.4f} | BalAcc: {val_m['bal_acc']:.4f} | AUC: {val_m['auc']:.4f}")

        score = val_m["auc"]
        if score > best_score:
            print(f"  ⭐ best 갱신! ({best_score:.4f} → {score:.4f})")
            best_score = score
            early_stop = 0
            torch.save({
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "best_val_auc": best_score,
                "val_metrics": val_m,
            }, best_path)
        else:
            early_stop += 1

        torch.save({
            "epoch": epoch + 1,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "best_score": best_score,
            "early_stop": early_stop,
            "val_metrics": val_m,
        }, last_path)

        if early_stop >= patience:
            print(f"  🛑 {patience}회 연속 개선 없음 → 조기 종료")
            break

    # =========================================================
    # 최종 평가
    # =========================================================
    print(f"\n{'='*60}\n📦 베스트 모델 로드 & 최종 평가")
    ckpt = torch.load(best_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])

    print("\n[ TEST: FF++ ]")
    m_ff = evaluate_model(model, test_ff_loader, criterion_eval, device, "TEST_FF")
    print(f"  AUC: {m_ff['auc']:.4f} | Acc: {m_ff['acc']*100:.2f}% | F1: {m_ff['f1']:.4f} | BalAcc: {m_ff['bal_acc']:.4f}")

    print("\n[ TEST: DFDC ]")
    m_dfdc = evaluate_model(model, test_dfdc_loader, criterion_eval, device, "TEST_DFDC")
    print(f"  AUC: {m_dfdc['auc']:.4f} | Acc: {m_dfdc['acc']*100:.2f}% | F1: {m_dfdc['f1']:.4f} | BalAcc: {m_dfdc['bal_acc']:.4f}")

    print(f"\n[ Cross-dataset gap ]")
    print(f"  AUC delta (FF - DFDC): {m_ff['auc'] - m_dfdc['auc']:+.4f}")

    print(f"\n[ v7 / v8 / v9 / v10 비교 ]")
    print(f"  v7  (RGB only,    Xception):   FF AUC 0.9659 | DFDC AUC 0.9878 | gap -0.0219")
    print(f"  v8  (RGB+Freq,    F3NetLite):  FF AUC 0.9737 | DFDC AUC 0.9881 | gap -0.0144")
    print(f"  v9  (RGB+Freq+Reg,F3NetLite):  (참조: v9_results.json)")
    print(f"  v10 (Freq only,   FAD+Xcep):   FF AUC {m_ff['auc']:.4f} | DFDC AUC {m_dfdc['auc']:.4f} | gap {m_ff['auc']-m_dfdc['auc']:+.4f}")

    results = {
        "version": "v10_freq_only_xception",
        "best_val_auc": float(best_score),
        "test_ff": m_ff,
        "test_dfdc": m_dfdc,
        "cross_gap_auc": float(m_ff["auc"] - m_dfdc["auc"]),
        "comparison": {
            "v7_ff_auc": 0.9659, "v7_dfdc_auc": 0.9878, "v7_gap": -0.0219,
            "v8_ff_auc": 0.9737, "v8_dfdc_auc": 0.9881, "v8_gap": -0.0144,
        },
        "model": "FAD + Xception(in_chans=9) + BN + Dropout 0.3 + Linear",
        "params_M": n_params / 1e6,
        "regularization": {
            "mixup_alpha": mixup_alpha,
            "label_smoothing": label_smoothing,
            "dropout": dropout,
        },
        "purpose": "Plan B (Triple Ensemble)의 Freq-only 축. v7(RGB) + v10(Freq) + v8/v9(Joint) 앙상블용.",
        "env": {
            "torch": torch.__version__,
            "cudnn": torch.backends.cudnn.version(),
            "cudnn_enabled": torch.backends.cudnn.enabled,
        },
    }
    with open(os.path.join(save_dir, "v10_results.json"), "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\n✅ 결과 저장: {save_dir}/v10_results.json")
    print(f"✅ 베스트 모델: {best_path}\n{'='*60}")


if __name__ == "__main__":
    train()
