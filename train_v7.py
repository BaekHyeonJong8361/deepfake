"""
train_v7.py
===========
v7: FF++ + DFDC 통합 학습 (데이터 다양성 효과 측정)
====================================================

v6 대비 변경점 (변수 통제):
  - 학습 데이터: train_ff_only.csv → train_combined.csv (FF++ + DFDC)
  - Sampler: 2-way (REAL/FAKE) → 4-way (REAL/FAKE × FF/DFDC)
  - 그 외 augmentation, 모델, loss, 하이퍼파라미터 모두 v6과 동일

목적:
  - v6 vs v7 비교를 통해 "DFDC를 학습에 넣은 효과"를 통제 실험으로 측정
  - cross-dataset gap 0.2766이 얼마나 좁혀지는가 측정

NOTE: 삭제된 원본을 train_v7.cpython-39.pyc 에서 복원. make_4way_balanced_sampler 등
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
from torch.utils.data import DataLoader, WeightedRandomSampler
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, balanced_accuracy_score
import timm
from dataset import FastDeepfakeDataset
from train_v6 import (
    BinaryFocalLoss, seed_worker, safe_auc, evaluate, warmup_cosine,
    get_v6_train_transforms, get_eval_transforms,
)


def make_4way_balanced_sampler(csv_path):
    """
    학습 시 매 batch에서 (FF-REAL, FF-FAKE, DFDC-REAL, DFDC-FAKE)
    4그룹이 균등하게 뽑히도록 가중치 부여.

    원리:
      - 각 (dataset, label) 조합별 샘플 수 카운트
      - sample_weight = 1 / count_in_group
      - 그룹 내 모든 샘플은 동일 가중치, 그룹 간은 크기 반비례
    """
    df = pd.read_csv(csv_path)
    if 'dataset' not in df.columns:
        raise ValueError(f"{csv_path}에 'dataset' 컬럼이 없습니다 (combined CSV 필요)")
    labels = df['label'].astype(int).values
    datasets = df['dataset'].astype(str).values
    group_keys = np.array([f"{d}_{l}" for d, l in zip(datasets, labels)])
    unique_groups, group_counts = np.unique(group_keys, return_counts=True)
    print('  [4-way Sampler] 그룹별 샘플 수:')
    for g, c in zip(unique_groups, group_counts):
        print(f"    {g}: {c:,}")
    group_weight_map = {g: 1.0 / c for g, c in zip(unique_groups, group_counts)}
    sample_weights = np.array([group_weight_map[g] for g in group_keys], dtype=np.float64)
    return WeightedRandomSampler(
        weights=torch.as_tensor(sample_weights, dtype=torch.double),
        num_samples=len(sample_weights),
        replacement=True,
    )


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.enabled = False


def train():
    # 원본 train() 루프는 .pyc 디컴파일이 truncated 되어 복원 불가. splits_v6 데이터도 삭제됨.
    raise NotImplementedError(
        "train_v7.train() 본문은 .pyc 바이트코드에서 복원 불가 (truncated decompilation)."
    )


if __name__ == '__main__':
    train()
