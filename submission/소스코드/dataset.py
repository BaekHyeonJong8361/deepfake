import os
import io
import random
from typing import Optional

import numpy as np
import pandas as pd
from PIL import Image, UnidentifiedImageError

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
import torchvision.transforms as transforms

# =========================================================
# 커스텀 Augmentation: 랜덤 JPEG 압축 (v5 핵심 무기)
# =========================================================
class RandomJPEGCompression(object):
    """
    딥페이크 영상이 온라인(유튜브, SNS)에 업로드될 때 발생하는
    '압축 노이즈'와 실제 '조작 노이즈'를 모델이 헷갈리지 않도록 
    학습 이미지에 랜덤하게 JPEG 압축 손실을 부여합니다.
    """
    def __init__(self, min_quality=60, max_quality=95):
        self.min_quality = min_quality
        self.max_quality = max_quality

    def __call__(self, img):
        # PIL 이미지가 들어오면 랜덤 화질로 메모리 버퍼에 JPEG로 저장했다가 다시 읽음
        quality = random.randint(self.min_quality, self.max_quality)
        output_io = io.BytesIO()
        img.save(output_io, format='JPEG', quality=quality)
        output_io.seek(0)
        return Image.open(output_io)

# =========================================================
# Dataset 클래스 (에러 방어형 로더)
# =========================================================
class FastDeepfakeDataset(Dataset):
    def __init__(
        self,
        csv_path: str,
        transform=None,
        label_dtype=torch.float32,
        max_retry: int = 20,
    ):
        self.csv_path = csv_path
        self.transform = transform
        self.label_dtype = label_dtype
        self.max_retry = max_retry

        self.df = pd.read_csv(csv_path)

        required_cols = ["image_path", "label"]
        missing = [c for c in required_cols if c not in self.df.columns]
        if missing:
            raise ValueError(f"{csv_path} 에 필수 컬럼이 없습니다: {missing}")

        # 실제 파일이 존재하는 것만 유지
        exists_mask = self.df["image_path"].map(os.path.exists)
        self.df = self.df[exists_mask].reset_index(drop=True)

        if len(self.df) == 0:
            raise ValueError(f"유효한 이미지가 없습니다: {csv_path}")

        self.image_paths = self.df["image_path"].tolist()
        self.labels = self.df["label"].astype(int).tolist()

    def __len__(self):
        return len(self.image_paths)

    def _load_image(self, img_path: str):
        with Image.open(img_path) as img:
            img = img.convert("RGB")
        return img

    def __getitem__(self, idx):
        n = len(self.image_paths)

        for offset in range(self.max_retry):
            cur_idx = (idx + offset) % n
            img_path = self.image_paths[cur_idx]
            label = self.labels[cur_idx]

            try:
                img = self._load_image(img_path)

                if self.transform is not None:
                    img = self.transform(img)

                label_tensor = torch.tensor(label, dtype=self.label_dtype)
                return img, label_tensor

            except (FileNotFoundError, UnidentifiedImageError, OSError):
                continue

        raise RuntimeError(
            f"이미지를 {self.max_retry}번 시도했지만 불러오지 못했습니다. "
            f"csv={self.csv_path}, start_idx={idx}"
        )

# =========================================================
# Transform 함수들 (v5 최종 폼)
# =========================================================
def get_xception_transforms(img_size: int = 299, train: bool = True):
    """v5용: 순정 RGB + JPEG 압축 + ImageNet Normalize"""
    if train:
        return transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(brightness=0.1, contrast=0.1),
            RandomJPEGCompression(min_quality=60, max_quality=95), # 🔥 핵심 무기 투입
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
        ])
    else:
        # 검증/테스트 시에는 압축이나 증강 없이 원본 그대로 평가
        return transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
        ])