# SNS 화면녹화 딥페이크 탐지 (RGB + 주파수 듀얼 브랜치)

틱톡·인스타그램 등 **SNS 화면녹화 환경**의 딥페이크/AI 생성 영상을 탐지하는 딥러닝 시스템입니다.
공개 벤치마크가 아닌 **실사용 도메인(압축·블러된 화면녹화)** 을 타깃으로,
자체 수집한 OOD 평가셋에서 **AUC 0.41 → 0.88** 로 끌어올렸습니다.

> 15주간의 모델 진화(v6~v36), 실패한 접근, 앙상블 최적화 과정을 모두 기록한 연구 프로젝트입니다.
> 팀: 백현종(AI/딥러닝), 권민재(웹), 김지혁(아키텍처)

---

## 핵심 성과

| 지표 | 기존 탐지기 | 본 시스템 (SHEN 모드) |
|------|------------|----------------------|
| **SNS OOD AUC** | 0.411 (동전 던지기) | **0.884** |
| 정확도 | – | 82.1% |
| 탐지율(Recall) | – | 81.4% |
| 오탐률(FPR) | – | 17.1% |

- **핵심 난제**: SNS 화면녹화 → 압축·블러 → 기존 딥페이크 탐지기가 OOD 도메인에서 붕괴(AUC 0.41)
- **해결 전략**: RGB(시각적 이상) + 주파수(FFT 합성 흔적) **듀얼 브랜치** + SNS 특화 학습 + 앙상블
- **정직한 한계**: SNS 특화 trade-off로 WildDF/DFDC는 0.58대, AUC 0.9 벽은 미돌파

---

## 아키텍처: F3NetLiteV20

```
입력 (299×299 RGB)
   ├─► RGB Branch  (Xception backbone, 2048-d)        ← 시각적 이상
   └─► FAD: FFT → num_bands=3 (저/중/고주파)
         └─► Freq Branch (Xception backbone, 2048-d)   ← 숨겨진 합성 흔적
   │
   └─► SEBlockFusion (BatchNorm1d + Channel Attention, 4096-d)
         └─► fc_head Linear(4096→1) → sigmoid → 딥페이크 확률
파라미터: ~44M
```

**2-Stage Cascade**: 고전 딥페이크(FF++/DFDC)용 앙상블(RYZE) + SNS 특화 앙상블(SHEN) 모드 분리.
최종 SHEN 모드 = `v24 + v25 + v29` 가중 앙상블(0.2 / 0.5 / 0.3), 임계값 0.081(Youden's J 최적).

자세한 모델 이력·실패 분석·앙상블 스윕은 [project_summary.md](project_summary.md) 참고.

---

## 디렉토리 구조

```
.
├── train_v*.py              # 모델 진화 이력 (v6~v38) — 버전별 학습 스크립트
├── dataset.py               # 데이터 로더 / 샘플러
├── preprocess*.py           # 얼굴 검출(Mediapipe/MTCNN) + 크롭 전처리
├── eval_*.py                # OOD 평가 / TTA / 앙상블 스윕
├── inference_server_v10.py  # FastAPI 추론 서버 (포트 60006, 4개 모드)
├── src/                     # 프론트엔드 (React + TypeScript + Vite)
├── results/, eval_cascade_v3/   # 실험 결과
├── submission/              # 제출본 (문서 · 소스코드 · 실험결과 · 시각화 · XAI)
│   ├── 문서/                #   설계서 · 완료보고서 · 실험이력 · 발표자료
│   ├── 소스코드/
│   ├── 중간산출물/          #   학습로그 · 실험결과 JSON · splits CSV
│   └── 최종산출물/          #   시각화(ROC/CM) · XAI(GradCAM 100장 · 샘플 25종)
├── requirements.txt
└── project_summary.md       # 프로젝트 종합 보고서 (전체 실험 이력)
```

> **저장소에 포함되지 않은 것** (용량·라이선스): 학습된 모델 가중치(`*.pth`),
> 원본/전처리 데이터셋(FaceForensics++, DFDC, DF40, 전처리 얼굴 프레임 등).
> 코드와 데이터 분할(splits CSV)·실험 로그·결과는 모두 포함되어 있어 재학습으로 재현 가능합니다.

---

## 재현 방법

### 1. 환경 설정

- **검증 환경**: Python 3.9, NVIDIA RTX 4500 Ada, CUDA 12.8, cuDNN 9.10.2

```bash
python -m venv .venv && source .venv/bin/activate

# PyTorch는 CUDA 12.8 빌드를 별도 인덱스에서 설치
pip install torch==2.8.0 torchvision==0.23.0 \
    --index-url https://download.pytorch.org/whl/cu128

# 나머지 의존성
pip install -r requirements.txt
```

> ⚠️ **cuDNN 워크어라운드**: 일부 서버 환경에서 시스템 cuDNN과 PyTorch 번들 cuDNN
> 심볼이 충돌하면, 학습 스크립트 `set_seed()`에 `torch.backends.cudnn.enabled = False`
> 를 설정하세요(속도 약간 저하, 정확도 영향 없음).

### 2. 데이터 준비

1. 데이터셋(FF++, DFDC, DF40, 자체 SNS 수집본)을 준비하고 `preprocess.py`로 얼굴 크롭 추출
   → 결과를 `preprocessed_faces/` 구조로 저장
2. 데이터 분할 CSV는 `submission/중간산출물/splits/` 에 포함됨
   (`splits_v24/`, `splits_v35/`, `dataset_train/val/test.csv` 등)
   - 학습 스크립트는 루트 기준 경로(예: `splits_v24/train_v24.csv`)를 참조하므로,
     해당 CSV를 루트로 복사하거나 스크립트 내 경로를 맞춰주세요.

### 3. 학습 / 평가 / 추론

```bash
# 학습 (예: v24 — 단일 OOD 최고 성능)  ※ GPU 인덱스는 환경에 맞게
nohup python -u train_v24.py > train_v24.log 2>&1 &

# OOD 평가 (140개 평가셋, first5 + mean 프로토콜)
python eval_all_models_ood_v2.py

# 추론 서버 (FastAPI, 포트 60006)
python inference_server_v10.py
```

### 4. 프론트엔드 (데모 UI)

```bash
npm install
npm run dev
```

---

## 주요 교훈 (실패에서 배운 것)

- **in-distribution 완벽 ≠ OOD 성능**: val AUC 0.9998이어도 OOD에서 0.586으로 붕괴(v33)
- **대형 frozen 백본의 한계**: DINOv2-large도 frozen이면 도메인 갭 극복 실패(v34/v35)
- **SOTA 단일 백본 < 도메인 특화 듀얼 브랜치**: EfficientNet/DINOv2 단일 백본이
  RGB+주파수 듀얼 브랜치를 못 넘음
- **외부 pretrained의 무용**: DF40 CLIP-Large가 SNS OOD에서 AUC 0.238

---

## 라이선스 / 데이터 사용

학습에 사용한 공개 데이터셋(FaceForensics++, DFDC, DF40 등)은 각자의 라이선스를 따릅니다.
자체 수집 SNS 데이터는 연구 목적으로만 사용되었습니다.
