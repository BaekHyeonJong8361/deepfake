# CLAUDE.md — 딥페이크 탐지 프로젝트 작업 규칙

이 파일은 Claude가 이 프로젝트에서 작업할 때 따르는 규칙입니다.
보편 원칙 + 본 프로젝트의 v5~v7 작업에서 실제로 효과 본 규칙을 정리.

---

## 1. 핵심 원칙 (Core Principles)

- **Simplicity First**: 변경은 최소·최단으로. 영향 범위가 작을수록 좋음.
- **No Laziness**: 임시 우회 금지. 근본 원인 찾고 정공법으로.
- **Minimal Impact**: 필요한 부분만 건드림. 새 버그 도입 금지.
- **Verification Before Done**: "테스트 통과 / 로그 확인 / 결과물 검증"되기 전엔 task 완료 표시 X.

---

## 2. 실험 설계 — 변수 통제 (v5~v7에서 검증됨)

- **한 번에 한 변수만 바꾼다**. v6은 augmentation만, v7은 데이터만 바꿔서 효과 분리 측정 성공.
- 비교 baseline은 항상 직전 버전 (v_n vs v_{n-1}).
- augmentation·loss·모델·하이퍼파라미터를 동시에 바꾸지 않는다. 그러면 무엇이 효과를 냈는지 모름.
- 새 실험 시작 전: "이번에 바꾸는 변수는 무엇이고, 통제되는 변수는 무엇인가?"를 명시.

---

## 3. GPU 및 학과서버 사용

- **GPU는 1번만 사용** (학과서버 공유, 다른 조 사용 가능성).
  - 코드: `device = torch.device("cuda:1")` 명시
- 학습은 **항상 `nohup` 백그라운드** 실행.
  - 명령: `nohup python -u train_*.py > train_*.log 2>&1 &`
  - SSH 끊겨도 계속 돌도록 (PPID=1, TT=? 확인)
- 학습 중 GPU 점유 확인: `nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader`
- 절대 두 학습 동시 실행 X (메모리 충돌)

---

## 4. cuDNN 워크어라운드 (학과서버 환경)

- 시스템 cuDNN과 PyTorch 번들 cuDNN 심볼 충돌으로 RuntimeError 발생.
- **모든 학습 스크립트의 `set_seed()`에 `torch.backends.cudnn.enabled = False` 필수.**
- 학습 속도 약간 저하되지만 정확도엔 영향 없음. v5~v7 모두 동일 처리.

---

## 5. 코드 수정 시 원칙

- **사용자(백현종)의 기존 학습/전처리 코드를 직접 수정하지 않는다**. 새 파일(`train_v6.py`, `train_v7.py` 등)로 분리.
  - 이유: malware 리마인더 시스템이 사용자 ML 코드 수정에 보수적으로 작동
  - 또한 v_n 코드를 보존해야 재현/비교 가능
- 공통 유틸은 import로 재사용 (예: `from train_v6 import BinaryFocalLoss`).
- `dataset.py` 등 코어 모듈은 신중히 수정. 새 augmentation은 학습 스크립트 안에 정의.

---

## 6. 산출물 저장 컨벤션

- 모델: `saved_models/xception_best_v{N}_{tag}.pth`
- 결과 JSON: `saved_models/v{N}_results.json` (test 메트릭 + 직전 버전 비교)
- 학습 로그: `train_v{N}.log`
- Split CSV: `splits_v{N}/` (재현성 위해 버전별 분리)
- 리포트: `notion_v{N}_report.md`

---

## 7. 자율적 버그 처리

- 명확한 에러(Stack trace, 로그)면 사용자에게 묻지 말고 즉시 수정 시도.
- 단, 다음은 반드시 사전 확인:
  - 학습 프로세스 강제 종료 (kill)
  - 디스크 다량 삭제
  - 기존 모델/CSV 덮어쓰기
- v6/v7 학습 시 cuDNN 에러 자율 해결한 패턴이 좋은 예시.

---

## 8. Elegance 체크 (균형)

- 비자명한 변경 시: "더 간단한 방법은 없나?" 한 번 자문
- 임시방편 느낌이 들면: "지금 아는 모든 걸로 다시 짠다면?" 다시 생각
- 단순·자명한 수정엔 적용 X (오버엔지니어링 방지)

---

## 9. 결과 보고 시 정직성

- 좋은 결과는 좋게, 나쁜 결과는 나쁘게 말한다.
- v6 DFDC AUC 0.6860을 "낮다"고 정직하게 말한 게 v7 전략 결정에 도움 됐음.
- 수치 비교 시 단위(AUC vs Accuracy)와 도메인(in/cross)을 항상 명시.

---

## 10. 메모리 시스템 활용

- `~/.claude/projects/-home-t26106-deepfake/memory/`의 `MEMORY.md` 참조.
- 사용자 정체성, 팀 구성, 프로젝트 컨텍스트는 이미 저장됨 (백현종, AI/딥러닝, 권민재/김지혁).
- 새로 알게 된 사용자 선호·교정 사항은 메모리에 추가 (단 ML 코드 패턴은 코드에서 직접 확인 가능하므로 저장 X).
