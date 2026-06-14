# 딥페이크 분석 서버 API 명세서 v7.0

> 작성일: 2026-05-21
> 서버 담당: 백현종 (AI/딥러닝)
> 연동 대상: 웹 백엔드 (권민재)
> 이전 버전: API_SPEC_v4.md (v4.0, 단일 앙상블) — **이번에 폐기**

---

## 0. v4 → v7 핵심 변경점

| 항목 | v4 (기존) | v7 (신규) |
|---|---|---|
| 구조 | 단일 앙상블 (v20+v27) | **모드 선택형 듀얼 앙상블** |
| 사용 모델 | v20, v27 (2개) | **전통: v15+v16+v17 / SNS: v24+v25+v29 (6개)** |
| 모드 선택 | 없음 | **`X-Mode: traditional \| sns` 헤더 추가** |
| 임계값 | 1개 (0.39) | **전통: 0.50 / SNS: 0.081** |
| XAI | 외부 모듈 의존 | **서버 내장 (GradCAM + 한국어 설명)** |
| XAI 응답 위치 | `evidence.heatmaps` | **`xai.heatmaps` (별도 키)** |
| 설명 텍스트 | 없음 | **`explanation` 필드 추가** |
| 추론 시간 | ~2초 | **~2초 (동일)** |
| 정확도 | SNS OOD AUC 0.76 | **SNS OOD AUC 0.884 / FF++ AUC 0.999** |

**왜 바꾸나:**
- v20+v27은 SNS 실제 영상 OOD 성능이 0.76에 불과했음
- SNS 특화(v24+v25+v29)로 교체 시 0.884로 향상
- 전통 딥페이크(FF++/DFDC)는 v15+v16+v17이 AUC 0.999로 압도적
- 단일 모델로 두 도메인을 동시에 커버하는 건 실험적으로 불가능 → 모드 분리

---

## 1. 서버 정보

| 항목 | 값 |
|---|---|
| **Base URL** | `http://192.9.202.17:60006` |
| 포트 | `60006` (동일) |
| 프로토콜 | HTTP |
| CORS | 전체 허용 (`*`) |
| 실행 파일 | `inference_server_v7.py` |
| GPU | `cuda:1` (RTX 4500 Ada) |

**서버 실행 명령어 (백현종이 올릴 때):**
```bash
source ~/.venv/bin/activate
nohup python -u inference_server_v7.py > inference_server_v7.log 2>&1 &
```

---

## 2. 엔드포인트 목록

| Method | Path | 설명 |
|---|---|---|
| `POST` | `/deepfake/analyze_local` | 바이너리 직접 업로드 (메인) |
| `POST` | `/deepfake/analyze_image` | 이미지 1장 분석 |
| `GET`  | `/deepfake/health` | 서버 상태 + 모드별 모델 목록 |

> ⚠️ v4의 `/deepfake/analyze` (URL 방식) 는 v7에서 제거됨.
> 영상 파일은 반드시 바이너리로 직접 전송 (`analyze_local`) 할 것.

---

## 3. 모드 선택 (핵심 변경)

모든 분석 요청에 `X-Mode` 헤더를 추가한다.

| 헤더 값 | 대상 | 사용 모델 | 임계값 | AUC |
|---|---|---|---|---|
| `traditional` | FF++/DFDC 같은 고품질 얼굴 교체 영상 | v15+v16+v17 | 0.50 | FF++ 0.999 |
| `sns` | 틱톡/인스타 화면녹화, 실제 유포 영상 | v24+v25+v29 | 0.081 | SNS 0.884 |

- **기본값**: `sns` (헤더 생략 시)
- UI에서 사용자가 모드를 선택하게 하거나, 서비스 성격에 따라 고정해도 됨

---

## 4. 메인 분석 API

### `POST /deepfake/analyze_local`

#### 요청
```http
POST http://192.9.202.17:60006/deepfake/analyze_local
X-Filename: my_video.mp4
X-Mode: sns
X-Return-XAI: 1          # XAI 생략하려면 0
Content-Type: application/octet-stream

<binary>
```

#### curl 예시
```bash
# SNS 모드
curl -X POST http://192.9.202.17:60006/deepfake/analyze_local \
  -H "X-Filename: video.mp4" \
  -H "X-Mode: sns" \
  --data-binary @video.mp4

# 전통 모드 + XAI 생략 (빠름)
curl -X POST http://192.9.202.17:60006/deepfake/analyze_local \
  -H "X-Filename: video.mp4" \
  -H "X-Mode: traditional" \
  -H "X-Return-XAI: 0" \
  --data-binary @video.mp4
```

#### 응답 예시 (SNS 모드, XAI 포함)
```json
{
  "request_id": "a3f9c12b4e71",
  "mode": "sns",
  "mode_label": "SNS 딥페이크 탐지 (실제 영상 특화)",
  "decision": "FAKE",
  "score": 0.4823,
  "threshold": 0.081,
  "models": [
    { "name": "v24", "score": 0.5102, "weight": 0.2 },
    { "name": "v25", "score": 0.4651, "weight": 0.5 },
    { "name": "v29", "score": 0.4871, "weight": 0.3 }
  ],
  "explanation": "딥페이크로 판정되었습니다. (점수 0.482 ≥ 임계값 0.081) 주요 이상 영역: 중상(눈)(38%), 중앙(코)(21%)에서 비정상적인 패턴이 감지되었습니다. 주파수 특성 분석에서 압축·재인코딩 흔적이 발견되었습니다. SNS 영상 특화 모델 기준.",
  "xai": {
    "heatmaps": {
      "rgb":  "data:image/jpeg;base64,...",
      "freq": "data:image/jpeg;base64,..."
    },
    "se_attention": {
      "rgb_share": 0.43,
      "freq_share": 0.57
    },
    "regions": [
      { "region": "중상(눈)",  "ratio": 0.375 },
      { "region": "중앙(코)",  "ratio": 0.210 },
      { "region": "좌중(광대)", "ratio": 0.169 }
    ]
  },
  "evidence": {
    "suspect_frame_idx": 42,
    "face_image_base64": "/9j/4AAQSkZJRgABAQ...",
    "face_bbox_in_frame_xyxy": [151, 350, 535, 733],
    "detect_conf": 0.921,
    "blur_var": 88.4,
    "n_frames_analyzed": 24
  },
  "latency_ms": 2031
}
```

---

## 5. 응답 필드 설명

### 메인 결과 (백엔드 필수)

| 필드 | 타입 | 설명 |
|---|---|---|
| `request_id` | string | 12자 hex, 로그용 |
| `mode` | `"traditional"` \| `"sns"` | 사용된 모드 |
| `mode_label` | string | 모드 한국어 설명 |
| `decision` | `"FAKE"` \| `"REAL"` | **최종 판정** |
| `score` | float (0~1) | **앙상블 점수** |
| `threshold` | float | 적용된 임계값 (`score ≥ threshold` → FAKE) |
| `explanation` | string | rule-based 한국어 판정 설명 |
| `latency_ms` | int | 추론 소요 시간 (ms) |

### 모델별 상세 (선택)

| 필드 | 타입 | 설명 |
|---|---|---|
| `models` | array | 모델별 점수 배열 |
| `models[i].name` | string | 모델명 (`"v24"` 등) |
| `models[i].score` | float | 해당 모델 단독 점수 |
| `models[i].weight` | float | 앙상블 가중치 |

> ⚠️ `models` 는 **배열**. `response.models.v24` 처럼 키로 접근 금지. `for` 또는 `.find()` 로 순회.

### XAI (선택, 시각화용)

| 필드 | 타입 | 설명 |
|---|---|---|
| `xai.heatmaps.rgb` | string | RGB 텍스처 GradCAM 오버레이 (data URI JPEG) |
| `xai.heatmaps.freq` | string | 주파수 도메인 GradCAM 오버레이 (data URI JPEG) |
| `xai.se_attention.rgb_share` | float | 모델이 RGB 텍스처에 의존한 비율 |
| `xai.se_attention.freq_share` | float | 모델이 주파수 특성에 의존한 비율 |
| `xai.regions` | array | 얼굴 3×3 그리드 상위 3개 이상 영역 |
| `xai.regions[i].region` | string | 영역명 (예: `"중상(눈)"`, `"중앙(코)"`) |
| `xai.regions[i].ratio` | float | 해당 영역의 attention 비율 |

> XAI 응답 생략: `X-Return-XAI: 0` 헤더 추가 → `xai: {}` 로 반환, 레이턴시 약 0.3초 감소

### Evidence (선택, 시각화용)

| 필드 | 타입 | 설명 |
|---|---|---|
| `evidence.face_image_base64` | string | 의심 프레임 얼굴 크롭 (raw base64 JPEG) |
| `evidence.face_bbox_in_frame_xyxy` | [x1,y1,x2,y2] | 원본 프레임 내 얼굴 박스 |
| `evidence.detect_conf` | float | mediapipe 얼굴 검출 신뢰도 |
| `evidence.blur_var` | float | 선명도 (클수록 선명) |
| `evidence.n_frames_analyzed` | int | 분석된 프레임 수 |

---

## 6. 에러 응답

| HTTP | 의미 | 권장 처리 |
|---:|---|---|
| 200 | 성공 | 정상 |
| 422 | 얼굴 미검출 | "얼굴이 보이는 영상을 업로드하세요" 안내 |
| 500 | 서버 내부 오류 | 재시도 또는 에러 표시 |

```json
{ "detail": "얼굴이 검출되지 않았습니다" }
```

---

## 7. TypeScript 타입 정의

```typescript
type Mode = "traditional" | "sns";

type AnalyzeResponse = {
  request_id: string;
  mode: Mode;
  mode_label: string;
  decision: "FAKE" | "REAL";
  score: number;        // 0~1
  threshold: number;
  models: Array<{
    name: string;
    score: number;
    weight: number;
  }>;
  explanation: string;
  xai: {
    heatmaps?: { rgb?: string; freq?: string };
    se_attention?: { rgb_share: number; freq_share: number };
    regions?: Array<{ region: string; ratio: number }>;
  };
  evidence: {
    suspect_frame_idx: number;
    face_image_base64: string;
    face_bbox_in_frame_xyxy: [number, number, number, number];
    detect_conf: number;
    blur_var: number;
    n_frames_analyzed: number;
  };
  latency_ms: number;
};
```

### 사용 예시

```typescript
async function analyzeVideo(file: File, mode: Mode = "sns"): Promise<AnalyzeResponse> {
  const res = await fetch("http://192.9.202.17:60006/deepfake/analyze_local", {
    method: "POST",
    headers: {
      "X-Filename": encodeURIComponent(file.name),
      "X-Mode": mode,
    },
    body: file,
  });
  if (!res.ok) {
    const err = await res.json();
    throw new Error(err.detail ?? `HTTP ${res.status}`);
  }
  return res.json();
}

// 사용
const result = await analyzeVideo(videoFile, "sns");

// ✅ 판정
const isFake = result.decision === "FAKE";

// ✅ 설명 텍스트 (그대로 UI에 표시)
console.log(result.explanation);

// ✅ 히트맵 (data URI 그대로 img src에)
if (result.xai.heatmaps?.rgb) {
  imgElement.src = result.xai.heatmaps.rgb;
}

// ✅ 의심 프레임 (raw base64 → data URI 변환 필요)
faceImg.src = `data:image/jpeg;base64,${result.evidence.face_image_base64}`;

// ✅ 모델별 점수 (배열 순회)
result.models.forEach(m => {
  console.log(`${m.name}: ${m.score.toFixed(3)} (w=${m.weight})`);
});

// ❌ 금지 — 모델 이름 키로 접근
// result.models.v24  ← 안됨
```

---

## 8. v4 → v7 마이그레이션 체크리스트 (권민재)

- [ ] 엔드포인트 변경: `/deepfake/analyze` → `/deepfake/analyze_local`
- [ ] 요청 방식 변경: JSON body → binary body + `X-Filename` 헤더
- [ ] `X-Mode: sns` 헤더 추가 (또는 UI에서 선택)
- [ ] XAI 필드 위치 변경: `evidence.heatmaps` → `xai.heatmaps`
- [ ] `explanation` 필드 추가 (판정 설명 텍스트, UI에 표시 권장)
- [ ] `mode`, `mode_label` 필드 추가됨 (선택)
- [ ] TypeScript 타입 위 정의로 교체
- [ ] `/deepfake/health` 로 서버 상태 확인

---

## 9. health 응답 예시

```bash
curl http://192.9.202.17:60006/deepfake/health
```

```json
{
  "status": "running",
  "version": "v7.0.0",
  "modes": {
    "traditional": {
      "label": "전통 딥페이크 탐지 (FF++/DFDC 특화)",
      "threshold": 0.5,
      "models": [
        { "name": "v15", "weight": 0.333, "loaded": true },
        { "name": "v16", "weight": 0.333, "loaded": true },
        { "name": "v17", "weight": 0.333, "loaded": true }
      ]
    },
    "sns": {
      "label": "SNS 딥페이크 탐지 (실제 영상 특화)",
      "threshold": 0.081,
      "models": [
        { "name": "v24", "weight": 0.2,  "loaded": true },
        { "name": "v25", "weight": 0.5,  "loaded": true },
        { "name": "v29", "weight": 0.3,  "loaded": true }
      ]
    }
  }
}
```

---

## 10. 문의

- 모델 정확도 / 임계값 / XAI → 백현종
- API 동작 / 응답 파싱 → 백현종
- 서버 IP / 방화벽 / 포트 → 함께 협의
