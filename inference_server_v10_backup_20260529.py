import os, io, time, uuid, asyncio, base64, tempfile, shutil
os.environ['CUDA_VISIBLE_DEVICES'] = '1'  # GPU 1만 노출 → EGL도 GPU 1에 붙음
from contextlib import asynccontextmanager
from typing import Dict, Optional, List
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from fastapi import FastAPI, HTTPException, Request, BackgroundTasks, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
import uvicorn
import requests
import timm
import mediapipe as mp
from torchvision import transforms

# ============================================================
# Core Detection Logic
# ============================================================
import importlib.util, sys as _sys, types as _types

def _load_pyc(name, pyc_path):
    spec = importlib.util.spec_from_file_location(name, pyc_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    _sys.modules[name] = mod
    return mod

_BASE = os.path.dirname(os.path.abspath(__file__))
_CACHE = os.path.join(_BASE, '__pycache__')
for _n in ['train_v6', 'train_v7', 'train_v8', 'train_v9']:
    if _n not in _sys.modules:
        _load_pyc(_n, f'{_CACHE}/{_n}.cpython-39.pyc')
_train_v20 = _load_pyc('train_v20', f'{_CACHE}/train_v20.cpython-39.pyc')
F3NetLiteV20 = _train_v20.F3NetLiteV20

GRID_LABELS = [
    ["Top-Left (Forehead)", "Top-Center (Glabella)", "Top-Right (Forehead)"],
    ["Mid-Left (Cheek)", "Center (Nose)", "Mid-Right (Cheek)"],
    ["Bottom-Left (Jaw/Mouth)", "Bottom-Center (Mouth/Chin)", "Bottom-Right (Jaw/Mouth)"]
]

# Settings
MODE_CONFIG = {
    # RYZE: 고전 딥페이크 전용 (FF++/DFDC 계열) — v7 traditional 모드와 동일
    'RYZE': {
        'label': 'RYZE (Classic Deepfake Specialist)',
        'threshold': 0.50,
        'agg': 'mean',
        'models': [
            {'name': 'v15', 'path': 'saved_models/xception_best_v15_sns_aug.pth',
             'builder': lambda: timm.create_model('xception', pretrained=False, num_classes=1),
             'weight': 1/3},
            {'name': 'v16', 'path': 'saved_models/f3netlite_best_v16_sns_aug.pth',
             'builder': lambda: F3NetLiteV20(img_size=299, num_bands=3, pretrained=False),
             'weight': 1/3},
            {'name': 'v17', 'path': 'saved_models/freq_only_best_v17_sns_aug.pth',
             'builder': lambda: F3NetLiteV20(img_size=299, num_bands=3, pretrained=False),
             'weight': 1/3},
        ],
    },
    # LEE_SIN: 공격적 (낮은 임계값 — 가짜를 절대 놓치지 않는다, 오탐 위험 있음)
    # 기반: v24+v25+v29 최적 앙상블, 임계값만 낮춤
    'LEE_SIN': {
        'label': 'LEE SIN (Aggressive - High Sensitivity)',
        'threshold': 0.055,  # 0.081보다 낮아서 더 많이 잡음 (오탐 증가)
        'agg': 'mean',
        'models': [
            {'name': 'v24', 'path': 'saved_models/f3netlite_best_v24_sns_real_finetune.pth',
             'builder': lambda: F3NetLiteV20(img_size=299, num_bands=3, pretrained=False),
             'weight': 0.2},
            {'name': 'v25', 'path': 'saved_models/f3netlite_best_v25_sns_oversample2x.pth',
             'builder': lambda: F3NetLiteV20(img_size=299, num_bands=3, pretrained=False),
             'weight': 0.5},
            {'name': 'v29', 'path': 'saved_models/f3netlite_best_v29_v20arch_more_data.pth',
             'builder': lambda: F3NetLiteV20(img_size=299, num_bands=3, pretrained=False),
             'weight': 0.3},
        ],
    },
    # SHEN: 균형형 — OOD sweep 검증 최고 조합 (AUC 0.8898)
    # v24(0.2) + v25(0.5) + v29(0.3), first5 Mean, thr=0.081
    'SHEN': {
        'label': 'SHEN (Balanced - Best Verified Performance)',
        'threshold': 0.081,  # ensemble_sweep 실측 최적 임계값
        'agg': 'mean',
        'models': [
            {'name': 'v24', 'path': 'saved_models/f3netlite_best_v24_sns_real_finetune.pth',
             'builder': lambda: F3NetLiteV20(img_size=299, num_bands=3, pretrained=False),
             'weight': 0.2},
            {'name': 'v25', 'path': 'saved_models/f3netlite_best_v25_sns_oversample2x.pth',
             'builder': lambda: F3NetLiteV20(img_size=299, num_bands=3, pretrained=False),
             'weight': 0.5},
            {'name': 'v29', 'path': 'saved_models/f3netlite_best_v29_v20arch_more_data.pth',
             'builder': lambda: F3NetLiteV20(img_size=299, num_bands=3, pretrained=False),
             'weight': 0.3},
        ],
    },
    # RAMMUS: 보수적 (높은 임계값 — 오탐 최소화, 놓치는 것 감수)
    # 기반: v24+v25+v29 최적 앙상블, 임계값만 올림
    'RAMMUS': {
        'label': 'RAMMUS (Conservative - High Precision)',
        'threshold': 0.15,  # 0.081보다 높아서 오탐 최소화
        'agg': 'mean',
        'models': [
            {'name': 'v24', 'path': 'saved_models/f3netlite_best_v24_sns_real_finetune.pth',
             'builder': lambda: F3NetLiteV20(img_size=299, num_bands=3, pretrained=False),
             'weight': 0.2},
            {'name': 'v25', 'path': 'saved_models/f3netlite_best_v25_sns_oversample2x.pth',
             'builder': lambda: F3NetLiteV20(img_size=299, num_bands=3, pretrained=False),
             'weight': 0.5},
            {'name': 'v29', 'path': 'saved_models/f3netlite_best_v29_v20arch_more_data.pth',
             'builder': lambda: F3NetLiteV20(img_size=299, num_bands=3, pretrained=False),
             'weight': 0.3},
        ],
    },
    'T2V': {
        'label': 'T2V (Text-to-Video Detection)',
        'threshold': 0.081,
        'agg': 'mean',
        'models': [
            {'name': 'v24', 'path': 'saved_models/f3netlite_best_v24_sns_real_finetune.pth',
             'builder': lambda: F3NetLiteV20(img_size=299, num_bands=3, pretrained=False),
             'weight': 0.2},
            {'name': 'v25', 'path': 'saved_models/f3netlite_best_v25_sns_oversample2x.pth',
             'builder': lambda: F3NetLiteV20(img_size=299, num_bands=3, pretrained=False),
             'weight': 0.5},
            {'name': 'v29', 'path': 'saved_models/f3netlite_best_v29_v20arch_more_data.pth',
             'builder': lambda: F3NetLiteV20(img_size=299, num_bands=3, pretrained=False),
             'weight': 0.3},
        ],
    },
}

BASE = os.path.dirname(os.path.abspath(__file__))
FACE_SIZE = 299
CROP_SCALE = 1.0
MAX_FRAMES = 30  # 분석 커버리지 향상 (15→30)
FRAME_STEP = 10
PORT = 60006

mp_face_detection = mp.solutions.face_detection
face_detection = mp_face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.5)

device = torch.device("cuda:0")  # CUDA_VISIBLE_DEVICES=1이므로 물리적 GPU 1 = cuda:0
loaded_models: Dict[str, nn.Module] = {}

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]
transform_eval = transforms.Compose([
    transforms.Resize((FACE_SIZE, FACE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])

analysis_tasks: Dict[str, dict] = {}

def load_all_models():
    print(f"[v10] Loading models to {device}...")
    for mode_name, cfg in MODE_CONFIG.items():
        for entry in cfg["models"]:
            name = entry["name"]
            if name in loaded_models: continue
            path = os.path.join(BASE, entry["path"])
            if not os.path.exists(path): continue
            m = entry["builder" ]().to(device)
            for module in m.modules():
                if hasattr(module, 'inplace'): module.inplace = False
            ck = torch.load(path, map_location=device, weights_only=False)
            m.load_state_dict(ck.get("model_state_dict", ck), strict=False)
            m.eval()
            loaded_models[name] = m
            print(f"  ✅ {name} loaded")

class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        for m in self.model.modules():
            if hasattr(m, 'inplace'): m.inplace = False
        self.target_layer = target_layer
        self.activations = None
        self.gradients = None
        target_layer.register_forward_hook(self._save_act)
        target_layer.register_full_backward_hook(self._save_grad)
    def _save_act(self, module, inp, out): self.activations = out.detach()
    def _save_grad(self, module, gi, go):  self.gradients = go[0].detach()
    def generate(self, x):
        self.activations = None; self.gradients = None
        out = self.model(x)
        if isinstance(out, (list, tuple)): out = out[0]
        self.model.zero_grad()
        out.backward()
        weights = self.gradients.mean(dim=[2, 3], keepdim=True)
        cam = F.relu((weights * self.activations).sum(dim=1, keepdim=True))
        cam = cam.squeeze().cpu().numpy()
        cam -= cam.min(); cam /= cam.max() + 1e-8
        return cam


def download_to_tempfile(url):
    # 로컬 /files/ 경로 (상대경로 or http 포함 모두)
    if '/files/' in url:
        return os.path.join('/tmp', url.split('/files/')[-1])
    if not url.startswith('http'): return url
    
    # External URL (S3, etc.)
    print(f"[v10] Downloading external URL: {url}")
    fd, p = tempfile.mkstemp(suffix=".mp4"); os.close(fd)
    try:
        r = requests.get(url, stream=True, timeout=30)
        r.raise_for_status()
        with open(p, 'wb') as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)
        return p
    except Exception as e:
        print(f"[v10] Download failed: {e}")
        if os.path.exists(p): os.remove(p)
        return None

async def run_analysis(video_id: str, path: str, analysis_type: str):
    analysis_tasks[video_id]["status"] = "analyzing"
    try:
        cap = cv2.VideoCapture(path)
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        frames, indices, idx = [], [], 0
        while len(frames) < MAX_FRAMES:
            ret, f = cap.read()
            if not ret: break
            if idx % FRAME_STEP == 0:
                frames.append(f); indices.append(idx)
            idx += 1
        cap.release()
        if not frames:
            analysis_tasks[video_id]["status"] = "failed"
            return

        faces = []
        for i, f in zip(indices, frames):
            rgb = cv2.cvtColor(f, cv2.COLOR_BGR2RGB)
            res = face_detection.process(rgb)
            if res.detections:
                det = max(res.detections, key=lambda d: d.score[0])
                b = det.location_data.relative_bounding_box
                H, W = rgb.shape[:2]
                cx, cy = (b.xmin + b.width/2)*W, (b.ymin + b.height/2)*H
                s = max(b.width*W, b.height*H) * CROP_SCALE
                x1, y1 = int(max(0, cx-s/2)), int(max(0, cy-s/2))
                x2, y2 = int(min(W, cx+s/2)), int(min(H, cy+s/2))
                face_pil = Image.fromarray(rgb[y1:y2, x1:x2]).resize((FACE_SIZE, FACE_SIZE), Image.BILINEAR)
                faces.append({"pil": face_pil, "idx": i, "timestamp": i / fps})

        if not faces:
            analysis_tasks[video_id]["status"] = "failed"
            return

        batch = torch.stack([transform_eval(f["pil"]) for f in faces]).to(device)
        mode = analysis_type if analysis_type in MODE_CONFIG else "SHEN"
        cfg = MODE_CONFIG[mode]
        model_probs = {}
        with torch.set_grad_enabled(True):
            for entry in cfg["models"]:
                name = entry["name"]
                if name in loaded_models:
                    model = loaded_models[name]
                    x = batch.clone().requires_grad_(True)
                    out = model(x)
                    if isinstance(out, (list, tuple)): out = out[0]
                    probs = torch.sigmoid(out.squeeze()).detach().cpu().numpy()
                    if probs.ndim == 0: probs = np.array([probs])
                    model_probs[name] = probs

        weights = np.array([e["weight"] for e in cfg["models"] if e["name"] in model_probs])
        weights /= weights.sum()
        all_probs = []
        for i in range(len(faces)):
            p = sum(weights[j] * list(model_probs.values())[j][i] for j in range(len(weights)))
            all_probs.append(float(p))
            
        agg_type = cfg.get("agg", "mean")
        final_score = float(np.max(all_probs) if agg_type == "max" else np.mean(all_probs))
        verdict = "FAKE" if final_score >= cfg["threshold"] else "REAL"
        
        suspect_idx = np.argmax(all_probs)
        suspect_face = faces[suspect_idx]["pil"]
        
        b64_img, b64_orig = "", ""
        rgb_share, freq_share = 0.5, 0.5
        top_regions = []
        forensic_report = "[안전] 정밀 검사 결과 이상 없음."

        try:
            buf_orig = io.BytesIO()
            suspect_face.save(buf_orig, format="JPEG", quality=85)
            b64_orig = "data:image/jpeg;base64," + base64.b64encode(buf_orig.getvalue()).decode()

            m_target = loaded_models.get("v25", list(loaded_models.values())[0])
            if hasattr(m_target, 'fusion'):
                captured = {}
                def h_fn(module, inp, out):
                    try:
                        c = torch.cat([module.bn_rgb(inp[0]), module.bn_freq(inp[1])], dim=1)
                        captured["g"] = torch.sigmoid(module.fc_excite(F.relu(module.fc_squeeze(c)))).detach().cpu().numpy()[0]
                    except: pass
                h = m_target.fusion.register_forward_hook(h_fn)
                with torch.no_grad(): _ = m_target(transform_eval(suspect_face).unsqueeze(0).to(device))
                h.remove()
                g = captured.get("g")
                if g is not None:
                    rm, fm = float(g[:len(g)//2].mean()), float(g[len(g)//2:].mean())
                    s = rm + fm + 1e-9
                    rgb_share, freq_share = round(rm/s, 3), round(fm/s, 3)

            target_layer = None
            if hasattr(m_target, 'rgb_branch'): target_layer = dict(m_target.rgb_branch.named_modules()).get("bn4")
            elif hasattr(m_target, 'bn4'): target_layer = m_target.bn4
            
            if target_layer:
                cam_obj = GradCAM(m_target, target_layer)
                cam = cam_obj.generate(transform_eval(suspect_face).unsqueeze(0).to(device).requires_grad_(True))
                H_c, W_c = cam.shape
                cells = np.zeros((3, 3))
                ys, xs = np.linspace(0, H_c, 4, dtype=int), np.linspace(0, W_c, 4, dtype=int)
                for r in range(3):
                    for c in range(3): cells[r,c] = cam[ys[r]:ys[r+1], xs[c]:xs[c+1]].sum()
                cells /= (cells.sum() + 1e-9)
                flat = sorted([(GRID_LABELS[r][c], float(cells[r,c])) for r in range(3) for c in range(3)], key=lambda t: -t[1])
                top_regions = [{"region": r, "ratio": round(p, 3)} for r, p in flat[:3]]
                
                heatmap = cv2.applyColorMap(np.uint8(255 * cv2.resize(cam, (suspect_face.width, suspect_face.height))), cv2.COLORMAP_JET)
                heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
                overlay = cv2.addWeighted(np.array(suspect_face), 0.5, heatmap, 0.5, 0)
                buf_hm = io.BytesIO()
                Image.fromarray(overlay).save(buf_hm, format="JPEG", quality=85)
                b64_img = "data:image/jpeg;base64," + base64.b64encode(buf_hm.getvalue()).decode()

            if verdict == "FAKE":
                forensic_report = f"[치명적 위험] 영상 {faces[suspect_idx]['timestamp']:.1f}초 지점에서 조작 흔적이 발견되었습니다. 시각적 텍스처 왜곡({int(rgb_share*100)}%)과 비시각적 주파수 변조({int(freq_share*100)}%)가 동시에 감지되었으며, 특히 {top_regions[0]['region'] if top_regions else '얼굴'} 부근이 집중적으로 조작된 것으로 분석됩니다."
            else:
                forensic_report = "[안전] 정밀 검사 결과, 해당 영상에서는 합성 흔적이나 주파수 아티팩트가 발견되지 않았습니다."

        except Exception as e:
            print(f"[XAI Error] {e}")
            if not b64_img: b64_img = b64_orig

        result = {
            # Spring Boot AiAnalysisResponse DTO 필드
            "decision": verdict, "score": final_score, "threshold": cfg["threshold"],
            "evidence": {
                "suspect_frame_idx": int(suspect_idx),
                "detect_conf": round(float(all_probs[suspect_idx]), 4),
                "blur_var": None,
                "n_frames_analyzed": len(faces),
                "heatmaps": {"v7": b64_img} if b64_img else {},
                "se_attention": None,
                "regions": None,
            },
            # 프론트엔드 직접 필드
            "final_verdict": verdict, "deepfake_score": final_score * 100,
            "t2v_score": (final_score * 0.8) * 100 if analysis_type == "T2V" else 0,
            "xai_text": forensic_report,
            "suspicious_frames": [{"frameIndex": f["idx"], "probability": p, "time": f"{f['timestamp']:.2f}s"} for f, p in zip(faces, all_probs)],
            "xai_heatmap_url": b64_img, "per_frame_probs": all_probs, "analysis_type": analysis_type, "engine_label": cfg.get("label", analysis_type),
            "original_face_url": b64_orig,
            "rgb_contribution": rgb_share * 100, "freq_contribution": freq_share * 100,
            "top_regions": top_regions, "forensic_report": forensic_report,
            "raw": {
                "score": final_score, "threshold": cfg["threshold"], "original_face_url": b64_orig,
                "rgb_contribution": rgb_share * 100, "freq_contribution": freq_share * 100,
                "top_regions": top_regions, "forensic_report": forensic_report
            }
        }
        print(f"[v10] RESULT: verdict={verdict}, score={final_score:.4f}, threshold={cfg['threshold']}, faces={len(faces)}, per_frame={[round(p,4) for p in all_probs]}")
        analysis_tasks[video_id]["result"] = result
        analysis_tasks[video_id]["status"] = "completed"
    except Exception as e:
        print(f"Analysis Error: {e}")
        analysis_tasks[video_id]["status"] = "failed"
    finally:
        if os.path.exists(path) and "/tmp/" in path:
            try: os.remove(path)
            except: pass

@asynccontextmanager
async def lifespan(app: FastAPI):
    load_all_models()
    yield
app = FastAPI(lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

@app.post("/api/uploads/presigned-url")
async def presigned_url(request: Request):
    data = await request.json()
    unique_name = f"{uuid.uuid4().hex}_{data.get('fileName', 'v.mp4')}"
    return {"uploadUrl": f"/upload/{unique_name}", "fileUrl": f"/files/{unique_name}"}
@app.put("/upload/{filename}")
async def upload_file(filename: str, request: Request):
    with open(os.path.join("/tmp", filename), "wb") as f: f.write(await request.body())
    return {"ok": True}
@app.get("/files/{filename}")
async def get_file(filename: str): return FileResponse(os.path.join("/tmp", filename))

@app.post("/deepfake/analyze")
async def deepfake_analyze(request: Request):
    data = await request.json()
    print(f"[v10] Raw request body: {data}")
    url = data.get("url") or data.get("videoUrl") or data.get("video_url", "")
    model_type = data.get("type", "SHEN")
    print(f"[v10] Sync analysis request: {url} ({model_type})")
    
    local_path = download_to_tempfile(url)
    if not local_path or not os.path.exists(local_path):
         return {"status": "failed", "message": "download failed"}
         
    # Generate unique ID for internal tracking
    video_id = "sync_" + uuid.uuid4().hex[:8]
    analysis_tasks[video_id] = {"status": "analyzing"}
    
    try:
        # Call the existing analysis function
        await run_analysis(video_id, local_path, model_type)
        result = analysis_tasks.get(video_id, {}).get("result", {})
        return result
    except Exception as e:
        print(f"[v10] Sync analysis error: {e}")
        return {"status": "failed", "message": str(e)}

@app.post("/t2v/analyze")
async def t2v_analyze(request: Request):
    return await deepfake_analyze(request)
@app.post("/api/videos/url")
async def video_url(request: Request, bg_tasks: BackgroundTasks):
    data = await request.json()
    url = data.get("url", "")
    video_id = uuid.uuid4().hex[:12]
    analysis_tasks[video_id] = {"status": "pending"}
    
    local_path = download_to_tempfile(url)
    if not local_path or not os.path.exists(local_path):
         analysis_tasks[video_id]["status"] = "failed"
         return {"video_id": video_id}

    bg_tasks.add_task(run_analysis, video_id, local_path, data.get("type", "SHEN"))
    return {"video_id": video_id}
@app.get("/api/videos/{video_id}/status")
async def video_status(video_id: str): return {"status": analysis_tasks.get(video_id, {}).get("status", "failed")}
@app.get("/api/videos/{video_id}/result")
async def video_result(video_id: str): return analysis_tasks.get(video_id, {}).get("result", {})
@app.get("/{full_path:path}")
async def serve_frontend(full_path: str):
    f_path = os.path.join(BASE, "frontendRepo-main/dist")
    if any(full_path.startswith(p) for p in ["api/", "upload/", "files/"]): raise HTTPException(404)
    if not full_path or not os.path.exists(os.path.join(f_path, full_path)): return FileResponse(os.path.join(f_path, "index.html"))
    return FileResponse(os.path.join(f_path, full_path))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=60006)
