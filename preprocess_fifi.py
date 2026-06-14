"""
preprocess_fifi.py
==================
final/fifi_tmp/ 의 SNS real 신규 영상 → 얼굴 크롭 → preprocessed_faces/sns_real_fifi/
- preprocess_insta_real.py 와 동일 파이프라인
- 다만 출력 디렉터리/CSV 분리 (기존 sns_real 과 구분)

기존 (이미 v24/v25/v26 학습에 사용된) sns_real:    66개 영상  → 2,774+702 프레임
신규                                               155개 영상  → 약 7,000~9,000 프레임 예상
"""
import os, sys, cv2, csv
import numpy as np
from PIL import Image
import mediapipe as mp

INPUT_DIR  = '/home/t26106/deepfake/final/fifi_tmp'
OUTPUT_DIR = '/home/t26106/deepfake/preprocessed_faces/sns_real_fifi'
CSV_OUT    = '/home/t26106/deepfake/splits/sns_real_fifi_frames.csv'
TARGET_FPS = 5
MAX_FRAMES = 60
FACE_SIZE  = 299
MIN_BLUR   = 30.0

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(os.path.dirname(CSV_OUT), exist_ok=True)

mp_fd = mp.solutions.face_detection
fd = mp_fd.FaceDetection(model_selection=1, min_detection_confidence=0.5)


def detect_and_crop(frame):
    H, W = frame.shape[:2]
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    res = fd.process(rgb)
    if not res.detections:
        return None
    det = max(res.detections, key=lambda d: d.score[0])
    b = det.location_data.relative_bounding_box
    cx, cy = (b.xmin + b.width/2)*W, (b.ymin + b.height/2)*H
    s = max(b.width*W, b.height*H) * 1.3
    x1, y1 = int(max(0, cx-s/2)), int(max(0, cy-s/2))
    x2, y2 = int(min(W, cx+s/2)), int(min(H, cy+s/2))
    if x2 <= x1 or y2 <= y1:
        return None
    face = Image.fromarray(rgb[y1:y2, x1:x2]).resize((FACE_SIZE, FACE_SIZE), Image.BILINEAR)
    return face


def is_blurry(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return cv2.Laplacian(gray, cv2.CV_64F).var() < MIN_BLUR


def process_video(video_path, video_name):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    step = max(1, int(round(fps / TARGET_FPS)))

    video_dir = os.path.join(OUTPUT_DIR, video_name)
    os.makedirs(video_dir, exist_ok=True)

    saved, idx, frame_idx = 0, 0, 0
    entries = []
    while saved < MAX_FRAMES:
        ret, frame = cap.read()
        if not ret:
            break
        if idx % step == 0:
            if not is_blurry(frame):
                face = detect_and_crop(frame)
                if face is not None:
                    fname = f'frame_{frame_idx:04d}.jpg'
                    fpath = os.path.join(video_dir, fname)
                    face.save(fpath, quality=95)
                    entries.append({
                        'split': 'train',
                        'image_path': fpath,
                        'label': 0,
                        'group': 'original',
                        'subtype': 'sns_real',
                        'video_name': video_name,
                        'holdout_key': abs(hash(video_name)) % 100,
                        'frame_dir': video_dir,
                        'dataset': 'sns'
                    })
                    saved += 1
                    frame_idx += 1
        idx += 1
    cap.release()
    return saved, entries


def main():
    videos = sorted([
        f for f in os.listdir(INPUT_DIR)
        if f.lower().endswith(('.mp4', '.mov', '.avi'))
        and not f.startswith('._')
    ])
    if not videos:
        print(f'[ERROR] {INPUT_DIR} 에 영상 없음')
        return

    print(f'총 {len(videos)}개 영상 전처리 시작')
    print(f'출력: {OUTPUT_DIR}')
    print('='*60)

    all_entries = []
    total_frames = 0
    fail_videos = []

    for i, vname in enumerate(videos):
        vpath = os.path.join(INPUT_DIR, vname)
        vkey = os.path.splitext(vname)[0]
        try:
            saved, entries = process_video(vpath, vkey)
            all_entries.extend(entries)
            total_frames += saved
            if saved == 0:
                fail_videos.append(vname)
            print(f'[{i+1:3d}/{len(videos)}] {vname[:40]:<40} → {saved}f')
        except Exception as e:
            fail_videos.append(vname)
            print(f'[{i+1:3d}/{len(videos)}] {vname[:40]:<40} → ERROR {e}')

    with open(CSV_OUT, 'w', newline='') as f:
        fieldnames = ['split','image_path','label','group','subtype','video_name','holdout_key','frame_dir','dataset']
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_entries)

    print('='*60)
    print(f'완료: {len(videos)}개 영상 → {total_frames} 프레임')
    print(f'성공 영상: {len(videos)-len(fail_videos)} / 실패: {len(fail_videos)}')
    if fail_videos:
        print('실패 영상 (얼굴 미검출):')
        for v in fail_videos[:10]:
            print(f'  - {v}')
    print(f'CSV: {CSV_OUT}')


if __name__ == '__main__':
    main()
