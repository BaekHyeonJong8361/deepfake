import cv2
import os
import glob
import torch
import time
import tarfile
import io
import csv
import multiprocessing as mp
from datetime import datetime
from facenet_pytorch import MTCNN

global_detector = None


def init_worker(device_str):
    global global_detector
    device = torch.device(device_str)
    global_detector = MTCNN(keep_all=True, device=device)
    global_detector.eval()


def get_enlarged_box_pytorch(box, scale=1.3, img_shape=None):
    x1, y1, x2, y2 = box
    w, h = x2 - x1, y2 - y1
    center_x, center_y = x1 + w / 2, y1 + h / 2
    new_w, new_h = w * scale, h * scale
    nx1, ny1 = int(center_x - new_w / 2), int(center_y - new_h / 2)
    nx2, ny2 = int(nx1 + new_w), int(ny1 + new_h)

    if img_shape is not None:
        h_max, w_max = img_shape[:2]
        nx1, ny1 = max(0, nx1), max(0, ny1)
        nx2, ny2 = min(w_max, nx2), min(h_max, ny2)

    return nx1, ny1, nx2, ny2


def calculate_iou(boxA, boxB):
    xA = max(float(boxA[0]), float(boxB[0]))
    yA = max(float(boxA[1]), float(boxB[1]))
    xB = min(float(boxA[2]), float(boxB[2]))
    yB = min(float(boxA[3]), float(boxB[3]))

    inter_w = max(0.0, xB - xA)
    inter_h = max(0.0, yB - yA)
    inter_area = inter_w * inter_h

    if inter_area <= 0:
        return 0.0

    boxA_area = max(0.0, float(boxA[2]) - float(boxA[0])) * max(0.0, float(boxA[3]) - float(boxA[1]))
    boxB_area = max(0.0, float(boxB[2]) - float(boxB[0])) * max(0.0, float(boxB[3]) - float(boxB[1]))

    union_area = boxA_area + boxB_area - inter_area
    if union_area <= 0:
        return 0.0

    return inter_area / union_area


def box_area(box):
    return max(0.0, float(box[2]) - float(box[0])) * max(0.0, float(box[3]) - float(box[1]))


def is_blurry(frame, blur_threshold):
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur_score = cv2.Laplacian(gray_frame, cv2.CV_64F).var()
    return blur_score < blur_threshold, blur_score


def write_debug_log(
    log_path,
    result_dict,
    settings
):
    with open(log_path, "w", encoding="utf-8") as f:
        f.write(f"status: {result_dict['status']}\\n")
        f.write(f"video_name: {result_dict['video_name']}\\n")
        f.write(f"video_path: {result_dict['video_path']}\\n")
        f.write(f"output_tar_path: {result_dict['tar_path']}\\n")
        f.write(f"elapsed_sec: {result_dict['elapsed_sec']:.2f}\\n")
        f.write(f"saved_count: {result_dict['saved_count']}\\n")
        f.write(f"label: {result_dict['label']}\\n")
        f.write(f"group: {result_dict['group']}\\n")
        f.write(f"subtype: {result_dict['subtype']}\\n")
        f.write(f"compression: {result_dict['compression']}\\n")
        if result_dict.get("message"):
            f.write(f"message: {result_dict['message']}\\n")

        f.write("\\n[stats]\\n")
        for k, v in result_dict["stats"].items():
            f.write(f"{k}: {v}\\n")

        f.write("\\n[settings]\\n")
        for k, v in settings.items():
            f.write(f"{k}: {v}\\n")


def get_output_base_dir(video_path, output_root):
    norm_path = os.path.normpath(video_path)
    parts = norm_path.split(os.sep)

    # ff_data/original_sequences/youtube/c23/videos/xxx.mp4
    if "original_sequences" in parts:
        idx = parts.index("original_sequences")
        source = parts[idx + 1] if idx + 1 < len(parts) else "unknown_source"
        compression = parts[idx + 2] if idx + 2 < len(parts) else "unknown_compression"
        return os.path.join(output_root, "original", source, compression)

    # ff_data/manipulated_sequences/FaceSwap/c23/videos/xxx.mp4
    if "manipulated_sequences" in parts:
        idx = parts.index("manipulated_sequences")
        method = parts[idx + 1] if idx + 1 < len(parts) else "unknown_method"
        compression = parts[idx + 2] if idx + 2 < len(parts) else "unknown_compression"
        return os.path.join(output_root, "manipulated", method, compression)

    return os.path.join(output_root, "unknown")


def parse_video_metadata(video_path):
    norm_path = os.path.normpath(video_path)
    parts = norm_path.split(os.sep)

    meta = {
        "label": -1,
        "group": "unknown",
        "subtype": "unknown",
        "compression": "unknown",
        "is_unknown": False,
    }

    if "original_sequences" in parts:
        idx = parts.index("original_sequences")
        meta["label"] = 0
        meta["group"] = "original"
        meta["subtype"] = parts[idx + 1] if idx + 1 < len(parts) else "unknown_source"
        meta["compression"] = parts[idx + 2] if idx + 2 < len(parts) else "unknown_compression"
        return meta

    if "manipulated_sequences" in parts:
        idx = parts.index("manipulated_sequences")
        meta["label"] = 1
        meta["group"] = "manipulated"
        meta["subtype"] = parts[idx + 1] if idx + 1 < len(parts) else "unknown_method"
        meta["compression"] = parts[idx + 2] if idx + 2 < len(parts) else "unknown_compression"
        return meta

    meta["is_unknown"] = True
    return meta


def make_result_dict(
    status,
    video_name,
    video_path,
    tar_path,
    log_path,
    elapsed_sec,
    saved_count,
    stats,
    label,
    group,
    subtype,
    compression,
    message=""
):
    return {
        "status": status,              # success / skipped / failed / no_frames
        "video_name": video_name,
        "video_path": video_path,
        "tar_path": tar_path,
        "log_path": log_path,
        "elapsed_sec": float(elapsed_sec),
        "saved_count": int(saved_count),
        "stats": stats,
        "label": int(label),
        "group": group,
        "subtype": subtype,
        "compression": compression,
        "message": message,
    }


def process_single_video(args):
    (
        video_path,
        output_base_dir,
        target_fps,
        blur_threshold,
        conf_threshold,
        crop_scale,
        output_size,
        jpeg_quality,
        iou_threshold,
        min_face_size_px,
        min_saved_frames,
        debug_mode,
        skip_existing,
        overwrite_existing,
    ) = args

    start_time = time.time()
    global global_detector

    video_name = os.path.splitext(os.path.basename(video_path))[0]
    os.makedirs(output_base_dir, exist_ok=True)

    tar_path = os.path.join(output_base_dir, f"{video_name}.tar")
    log_path = os.path.join(output_base_dir, f"{video_name}_log.txt")

    meta = parse_video_metadata(video_path)

    settings = {
        "target_fps": target_fps,
        "blur_threshold": blur_threshold,
        "conf_threshold": conf_threshold,
        "crop_scale": crop_scale,
        "output_size": output_size,
        "jpeg_quality": jpeg_quality,
        "iou_threshold": iou_threshold,
        "min_face_size_px": min_face_size_px,
        "min_saved_frames": min_saved_frames,
        "debug_mode": debug_mode,
        "skip_existing": skip_existing,
        "overwrite_existing": overwrite_existing,
    }

    # overwrite 우선
    if overwrite_existing and os.path.exists(tar_path):
        try:
            os.remove(tar_path)
        except Exception:
            pass

    # 이미 전처리 결과가 있으면 skip
    if skip_existing and os.path.exists(tar_path) and os.path.getsize(tar_path) > 0:
        elapsed = time.time() - start_time
        result = make_result_dict(
            status="skipped",
            video_name=video_name,
            video_path=video_path,
            tar_path=tar_path,
            log_path=log_path,
            elapsed_sec=elapsed,
            saved_count=0,
            stats={},
            label=meta["label"],
            group=meta["group"],
            subtype=meta["subtype"],
            compression=meta["compression"],
            message="이미 전처리된 파일 존재"
        )
        return result

    stats = {
        "total_frames_read": 0,
        "sampled_frames": 0,
        "blur_skipped": 0,
        "no_face_detected": 0,
        "low_confidence_faces_filtered": 0,
        "small_face_filtered": 0,
        "tracking_failed": 0,
        "invalid_crop_skipped": 0,
        "encode_failed": 0,
        "saved_frames": 0,
        "detector_none_output": 0,
    }

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        elapsed = time.time() - start_time
        result = make_result_dict(
            status="failed",
            video_name=video_name,
            video_path=video_path,
            tar_path=tar_path,
            log_path=log_path,
            elapsed_sec=elapsed,
            saved_count=0,
            stats=stats,
            label=meta["label"],
            group=meta["group"],
            subtype=meta["subtype"],
            compression=meta["compression"],
            message="비디오 열기 실패"
        )
        write_debug_log(log_path, result, settings)
        return result

    original_fps = cap.get(cv2.CAP_PROP_FPS)
    if not original_fps or original_fps <= 0:
        original_fps = 30.0

    frame_interval = max(1, int(round(original_fps / target_fps)))

    settings["original_fps"] = original_fps
    settings["frame_interval"] = frame_interval

    frame_count = 0
    saved_count = 0
    target_box = None

    try:
        with tarfile.open(tar_path, "w") as tar:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                stats["total_frames_read"] += 1

                if frame_count % frame_interval == 0:
                    stats["sampled_frames"] += 1

                    blurry, blur_score = is_blurry(frame, blur_threshold)
                    if blurry:
                        stats["blur_skipped"] += 1
                        frame_count += 1
                        continue

                    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    with torch.no_grad():
                        boxes, probs = global_detector.detect(rgb_frame)

                    if boxes is None or probs is None:
                        stats["no_face_detected"] += 1
                        stats["detector_none_output"] += 1
                        frame_count += 1
                        continue

                    valid_faces = []
                    for b, p in zip(boxes, probs):
                        if b is None or p is None:
                            stats["low_confidence_faces_filtered"] += 1
                            continue

                        if float(p) < conf_threshold:
                            stats["low_confidence_faces_filtered"] += 1
                            continue

                        if box_area(b) < (min_face_size_px * min_face_size_px):
                            stats["small_face_filtered"] += 1
                            continue

                        valid_faces.append((b, p))

                    if not valid_faces:
                        stats["no_face_detected"] += 1
                        frame_count += 1
                        continue

                    best_box = None

                    if target_box is None:
                        best_box, _ = max(
                            valid_faces,
                            key=lambda item: box_area(item[0])
                        )
                    else:
                        max_iou = -1.0
                        for b, p in valid_faces:
                            iou = calculate_iou(target_box, b)
                            if iou > max_iou:
                                max_iou = iou
                                best_box = b

                        if max_iou < iou_threshold:
                            stats["tracking_failed"] += 1
                            best_box, _ = max(
                                valid_faces,
                                key=lambda item: box_area(item[0])
                            )

                    if best_box is not None:
                        target_box = best_box
                        nx1, ny1, nx2, ny2 = get_enlarged_box_pytorch(best_box, crop_scale, frame.shape)

                        if nx2 <= nx1 or ny2 <= ny1:
                            stats["invalid_crop_skipped"] += 1
                            frame_count += 1
                            continue

                        face = frame[ny1:ny2, nx1:nx2]
                        if face.size == 0:
                            stats["invalid_crop_skipped"] += 1
                            frame_count += 1
                            continue

                        face = cv2.resize(face, output_size)
                        is_success, buffer = cv2.imencode(
                            ".jpg",
                            face,
                            [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality]
                        )

                        if not is_success:
                            stats["encode_failed"] += 1
                            frame_count += 1
                            continue

                        buffer_bytes = buffer.tobytes()
                        io_buf = io.BytesIO(buffer_bytes)
                        info = tarfile.TarInfo(name=f"frame_{saved_count:04d}.jpg")
                        info.size = len(buffer_bytes)
                        tar.addfile(tarinfo=info, fileobj=io_buf)

                        saved_count += 1
                        stats["saved_frames"] = saved_count

                frame_count += 1

        cap.release()

        # 0장 또는 너무 적게 저장되면 no_frames 처리
        if os.path.exists(tar_path) and (saved_count == 0 or saved_count < min_saved_frames):
            os.remove(tar_path)

            elapsed = time.time() - start_time
            reason = f"저장 프레임 부족 ({saved_count}장)"
            result = make_result_dict(
                status="no_frames",
                video_name=video_name,
                video_path=video_path,
                tar_path=tar_path,
                log_path=log_path,
                elapsed_sec=elapsed,
                saved_count=saved_count,
                stats=stats,
                label=meta["label"],
                group=meta["group"],
                subtype=meta["subtype"],
                compression=meta["compression"],
                message=reason
            )
            write_debug_log(log_path, result, settings)
            return result

        elapsed = time.time() - start_time
        result = make_result_dict(
            status="success",
            video_name=video_name,
            video_path=video_path,
            tar_path=tar_path,
            log_path=log_path,
            elapsed_sec=elapsed,
            saved_count=saved_count,
            stats=stats,
            label=meta["label"],
            group=meta["group"],
            subtype=meta["subtype"],
            compression=meta["compression"],
            message=f"{saved_count}장 추출 완료"
        )
        write_debug_log(log_path, result, settings)
        return result

    except Exception as e:
        cap.release()
        elapsed = time.time() - start_time

        # 비정상 생성 tar 정리
        if os.path.exists(tar_path):
            try:
                os.remove(tar_path)
            except Exception:
                pass

        stats["error"] = str(e)
        result = make_result_dict(
            status="failed",
            video_name=video_name,
            video_path=video_path,
            tar_path=tar_path,
            log_path=log_path,
            elapsed_sec=elapsed,
            saved_count=saved_count,
            stats=stats,
            label=meta["label"],
            group=meta["group"],
            subtype=meta["subtype"],
            compression=meta["compression"],
            message=str(e)
        )
        write_debug_log(log_path, result, settings)
        return result
def write_summary_log(summary_path, total_videos, results, settings, total_elapsed):
    success_count = sum(r["status"] == "success" for r in results)
    skipped_count = sum(r["status"] == "skipped" for r in results)
    failed_count = sum(r["status"] == "failed" for r in results)
    no_frames_count = sum(r["status"] == "no_frames" for r in results)
    unknown_count = sum(r["group"] == "unknown" for r in results)

    total_saved_frames = sum(r["saved_count"] for r in results if r["status"] == "success")
    avg_saved_frames_success = total_saved_frames / max(1, success_count)

    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("[summary]\\n")
        f.write(f"total_videos: {total_videos}\\n")
        f.write(f"success_count: {success_count}\\n")
        f.write(f"skipped_count: {skipped_count}\\n")
        f.write(f"failed_count: {failed_count}\\n")
        f.write(f"no_frames_count: {no_frames_count}\\n")
        f.write(f"unknown_path_count: {unknown_count}\\n")
        f.write(f"total_saved_frames(success_only): {total_saved_frames}\\n")
        f.write(f"avg_saved_frames_per_success_video: {avg_saved_frames_success:.2f}\\n")
        f.write(f"total_elapsed_sec: {total_elapsed:.2f}\\n")
        if total_videos > 0:
            f.write(f"avg_sec_per_video: {total_elapsed / total_videos:.2f}\\n")

        f.write("\\n[settings]\\n")
        for k, v in settings.items():
            f.write(f"{k}: {v}\\n")

        f.write("\\n[results]\\n")
        for r in results:
            f.write(
                f"{r['status']} | "
                f"{r['video_name']} | "
                f"saved={r['saved_count']} | "
                f"group={r['group']} | "
                f"subtype={r['subtype']} | "
                f"compression={r['compression']} | "
                f"elapsed={r['elapsed_sec']:.2f}s | "
                f"message={r['message']}\\n"
            )


def write_manifest_csv(manifest_path, results):
    fieldnames = [
        "status",
        "video_name",
        "video_path",
        "tar_path",
        "log_path",
        "elapsed_sec",
        "saved_count",
        "label",
        "group",
        "subtype",
        "compression",
        "message",
    ]

    with open(manifest_path, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in results:
            row = {k: r.get(k, "") for k in fieldnames}
            writer.writerow(row)

def collect_input_videos():
    original_videos = glob.glob("ff_data/original_sequences/**/*.mp4", recursive=True)
    manipulated_videos = glob.glob("ff_data/manipulated_sequences/**/*.mp4", recursive=True)

    input_videos = sorted(original_videos) + sorted(manipulated_videos)
    return input_videos


def is_already_processed(video_path, output_root):
    output_base_dir = get_output_base_dir(video_path, output_root)
    video_name = os.path.splitext(os.path.basename(video_path))[0]

    tar_path = os.path.join(output_base_dir, f"{video_name}.tar")
    log_path = os.path.join(output_base_dir, f"{video_name}_log.txt")

    # 성공적으로 tar가 있으면 이미 처리된 것으로 간주
    if os.path.exists(tar_path) and os.path.getsize(tar_path) > 0:
        return True

    # tar는 없어도 log가 있으면 이미 한 번 시도한 것으로 간주
    if os.path.exists(log_path) and os.path.getsize(log_path) > 0:
        return True

    return False


def collect_unprocessed_videos(all_videos, output_root):
    unprocessed_videos = []

    for vid in all_videos:
        if not is_already_processed(vid, output_root):
            unprocessed_videos.append(vid)

    return unprocessed_videos


if __name__ == '__main__':
    mp.set_start_method('spawn', force=True)

    # =========================
    # Paths / run options
    # =========================
    output_root = "preprocessed_faces"
    os.makedirs(output_root, exist_ok=True)

    skip_existing = True
    overwrite_existing = False   # True면 기존 tar 삭제 후 다시 생성
    debug_mode = True

    # =========================
    # Preprocess settings
    # =========================
    target_fps = 5
    blur_threshold = 23.0
    conf_threshold = 0.90
    crop_scale = 1.3
    output_size = (256, 256)
    jpeg_quality = 95
    iou_threshold = 0.20
    min_face_size_px = 24
    min_saved_frames = 1   # 1보다 작으면 사실상 0장만 제거, 5 등으로 올리면 너무 적은 영상 제거 가능

    # =========================
    # Collect inputs
    # =========================
    all_videos = collect_input_videos()
    unprocessed_videos = collect_unprocessed_videos(all_videos, output_root)

    processed_count = len(all_videos) - len(unprocessed_videos)

    print(f"📦 전체 영상 수: {len(all_videos)}개")
    print(f"✅ 이미 처리된 영상 수: {processed_count}개")
    print(f"🕒 미처리 영상 수: {len(unprocessed_videos)}개")

    if len(unprocessed_videos) == 0:
        print("✅ 모든 파일 전처리 완료")
        raise SystemExit

    batch_size = 500
    input_videos = unprocessed_videos[:batch_size]

    print(f"🚀 이번 실행  대상: {len(input_videos)}개")

    run_settings = {
        "output_root": output_root,
        "skip_existing": skip_existing,
        "overwrite_existing": overwrite_existing,
        "debug_mode": debug_mode,
        "target_fps": target_fps,
        "blur_threshold": blur_threshold,
        "conf_threshold": conf_threshold,
        "crop_scale": crop_scale,
        "output_size": output_size,
        "jpeg_quality": jpeg_quality,
        "iou_threshold": iou_threshold,
        "min_face_size_px": min_face_size_px,
        "min_saved_frames": min_saved_frames,
        "batch_size": batch_size,
        "torch_cuda_available": torch.cuda.is_available(),
        "torch_version": torch.__version__,
        "run_datetime": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }

    tasks = []
    for vid in input_videos:
        output_base_dir = get_output_base_dir(vid, output_root)
        tasks.append(
            (
                vid,
                output_base_dir,
                target_fps,
                blur_threshold,
                conf_threshold,
                crop_scale,
                output_size,
                jpeg_quality,
                iou_threshold,
                min_face_size_px,
                min_saved_frames,
                debug_mode,
                skip_existing,
                overwrite_existing,
            )
        )

    print(f"🚀 총 {len(input_videos)}개 작업 시작...")
    total_start_time = time.time()

    if torch.cuda.is_available():
        device_str = "cuda:0"
        num_processes = 2
    else:
        device_str = "cpu"
        num_processes = min(len(tasks), max(1, mp.cpu_count() // 2))

    print(f"🖥️ device: {device_str}")
    print(f"⚙️ num_processes: {num_processes}")

    with mp.Pool(
        processes=num_processes,
        initializer=init_worker,
        initargs=(device_str,)
    ) as pool:
        results = pool.map(process_single_video, tasks)

    total_elapsed = time.time() - total_start_time

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    summary_path = os.path.join(output_root, f"preprocess_summary_{timestamp}.txt")
    manifest_path = os.path.join(output_root, f"preprocess_manifest_{timestamp}.csv")

    write_summary_log(
        summary_path=summary_path,
        total_videos=len(input_videos),
        results=results,
        settings=run_settings,
        total_elapsed=total_elapsed,
    )

    write_manifest_csv(
        manifest_path=manifest_path,
        results=results
    )

    print("\\n" + "=" * 60)
    for r in results:
        if r["status"] == "success":
            print(f"[SUCCESS] {r['video_name']} - {r['saved_count']}장 저장")
        elif r["status"] == "skipped":
            print(f"[SKIPPED] {r['video_name']} - {r['message']}")
        elif r["status"] == "no_frames":
            print(f"[NO_FRAMES] {r['video_name']} - {r['message']}")
        else:
            print(f"[FAILED] {r['video_name']} - {r['message']}")
    print("=" * 60)
    print(f"✨ 전체 총 소요 시간: {total_elapsed:.2f}초")
    if len(input_videos) > 0:
        print(f"📈 평균 처리 속도: {total_elapsed / len(input_videos):.2f}초/영상")
    print(f"📝 summary 저장 완료: {summary_path}")
    print(f"🧾 manifest 저장 완료: {manifest_path}")
    print("=" * 60)
