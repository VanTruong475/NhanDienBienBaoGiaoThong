import cv2
import torch
import numpy as np
from ultralytics import YOLO
import time
import tempfile
import os
from utils.model_cache import get_optimized_model, get_inference_config


def process_video(
    video_path,
    model_path,
    class_names,
    class_names_full=None,
    output_path="output.mp4",
    conf_threshold=0.5,
    stop_flag=None
):
    """
    Xử lý video sử dụng YOLOv8 và trả về thống kê biển báo
    
    Returns:
        Generator yielding: (progress, detected_signs_summary)
        Final return: (output_path, statistics)
    """
    # Khởi tạo mô hình YOLOv8 với cache và optimization
    model = get_optimized_model(model_path)
    inference_config = get_inference_config()
    device = inference_config['device']
    print(f"Đang sử dụng thiết bị: {device}")

    # Mở video
    cap = cv2.VideoCapture(video_path)
    
    # Kiểm tra video có mở được không
    if not cap.isOpened():
        raise ValueError(f"Không thể mở video: {video_path}")
    
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Kiểm tra các thông số video hợp lệ
    if width <= 0 or height <= 0 or fps <= 0:
        cap.release()
        raise ValueError(f"Video có thông số không hợp lệ: {width}x{height} @ {fps}fps")

    # Tạo thư mục output nếu chưa tồn tại
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Khởi tạo VideoWriter
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    # Kiểm tra VideoWriter đã mở được không
    if not out.isOpened():
        cap.release()
        raise ValueError(f"Không thể tạo video output: {output_path}")

    frame_count = 0
    start_time = time.time()
    
    # Tracking biển báo đã phát hiện
    detected_signs = {}  # {code: {'name': ..., 'count': ..., 'max_conf': ..., 'frames': [...]}}
    all_detections = []  # Lưu tất cả detections theo thời gian

    # Màu sắc cho các bounding box
    colors = [
        (0, 255, 0),   # Xanh lá
        (0, 0, 255),   # Đỏ
        (255, 0, 0),   # Xanh dương
        (0, 255, 255), # Vàng
        (255, 255, 255), # Trắng
        (0, 165, 255), # Cam
        (128, 0, 128)  # Tím
    ]

    try:
        while True:
            # Kiểm tra nếu có yêu cầu dừng
            if stop_flag and stop_flag.is_set():
                print("Đã nhận lệnh dừng xử lý")
                break
            
            # Kiểm tra capture vẫn còn mở
            if not cap.isOpened():
                print("Video capture đã bị đóng")
                break

            ret, frame = cap.read()
            if not ret:
                break

            # Suy luận với YOLOv8 (optimized with half precision on GPU)
            results = model(frame, conf=conf_threshold, **inference_config)[0]
            
            frame_detections = []
            
            # Vẽ kết quả
            for box in results.boxes:
                # Lấy tọa độ
                x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
                # Lấy confidence và class
                conf = float(box.conf[0].cpu().numpy())
                class_id = int(box.cls[0].cpu().numpy())
                
                # Kiểm tra class_id có hợp lệ không
                if class_id >= len(class_names):
                    print(f"⚠️  Warning: class_id {class_id} vượt quá số lượng classes ({len(class_names)}). Bỏ qua detection này.")
                    continue
                
                code = class_names[class_id]
                sign_name = class_names_full.get(code, code) if class_names_full else code

                # Lưu thông tin detection
                frame_detections.append({
                    'code': code,
                    'name': sign_name,
                    'confidence': conf,
                    'frame': frame_count
                })
                
                # Cập nhật thống kê
                if code not in detected_signs:
                    detected_signs[code] = {
                        'name': sign_name,
                        'count': 0,
                        'max_conf': 0,
                        'frames': [],
                        'first_seen': frame_count,
                        'last_seen': frame_count
                    }
                
                detected_signs[code]['count'] += 1
                detected_signs[code]['max_conf'] = max(detected_signs[code]['max_conf'], conf)
                detected_signs[code]['frames'].append(frame_count)
                detected_signs[code]['last_seen'] = frame_count

                # Tạo label với tên đầy đủ
                label = f"{code}: {sign_name} {conf:.2f}"

                # Vẽ bounding box
                cv2.rectangle(frame, (x1, y1), (x2, y2), colors[class_id % len(colors)], 2)
                
                # Vẽ text với tên đầy đủ
                from utils.inference import draw_text_unicode
                frame = draw_text_unicode(frame, label, (x1, y1-30), color=colors[class_id % len(colors)])

            # Lưu detections của frame này
            if frame_detections:
                all_detections.extend(frame_detections)

            # Ghi frame đã xử lý vào file
            out.write(frame)

            frame_count += 1
            progress = (frame_count / total_frames) * 100
            
            # Trả về tiến độ và số biển báo unique đã tìm thấy
            yield (progress, len(detected_signs))

    except Exception as e:
        print(f"Lỗi trong quá trình xử lý video: {e}")
        raise
    finally:
        # Đảm bảo luôn giải phóng tài nguyên
        try:
            if cap is not None and cap.isOpened():
                cap.release()
                print("✅ Đã giải phóng video capture")
        except Exception as e:
            print(f"Lỗi khi giải phóng capture: {e}")
        
        try:
            if out is not None and out.isOpened():
                out.release()
                print("✅ Đã giải phóng video writer")
        except Exception as e:
            print(f"Lỗi khi giải phóng writer: {e}")
        
        processing_time = time.time() - start_time
        print(f"Thời gian xử lý: {processing_time:.2f} giây")
        print(f"Video đã xử lý được lưu tại: {output_path}")
        
        # Tạo thống kê chi tiết
        statistics = {
            'output_path': output_path,
            'total_frames': total_frames,
            'processed_frames': frame_count,
            'processing_time': processing_time,
            'fps': fps,
            'video_duration': total_frames / fps if fps > 0 else 0,
            'detected_signs': detected_signs,
            'total_detections': len(all_detections),
            'unique_signs': len(detected_signs),
            'all_detections': all_detections
        }
        
        # Yield statistics cuối cùng thay vì return
        yield statistics