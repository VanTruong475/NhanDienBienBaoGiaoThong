import cv2
import torch
import numpy as np
from ultralytics import YOLO
import time
import tempfile
import os


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
    Xử lý video sử dụng YOLOv8
    """
    # Khởi tạo mô hình YOLOv8
    model = YOLO(model_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Đang sử dụng thiết bị: {device}")

    # Mở video
    cap = cv2.VideoCapture(video_path)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Tạo thư mục output nếu chưa tồn tại
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Khởi tạo VideoWriter
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    frame_count = 0
    start_time = time.time()

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
        while cap.isOpened():
            # Kiểm tra nếu có yêu cầu dừng
            if stop_flag and stop_flag.is_set():
                print("Đã nhận lệnh dừng xử lý")
                break

            ret, frame = cap.read()
            if not ret:
                break

            # Suy luận với YOLOv8
            results = model(frame, conf=conf_threshold)[0]
            
            # Vẽ kết quả
            for box in results.boxes:
                # Lấy tọa độ
                x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
                # Lấy confidence và class
                conf = float(box.conf[0].cpu().numpy())
                class_id = int(box.cls[0].cpu().numpy())
                code = class_names[class_id]

                # Tạo label với tên đầy đủ
                if class_names_full and code in class_names_full:
                    label = f"{code}: {class_names_full[code]} {conf:.2f}"
                else:
                    label = f"{code} {conf:.2f}"

                # Vẽ bounding box
                cv2.rectangle(frame, (x1, y1), (x2, y2), colors[class_id % len(colors)], 2)
                
                # Vẽ text với tên đầy đủ
                from utils.inference import draw_text_unicode
                frame = draw_text_unicode(frame, label, (x1, y1-30), color=colors[class_id % len(colors)])

            # Ghi frame đã xử lý vào file
            out.write(frame)

            frame_count += 1
            progress = (frame_count / total_frames) * 100
            print(f"Đã xử lý frame {frame_count}/{total_frames} ({progress:.1f}%)")

            # Trả về tiến độ xử lý
            yield progress

    finally:
        cap.release()
        out.release()
        print(f"Thời gian xử lý: {time.time() - start_time:.2f} giây")
        print(f"Video đã xử lý được lưu tại: {output_path}")
        
        # Trả về đường dẫn file video đã xử lý
        return output_path