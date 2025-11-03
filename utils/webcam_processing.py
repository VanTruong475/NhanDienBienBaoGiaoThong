import cv2
import numpy as np
from ultralytics import YOLO
import threading
from PIL import Image, ImageDraw, ImageFont
import os
from datetime import datetime

def get_font(font_size=24):
    arial_path = r'C:\Windows\Fonts\arial.ttf'
    times_path = r'C:\Windows\Fonts\times.ttf'
    if os.path.exists(arial_path):
        return ImageFont.truetype(arial_path, font_size)
    elif os.path.exists(times_path):
        return ImageFont.truetype(times_path, font_size)
    else:
        return ImageFont.load_default()

def draw_text_unicode(img, text, position, color=(255,255,255), font_size=20):
    """Vẽ text Unicode lên ảnh"""
    img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img_pil)
    font = get_font(font_size)
    x, y = position
    
    # Lấy kích thước vùng chữ
    bbox = draw.textbbox((x, y), text, font=font)
    text_w = bbox[2] - bbox[0]
    text_h = bbox[3] - bbox[1]
    
    # Điều chỉnh vị trí để tránh hiển thị ra ngoài khung ảnh
    img_h, img_w = img.shape[:2]
    if x + text_w > img_w:
        x = img_w - text_w
    if y + text_h > img_h:
        y = img_h - text_h
    if y < 0:
        y = 0
    
    # Vẽ outline đen
    color_rgb = (color[2], color[1], color[0])
    outline_range = 2
    for dx in range(-outline_range, outline_range+1):
        for dy in range(-outline_range, outline_range+1):
            draw.text((x+dx, y+dy), text, font=font, fill=(0,0,0))
    
    # Vẽ text chính
    draw.text((x, y), text, font=font, fill=color_rgb)
    return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)


class WebcamProcessor:
    def __init__(self, model_path, class_names, class_names_full, conf_threshold=0.5):
        """
        Khởi tạo xử lý webcam
        
        Args:
            model_path: Đường dẫn đến model YOLO
            class_names: Danh sách tên class
            class_names_full: Dictionary ánh xạ mã -> tên đầy đủ
            conf_threshold: Ngưỡng confidence
        """
        self.model = YOLO(model_path)
        self.class_names = class_names
        self.class_names_full = class_names_full
        self.conf_threshold = conf_threshold
        self.stop_flag = threading.Event()
        self.cap = None
        
        # Recording
        self.is_recording = False
        self.video_writer = None
        self.recording_path = None
        
        # Capture
        self.captured_frames = []
        
        # Màu sắc cho bounding box
        self.colors = [
            (0, 255, 0),   # Xanh lá
            (0, 0, 255),   # Đỏ
            (255, 0, 0),   # Xanh dương
            (0, 255, 255), # Vàng
            (255, 255, 255), # Trắng
            (0, 165, 255), # Cam
            (128, 0, 128)  # Tím
        ]
    
    def start(self, camera_id=0):
        """Bắt đầu capture từ webcam"""
        self.cap = cv2.VideoCapture(camera_id)
        if not self.cap.isOpened():
            raise ValueError(f"Không thể mở webcam {camera_id}")
        self.stop_flag.clear()
        return True
    
    def process_frame(self, frame):
        """
        Xử lý một frame và trả về frame đã được vẽ kết quả
        
        Args:
            frame: Frame từ webcam
            
        Returns:
            processed_frame: Frame đã được vẽ bounding box và label
            detected_signs: List các biển báo được phát hiện
        """
        # Suy luận với YOLO
        results = self.model(frame, conf=self.conf_threshold, verbose=False)
        
        detected_signs = []
        
        # Vẽ kết quả lên frame
        for result in results:
            for idx, box in enumerate(result.boxes):
                # Lấy thông tin box
                x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
                conf = float(box.conf[0].cpu().numpy())
                class_id = int(box.cls[0].cpu().numpy())
                code = self.class_names[class_id]
                
                # Lưu thông tin biển báo phát hiện
                detected_signs.append({
                    'code': code,
                    'name': self.class_names_full.get(code, code),
                    'confidence': conf
                })
                
                # Tạo label
                if self.class_names_full and code in self.class_names_full:
                    label = f"{code}: {self.class_names_full[code]} {conf:.2f}"
                else:
                    label = f"{code} {conf:.2f}"
                
                # Vẽ bounding box
                color = self.colors[class_id % len(self.colors)]
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                
                # Vẽ label
                label_y = y1 - 10
                if label_y < 0:
                    label_y = y2 + 20
                    
                frame = draw_text_unicode(frame, label, (x1, label_y), color=color, font_size=16)
        
        return frame, detected_signs
    
    def read_frame(self):
        """Đọc frame từ webcam"""
        if self.cap is None or not self.cap.isOpened():
            return None
        
        ret, frame = self.cap.read()
        if ret:
            return frame
        return None
    
    def capture_frame(self, frame, save_dir="captured_images"):
        """
        Chụp và lưu frame hiện tại
        
        Args:
            frame: Frame cần lưu
            save_dir: Thư mục lưu ảnh
            
        Returns:
            path: Đường dẫn file đã lưu
        """
        # Tạo thư mục nếu chưa có
        os.makedirs(save_dir, exist_ok=True)
        
        # Tạo tên file với timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"capture_{timestamp}.jpg"
        filepath = os.path.join(save_dir, filename)
        
        # Lưu ảnh
        cv2.imwrite(filepath, frame)
        
        # Lưu vào list captured frames
        self.captured_frames.append({
            'path': filepath,
            'timestamp': timestamp
        })
        
        return filepath
    
    def start_recording(self, output_dir="recorded_videos"):
        """
        Bắt đầu ghi video
        
        Args:
            output_dir: Thư mục lưu video
            
        Returns:
            bool: True nếu thành công
        """
        if self.is_recording:
            return False
        
        if self.cap is None or not self.cap.isOpened():
            return False
        
        # Tạo thư mục nếu chưa có
        os.makedirs(output_dir, exist_ok=True)
        
        # Lấy thông tin video
        width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(self.cap.get(cv2.CAP_PROP_FPS))
        if fps == 0:
            fps = 30  # Default FPS
        
        # Tạo tên file với timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"recording_{timestamp}.mp4"
        self.recording_path = os.path.join(output_dir, filename)
        
        # Khởi tạo VideoWriter
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        self.video_writer = cv2.VideoWriter(self.recording_path, fourcc, fps, (width, height))
        
        if self.video_writer.isOpened():
            self.is_recording = True
            return True
        else:
            self.video_writer = None
            return False
    
    def write_frame(self, frame):
        """
        Ghi frame vào video (nếu đang recording)
        
        Args:
            frame: Frame cần ghi
        """
        if self.is_recording and self.video_writer is not None:
            self.video_writer.write(frame)
    
    def stop_recording(self):
        """
        Dừng ghi video
        
        Returns:
            path: Đường dẫn file video đã lưu
        """
        if not self.is_recording:
            return None
        
        self.is_recording = False
        
        if self.video_writer is not None:
            self.video_writer.release()
            self.video_writer = None
        
        path = self.recording_path
        self.recording_path = None
        
        return path
    
    def get_captured_frames(self):
        """Lấy danh sách các frame đã chụp"""
        return self.captured_frames
    
    def clear_captured_frames(self):
        """Xóa danh sách frame đã chụp"""
        self.captured_frames = []
    
    def stop(self):
        """Dừng capture webcam"""
        # Dừng recording nếu đang record
        if self.is_recording:
            self.stop_recording()
        
        self.stop_flag.set()
        if self.cap is not None:
            self.cap.release()
            self.cap = None
    
    def is_stopped(self):
        """Kiểm tra đã dừng chưa"""
        return self.stop_flag.is_set()
    
    def __del__(self):
        """Cleanup khi object bị xóa"""
        self.stop()

