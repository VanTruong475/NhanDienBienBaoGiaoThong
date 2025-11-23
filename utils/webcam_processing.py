import cv2
import numpy as np
from ultralytics import YOLO
import threading
from PIL import Image, ImageDraw
import os
from datetime import datetime
from utils.model_cache import get_optimized_model, get_inference_config
from utils.font_utils import get_cached_font

def get_font(font_size=24):
    """Get cross-platform font"""
    return get_cached_font(font_size)

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
    def __init__(self, model_path, class_names, class_names_full, conf_threshold=0.5, 
                 skip_frames=2, inference_size=640):
        """
        Khởi tạo xử lý webcam
        
        Args:
            model_path: Đường dẫn đến model YOLO
            class_names: Danh sách tên class
            class_names_full: Dictionary ánh xạ mã -> tên đầy đủ
            conf_threshold: Ngưỡng confidence
            skip_frames: Số frames bỏ qua giữa mỗi lần inference (mặc định 2)
            inference_size: Kích thước ảnh cho inference (mặc định 640)
        """
        # Use cached and optimized model
        self.model = get_optimized_model(model_path)
        self.inference_config = get_inference_config()
        self.class_names = class_names
        self.class_names_full = class_names_full
        self.conf_threshold = conf_threshold
        self.stop_flag = threading.Event()
        self.cap = None
        
        # Performance optimization
        self.skip_frames = skip_frames
        self.inference_size = inference_size
        self.frame_count = 0
        self.last_detections = []  # Cache kết quả detection cuối cùng
        
        # Recording
        self.is_recording = False
        self.video_writer = None
        self.recording_path = None
        
        # Capture
        self.captured_frames = []
        
        # Cache font để không load lại mỗi lần
        self._font_cache = {}
        
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
        try:
            self.cap = cv2.VideoCapture(camera_id)
            if not self.cap.isOpened():
                raise ValueError(f"Không thể mở webcam {camera_id}")
            
            # Kiểm tra có đọc được frame không
            ret, test_frame = self.cap.read()
            if not ret:
                self.cap.release()
                raise ValueError(f"Không thể đọc frame từ webcam {camera_id}")
            
            self.stop_flag.clear()
            return True
        except Exception as e:
            if self.cap is not None:
                self.cap.release()
                self.cap = None
            raise e
    
    def process_frame(self, frame):
        """
        Xử lý một frame và trả về frame đã được vẽ kết quả
        
        Args:
            frame: Frame từ webcam
            
        Returns:
            processed_frame: Frame đã được vẽ bounding box và label
            detected_signs: List các biển báo được phát hiện
        """
        self.frame_count += 1
        original_frame = frame.copy()
        
        # Chỉ chạy inference mỗi N frames để tăng FPS
        should_inference = (self.frame_count % (self.skip_frames + 1)) == 0
        
        if should_inference:
            # Lấy kích thước gốc
            original_h, original_w = frame.shape[:2]
            
            # Xử lý ảnh grayscale (trắng đen) - convert sang BGR nếu cần
            if len(frame.shape) == 2:  # Grayscale (1 channel)
                frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
            elif frame.shape[2] == 4:  # RGBA -> BGR
                frame = cv2.cvtColor(frame, cv2.COLOR_RGBA2BGR)
            
            # Resize nhỏ hơn cho inference nếu ảnh quá lớn
            if max(original_w, original_h) > self.inference_size:
                scale = self.inference_size / max(original_w, original_h)
                new_w = int(original_w * scale)
                new_h = int(original_h * scale)
                frame_resized = cv2.resize(frame, (new_w, new_h))
            else:
                frame_resized = frame
                scale = 1.0
            
            # Suy luận với YOLO trên ảnh đã resize (with optimizations)
            results = self.model(frame_resized, conf=self.conf_threshold, **self.inference_config)
            
            detected_signs = []
            detection_boxes = []  # Lưu thông tin boxes để vẽ lại
            
            # Xử lý kết quả
            for result in results:
                for idx, box in enumerate(result.boxes):
                    # Lấy thông tin box (scale về kích thước gốc)
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    x1, y1, x2, y2 = int(x1/scale), int(y1/scale), int(x2/scale), int(y2/scale)
                    
                    conf = float(box.conf[0].cpu().numpy())
                    class_id = int(box.cls[0].cpu().numpy())
                    
                    # Kiểm tra class_id có hợp lệ không
                    if class_id >= len(self.class_names):
                        print(f"⚠️  Warning: class_id {class_id} vượt quá số lượng classes ({len(self.class_names)}). Bỏ qua detection này.")
                        continue
                    
                    code = self.class_names[class_id]
                    
                    # Lưu thông tin biển báo phát hiện
                    sign_info = {
                        'code': code,
                        'name': self.class_names_full.get(code, code),
                        'confidence': conf
                    }
                    detected_signs.append(sign_info)
                    
                    # Lưu box info để vẽ lại cho các frame sau
                    detection_boxes.append({
                        'box': (x1, y1, x2, y2),
                        'class_id': class_id,
                        'code': code,
                        'conf': conf,
                        'label': f"{code}: {self.class_names_full.get(code, code)} {conf:.2f}"
                    })
            
            # Cache kết quả
            self.last_detections = detection_boxes
        else:
            # Sử dụng kết quả detection từ frame trước (không chạy inference)
            detected_signs = [
                {
                    'code': det['code'],
                    'name': self.class_names_full.get(det['code'], det['code']),
                    'confidence': det['conf']
                }
                for det in self.last_detections
            ]
        
        # Vẽ bounding boxes (dù inference hay không)
        for det in self.last_detections:
            x1, y1, x2, y2 = det['box']
            color = self.colors[det['class_id'] % len(self.colors)]
            
            # Vẽ bounding box
            cv2.rectangle(original_frame, (x1, y1), (x2, y2), color, 2)
            
            # Vẽ label
            label_y = y1 - 10
            if label_y < 0:
                label_y = y2 + 20
            
            original_frame = draw_text_unicode(original_frame, det['label'], (x1, label_y), 
                                             color=color, font_size=16)
        
        return original_frame, detected_signs
    
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
        
        # Giải phóng video writer
        if self.video_writer is not None:
            try:
                if self.video_writer.isOpened():
                    self.video_writer.release()
                    print("✅ Đã giải phóng video writer")
            except Exception as e:
                print(f"Lỗi khi giải phóng video writer: {e}")
            finally:
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
        try:
            # Dừng recording nếu đang record
            if self.is_recording:
                self.stop_recording()
            
            self.stop_flag.set()
            
            # Giải phóng capture
            if self.cap is not None:
                try:
                    if self.cap.isOpened():
                        self.cap.release()
                        print("✅ Đã giải phóng webcam capture")
                except Exception as e:
                    print(f"Lỗi khi giải phóng webcam: {e}")
                finally:
                    self.cap = None
        except Exception as e:
            print(f"Lỗi khi dừng webcam: {e}")
    
    def is_stopped(self):
        """Kiểm tra đã dừng chưa"""
        return self.stop_flag.is_set()
    
    def set_skip_frames(self, skip_frames):
        """
        Điều chỉnh số frames bỏ qua
        
        Args:
            skip_frames: Số frames bỏ qua (0 = xử lý mọi frame, 1 = xử lý mỗi frame thứ 2, v.v.)
        """
        self.skip_frames = max(0, skip_frames)
    
    def set_inference_size(self, size):
        """
        Điều chỉnh kích thước inference
        
        Args:
            size: Kích thước tối đa (320, 416, 640, 1280)
        """
        self.inference_size = size
    
    def get_performance_stats(self):
        """
        Lấy thông tin hiệu năng
        
        Returns:
            dict: Thông tin về skip_frames, inference_size, frame_count
        """
        return {
            'skip_frames': self.skip_frames,
            'inference_size': self.inference_size,
            'frame_count': self.frame_count,
            'inference_rate': f"1/{self.skip_frames + 1} frames"
        }
    
    def __del__(self):
        """Cleanup khi object bị xóa"""
        try:
            self.stop()
        except:
            pass  # Ignore errors during cleanup

