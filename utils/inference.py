import cv2
import numpy as np
from ultralytics import YOLO
from PIL import Image, ImageDraw, ImageFont
import os

def rects_overlap(rect1, rect2):
    # rect: (x, y, w, h)
    x1, y1, w1, h1 = rect1
    x2, y2, w2, h2 = rect2
    return not (x1 + w1 < x2 or x2 + w2 < x1 or y1 + h1 < y2 or y2 + h2 < y1)

def get_font(font_size=24):
    arial_path = r'C:\Windows\Fonts\arial.ttf'
    times_path = r'C:\Windows\Fonts\times.ttf'
    if os.path.exists(arial_path):
        return ImageFont.truetype(arial_path, font_size)
    elif os.path.exists(times_path):
        return ImageFont.truetype(times_path, font_size)
    else:
        return ImageFont.load_default()

def draw_text_unicode(img, text, position, color=(255,255,255), font_size=24, used_rects=None):
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
    rect = (x, y, text_w, text_h)
    # Tránh ghi đè chữ
    if used_rects is not None:
        while any(rects_overlap(rect, r) for r in used_rects):
            y += text_h + 2
            rect = (x, y, text_w, text_h)
        used_rects.append(rect)
    color_rgb = (color[2], color[1], color[0])
    outline_range = 2
    for dx in range(-outline_range, outline_range+1):
        for dy in range(-outline_range, outline_range+1):
            draw.text((x+dx, y+dy), text, font=font, fill=(0,0,0))
    draw.text((x, y), text, font=font, fill=color_rgb)
    return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

def process_image(image_path, model_path, class_names, class_names_full, conf_threshold=0.5):
    model = YOLO(model_path)
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Không thể đọc ảnh từ {image_path}")
    results = model(img, conf=conf_threshold)
    colors = [
        (0, 255, 0),   # Xanh lá
        (0, 0, 255),   # Đỏ
        (255, 0, 0),   # Xanh dương
        (0, 255, 255), # Vàng
        (255, 255, 255), # Trắng
        (0, 165, 255), # Cam
        (128, 0, 128)  # Tím
    ]
    detected_codes = []
    used_rects = []
    for result in results:
        for idx, box in enumerate(result.boxes):
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = box.conf[0]
            class_id = int(box.cls[0])
            code = class_names[class_id]
            detected_codes.append(code)
            label = f"{code}: {class_names_full.get(code, code)} {conf:.2f}"
            color = colors[idx % len(colors)]
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
            # --- FIX vị trí text ---
            label_y = y1 - 30
            if label_y < 0:   # nếu tràn ra ngoài ảnh
                label_y = y1 + 30   # vẽ xuống dưới
            img = draw_text_unicode(img, label, (x1, y1-30), color=color, used_rects=used_rects)
            return img, detected_codes

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
    Xử lý video và trả về đường dẫn video đã xử lý
    """
    from utils.video_processing import process_video as process_video_func
    return process_video_func(
        video_path=video_path,
        model_path=model_path,
        class_names=class_names,
        output_path=output_path,
        conf_threshold=conf_threshold,
        stop_flag=stop_flag
    )