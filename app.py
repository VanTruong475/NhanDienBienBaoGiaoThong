import streamlit as st
import cv2
import os
import yaml
import json
from utils.inference import process_image, process_video
from utils.webcam_processing import WebcamProcessor
from utils.database import TrafficSignDatabase
from utils.audio_alert import AudioAlert
from utils.path_utils import get_absolute_path, get_data_path, get_model_path, get_output_path, get_temp_path, ensure_dir
import time
import threading
import numpy as np
import pandas as pd
from datetime import datetime

st.set_page_config(
    page_title="Traffic Sign Recognition System",
    page_icon="🚦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main-title {
        text-align: center;
        color: #1E88E5;
        font-size: 3rem;
        font-weight: bold;
        margin-bottom: 1rem;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 20px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        padding-left: 20px;
        padding-right: 20px;
        font-size: 16px;
        font-weight: 600;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 20px;
        border-radius: 10px;
        color: white;
        text-align: center;
    }
    .metric-value {
        font-size: 2.5rem;
        font-weight: bold;
        margin: 10px 0;
    }
    .metric-label {
        font-size: 1rem;
        opacity: 0.9;
    }
    </style>
    """, unsafe_allow_html=True)

st.markdown('<h1 class="main-title">🚦 Hệ Thống Nhận Diện Biển Báo Giao Thông 🚦</h1>', unsafe_allow_html=True)

# ============================================
# KHỞI TẠO & CẤU HÌNH
# ============================================

# Note: get_absolute_path is now imported from utils.path_utils

# Đọc cấu hình dataset
try:
    with open(get_data_path("data.yaml"), "r", encoding="utf-8") as f:
        data_config = yaml.safe_load(f)
    class_names = data_config["names"]
except FileNotFoundError:
    st.error("❌ Không tìm thấy file data.yaml. Vui lòng kiểm tra thư mục data/")
    st.stop()

# Đọc tên đầy đủ biển báo từ JSON file
try:
    with open(get_data_path("sign_labels_vi.json"), "r", encoding="utf-8") as f:
        class_names_full = json.load(f)
except FileNotFoundError:
    st.error("❌ Không tìm thấy file sign_labels_vi.json. Vui lòng kiểm tra thư mục data/")
    st.stop()

# Khởi tạo Database và Audio Alert
if 'database' not in st.session_state:
    st.session_state.database = TrafficSignDatabase(get_data_path("traffic_signs.db"))

if 'audio_alert' not in st.session_state:
    # Initialize with audio disabled by default (user can enable via checkbox)
    st.session_state.audio_alert = AudioAlert(enabled=False)

db = st.session_state.database
audio = st.session_state.audio_alert

# ============================================
# SIDEBAR - CÀI ĐẶT
# ============================================

st.sidebar.header("⚙️ Cài đặt")

# Model settings
model_path = get_model_path("best.pt")
if not os.path.exists(model_path):
    st.sidebar.error(f"❌ Không tìm thấy model tại: {model_path}")
    st.stop()
else:
    st.sidebar.success(f"✅ Model: {os.path.basename(model_path)}")

conf_threshold = st.sidebar.slider("Ngưỡng độ tin cậy", 0.0, 1.0, 0.5, 0.05)

# Performance settings
st.sidebar.markdown("---")
st.sidebar.subheader("⚡ Tối Ưu Hiệu Năng")
skip_frames = st.sidebar.slider(
    "Skip Frames (càng cao càng nhanh)", 
    0, 5, 2, 
    help="Số frames bỏ qua giữa mỗi lần nhận diện. 0=xử lý mọi frame, 2=xử lý mỗi frame thứ 3"
)
inference_size = st.sidebar.select_slider(
    "Kích thước Inference",
    options=[320, 416, 640, 1280],
    value=640,
    help="Kích thước ảnh cho inference. Nhỏ hơn = nhanh hơn nhưng ít chính xác hơn"
)

# Audio settings
st.sidebar.markdown("---")
st.sidebar.subheader("🔊 Cảnh Báo Âm Thanh")

# Show audio backend info
if audio.is_available():
    st.sidebar.info(f"🔊 Audio: {audio.get_audio_backend()}")
else:
    st.sidebar.warning("⚠️  Audio không khả dụng trên hệ thống này")

audio_enabled = st.sidebar.checkbox("Bật cảnh báo âm thanh", value=audio.enabled, 
                                    disabled=not audio.is_available())
if audio_enabled != audio.enabled:
    if audio_enabled:
        audio.enable()
    else:
        audio.disable()

if audio_enabled:
    cooldown = st.sidebar.slider("Thời gian giữa cảnh báo (giây)", 1, 10, 3)
    audio.set_cooldown(cooldown)

# Database settings
st.sidebar.markdown("---")
st.sidebar.subheader("💾 Cơ Sở Dữ Liệu")
stats = db.get_statistics()
st.sidebar.metric("Tổng phát hiện", stats['total_detections'])
st.sidebar.metric("Loại biển báo", stats['unique_signs'])

if st.sidebar.button("🗑️ Xóa lịch sử", help="Xóa toàn bộ dữ liệu"):
    db.clear_history()
    st.sidebar.success("✅ Đã xóa lịch sử!")
    time.sleep(1)
    st.rerun()

# ============================================
# TABS CHÍNH
# ============================================

tab1, tab2, tab3, tab4 = st.tabs([
    "📊 Dashboard",
    "📷 Hình Ảnh",
    "🎥 Video",
    "📹 Webcam"
])

# ============================================
# TAB 1: DASHBOARD THỐNG KÊ
# ============================================
with tab1:
    st.header("📊 Dashboard Thống Kê")
    st.markdown("---")
    
    # Metrics row
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            label="🎯 Tổng Phát Hiện",
            value=stats['total_detections'],
            delta=f"+{len(stats['by_date'])} (7 ngày)" if stats['by_date'] else None
        )
    
    with col2:
        st.metric(
            label="🚦 Loại Biển Báo",
            value=stats['unique_signs'],
            delta=f"/{len(class_names)} classes"
        )
    
    with col3:
        image_count = stats['by_source'].get('image', 0)
        st.metric(
            label="📷 Từ Hình Ảnh",
            value=image_count
        )
    
    with col4:
        webcam_count = stats['by_source'].get('webcam', 0)
        st.metric(
            label="📹 Từ Webcam",
            value=webcam_count
        )
    
    st.markdown("---")
    
    # Charts row
    col_left, col_right = st.columns([2, 1])
    
    with col_left:
        st.subheader("📈 Top 10 Biển Báo Phổ Biến Nhất")
        if stats['top_signs']:
            # Tạo dataframe
            df_top = pd.DataFrame(stats['top_signs'])
            df_top['label'] = df_top['code'] + ' - ' + df_top['name'].str[:30]
            
            # Bar chart
            st.bar_chart(df_top.set_index('label')['count'])
        else:
            st.info("Chưa có dữ liệu để hiển thị")
    
    with col_right:
        st.subheader("📊 Thống Kê Theo Loại")
        categories = db.get_statistics_by_category()
        
        cat_data = []
        for cat_code, cat_info in categories.items():
            if cat_info['count'] > 0:
                cat_data.append({
                    'Loại': cat_info['name'],
                    'Số lượng': cat_info['count']
                })
        
        if cat_data:
            df_cat = pd.DataFrame(cat_data)
            st.dataframe(df_cat, hide_index=True, use_container_width=True)
        else:
            st.info("Chưa có dữ liệu")
    
    st.markdown("---")
    
    # Recent detections
    st.subheader("🕒 Lịch Sử Phát Hiện Gần Đây")
    recent = db.get_recent_detections(20)
    
    if recent:
        df_recent = pd.DataFrame(recent)
        df_recent['timestamp'] = pd.to_datetime(df_recent['timestamp'])
        df_recent['confidence'] = df_recent['confidence'].apply(lambda x: f"{x:.2%}")
        
        # Rename columns
        df_recent = df_recent.rename(columns={
            'timestamp': 'Thời gian',
            'source_type': 'Nguồn',
            'sign_code': 'Mã',
            'sign_name': 'Tên biển báo',
            'confidence': 'Độ tin cậy'
        })
        
        st.dataframe(
            df_recent[['Thời gian', 'Nguồn', 'Mã', 'Tên biển báo', 'Độ tin cậy']],
            hide_index=True,
            use_container_width=True
        )
        
        # Export button
        if st.button("📥 Export dữ liệu ra CSV"):
            export_path = get_output_path(f"export_detections_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")
            csv_path = db.export_to_csv(export_path)
            with open(csv_path, 'rb') as f:
                st.download_button(
                    label="⬇️ Tải file CSV",
                    data=f.read(),
                    file_name=os.path.basename(export_path),
                    mime="text/csv"
                )
            st.success(f"✅ Đã export {len(recent)} records!")
    else:
        st.info("👆 Chưa có lịch sử phát hiện. Hãy bắt đầu nhận diện!")

# ============================================
# TAB 2: XỬ LÝ HÌNH ẢNH
# ============================================
with tab2:
    st.header("📷 Nhận Diện Biển Báo Từ Hình Ảnh")
    st.markdown("---")
    
    uploaded_image = st.file_uploader(
        "Tải lên hình ảnh của bạn",
        type=["jpg", "jpeg", "png"],
        key="image_uploader"
    )
    
    if uploaded_image is not None:
        col1, col2 = st.columns([2, 1])
        
        # Initialize variable
        detected_detections = []
        
        with col1:
            try:
                temp_image_path = get_temp_path("temp_image.jpg")
                with open(temp_image_path, "wb") as f:
                    f.write(uploaded_image.read())
                
                with st.spinner("🔄 Đang xử lý hình ảnh..."):
                    result_img, detected_detections = process_image(
                        image_path=temp_image_path,
                        model_path=model_path,
                        class_names=class_names,
                        class_names_full=class_names_full,
                        conf_threshold=conf_threshold,
                    )
                
                st.image(result_img, channels="BGR", caption="Kết quả nhận diện", width="stretch")
                
                # Lưu vào database với confidence thực tế từ YOLO
                if detected_detections:
                    for detection in detected_detections:
                        db.add_detection(
                            sign_code=detection['code'],
                            sign_name=class_names_full.get(detection['code'], detection['code']),
                            confidence=detection['confidence'],  # Use actual YOLO confidence
                            source_type="image"
                        )
                
                os.remove(temp_image_path)
                
            except Exception as e:
                st.error(f"❌ Lỗi khi xử lý hình ảnh: {str(e)}")
        
        with col2:
            if detected_detections:
                st.subheader("📋 Biển báo phát hiện:")
                for idx, detection in enumerate(detected_detections, 1):
                    code = detection['code']
                    conf = detection['confidence']
                    full_name = class_names_full.get(code, code)
                    st.success(f"**{idx}. {code}** (Conf: {conf:.2%})\n\n{full_name}")
            else:
                st.info("Không phát hiện biển báo nào")
    else:
        st.info("👆 Vui lòng tải lên hình ảnh để bắt đầu nhận diện")

# ============================================
# TAB 3: XỬ LÝ VIDEO
# ============================================
with tab3:
    st.header("🎥 Nhận Diện Biển Báo Từ Video")
    st.markdown("---")
    
    if 'video_processing' not in st.session_state:
        st.session_state.video_processing = False
    if 'video_stop_flag' not in st.session_state:
        st.session_state.video_stop_flag = threading.Event()
    if 'video_statistics' not in st.session_state:
        st.session_state.video_statistics = None
    
    uploaded_video = st.file_uploader(
        "Tải lên video của bạn",
        type=["mp4", "avi", "mov"],
        key="video_uploader"
    )
    
    if uploaded_video is not None:
        temp_video_path = get_temp_path("temp_video.mp4")
        
        if not os.path.exists(temp_video_path) or st.session_state.get('last_video') != uploaded_video.name:
            with open(temp_video_path, "wb") as f:
                f.write(uploaded_video.read())
            st.session_state.last_video = uploaded_video.name
            st.session_state.video_statistics = None  # Reset statistics
        
        # Layout 2 cột: Video bên trái (nhỏ hơn), Thông tin bên phải
        col_video, col_info = st.columns([1.5, 1])
        
        with col_video:
            st.subheader("📹 Video gốc")
            st.video(uploaded_video)
        
        with col_info:
            st.subheader("ℹ️ Thông tin Video")
            
            # Hiển thị thông tin cơ bản
            file_size = len(uploaded_video.getvalue()) / (1024 * 1024)  # MB
            st.metric("📦 Kích thước", f"{file_size:.1f} MB")
            st.metric("📄 Tên file", uploaded_video.name)
            
            # Instructions
            st.info("💡 **Hướng dẫn:**\n\n"
                   "1. Nhấn 'Bắt đầu xử lý'\n"
                   "2. Đợi quá trình hoàn tất\n"
                   "3. Xem kết quả bên dưới")
        
        st.markdown("---")
        
        # Control buttons
        col_btn1, col_btn2, col_btn3 = st.columns(3)
        with col_btn1:
            start_btn =             st.button(
                "▶️ Bắt đầu xử lý", 
                disabled=st.session_state.video_processing, 
                key="start_video_btn",
                width="stretch"
            )
        
        with col_btn2:
            stop_btn = st.button(
                "⏹️ Dừng xử lý", 
                disabled=not st.session_state.video_processing, 
                key="stop_video_btn",
                width="stretch"
            )
        
        with col_btn3:
            if st.session_state.video_statistics:
                clear_btn = st.button(
                    "🗑️ Xóa kết quả",
                    key="clear_results_btn",
                    width="stretch"
                )
                if clear_btn:
                    st.session_state.video_statistics = None
                    st.rerun()
        
        # Processing
        if start_btn:
            st.session_state.video_processing = True
            st.session_state.video_stop_flag.clear()
        
        if stop_btn:
            st.session_state.video_stop_flag.set()
        
        if st.session_state.video_processing:
            st.markdown("---")
            
            # Processing UI
            col_prog1, col_prog2 = st.columns([2, 1])
            
            with col_prog1:
                status_placeholder = st.empty()
                progress_bar = st.progress(0)
                progress_text = st.empty()
            
            with col_prog2:
                signs_counter = st.empty()
            
            status_placeholder.info("⏳ Đang xử lý video...")
            
            output_path = get_output_path("output.mp4")
            
            try:
                unique_signs_count = 0
                statistics = None
                
                # Tạo generator
                video_generator = process_video(
                    video_path=temp_video_path,
                    model_path=model_path,
                    class_names=class_names,
                    class_names_full=class_names_full,
                    output_path=output_path,
                    conf_threshold=conf_threshold,
                    stop_flag=st.session_state.video_stop_flag
                )
                
                # Lặp qua progress updates
                for progress_data in video_generator:
                    if isinstance(progress_data, tuple):
                        progress, unique_signs_count = progress_data
                        
                        progress_bar.progress(progress / 100)
                        progress_text.text(f"Tiến độ: {progress:.1f}%")
                        
                        with signs_counter.container():
                            st.metric("🚦 Biển báo tìm thấy", unique_signs_count)
                    elif isinstance(progress_data, dict):
                        # Đây là statistics cuối cùng
                        statistics = progress_data
                
                st.session_state.video_processing = False
                
                if not st.session_state.video_stop_flag.is_set():
                    status_placeholder.success("✅ Xử lý video hoàn tất!")
                    
                    # Lưu statistics
                    if statistics:
                        st.session_state.video_statistics = statistics
                    else:
                        # Fallback nếu không có statistics
                        st.session_state.video_statistics = {
                            'unique_signs': unique_signs_count,
                            'total_detections': 0,
                            'detected_signs': {},
                            'fps': 30,
                            'video_duration': 0,
                            'processing_time': 0
                        }
                    
                    st.rerun()
                else:
                    status_placeholder.warning("⚠️ Đã dừng xử lý video!")
                    
            except Exception as e:
                st.error(f"❌ Lỗi khi xử lý video: {str(e)}")
                import traceback
                st.error(traceback.format_exc())
                st.session_state.video_processing = False
        
        # Hiển thị kết quả sau khi xử lý xong
        if st.session_state.video_statistics and not st.session_state.video_processing:
            st.markdown("---")
            st.subheader("📊 Kết Quả Nhận Diện")
            
            stats = st.session_state.video_statistics
            
            # Metrics row
            metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)
            
            with metric_col1:
                st.metric(
                    "🚦 Loại biển báo",
                    stats.get('unique_signs', 0),
                    help="Số loại biển báo khác nhau"
                )
            
            with metric_col2:
                st.metric(
                    "🎯 Tổng phát hiện",
                    stats.get('total_detections', 0),
                    help="Tổng số lần phát hiện biển báo"
                )
            
            with metric_col3:
                duration = stats.get('video_duration', 0)
                st.metric(
                    "⏱️ Thời lượng",
                    f"{duration:.1f}s",
                    help="Độ dài video"
                )
            
            with metric_col4:
                proc_time = stats.get('processing_time', 0)
                st.metric(
                    "⚡ Xử lý",
                    f"{proc_time:.1f}s",
                    help="Thời gian xử lý"
                )
            
            st.markdown("---")
            
            # Video và danh sách biển báo
            result_col1, result_col2 = st.columns([1.5, 1])
            
            with result_col1:
                st.subheader("🎬 Video đã xử lý")
                output_path = get_output_path("output.mp4")
                if os.path.exists(output_path):
                    with open(output_path, "rb") as video_file:
                        video_bytes = video_file.read()
                        st.video(video_bytes)
                    
                    st.download_button(
                        label="⬇️ Tải xuống video",
                        data=video_bytes,
                        file_name=f"detected_{uploaded_video.name}",
                        mime="video/mp4",
                        width="stretch"
                    )
            
            with result_col2:
                st.subheader("📋 Biển báo phát hiện")
                
                detected_signs = stats.get('detected_signs', {})
                
                if detected_signs:
                    # Sắp xếp theo số lần xuất hiện
                    sorted_signs = sorted(
                        detected_signs.items(),
                        key=lambda x: x[1]['count'],
                        reverse=True
                    )
                    
                    for idx, (code, info) in enumerate(sorted_signs, 1):
                        # Tạo box hiển thị rõ ràng với tên đầy đủ
                        with st.container():
                            # Header với mã và tên
                            st.markdown(f"### {idx}. 🚦 **{code}**")
                            st.markdown(f"**{info['name']}**")
                            
                            # Thông tin chi tiết trong columns
                            col1, col2, col3 = st.columns(3)
                            with col1:
                                st.metric("Số lần", f"{info['count']} lần")
                            with col2:
                                st.metric("Độ tin cậy", f"{info['max_conf']:.1%}")
                            with col3:
                                first_time = info['first_seen'] / stats['fps']
                                last_time = info['last_seen'] / stats['fps']
                                duration = last_time - first_time
                                st.metric("Thời lượng", f"{duration:.1f}s")
                            
                            # Thông tin thời gian chi tiết
                            with st.expander("📊 Chi tiết thời gian", expanded=False):
                                st.write(f"**⏱️ Xuất hiện lần đầu:** {first_time:.1f}s")
                                st.write(f"**⏱️ Xuất hiện lần cuối:** {last_time:.1f}s")
                                st.write(f"**📏 Tổng thời lượng:** {duration:.1f} giây")
                                st.write(f"**📊 Số lần phát hiện:** {info['count']} lần")
                            
                            st.markdown("---")
                            
                            # Lưu vào database
                            db.add_detection(
                                sign_code=code,
                                sign_name=info['name'],
                                confidence=info['max_conf'],
                                source_type="video"
                            )
                else:
                    st.info("Không phát hiện biển báo nào trong video")
            
            # Chi tiết thống kê
            with st.expander("📈 Bảng Thống Kê Chi Tiết", expanded=False):
                if detected_signs:
                    # Sắp xếp theo số lần xuất hiện
                    sorted_for_table = sorted(
                        detected_signs.items(),
                        key=lambda x: x[1]['count'],
                        reverse=True
                    )
                    
                    df_stats = pd.DataFrame([
                        {
                            'STT': idx,
                            'Mã biển': code,
                            'Tên đầy đủ': info['name'],
                            'Số lần': info['count'],
                            'Độ tin cậy': f"{info['max_conf']:.1%}",
                            'Bắt đầu (s)': f"{info['first_seen']/stats['fps']:.1f}",
                            'Kết thúc (s)': f"{info['last_seen']/stats['fps']:.1f}",
                            'Thời lượng (s)': f"{(info['last_seen'] - info['first_seen'])/stats['fps']:.1f}"
                        }
                        for idx, (code, info) in enumerate(sorted_for_table, 1)
                    ])
                    
                    st.dataframe(
                        df_stats, 
                        use_container_width=True, 
                        hide_index=True,
                        height=400
                    )
                    
                    # Thêm tóm tắt
                    st.markdown("---")
                    col_sum1, col_sum2, col_sum3 = st.columns(3)
                    with col_sum1:
                        st.metric("📊 Tổng loại", len(detected_signs))
                    with col_sum2:
                        total_count = sum(info['count'] for info in detected_signs.values())
                        st.metric("🎯 Tổng phát hiện", total_count)
                    with col_sum3:
                        avg_conf = sum(info['max_conf'] for info in detected_signs.values()) / len(detected_signs)
                        st.metric("📈 Độ tin cậy TB", f"{avg_conf:.1%}")
                else:
                    st.info("Không có dữ liệu")
    
    else:
        st.info("👆 Vui lòng tải lên video để bắt đầu nhận diện")

# ============================================
# TAB 4: XỬ LÝ WEBCAM
# ============================================
with tab4:
    st.header("📹 Nhận Diện Biển Báo Từ Webcam")
    st.markdown("---")
    
    if 'webcam_running' not in st.session_state:
        st.session_state.webcam_running = False
    if 'webcam_processor' not in st.session_state:
        st.session_state.webcam_processor = None
    if 'captured_images' not in st.session_state:
        st.session_state.captured_images = []
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        video_placeholder = st.empty()
        
        # Control buttons
        col_btn1, col_btn2, col_btn3, col_btn4 = st.columns(4)
        
        with col_btn1:
            if st.button("📹 Bật Webcam", disabled=st.session_state.webcam_running, key="start_webcam_btn"):
                try:
                    st.session_state.webcam_processor = WebcamProcessor(
                        model_path=model_path,
                        class_names=class_names,
                        class_names_full=class_names_full,
                        conf_threshold=conf_threshold,
                        skip_frames=skip_frames,
                        inference_size=inference_size
                    )
                    
                    if st.session_state.webcam_processor.start(camera_id=0):
                        st.session_state.webcam_running = True
                        st.success("✅ Webcam đã được bật!")
                        st.rerun()
                except Exception as e:
                    st.error(f"❌ Không thể khởi động webcam: {str(e)}")
        
        with col_btn2:
            if st.button("⏹️ Tắt Webcam", disabled=not st.session_state.webcam_running, key="stop_webcam_btn"):
                if st.session_state.webcam_processor is not None:
                    st.session_state.webcam_processor.stop()
                st.session_state.webcam_running = False
                st.session_state.webcam_processor = None
                st.info("🛑 Webcam đã được tắt!")
                st.rerun()
        
        with col_btn3:
            capture_btn = st.button(
                "📸 Chụp Ảnh",
                disabled=not st.session_state.webcam_running,
                key="capture_btn"
            )
        
        with col_btn4:
            if not st.session_state.get('is_recording', False):
                record_btn = st.button(
                    "🔴 Bắt đầu Ghi",
                    disabled=not st.session_state.webcam_running,
                    key="start_record_btn"
                )
            else:
                stop_record_btn = st.button(
                    "⏹️ Dừng Ghi",
                    disabled=not st.session_state.webcam_running,
                    key="stop_record_btn"
                )
    
    with col2:
        detected_list_placeholder = st.empty()
        fps_placeholder = st.empty()
        perf_placeholder = st.empty()
    
    # Webcam stream processing
    if st.session_state.webcam_running and st.session_state.webcam_processor is not None:
        try:
            # Cập nhật settings realtime
            st.session_state.webcam_processor.set_skip_frames(skip_frames)
            st.session_state.webcam_processor.set_inference_size(inference_size)
            st.session_state.webcam_processor.conf_threshold = conf_threshold
            
            frame_count = 0
            start_time = time.time()
            current_frame = None
            
            while st.session_state.webcam_running:
                frame = st.session_state.webcam_processor.read_frame()
                
                if frame is None:
                    st.error("❌ Không thể đọc frame từ webcam!")
                    break
                
                # Process frame
                processed_frame, detected_signs = st.session_state.webcam_processor.process_frame(frame)
                current_frame = processed_frame.copy()
                
                # Write to video if recording
                if st.session_state.get('is_recording', False):
                    st.session_state.webcam_processor.write_frame(processed_frame)
                
                # Handle capture button
                if capture_btn:
                    capture_dir = ensure_dir(get_absolute_path("captured_images"))
                    filepath = st.session_state.webcam_processor.capture_frame(
                        processed_frame,
                        save_dir=str(capture_dir)
                    )
                    st.session_state.captured_images.append(filepath)
                    st.success(f"📸 Đã chụp: {os.path.basename(filepath)}")
                    audio.play_success_sound()
                
                # Handle recording buttons
                if 'record_btn' in locals() and record_btn:
                    record_dir = ensure_dir(get_absolute_path("recorded_videos"))
                    if st.session_state.webcam_processor.start_recording(
                        output_dir=str(record_dir)
                    ):
                        st.session_state.is_recording = True
                        st.success("🔴 Đang ghi video...")
                        st.rerun()
                
                if 'stop_record_btn' in locals() and stop_record_btn:
                    video_path = st.session_state.webcam_processor.stop_recording()
                    st.session_state.is_recording = False
                    if video_path:
                        st.success(f"✅ Đã lưu video: {os.path.basename(video_path)}")
                    st.rerun()
                
                # Display frame
                processed_frame_rgb = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
                
                # Add recording indicator
                if st.session_state.get('is_recording', False):
                    cv2.circle(processed_frame_rgb, (30, 30), 15, (255, 0, 0), -1)
                    cv2.putText(processed_frame_rgb, "REC", (50, 40), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
                
                video_placeholder.image(processed_frame_rgb, channels="RGB", width="stretch")
                
                # Display detected signs and play alerts
                with detected_list_placeholder.container():
                    if detected_signs:
                        st.subheader("🚦 Đang phát hiện:")
                        for idx, sign in enumerate(detected_signs, 1):
                            st.success(f"**{sign['code']}**\n{sign['name']}\n*Conf: {sign['confidence']:.2f}*")
                            
                            # Play audio alert
                            audio.play_alert(sign['code'], sign['name'])
                            
                            # Save to database
                            db.add_detection(
                                sign_code=sign['code'],
                                sign_name=sign['name'],
                                confidence=sign['confidence'],
                                source_type="webcam"
                            )
                    else:
                        st.info("Không phát hiện biển báo")
                
                # Calculate and display FPS
                frame_count += 1
                if frame_count % 10 == 0:
                    elapsed_time = time.time() - start_time
                    fps = frame_count / elapsed_time
                    fps_placeholder.metric("FPS", f"{fps:.1f}")
                    
                    # Hiển thị performance stats
                    perf_stats = st.session_state.webcam_processor.get_performance_stats()
                    with perf_placeholder.container():
                        st.caption("⚡ **Hiệu năng**")
                        st.text(f"Skip: {perf_stats['skip_frames']} frames")
                        st.text(f"Size: {perf_stats['inference_size']}px")
                        st.text(f"Rate: {perf_stats['inference_rate']}")
                
                time.sleep(0.01)  # Reduce CPU usage (giảm từ 0.03 xuống 0.01)
                
        except Exception as e:
            st.error(f"❌ Lỗi khi xử lý webcam: {str(e)}")
            if st.session_state.webcam_processor is not None:
                st.session_state.webcam_processor.stop()
            st.session_state.webcam_running = False
            st.session_state.webcam_processor = None
    else:
        video_placeholder.info("📹 Nhấn 'Bật Webcam' để bắt đầu nhận diện realtime")
    
    # Display captured images
    if st.session_state.captured_images:
        st.markdown("---")
        st.subheader("📸 Ảnh Đã Chụp")
        cols = st.columns(4)
        for idx, img_path in enumerate(st.session_state.captured_images[-8:]):  # Show last 8
            with cols[idx % 4]:
                if os.path.exists(img_path):
                    st.image(img_path, caption=os.path.basename(img_path), width="stretch")

# Footer
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: #666;'>
        <p>🚦 <b>Hệ Thống Nhận Diện Biển Báo Giao Thông Việt Nam</b> 🇻🇳</p>
        <p>Powered by YOLOv8 | 58 loại biển báo | Realtime Detection</p>
        <p style='font-size: 0.8rem; margin-top: 10px;'>
            ✨ Dashboard | 📸 Capture | 🎥 Recording | 🔊 Audio Alert | 💾 Database
        </p>
    </div>
    """,
    unsafe_allow_html=True
)
