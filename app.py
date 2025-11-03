import streamlit as st
import cv2
import os
import yaml
from utils.inference import process_image, process_video
from utils.webcam_processing import WebcamProcessor
from utils.database import TrafficSignDatabase
from utils.audio_alert import AudioAlert
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

def get_absolute_path(relative_path):
    """Lấy đường dẫn tuyệt đối"""
    return os.path.abspath(os.path.join(os.path.dirname(__file__), relative_path))

# Đọc cấu hình dataset
try:
    with open(get_absolute_path("./data/data.yaml"), "r") as f:
        data_config = yaml.safe_load(f)
    class_names = data_config["names"]
except FileNotFoundError:
    st.error("❌ Không tìm thấy file data.yaml. Vui lòng kiểm tra thư mục data/")
    st.stop()

# Dictionary ánh xạ mã biển báo sang tên đầy đủ
class_names_full = {
    "DP.135": "Hết tất cả các lệnh cấm",
    "P.102": "Cấm đi ngược chiều",
    "P.103a": "Cấm ô tô",
    "P.103b": "Cấm ô tô rẽ phải",
    "P.103c": "Cấm ô tô rẽ trái",
    "P.104": "Cấm mô tô",
    "P.106a": "Cấm xe tải",
    "P.106b": "Cấm xe tải trên 2,5 tấn",
    "P.107a": "Cấm ô tô khách và ô tô tải",
    "P.112": "Cấm người đi bộ",
    "P.115": "Hạn chế trọng lượng xe",
    "P.117": "Hạn chế chiều cao",
    "P.123a": "Cấm rẽ trái",
    "P.123b": "Cấm rẽ phải",
    "P.124a": "Cấm quay đầu",
    "P.124b": "Cấm ô tô quay đầu",
    "P.124c": "Cấm rẽ trái và quay đầu",
    "P.125": "Cấm vượt",
    "P.127": "Tốc độ tối đa cho phép",
    "P.128": "Cấm bóp còi",
    "P.130": "Cấm dừng và đỗ xe",
    "P.131a": "Cấm đỗ xe",
    "P.137": "Cấm đi thẳng và rẽ trái",
    "P.245a": "Cấm xe đạp",
    "R.301c": "Hướng đi thẳng phải theo",
    "R.301d": "Các xe chỉ được phép rẽ phải",
    "R.301e": "Các xe chỉ được phép rẽ trái",
    "R.302a": "Chỉ hướng đi phải theo vòng chướng ngại vật",
    "R.302b": "Chỉ hướng đi trái theo vòng chướng ngại vật",
    "R.303": "Giao nhau chạy theo vòng xuyến",
    "R.407a": "Đường 1 chiều",
    "R.409": "Chỗ quay xe",
    "R.425": "Bệnh viện",
    "R.434": "Bến xe buýt",
    "S.509a": "Chỗ đường sắt cắt đường bộ",
    "W.201a": "Chỗ ngặt nguy hiểm",
    "W.201b": "Chỗ ngặt nguy hiểm",
    "W.202a": "Nhiều chỗ ngoặt nguy hiểm liên tiếp",
    "W.202b": "Nhiều chỗ ngoặt nguy hiểm liên tiếp",
    "W.203b": "Đường bị hẹp bên trái",
    "W.203c": "Đường hẹp bên trái",
    "W.205a": "Đường hẹp bên phải",
    "W.205b": "Nơi giao nhau của đường cùng cấp",
    "W.205d": "Nơi giao nhau của đường cùng cấp",
    "W.207a": "Giao nhau với đường không ưu tiên",
    "W.207b": "Giao nhau với đường không ưu tiên",
    "W.207c": "Giao nhau với đường không ưu tiên",
    "W.208": "Giao nhau với đường ưu tiên",
    "W.209": "Giao nhau có tín hiệu đèn",
    "W.210": "Giao nhau với đường sắt có rào chắn",
    "W.219": "Dốc xuống nguy hiểm",
    "W.221b": "Đường không bằng phẳng",
    "W.224": "Người đi bộ cắt ngang",
    "W.225": "Trẻ em",
    "W.227": "Công trường",
    "W.233": "Nguy hiểm khắc",
    "W.235": "Đường đôi",
    "W.245a": "Chú ý chướng ngại vật phía trước"
}

# Khởi tạo Database và Audio Alert
if 'database' not in st.session_state:
    st.session_state.database = TrafficSignDatabase(get_absolute_path("data/traffic_signs.db"))

if 'audio_alert' not in st.session_state:
    st.session_state.audio_alert = AudioAlert()

db = st.session_state.database
audio = st.session_state.audio_alert

# ============================================
# SIDEBAR - CÀI ĐẶT
# ============================================

st.sidebar.header("⚙️ Cài đặt")

# Model settings
model_path = get_absolute_path("runs/train/exp/weights/best.pt")
if not os.path.exists(model_path):
    st.sidebar.error(f"❌ Không tìm thấy model!")
    st.stop()
else:
    st.sidebar.success("✅ Model đã sẵn sàng!")

conf_threshold = st.sidebar.slider("Ngưỡng độ tin cậy", 0.0, 1.0, 0.5, 0.05)

# Audio settings
st.sidebar.markdown("---")
st.sidebar.subheader("🔊 Cảnh Báo Âm Thanh")
audio_enabled = st.sidebar.checkbox("Bật cảnh báo âm thanh", value=audio.enabled)
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
            csv_path = db.export_to_csv(get_absolute_path("export_detections.csv"))
            with open(csv_path, 'rb') as f:
                st.download_button(
                    label="⬇️ Tải file CSV",
                    data=f.read(),
                    file_name=f"traffic_signs_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
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
        
        with col1:
            try:
                temp_image_path = get_absolute_path("temp_image.jpg")
                with open(temp_image_path, "wb") as f:
                    f.write(uploaded_image.read())
                
                with st.spinner("🔄 Đang xử lý hình ảnh..."):
                    result_img, detected_codes = process_image(
                        image_path=temp_image_path,
                        model_path=model_path,
                        class_names=class_names,
                        class_names_full=class_names_full,
                        conf_threshold=conf_threshold,
                    )
                
                st.image(result_img, channels="BGR", caption="Kết quả nhận diện", use_column_width=True)
                
                # Lưu vào database
                if detected_codes:
                    for code in detected_codes:
                        db.add_detection(
                            sign_code=code,
                            sign_name=class_names_full.get(code, code),
                            confidence=conf_threshold,
                            source_type="image"
                        )
                
                os.remove(temp_image_path)
                
            except Exception as e:
                st.error(f"❌ Lỗi khi xử lý hình ảnh: {str(e)}")
        
        with col2:
            if detected_codes:
                st.subheader("📋 Biển báo phát hiện:")
                for idx, code in enumerate(detected_codes, 1):
                    full_name = class_names_full.get(code, code)
                    st.success(f"**{idx}. {code}**\n\n{full_name}")
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
    
    uploaded_video = st.file_uploader(
        "Tải lên video của bạn",
        type=["mp4", "avi", "mov"],
        key="video_uploader"
    )
    
    if uploaded_video is not None:
        temp_video_path = get_absolute_path("temp_video.mp4")
        
        if not os.path.exists(temp_video_path) or st.session_state.get('last_video') != uploaded_video.name:
            with open(temp_video_path, "wb") as f:
                f.write(uploaded_video.read())
            st.session_state.last_video = uploaded_video.name
        
        st.subheader("📹 Video gốc:")
        st.video(uploaded_video)
        
        st.markdown("---")
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("▶️ Bắt đầu xử lý", disabled=st.session_state.video_processing, key="start_video_btn"):
                st.session_state.video_processing = True
                st.session_state.video_stop_flag.clear()
        
        with col2:
            if st.button("⏹️ Dừng xử lý", disabled=not st.session_state.video_processing, key="stop_video_btn"):
                st.session_state.video_stop_flag.set()
        
        if st.session_state.video_processing:
            status_placeholder = st.empty()
            progress_bar = st.progress(0)
            progress_text = st.empty()
            
            status_placeholder.info("⏳ Đang xử lý video...")
            
            output_path = get_absolute_path("output/output.mp4")
            
            try:
                for progress in process_video(
                    video_path=temp_video_path,
                    model_path=model_path,
                    class_names=class_names,
                    class_names_full=class_names_full,
                    output_path=output_path,
                    conf_threshold=conf_threshold,
                    stop_flag=st.session_state.video_stop_flag
                ):
                    progress_bar.progress(progress / 100)
                    progress_text.text(f"Tiến độ: {progress:.1f}%")
                
                st.session_state.video_processing = False
                
                if not st.session_state.video_stop_flag.is_set():
                    status_placeholder.success("✅ Xử lý video hoàn tất!")
                    
                    st.subheader("🎬 Video đã xử lý:")
                    if os.path.exists(output_path):
                        with open(output_path, "rb") as video_file:
                            video_bytes = video_file.read()
                            st.video(video_bytes)
                        
                        st.download_button(
                            label="⬇️ Tải xuống video",
                            data=video_bytes,
                            file_name="traffic_signs_detected.mp4",
                            mime="video/mp4"
                        )
                else:
                    status_placeholder.warning("⚠️ Đã dừng xử lý video!")
                    
            except Exception as e:
                st.error(f"❌ Lỗi khi xử lý video: {str(e)}")
                st.session_state.video_processing = False
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
                        conf_threshold=conf_threshold
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
    
    # Webcam stream processing
    if st.session_state.webcam_running and st.session_state.webcam_processor is not None:
        try:
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
                    filepath = st.session_state.webcam_processor.capture_frame(
                        processed_frame,
                        save_dir=get_absolute_path("captured_images")
                    )
                    st.session_state.captured_images.append(filepath)
                    st.success(f"📸 Đã chụp: {os.path.basename(filepath)}")
                    audio.play_success_sound()
                
                # Handle recording buttons
                if 'record_btn' in locals() and record_btn:
                    if st.session_state.webcam_processor.start_recording(
                        output_dir=get_absolute_path("recorded_videos")
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
                
                video_placeholder.image(processed_frame_rgb, channels="RGB", use_column_width=True)
                
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
                
                time.sleep(0.03)  # Reduce CPU usage
                
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
                    st.image(img_path, caption=os.path.basename(img_path), use_column_width=True)

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
