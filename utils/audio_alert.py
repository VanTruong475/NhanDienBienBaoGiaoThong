import threading
import time
try:
    import winsound  # Windows only
    WINDOWS_AUDIO = True
except ImportError:
    WINDOWS_AUDIO = False

class AudioAlert:
    """Phát cảnh báo âm thanh khi phát hiện biển báo"""
    
    def __init__(self):
        self.enabled = True
        self.last_alert_time = {}
        self.alert_cooldown = 3  # Giây - tránh spam âm thanh
        
        # Định nghĩa các biển báo quan trọng cần cảnh báo
        self.critical_signs = {
            'P.102': 'Cấm đi ngược chiều',
            'P.125': 'Cấm vượt',
            'P.127': 'Tốc độ tối đa',
            'P.130': 'Cấm dừng và đỗ xe',
            'W.224': 'Người đi bộ',
            'W.225': 'Trẻ em',
            'W.227': 'Công trường',
            'W.233': 'Nguy hiểm'
        }
        
        # Âm thanh cho từng mức độ
        self.alert_sounds = {
            'critical': (1000, 200),  # Tần số cao, ngắn - nguy hiểm
            'warning': (800, 150),     # Tần số trung bình - cảnh báo
            'info': (600, 100)         # Tần số thấp - thông tin
        }
    
    def should_alert(self, sign_code):
        """
        Kiểm tra có nên phát cảnh báo cho biển này không
        
        Args:
            sign_code: Mã biển báo
            
        Returns:
            bool: True nếu nên phát cảnh báo
        """
        if not self.enabled:
            return False
        
        current_time = time.time()
        
        # Kiểm tra cooldown
        if sign_code in self.last_alert_time:
            if current_time - self.last_alert_time[sign_code] < self.alert_cooldown:
                return False
        
        # Cập nhật thời gian alert
        self.last_alert_time[sign_code] = current_time
        return True
    
    def get_alert_level(self, sign_code):
        """
        Xác định mức độ cảnh báo của biển
        
        Args:
            sign_code: Mã biển báo
            
        Returns:
            str: 'critical', 'warning', hoặc 'info'
        """
        if sign_code in self.critical_signs:
            return 'critical'
        elif sign_code.startswith('P.'):
            return 'warning'  # Biển cấm
        elif sign_code.startswith('W.'):
            return 'warning'  # Biển cảnh báo
        else:
            return 'info'     # Biển hiệu lệnh, chỉ dẫn
    
    def play_alert(self, sign_code, sign_name=None):
        """
        Phát âm thanh cảnh báo
        
        Args:
            sign_code: Mã biển báo
            sign_name: Tên biển báo (tùy chọn)
        """
        if not self.should_alert(sign_code):
            return
        
        # Xác định mức độ
        level = self.get_alert_level(sign_code)
        
        # Phát âm thanh trong thread riêng để không block
        thread = threading.Thread(
            target=self._play_sound_thread,
            args=(level,),
            daemon=True
        )
        thread.start()
    
    def _play_sound_thread(self, level):
        """Thread để phát âm thanh"""
        if not WINDOWS_AUDIO:
            print(f"\a")  # Fallback: system beep
            return
        
        try:
            frequency, duration = self.alert_sounds.get(level, self.alert_sounds['info'])
            
            # Phát âm thanh
            if level == 'critical':
                # Âm thanh nguy hiểm: 2 tiếng beep nhanh
                winsound.Beep(frequency, duration)
                time.sleep(0.05)
                winsound.Beep(frequency, duration)
            elif level == 'warning':
                # Âm thanh cảnh báo: 1 tiếng beep
                winsound.Beep(frequency, duration)
            else:
                # Âm thanh thông tin: beep ngắn
                winsound.Beep(frequency, duration)
                
        except Exception as e:
            print(f"Không thể phát âm thanh: {e}")
    
    def play_success_sound(self):
        """Phát âm thanh thành công (ví dụ: sau khi lưu ảnh)"""
        if not WINDOWS_AUDIO:
            return
        
        thread = threading.Thread(
            target=self._play_success_thread,
            daemon=True
        )
        thread.start()
    
    def _play_success_thread(self):
        """Thread phát âm thanh thành công"""
        try:
            # Âm thanh vui: tăng dần
            winsound.Beep(523, 100)  # C
            time.sleep(0.05)
            winsound.Beep(659, 100)  # E
            time.sleep(0.05)
            winsound.Beep(784, 150)  # G
        except:
            pass
    
    def enable(self):
        """Bật cảnh báo âm thanh"""
        self.enabled = True
    
    def disable(self):
        """Tắt cảnh báo âm thanh"""
        self.enabled = False
    
    def toggle(self):
        """Chuyển đổi bật/tắt"""
        self.enabled = not self.enabled
        return self.enabled
    
    def set_cooldown(self, seconds):
        """
        Đặt thời gian cooldown giữa các cảnh báo
        
        Args:
            seconds: Số giây
        """
        self.alert_cooldown = max(0, seconds)
    
    def clear_history(self):
        """Xóa lịch sử cảnh báo (reset cooldown)"""
        self.last_alert_time = {}

