import threading
import time
import sys
import os

# Cross-platform audio support
try:
    import winsound  # Windows only
    WINDOWS_AUDIO = True
except ImportError:
    WINDOWS_AUDIO = False

# Try to import pygame for cross-platform audio (optional)
try:
    import pygame
    PYGAME_AUDIO = True
except ImportError:
    PYGAME_AUDIO = False

class AudioAlert:
    """Phát cảnh báo âm thanh khi phát hiện biển báo"""
    
    def __init__(self, enabled=True):
        self.enabled = enabled
        self._audio_available = WINDOWS_AUDIO or PYGAME_AUDIO
        self.last_alert_time = {}
        self.alert_cooldown = 3  # Giây - tránh spam âm thanh
        
        # Initialize pygame mixer if available (for cross-platform)
        if PYGAME_AUDIO and not WINDOWS_AUDIO:
            try:
                pygame.mixer.init(frequency=22050, size=-16, channels=2, buffer=512)
            except Exception as e:
                print(f"⚠️  Could not initialize pygame audio: {e}")
                self._audio_available = False
        
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
        """Thread để phát âm thanh (cross-platform)"""
        if not self._audio_available:
            # Fallback: system beep (works on most terminals)
            try:
                print(f"\a", end='', flush=True)
            except:
                pass
            return
        
        try:
            frequency, duration = self.alert_sounds.get(level, self.alert_sounds['info'])
            
            if WINDOWS_AUDIO:
                # Windows winsound
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
            elif PYGAME_AUDIO:
                # Cross-platform pygame (generate beep programmatically)
                self._play_beep_pygame(frequency, duration, level)
            else:
                # Final fallback
                print(f"\a", end='', flush=True)
                
        except Exception as e:
            # Silently fail if audio disabled or error occurs
            pass
    
    def _play_beep_pygame(self, frequency, duration, level):
        """Generate and play beep using pygame (cross-platform)"""
        try:
            import numpy as np
            sample_rate = 22050
            duration_sec = duration / 1000.0
            
            # Generate sine wave
            t = np.linspace(0, duration_sec, int(sample_rate * duration_sec))
            wave = np.sin(2 * np.pi * frequency * t)
            
            # Apply envelope to avoid clicks
            envelope = np.ones_like(wave)
            fade_samples = int(0.01 * sample_rate)  # 10ms fade
            envelope[:fade_samples] = np.linspace(0, 1, fade_samples)
            envelope[-fade_samples:] = np.linspace(1, 0, fade_samples)
            wave = wave * envelope
            
            # Convert to 16-bit
            wave = (wave * 32767).astype(np.int16)
            
            # Create stereo
            stereo_wave = np.column_stack((wave, wave))
            
            # Play sound
            sound = pygame.sndarray.make_sound(stereo_wave)
            sound.play()
            
            if level == 'critical':
                time.sleep(0.05)
                sound.play()
            
            time.sleep(duration_sec)
        except Exception:
            # Fallback if numpy not available or error
            print(f"\a", end='', flush=True)
    
    def play_success_sound(self):
        """Phát âm thanh thành công (ví dụ: sau khi lưu ảnh)"""
        if not self.enabled or not self._audio_available:
            return
        
        thread = threading.Thread(
            target=self._play_success_thread,
            daemon=True
        )
        thread.start()
    
    def _play_success_thread(self):
        """Thread phát âm thanh thành công"""
        try:
            if WINDOWS_AUDIO:
                # Âm thanh vui: tăng dần
                winsound.Beep(523, 100)  # C
                time.sleep(0.05)
                winsound.Beep(659, 100)  # E
                time.sleep(0.05)
                winsound.Beep(784, 150)  # G
            elif PYGAME_AUDIO:
                # Use pygame for success sound
                self._play_beep_pygame(523, 100, 'info')
                time.sleep(0.05)
                self._play_beep_pygame(659, 100, 'info')
                time.sleep(0.05)
                self._play_beep_pygame(784, 150, 'info')
            else:
                print(f"\a", end='', flush=True)
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
    
    def is_available(self):
        """Check if audio is available on this system"""
        return self._audio_available
    
    def get_audio_backend(self):
        """Get current audio backend"""
        if WINDOWS_AUDIO:
            return "winsound (Windows)"
        elif PYGAME_AUDIO:
            return "pygame (cross-platform)"
        else:
            return "system beep (fallback)"
