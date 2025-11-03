import sqlite3
import os
from datetime import datetime
import json

class TrafficSignDatabase:
    """Database để lưu trữ lịch sử nhận diện biển báo"""
    
    def __init__(self, db_path="data/traffic_signs.db"):
        """Khởi tạo database"""
        self.db_path = db_path
        
        # Tạo thư mục nếu chưa tồn tại
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        
        # Khởi tạo database
        self._init_database()
    
    def _init_database(self):
        """Tạo các bảng trong database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Bảng lưu thông tin phát hiện
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS detections (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                source_type TEXT NOT NULL,
                sign_code TEXT NOT NULL,
                sign_name TEXT NOT NULL,
                confidence REAL NOT NULL,
                image_path TEXT,
                session_id TEXT
            )
        """)
        
        # Bảng lưu thông tin session
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS sessions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT UNIQUE NOT NULL,
                start_time DATETIME DEFAULT CURRENT_TIMESTAMP,
                end_time DATETIME,
                source_type TEXT NOT NULL,
                total_detections INTEGER DEFAULT 0
            )
        """)
        
        # Bảng thống kê theo biển báo
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS sign_statistics (
                sign_code TEXT PRIMARY KEY,
                sign_name TEXT NOT NULL,
                total_count INTEGER DEFAULT 0,
                last_seen DATETIME,
                avg_confidence REAL
            )
        """)
        
        conn.commit()
        conn.close()
    
    def add_detection(self, sign_code, sign_name, confidence, source_type="image", 
                     image_path=None, session_id=None):
        """
        Thêm một lần phát hiện vào database
        
        Args:
            sign_code: Mã biển báo (vd: P.102)
            sign_name: Tên đầy đủ biển báo
            confidence: Độ tin cậy (0-1)
            source_type: Loại nguồn (image/video/webcam)
            image_path: Đường dẫn ảnh (tùy chọn)
            session_id: ID của session (tùy chọn)
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Thêm detection
        cursor.execute("""
            INSERT INTO detections (sign_code, sign_name, confidence, source_type, image_path, session_id)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (sign_code, sign_name, confidence, source_type, image_path, session_id))
        
        # Cập nhật thống kê
        cursor.execute("""
            INSERT INTO sign_statistics (sign_code, sign_name, total_count, last_seen, avg_confidence)
            VALUES (?, ?, 1, datetime('now'), ?)
            ON CONFLICT(sign_code) DO UPDATE SET
                total_count = total_count + 1,
                last_seen = datetime('now'),
                avg_confidence = ((avg_confidence * (total_count - 1)) + ?) / total_count
        """, (sign_code, sign_name, confidence, confidence))
        
        conn.commit()
        conn.close()
    
    def add_batch_detections(self, detections, source_type="image", session_id=None):
        """
        Thêm nhiều detections cùng lúc
        
        Args:
            detections: List of dict [{'code': 'P.102', 'name': '...', 'confidence': 0.9}, ...]
            source_type: Loại nguồn
            session_id: ID session
        """
        for det in detections:
            self.add_detection(
                sign_code=det['code'],
                sign_name=det['name'],
                confidence=det['confidence'],
                source_type=source_type,
                session_id=session_id
            )
    
    def get_recent_detections(self, limit=50):
        """Lấy các phát hiện gần đây"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT id, timestamp, source_type, sign_code, sign_name, confidence
            FROM detections
            ORDER BY timestamp DESC
            LIMIT ?
        """, (limit,))
        
        results = cursor.fetchall()
        conn.close()
        
        return [
            {
                'id': r[0],
                'timestamp': r[1],
                'source_type': r[2],
                'sign_code': r[3],
                'sign_name': r[4],
                'confidence': r[5]
            }
            for r in results
        ]
    
    def get_statistics(self):
        """Lấy thống kê tổng quan"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Tổng số phát hiện
        cursor.execute("SELECT COUNT(*) FROM detections")
        total_detections = cursor.fetchone()[0]
        
        # Số loại biển báo khác nhau
        cursor.execute("SELECT COUNT(DISTINCT sign_code) FROM detections")
        unique_signs = cursor.fetchone()[0]
        
        # Top 10 biển báo phổ biến nhất
        cursor.execute("""
            SELECT sign_code, sign_name, total_count, avg_confidence
            FROM sign_statistics
            ORDER BY total_count DESC
            LIMIT 10
        """)
        top_signs = [
            {
                'code': r[0],
                'name': r[1],
                'count': r[2],
                'avg_conf': r[3]
            }
            for r in cursor.fetchall()
        ]
        
        # Thống kê theo nguồn
        cursor.execute("""
            SELECT source_type, COUNT(*) as count
            FROM detections
            GROUP BY source_type
        """)
        by_source = {r[0]: r[1] for r in cursor.fetchall()}
        
        # Thống kê theo ngày (7 ngày gần nhất)
        cursor.execute("""
            SELECT DATE(timestamp) as date, COUNT(*) as count
            FROM detections
            WHERE timestamp >= datetime('now', '-7 days')
            GROUP BY DATE(timestamp)
            ORDER BY date DESC
        """)
        by_date = [
            {'date': r[0], 'count': r[1]}
            for r in cursor.fetchall()
        ]
        
        conn.close()
        
        return {
            'total_detections': total_detections,
            'unique_signs': unique_signs,
            'top_signs': top_signs,
            'by_source': by_source,
            'by_date': by_date
        }
    
    def get_sign_details(self, sign_code):
        """Lấy thông tin chi tiết về một biển báo"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT sign_name, total_count, last_seen, avg_confidence
            FROM sign_statistics
            WHERE sign_code = ?
        """, (sign_code,))
        
        result = cursor.fetchone()
        conn.close()
        
        if result:
            return {
                'code': sign_code,
                'name': result[0],
                'total_count': result[1],
                'last_seen': result[2],
                'avg_confidence': result[3]
            }
        return None
    
    def clear_history(self):
        """Xóa toàn bộ lịch sử"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("DELETE FROM detections")
        cursor.execute("DELETE FROM sessions")
        cursor.execute("DELETE FROM sign_statistics")
        
        conn.commit()
        conn.close()
    
    def export_to_csv(self, output_path="export_detections.csv"):
        """Export dữ liệu ra CSV"""
        import csv
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT timestamp, source_type, sign_code, sign_name, confidence
            FROM detections
            ORDER BY timestamp DESC
        """)
        
        results = cursor.fetchall()
        conn.close()
        
        with open(output_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['Timestamp', 'Source', 'Code', 'Name', 'Confidence'])
            writer.writerows(results)
        
        return output_path
    
    def get_statistics_by_category(self):
        """Thống kê theo loại biển báo (P, R, W, S, DP)"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT sign_code, total_count
            FROM sign_statistics
        """)
        
        results = cursor.fetchall()
        conn.close()
        
        # Phân loại
        categories = {
            'P': {'name': 'Biển Cấm', 'count': 0, 'signs': []},
            'R': {'name': 'Biển Hiệu Lệnh', 'count': 0, 'signs': []},
            'W': {'name': 'Biển Cảnh Báo', 'count': 0, 'signs': []},
            'S': {'name': 'Biển Chỉ Dẫn', 'count': 0, 'signs': []},
            'DP': {'name': 'Hết Cấm', 'count': 0, 'signs': []}
        }
        
        for code, count in results:
            # Xác định category
            if code.startswith('DP'):
                cat = 'DP'
            else:
                cat = code.split('.')[0]
            
            if cat in categories:
                categories[cat]['count'] += count
                categories[cat]['signs'].append({'code': code, 'count': count})
        
        return categories

