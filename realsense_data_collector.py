import sys
import cv2
import numpy as np
import pyrealsense2 as rs
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                           QHBoxLayout, QPushButton, QLabel, QProgressBar,
                           QComboBox, QGridLayout, QGroupBox, QSpinBox)
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QImage, QPixmap
import os
import time
from pathlib import Path

# 全域變數，用於切換分析角度 (可選: 45 或 60 或 90)
ANGLE_MODE = 60
# 專案根目錄（相對於此檔案）
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

class RealSenseApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.initUI()
        self.initCamera()
        
    def initUI(self):
        self.setWindowTitle(f'RealSense 資料集採集工具 (角度模式: {ANGLE_MODE}°)')
        self.setGeometry(100, 100, 1200, 800)
        
        # 設定組別名稱與參數
        self.groups = [f"Group_{i+1}" for i in range(8)]  # 修改為 8 個組別 (Group_1 到 Group_8)
        self.current_group = self.groups[0]
        self.images_per_group = 50  # 每組50張
        self.group_progress = {group: 0 for group in self.groups}
        self.capture_count = 0

        # 建立中央視窗
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QHBoxLayout(central_widget)

        # 左側區域 - 相機預覽和控制
        left_widget = QWidget()
        left_layout = QVBoxLayout(left_widget)
        
        # 相機預覽
        self.preview_label = QLabel()
        self.preview_label.setMinimumSize(640, 480)
        self.preview_label.setAlignment(Qt.AlignCenter)
        left_layout.addWidget(self.preview_label)

        # 控制區域
        control_widget = QWidget()
        control_layout = QGridLayout(control_widget)
        
        # 組別選擇
        group_box = QGroupBox(f"資料集控制 ({ANGLE_MODE}° 角度)")
        group_layout = QVBoxLayout()
        
        # 組別選擇下拉選單
        self.group_combo = QComboBox()
        self.group_combo.addItems(self.groups)
        self.group_combo.currentTextChanged.connect(self.on_group_changed)
        group_layout.addWidget(QLabel("選擇組別:"))
        group_layout.addWidget(self.group_combo)
        
        # 組別資訊
        self.group_info_label = QLabel()
        self.update_group_info()
        group_layout.addWidget(self.group_info_label)
        
        group_box.setLayout(group_layout)
        control_layout.addWidget(group_box, 0, 0)
        
        # 拍攝控制
        capture_box = QGroupBox(f"拍攝控制 ({ANGLE_MODE}° 角度)")
        capture_layout = QVBoxLayout()
        
        # 拍攝按鈕
        self.capture_button = QPushButton('開始拍攝')
        self.capture_button.clicked.connect(self.toggle_capture)
        capture_layout.addWidget(self.capture_button)
        
        # 當前組別進度條
        self.progress_bar = QProgressBar()
        self.progress_bar.setMaximum(self.images_per_group)
        self.progress_bar.setValue(0)
        capture_layout.addWidget(QLabel("當前組別進度:"))
        capture_layout.addWidget(self.progress_bar)
        
        # 總體進度條
        self.total_progress_bar = QProgressBar()
        self.total_progress_bar.setMaximum(len(self.groups) * self.images_per_group)
        self.total_progress_bar.setValue(0)
        capture_layout.addWidget(QLabel("總體進度:"))
        capture_layout.addWidget(self.total_progress_bar)
        
        capture_box.setLayout(capture_layout)
        control_layout.addWidget(capture_box, 0, 1)
        
        left_layout.addWidget(control_widget)
        layout.addWidget(left_widget)

        # 右側區域 - 預覽和資訊
        right_widget = QWidget()
        right_layout = QVBoxLayout(right_widget)
        
        # 最後拍攝的照片預覽
        preview_box = QGroupBox(f"最後拍攝的照片 ({ANGLE_MODE}°)")
        preview_layout = QVBoxLayout()
        self.last_capture_label = QLabel()
        self.last_capture_label.setAlignment(Qt.AlignCenter)
        self.last_capture_label.setMinimumSize(320, 240)
        preview_layout.addWidget(self.last_capture_label)
        preview_box.setLayout(preview_layout)
        right_layout.addWidget(preview_box)
        
        # 組別預覽
        group_preview_box = QGroupBox(f"組別預覽 ({ANGLE_MODE}°)")
        group_preview_layout = QVBoxLayout()
        self.group_preview_label = QLabel("尚無照片")
        self.group_preview_label.setAlignment(Qt.AlignCenter)
        self.group_preview_label.setMinimumSize(320, 240)
        group_preview_layout.addWidget(self.group_preview_label)
        group_preview_box.setLayout(group_preview_layout)
        right_layout.addWidget(group_preview_box)
        
        layout.addWidget(right_widget)

        # 狀態列
        self.statusBar().showMessage(f'就緒 (角度模式: {ANGLE_MODE}°)')

        # 設定計時器用於更新預覽
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_preview)
        
        # 初始化變數
        self.is_capturing = False
        self.capture_count = 0
        # 將資料集路徑設為專案目錄下的相對路徑
        self.output_dir = os.path.join(PROJECT_ROOT, f"angle_{ANGLE_MODE}_dataset")

        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

    def initCamera(self):
        # 初始化 RealSense 相機
        self.pipeline = rs.pipeline()
        self.config = rs.config()
        
        # 檢查是否有 RealSense 裝置連接
        ctx = rs.context()
        devices = ctx.query_devices()
        if len(devices) == 0:
            self.statusBar().showMessage('錯誤：未偵測到 RealSense 相機')
            return
            
        # 顯示相機資訊
        device = devices[0]
        self.statusBar().showMessage(
            f'已連接相機: {device.get_info(rs.camera_info.name)} '
            f'(序號: {device.get_info(rs.camera_info.serial_number)}) '
            f'角度模式: {ANGLE_MODE}°'
        )

        # 配置串流
        self.config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

        try:
            self.pipeline.start(self.config)
            # 啟動預覽更新計時器
            self.timer.start(33)  # 約 30 FPS
        except RuntimeError as e:
            self.statusBar().showMessage(f'錯誤：無法啟動相機 - {str(e)}')

    def update_preview(self):
        try:
            # 取得影像幀
            frames = self.pipeline.wait_for_frames()
            color_frame = frames.get_color_frame()
            
            if color_frame:
                # 轉換為 NumPy 陣列
                color_image = np.asanyarray(color_frame.get_data())
                
                # 轉換為 Qt 影像
                h, w, ch = color_image.shape
                bytes_per_line = ch * w
                qt_image = QImage(color_image.data, w, h, bytes_per_line, QImage.Format_BGR888)
                
                # 更新預覽
                self.preview_label.setPixmap(QPixmap.fromImage(qt_image))

                # 如果正在拍攝模式，進行拍攝
                if self.is_capturing and self.capture_count < self.images_per_group:
                    self.capture_frame(color_image)
                
        except Exception as e:
            self.statusBar().showMessage(f'預覽更新錯誤：{str(e)}')

    def capture_frame(self, frame):
        try:
            # 建立組別目錄
            group_dir = os.path.join(self.output_dir, self.current_group)
            if not os.path.exists(group_dir):
                os.makedirs(group_dir)
            
            # 儲存影像
            filename = os.path.join(group_dir, f"capture_{self.capture_count + 1:03d}.png")
            cv2.imwrite(filename, frame)
            
            # 更新最後拍攝的照片預覽
            h, w, ch = frame.shape
            bytes_per_line = ch * w
            qt_image = QImage(frame.data, w, h, bytes_per_line, QImage.Format_BGR888)
            scaled_pixmap = QPixmap.fromImage(qt_image).scaled(
                320, 240, Qt.KeepAspectRatio, Qt.SmoothTransformation)
            self.last_capture_label.setPixmap(scaled_pixmap)
            
            # 更新計數和進度
            self.capture_count += 1
            self.group_progress[self.current_group] = self.capture_count
            
            # 更新進度條
            self.progress_bar.setValue(self.capture_count)
            total_captures = sum(self.group_progress.values())
            self.total_progress_bar.setValue(total_captures)
            
            # 更新狀態訊息
            self.statusBar().showMessage(
                f'組別 {self.current_group}: {self.capture_count}/{self.images_per_group} 張 | '
                f'總進度: {total_captures}/{len(self.groups) * self.images_per_group} 張'
            )
            
            # 更新組別資訊
            self.update_group_info()
            
            # 檢查是否完成當前組別
            if self.capture_count >= self.images_per_group:
                self.stop_capture()
                if total_captures >= len(self.groups) * self.images_per_group:
                    self.statusBar().showMessage(f'所有組別拍攝完成！({ANGLE_MODE}° 角度數據集)')
                else:
                    # 自動切換到下一個未完成的組別
                    self.switch_to_next_incomplete_group()
                
        except Exception as e:
            self.statusBar().showMessage(f'儲存影像時發生錯誤：{str(e)}')

    def toggle_capture(self):
        if not self.is_capturing:
            self.start_capture()
        else:
            self.stop_capture()

    def start_capture(self):
        self.is_capturing = True
        self.capture_button.setText('停止拍攝')
        self.statusBar().showMessage(f'開始拍攝 {ANGLE_MODE}° 角度數據...')

    def stop_capture(self):
        self.is_capturing = False
        self.capture_button.setText('開始拍攝')
        if self.capture_count < 50:
            self.statusBar().showMessage(f'拍攝已停止 (角度模式: {ANGLE_MODE}°)')

    def on_group_changed(self, group_name):
        self.current_group = group_name
        self.capture_count = self.group_progress[group_name]
        self.progress_bar.setValue(self.capture_count)
        self.update_group_info()
        self.update_group_preview()
        
    def update_group_info(self):
        completed_groups = sum(1 for count in self.group_progress.values() if count >= self.images_per_group)
        current_progress = self.group_progress[self.current_group]
        
        info_text = (f"角度模式: {ANGLE_MODE}°\n"
                    f"已完成組別：{completed_groups}/8\n"
                    f"當前組別進度：{current_progress}/{self.images_per_group}")
        self.group_info_label.setText(info_text)
        
    def update_group_preview(self):
        group_dir = os.path.join(self.output_dir, self.current_group)
        if os.path.exists(group_dir):
            # 找出最後拍攝的照片
            files = sorted([f for f in os.listdir(group_dir) if f.endswith('.png')])
            if files:
                latest_file = os.path.join(group_dir, files[-1])
                pixmap = QPixmap(latest_file)
                if not pixmap.isNull():
                    scaled_pixmap = pixmap.scaled(
                        320, 240, Qt.KeepAspectRatio, Qt.SmoothTransformation)
                    self.group_preview_label.setPixmap(scaled_pixmap)
                    return
        self.group_preview_label.setText("尚無照片")
        
    def switch_to_next_incomplete_group(self):
        current_index = self.groups.index(self.current_group)
        for i in range(current_index + 1, len(self.groups)):
            if self.group_progress[self.groups[i]] < self.images_per_group:
                self.group_combo.setCurrentText(self.groups[i])
                return
                
    def closeEvent(self, event):
        # 關閉視窗時停止相機串流
        self.timer.stop()
        self.pipeline.stop()
        event.accept()

if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = RealSenseApp()
    ex.show()
    sys.exit(app.exec_())