import cv2
import numpy as np
import os
import json
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from matplotlib.widgets import Button
import matplotlib

# 設定 Matplotlib 使用中文字體
try:
    # 嘗試不同的中文字體，提高相容性
    plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei', 'Microsoft YaHei', 'SimHei', 'Arial Unicode MS', 
                                      'STSong', 'NSimSun', 'FangSong', 'SimSun', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
    print("已設定中文字體支援")
except Exception as e:
    print(f"設定中文字體時發生錯誤: {e}")
    print("可能會導致中文顯示為亂碼")

# 全域變數，用於切換分析角度 (可選: 45 或 60 或 90)
ANGLE_MODE = 45
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

class ROISelector:
    def __init__(self, dataset_path):
        self.dataset_path = dataset_path
        self.groups = [f"Group_{i+1}" for i in range(8)]  # 更新為 8 個組別
        self.roi_data = {}  # 儲存每個組別的 ROI 座標
        self.roi_file = os.path.join(PROJECT_ROOT, f"angle_{ANGLE_MODE}_roi_coordinates.json")
        
        # 嘗試載入已存在的 ROI 資料
        self.load_roi_data()
        
    def load_roi_data(self):
        """載入已儲存的 ROI 資料"""
        if os.path.exists(self.roi_file):
            try:
                with open(self.roi_file, 'r') as f:
                    self.roi_data = json.load(f)
                print(f"已載入 ROI 資料：{list(self.roi_data.keys())}")
            except:
                print("載入 ROI 資料失敗，將重新建立")
                self.roi_data = {}
    
    def save_roi_data(self):
        """儲存 ROI 資料到檔案"""
        with open(self.roi_file, 'w') as f:
            json.dump(self.roi_data, f, indent=2)
        print(f"ROI 資料已儲存到 {self.roi_file}")
    
    def get_sample_image(self, group_name):
        """取得組別的第一張圖片作為參考"""
        group_path = os.path.join(self.dataset_path, group_name)
        if not os.path.exists(group_path):
            print(f"找不到組別 {group_name}")
            return None
            
        image_files = sorted([f for f in os.listdir(group_path) if f.endswith('.png')])
        if not image_files:
            print(f"組別 {group_name} 中沒有圖片")
            return None
            
        img_path = os.path.join(group_path, image_files[0])
        try:
            # 使用 np.fromfile 和 cv2.imdecode 來處理包含非 ASCII 字元的路徑
            n = np.fromfile(img_path, np.uint8)
            img = cv2.imdecode(n, cv2.IMREAD_COLOR)
            if img is not None:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        except Exception as e:
            print(f"使用 cv2.imdecode 讀取圖片失敗: {e}")
            # Fallback to cv2.imread, which might fail
            img = cv2.imread(img_path)
            if img is not None:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img
    
    def select_roi_for_group(self, group_name):
        """為特定組別選擇 ROI"""
        print(f"\n正在為 {group_name} 選擇 ROI（角度模式: {ANGLE_MODE}°）...")
        
        # 檢查是否已有 ROI 資料
        if group_name in self.roi_data:
            print(f"{group_name} 已有 ROI 資料")
            return True
            
        img = self.get_sample_image(group_name)
        if img is None:
            return False
            
        print("請依序點選四個角點來定義 ROI 區域")
        print("建議順序：左上 -> 右上 -> 右下 -> 左下")
        print("點選完成後按 Enter 確認，按 'r' 重新選擇")
        
        # 使用 matplotlib 進行互動式點選
        fig, ax = plt.subplots(figsize=(10, 7))                                                # 縮小圖片尺寸
        ax.imshow(img)
        ax.set_title(f'{group_name} - 請點選四個角點定義 ROI（{ANGLE_MODE}°）', fontsize=12)
        
        points = []
        
        def onclick(event):
            if event.inaxes != ax:
                return
            if len(points) < 4:
                x, y = int(event.xdata), int(event.ydata)
                points.append([x, y])
                ax.plot(x, y, 'ro', markersize=8)
                ax.annotate(f'P{len(points)}', (x, y), xytext=(5, 5), 
                           textcoords='offset points', color='red', fontsize=12)
                
                # 如果有兩個以上的點，畫線連接
                if len(points) > 1:
                    ax.plot([points[-2][0], points[-1][0]], 
                           [points[-2][1], points[-1][1]], 'r-', linewidth=2)
                
                # 如果四個點都選完了，閉合多邊形
                if len(points) == 4:
                    ax.plot([points[-1][0], points[0][0]], 
                           [points[-1][1], points[0][1]], 'r-', linewidth=2)
                    
                    # 填充半透明區域
                    polygon = Polygon(points, alpha=0.3, facecolor='red')
                    ax.add_patch(polygon)
                    
                plt.draw()
        
        def on_key(event):
            nonlocal points
            if event.key == 'r':  # 重新選擇
                points = []
                ax.clear()
                ax.imshow(img)
                ax.set_title(f'{group_name} - 請點選四個角點定義 ROI（{ANGLE_MODE}°）', fontsize=12)
                plt.draw()
            elif (event.key == 'enter' or event.key == '\r') and len(points) == 4:  # 確認 (兼容不同平台的Enter鍵)
                plt.close()
        
        # 連接事件
        fig.canvas.mpl_connect('button_press_event', onclick)
        fig.canvas.mpl_connect('key_press_event', on_key)
        
        # 添加說明文字 (使用具有中文支持的字體)
        fig.text(0.02, 0.02, "操作說明：\n1. 點選四個角點\n2. 按 Enter 確認\n3. 按 'r' 重新選擇", 
                fontsize=10, bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray"),
                family=plt.rcParams['font.sans-serif'][0])
        
        plt.tight_layout()
        plt.show()
        
        if len(points) == 4:
            self.roi_data[group_name] = points
            self.save_roi_data()
            print(f"{group_name} ROI 選擇完成！")
            return True
        else:
            print(f"{group_name} ROI 選擇取消")
            return False
    
    def select_all_rois(self):
        """為所有組別選擇 ROI"""
        print(f"開始為所有組別選擇 ROI（角度模式: {ANGLE_MODE}°）...")
        
        for group in self.groups:
            if group in self.roi_data:
                print(f"{group} 已有 ROI 資料，跳過")
                continue
                
            success = self.select_roi_for_group(group)
            if not success:
                print(f"警告：{group} ROI 選擇失敗")
        
        print("\nROI 選擇完成！")
        self.show_roi_summary()
    
    def show_roi_summary(self):
        """顯示所有 ROI 的摘要"""
        print("\n" + "="*50)
        print(f"ROI 選擇摘要（角度模式: {ANGLE_MODE}°）")
        print("="*50)
        
        for group in self.groups:
            if group in self.roi_data:
                roi = self.roi_data[group]
                print(f"{group:8s}: 已選擇 ROI ({len(roi)} 個點)")
            else:
                print(f"{group:8s}: 尚未選擇 ROI")
    
    def apply_roi_to_image(self, image, roi_points):
        """將 ROI 套用到圖片上"""
        if len(roi_points) != 4:
            return None
            
        # 建立遮罩
        mask = np.zeros(image.shape[:2], dtype=np.uint8)
        roi_array = np.array(roi_points, dtype=np.int32)
        cv2.fillPoly(mask, [roi_array], 255)
        
        # 套用遮罩
        roi_image = cv2.bitwise_and(image, image, mask=mask)
        
        return roi_image, mask
    
    def extract_roi_from_group(self, group_name):
        """從組別中提取所有圖片的 ROI"""
        if group_name not in self.roi_data:
            print(f"錯誤：{group_name} 沒有 ROI 資料")
            return None
            
        group_path = os.path.join(self.dataset_path, group_name)
        if not os.path.exists(group_path):
            print(f"錯誤：找不到組別 {group_name}")
            return None
            
        # 建立輸出目錄（放在專案目錄下）
        output_dir = os.path.join(PROJECT_ROOT, f"roi_extracted_angle_{ANGLE_MODE}", group_name)
        os.makedirs(output_dir, exist_ok=True)
        
        roi_points = self.roi_data[group_name]
        image_files = sorted([f for f in os.listdir(group_path) if f.endswith('.png')])
        
        roi_images = []
        masks = []
        
        print(f"正在處理 {group_name} 的 {len(image_files)} 張圖片...")
        
        for i, filename in enumerate(image_files):
            img_path = os.path.join(group_path, filename)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            
            if img is not None:
                roi_img, mask = self.apply_roi_to_image(img, roi_points)
                
                # 儲存 ROI 圖片
                output_path = os.path.join(output_dir, filename)
                cv2.imwrite(output_path, roi_img)
                
                roi_images.append(roi_img)
                masks.append(mask)
                
                if (i + 1) % 10 == 0:
                    print(f"  已處理 {i + 1}/{len(image_files)} 張")
        
        print(f"{group_name} ROI 提取完成！")
        return roi_images, masks
    
    def extract_all_rois(self):
        """為所有組別提取 ROI"""
        print(f"開始提取所有組別的 ROI（角度模式: {ANGLE_MODE}°）...")
        
        for group in self.groups:
            if group in self.roi_data:
                self.extract_roi_from_group(group)
            else:
                print(f"警告：{group} 沒有 ROI 資料，跳過")
        
        print("所有 ROI 提取完成！")
    
    def preview_roi(self, group_name):
        """預覽組別的 ROI 選擇結果"""
        if group_name not in self.roi_data:
            print(f"錯誤：{group_name} 沒有 ROI 資料")
            return
            
        img = self.get_sample_image(group_name)
        if img is None:
            return
            
        roi_points = self.roi_data[group_name]
        
        # 建立原圖和 ROI 的對比圖
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))                                   # 縮小圖片尺寸
        
        # 原圖 + ROI 框
        ax1.imshow(img)
        roi_array = np.array(roi_points + [roi_points[0]])  # 閉合多邊形
        ax1.plot(roi_array[:, 0], roi_array[:, 1], 'r-', linewidth=3)
        ax1.fill(roi_array[:, 0], roi_array[:, 1], alpha=0.3, color='red')
        
        # 使用 Unicode 字符確保中文正確顯示
        ax1.set_title(f'{group_name} - 原圖 + ROI（{ANGLE_MODE}°）')
        
        # ROI 結果
        img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        roi_img, _ = self.apply_roi_to_image(img_gray, roi_points)
        ax2.imshow(roi_img, cmap='gray')
        ax2.set_title(f'{group_name} - ROI 結果（{ANGLE_MODE}°）')
        
        # 調整佈局並添加總標題
        fig.suptitle(f"ROI 預覽 - 角度模式: {ANGLE_MODE}°", fontsize=14)
        plt.tight_layout(rect=[0, 0, 1, 0.95])  # 為總標題留出空間
        plt.show()
        
    def preview_all_rois(self):
        """預覽所有已選擇的 ROI"""
        if not self.roi_data:
            print("沒有任何 ROI 資料可以預覽")
            return
            
        groups_with_roi = [g for g in self.groups if g in self.roi_data]
        if not groups_with_roi:
            print("沒有任何組別有 ROI 資料")
            return
            
        print(f"正在預覽所有 ROI（角度模式: {ANGLE_MODE}°）...")
        
        # 計算需要的行數和列數 (最多4列)
        n_groups = len(groups_with_roi)
        n_cols = min(4, n_groups)
        n_rows = (n_groups + n_cols - 1) // n_cols
        
        # 創建足夠大的圖表
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 4, n_rows * 3))
        
        # 確保axes是二維數組，即使只有一行或一列
        if n_rows == 1 and n_cols == 1:
            axes = np.array([[axes]])
        elif n_rows == 1:
            axes = np.array([axes])
        elif n_cols == 1:
            axes = np.array([[ax] for ax in axes])
            
        # 填充每個子圖
        for i, group_name in enumerate(groups_with_roi):
            row = i // n_cols
            col = i % n_cols
            ax = axes[row, col]
            
            # 獲取圖片和ROI
            img = self.get_sample_image(group_name)
            if img is None:
                ax.text(0.5, 0.5, f"{group_name}: 找不到圖片", 
                       ha='center', va='center', fontsize=10, 
                       family=plt.rcParams['font.sans-serif'][0])
                ax.axis('off')
                continue
                
            roi_points = self.roi_data[group_name]
            
            # 顯示圖片和ROI
            ax.imshow(img)
            roi_array = np.array(roi_points + [roi_points[0]])  # 閉合多邊形
            ax.plot(roi_array[:, 0], roi_array[:, 1], 'r-', linewidth=2)
            ax.fill(roi_array[:, 0], roi_array[:, 1], alpha=0.3, color='red')
            ax.set_title(f'{group_name}', fontsize=10)
            ax.axis('off')  # 隱藏坐標軸，使圖片更乾淨
        
        # 隱藏空白子圖
        for i in range(len(groups_with_roi), n_rows * n_cols):
            row = i // n_cols
            col = i % n_cols
            axes[row, col].axis('off')
            
        # 添加總標題
        fig.suptitle(f"所有 ROI 預覽 - 角度模式: {ANGLE_MODE}°", fontsize=14)
        plt.tight_layout(rect=[0, 0, 1, 0.95])  # 為總標題留出空間
        plt.show()

if __name__ == "__main__":
    # 設定資料集路徑
    dataset_path = os.path.join(PROJECT_ROOT, f"angle_{ANGLE_MODE}_dataset")
    
    # 建立 ROI 選擇器
    roi_selector = ROISelector(dataset_path)
    
    # 顯示目前狀態
    roi_selector.show_roi_summary()
    
    print("\n" + "="*50)
    print(f"ROI 選擇器 - 角度模式: {ANGLE_MODE}°")
    print("="*50)
    print("選項：")
    print("1. 為所有組別選擇 ROI")
    print("2. 為特定組別選擇 ROI")
    print("3. 預覽已選擇的 ROI")
    print("4. 提取所有 ROI")
    print("5. 預覽所有 ROI")
    
    choice = input("\n請選擇操作 (1-5): ").strip()
    
    if choice == "1":
        roi_selector.select_all_rois()
    elif choice == "2":
        print("可用組別：", roi_selector.groups)
        group = input("請輸入組別名稱 (例如 Group_1): ").strip()
        if group in roi_selector.groups:
            roi_selector.select_roi_for_group(group)
        else:
            print("無效的組別名稱")
    elif choice == "3":
        if roi_selector.roi_data:
            print("已有 ROI 的組別：", list(roi_selector.roi_data.keys()))
            group = input("請輸入要預覽的組別名稱: ").strip()
            if group in roi_selector.roi_data:
                roi_selector.preview_roi(group)
            else:
                print("該組別沒有 ROI 資料")
        else:
            print("沒有任何 ROI 資料")
    elif choice == "4":
        roi_selector.extract_all_rois()
    elif choice == "5":
        roi_selector.preview_all_rois()
    else:
        print("無效的選擇")