import os
import json
import glob
import cv2
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

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

# 全域變數，用於切換分析角度 (可選: 45 或 60 或 90  )
ANGLE_MODE = 90

def main():
    
    results = {}                                 # 存放每組的 SNR 分數
    S_j_list = {}                                # 存放每組的訊號強度
    N_j_list = {}                                # 存放每組的雜訊強度
    visualization_data = []                      # 存放每組的視覺化資料
    
    # ==================================================
    # ====== Step 1 設定資料集路徑和 ROI 邊界參數 ======
    # ==================================================
    dataset_path = f"angle_{ANGLE_MODE}_dataset"
    roi_file = f"angle_{ANGLE_MODE}_roi_coordinates.json"
    
    # 1.1 讀取 ROI 座標資料
    try:
        with open(roi_file, 'r') as f:
            all_roi_data = json.load(f)
    except FileNotFoundError:
        print(f"錯誤：找不到 ROI 檔案 '{roi_file}'。請先執行 roi_selector.py 產生 ROI。")
        return

    # 1.2 讀取一張影像以取得影像尺寸
    try:
        sample_img_path = glob.glob(os.path.join(dataset_path, "Group_1", "*.png"))[0]
        sample_img = cv2.imread(sample_img_path, cv2.IMREAD_GRAYSCALE)
        img_shape = sample_img.shape
    except (IndexError, AttributeError):
        print("錯誤：在 Group_1 中找不到任何影像或無法讀取，無法確定影像尺寸。")
        return
    
    # ==================================================
    # ============== Step 2 讀取影像 =================
    # ==================================================

    print("--- 開始分析並收集數據 ---")
    
    # 處理 Group_1 到 Group_8，分別對應 0° 到 315° ，斜下 45° 的光源角度
    for i in range(1, 9):
        group_name = f"Group_{i}"
        group_path = os.path.join(dataset_path, group_name)
        
        # 2.1 檢查資料夾和 ROI 資料是否存在
        if not os.path.exists(group_path):
            print(f"警告：找不到資料夾 {group_path}，跳過此組。")
            continue
        if group_name not in all_roi_data:
            print(f"警告：在 '{roi_file}' 中找不到 {group_name} 的 ROI 資料，跳過此組。")
            continue
        print(f"--- 正在處理 {group_name} ---")

        # 2.2 讀取影像檔案
        img_files = sorted(glob.glob(os.path.join(group_path, "*.png")))
        if not img_files:
            print(f"警告：在 {group_path} 中沒有找到 .png 影像，跳過此組。")
            continue
        
        # 2.3 建立遮罩並計算 SNR 分數
        roi_coords = all_roi_data[group_name]        # 取得該組的 ROI 座標
        mask = np.zeros(img_shape, dtype=np.uint8)   # 建立全黑的遮罩
        pts = np.array(roi_coords, dtype=np.int32)   # 將 ROI 座標轉為 numpy 陣列
        cv2.fillPoly(mask, [pts], 255)               # 使用多邊形座標填充遮罩
        mask = mask.astype(bool)                     # 將遮罩轉為布林值陣列     
        
        # ==============================================
        # =========== Step 3 計算 SNR 分數 ===========
        # ==============================================
        
        # 3.0 堆疊影像並準備計算
        imgs = []
        for f in img_files:
            I = cv2.imread(f, cv2.IMREAD_GRAYSCALE)  # 以灰階讀取影像
            if I is None:
                print(f"警告：無法讀取影像 {f}，將跳過。")
                continue
            imgs.append(I.astype(np.float32))        # 轉為 float32 以避免溢位，並存入 imgs 陣列

        if not imgs:
            print("錯誤：沒有成功載入任何影像。")
            return 0.0, None, None, None

        stack = np.stack(imgs, axis=-1)              # 將 50 張影像堆疊成 (H, W, 50) 的陣列
        
        # 3.1 計算每個像素的平均值和標準差影像
        mu = np.mean(stack, axis=-1)                 # 先計算整個影像的平均值
        tau = np.std(stack, axis=-1)                 # 先計算整個影像的標準差
        
        # 3.2 只從 ROI 區域中獲取雜訊水平
        N_j = np.median(tau[mask])                   # 計算 ROI 區域內標準差的中位數作為雜訊強度

        # 3.3 masking
        masking_mu = mu.copy()                       # 複製 mu 影像
        masking_mu[~mask] = 0                        # 將 mu 影像 ROI 外的區域設為零值，用於視覺化
        masking_tau = tau.copy()                     # 複製 tau 影像
        masking_tau[~mask] = 0                       # 將 tau 影像 ROI 外的區域設為零值，用於視覺化

        # 3.4 padding
        padding_value = np.mean(mu[mask])            # 使用 ROI 內的平均值來填充 ROI 外的區域
        padding_mu = mu.copy()                       # 複製 mu 影像
        padding_mu[~mask] = padding_value            # 填充 mu 影像 ROI 外的區域，用來做高斯平滑

        # 3.5 使用高斯平滑來估計背景趨勢
        sigma = 20                                   # 高斯平滑的標準差參數
        mu_Gaussian = cv2.GaussianBlur(padding_mu, (0, 0), sigmaX=sigma, sigmaY=sigma) # 使用高斯平滑

        # 3.6 計算 SNR 分數
        residual = mu - mu_Gaussian                  # 計算殘差影像
        masking_residual = residual.copy()           # 複製殘差影像
        masking_residual[~mask] = 0                  # 將殘差影像 ROI 外的區域設為零值，用於視覺化
        residual_masked = residual[mask]             # 只取 ROI 區域的殘差值
        S_j = np.sqrt(np.mean(residual_masked**2))   # 計算殘差的 RMS 作為訊號強度
        R_j = S_j / max(N_j, 1e-8)                   # 計算 SNR 分數，避免除以零，物理意義為瑕疵強度與雜訊強度的比值
        
        # 3.7 儲存結果
        results[group_name] = R_j                    # 存儲 SNR 結果
        S_j_list[group_name] = S_j                   # 存儲訊號強度
        N_j_list[group_name] = N_j                   # 存儲雜訊強度
        print(f"{group_name} 的 SNR 分數: {R_j:.4f}, {group_name} 的訊號強度: {S_j:.4f}, {group_name} 的雜訊強度: {N_j:.4f}")

        # 3.8 準備視覺化資料
        
        # a. 建立平均疊加圖 (將 mask 輪廓畫在 mu 上)
        IMG_mu = cv2.cvtColor(masking_mu.astype(np.uint8), cv2.COLOR_GRAY2BGR)
        contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(IMG_mu, contours, -1, (0, 0, 255), 2)          # 畫上紅色輪廓

        # b. 建立標準差影像的疊加圖 (標準化標準差影像以便更好地可視化，因為標準差值範圍可能很小）
        tau_normalized = cv2.normalize(masking_tau.astype(np.uint8), None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        IMG_tau = cv2.cvtColor(tau_normalized, cv2.COLOR_GRAY2BGR)
        cv2.drawContours(IMG_tau, contours, -1, (0, 0, 255), 2)         # 畫上紅色輪廓
        
        # c. 建立 padding_mu 的疊加圖 (將 mask 輪廓畫在 padding_mu 上)
        IMG_padding_mu = cv2.cvtColor(padding_mu.astype(np.uint8), cv2.COLOR_GRAY2BGR)
        cv2.drawContours(IMG_padding_mu, contours, -1, (0, 0, 255), 2)  # 畫上紅色輪廓

        # d. 建立 mu_Gaussian 的疊加圖 (將 mask 輪廓畫在 mu_Gaussian 上)
        IMG_muG = cv2.cvtColor(mu_Gaussian.astype(np.uint8), cv2.COLOR_GRAY2BGR)
        cv2.drawContours(IMG_muG, contours, -1, (0, 0, 255), 2)         # 畫上紅色輪廓

        # e. 建立殘差影像的疊加圖 (將 mask 輪廓畫在 residual 上)
        IMG_residual = cv2.cvtColor(masking_residual.astype(np.uint8), cv2.COLOR_GRAY2BGR)
        cv2.drawContours(IMG_residual, contours, -1, (0, 0, 255), 2)    # 畫上紅色輪廓
        
        # f. 將這一組的資料存入列表
        visualization_data.append({
            "name": group_name,
            "IMG_mu": IMG_mu,
            "IMG_tau": IMG_tau,
            "IMG_padding": IMG_padding_mu,
            "IMG_muG": IMG_muG,
            "IMG_residual": IMG_residual
        })

    # ==================================================
    # ============ Step 4 視覺化資料 ==================
    # ==================================================    
    
    if not visualization_data:
        print("沒有可供視覺化的資料。")
    else:
        # 4.1 圖一 疊加平均圖
        fig1, axes1 = plt.subplots(2, 4, figsize=(12, 7))                                       # 縮小圖片尺寸
        fig1.suptitle('Figure 1: Average Image with Region of Interest (ROI) Overlay', fontsize=18)
        
        for idx, data in enumerate(visualization_data):                                         # 遍歷每個視覺化資料
            row = idx // 4                                                                      # 計算行
            col = idx % 4                                                                       # 計算欄
            ax_mu_overlay = axes1[row, col]                                                     # 選擇子圖
            ax_mu_overlay.imshow(data["IMG_mu"])                                                # 顯示平均影像疊加圖
            angle = (int(data["name"].split("_")[1]) - 1) * 45                                  # 固定使用 45 度間隔計算角度
            ax_mu_overlay.set_title(f'{data["name"]} (角度 {angle}°)', fontsize=9, family=plt.rcParams['font.sans-serif'][0])  # 設定子圖標題，縮小字體
            ax_mu_overlay.axis('off')                                                           # 不顯示座標軸

        for i in range(len(visualization_data), 8):                                             # 隱藏未使用的子圖
            row = i // 4                                                                        # 計算行
            col = i % 4                                                                         # 計算欄
            if row < 2 and col < 4:                                                             # 確保索引有效
                fig1.delaxes(axes1[row, col])                                                   # 刪除子圖

        # 確保輸出目錄存在
        output_dir = f"angle_{ANGLE_MODE}_analysis_img"
        os.makedirs(output_dir, exist_ok=True)
        
        plt.tight_layout(rect=[0, 0, 1, 0.95])                                                  # 調整佈局以適應標題
        plt.savefig(f"{output_dir}/fig1_average_image_with_mask.png")                           # 儲存圖片至指定資料夾
        plt.show()

        # 4.2 圖二 疊加標準差圖
        fig2, axes2 = plt.subplots(2, 4, figsize=(12, 7))                                       # 縮小圖片尺寸
        fig2.suptitle('Figure 2: Image Noise Map (Standard Deviation) with ROI Overlay', fontsize=18)

        for idx, data in enumerate(visualization_data):
            row = idx // 4                                                                      # 計算行
            col = idx % 4                                                                       # 計算欄
            ax_tau_overlay = axes2[row, col]                                                    # 選擇子圖
            ax_tau_overlay.imshow(data["IMG_tau"])                                              # 顯示標準差影像疊加圖
            angle = (int(data["name"].split("_")[1]) - 1) * 45                                  # 固定使用 45 度間隔計算角度
            ax_tau_overlay.set_title(f'{data["name"]} (角度 {angle}°)', family=plt.rcParams['font.sans-serif'][0])  # 設定子圖標題
            ax_tau_overlay.axis('off')                                                          # 不顯示座標軸

        for i in range(len(visualization_data), 8):
            row = i // 4                                                                        # 計算行
            col = i % 4                                                                         # 計算欄
            if row < 2 and col < 4:                                                             # 確保索引有效
                fig2.delaxes(axes2[row, col])

        plt.tight_layout(rect=[0, 0, 1, 0.96])                                                  # 調整佈局以適應標題
        plt.savefig(f"{output_dir}/fig2_standard_deviation_image_with_mask.png")                # 儲存圖片
        plt.show()

        # 4.3 圖三 padding 過的疊加平均圖
        fig3, axes3 = plt.subplots(2, 4, figsize=(12, 7))                                       # 縮小圖片尺寸
        fig3.suptitle('Figure 3: Average Image with Region of Interest (ROI) Overlay with Padding', fontsize=18)
        
        for idx, data in enumerate(visualization_data):
            row = idx // 4                                                                      # 計算行
            col = idx % 4                                                                       # 計算欄
            ax_padding_overlay = axes3[row, col]                                                # 選擇子圖
            ax_padding_overlay.imshow(data["IMG_padding"])                                      # 顯示 padding 疊加平均圖
            angle = (int(data["name"].split("_")[1]) - 1) * 45                                  # 固定使用 45 度間隔計算角度
            ax_padding_overlay.set_title(f'{data["name"]} (角度 {angle}°)', family=plt.rcParams['font.sans-serif'][0])  # 設定子圖標題
            ax_padding_overlay.axis('off')                                                      # 不顯示座標軸
        
        for i in range(len(visualization_data), 8):
            row = i // 4                                                                        # 計算行
            col = i % 4                                                                         # 計算欄
            if row < 2 and col < 4:                                                             # 確保索引有效
                fig3.delaxes(axes3[row, col])
        
        plt.tight_layout(rect=[0, 0, 1, 0.96])                                                  # 調整佈局以適應標題
        plt.savefig(f"{output_dir}/fig3_average_image_with_padding.png")                        # 儲存圖片
        plt.show()

        # 4.4 圖四 疊加 Gaussian 平滑圖
        fig4, axes4 = plt.subplots(2, 4, figsize=(12, 7))                                       # 縮小圖片尺寸
        fig4.suptitle('Figure 4: Gaussian Smoothed Image with ROI Overlay', fontsize=18)
        
        for idx, data in enumerate(visualization_data):
            row = idx // 4                                                                      # 計算行
            col = idx % 4                                                                       # 計算欄
            ax_muG_overlay = axes4[row, col]                                                    # 選擇子圖
            ax_muG_overlay.imshow(data["IMG_muG"])                                              # 顯示 Gaussian 平滑影像疊加圖
            angle = (int(data["name"].split("_")[1]) - 1) * 45                                  # 固定使用 45 度間隔計算角度
            ax_muG_overlay.set_title(f'{data["name"]} (Angle {angle}°)')                        # 設定子圖標題
            ax_muG_overlay.axis('off')                                                          # 不顯示座標軸
        
        for i in range(len(visualization_data), 8):
            row = i // 4                                                                        # 計算行
            col = i % 4                                                                         # 計算欄
            if row < 2 and col < 4:                                                             # 確保索引有效
                fig4.delaxes(axes4[row, col])
        
        plt.tight_layout(rect=[0, 0, 1, 0.96])                                                  # 調整佈局以適應標題
        plt.savefig(f"{output_dir}/fig4_gaussian_smoothed_image_with_mask.png")                 # 儲存圖片
        plt.show()

        # 4.5 圖五 疊加殘差圖
        fig5, axes5 = plt.subplots(2, 4, figsize=(12, 7))                                       # 縮小圖片尺寸
        fig5.suptitle('Figure 5: Residual Image with ROI Overlay', fontsize=18)
        
        for idx, data in enumerate(visualization_data):
            row = idx // 4                                                                      # 計算行
            col = idx % 4                                                                       # 計算欄
            ax_residual_overlay = axes5[row, col]                                               # 選擇子圖
            ax_residual_overlay.imshow(data["IMG_residual"])                                    # 顯示殘差影像疊加圖
            angle = (int(data["name"].split("_")[1]) - 1) * 45                                  # 固定使用 45 度間隔計算角度
            ax_residual_overlay.set_title(f'{data["name"]} (Angle {angle}°)')                   # 設定子圖標題
            ax_residual_overlay.axis('off')                                                     # 不顯示座標軸
       
        for i in range(len(visualization_data), 8):
            row = i // 4                                                                        # 計算行
            col = i % 4                                                                         # 計算欄
            if row < 2 and col < 4:                                                             # 確保索引有效
                fig5.delaxes(axes5[row, col])
        
        plt.tight_layout(rect=[0, 0, 1, 0.96])                                                  # 調整佈局以適應標題
        plt.savefig(f"{output_dir}/fig5_residual_image_with_mask.png")                          # 儲存圖片
        plt.show()

    # ================================================
    # ============ Step 5 報告分析結果 ================
    # ================================================
    
    # 5.1 找出 SNR 分數最高的組別
    if results:
        best_group = max(results, key=results.get)
        best_score = results[best_group]
        
        print("\n================== 分析完成 ==================")
        print(f"分析角度: {ANGLE_MODE}°")
        print("各組 SNR 分數總結：")
        for group, score in sorted(results.items()):
            angle = (int(group.split('_')[1]) - 1) * 45
            print(f"  - {group} (角度 {angle}°): {score:.4f}")
            
        print(f"\n分析結果：'{best_group}' (角度 {(int(best_group.split('_')[1]) - 1) * 45}°) 的 SNR 分數最高 ({best_score:.4f})。")
        print("這表示此角度的光源最有可能突顯出產品的瑕疵。")
        print("============================================")
    else:
        print("\n沒有計算出任何結果。請檢查您的資料夾結構與檔案。")
    
    # 5.2 存入 CSV 檔案
    df = pd.DataFrame({
        "Group": list(results.keys()),
        "Angle (°)": [(int(name.split('_')[1]) - 1) * 45 for name in results.keys()],
        "SNR Score": list(results.values()),
        "Signal Strength (S_j)": [S_j_list[name] for name in results.keys()],
        "Noise Strength (N_j)": [N_j_list[name] for name in results.keys()]
    })
    csv_file = f"snr_angle_{ANGLE_MODE}_results_summary.csv"
    df.to_csv(csv_file, index=False)
    print(f"\n已將結果存入 '{csv_file}'。")
    

if __name__ == "__main__":
    main()