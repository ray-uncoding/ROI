import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from pathlib import Path

# 全域變數，用於切換分析角度 (可選: 45 或 60 或 90)
ANGLE_MODE = 45
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

def main():

    print(f"=== SNR {ANGLE_MODE}度仰角分析與優化 ===")
    
    # === 設置中文字體支援 ===
    try:
        # 嘗試不同的中文字體，提高相容性
        plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei', 'Microsoft YaHei', 'SimHei', 'Arial Unicode MS', 
                                          'STSong', 'NSimSun', 'FangSong', 'SimSun', 'DejaVu Sans']
        plt.rcParams['axes.unicode_minus'] = False
        print("已設定中文字體支援")
    except Exception as e:
        print(f"設定中文字體時發生錯誤: {e}")
        print("警告: 可能會導致中文顯示為亂碼")
    
    # ===========================================
    # ============ Step 1 數據設定 ============
    # ===========================================
   
    # 1.1 載入測量數據
    csv_file = os.path.join(PROJECT_ROOT, f'snr_angle_{ANGLE_MODE}_results_summary.csv')
    print(f"載入數據: {csv_file}")
    snr_data = pd.read_csv(csv_file)                               # 讀取 CSV 文件
    deg_theta = np.full(len(snr_data), ANGLE_MODE)                 # 仰角 (使用全域變數)
    deg_phi = snr_data['Angle (°)'].values                         # 方位角 (從 CSV 讀取)
    R_meas = snr_data['SNR Score'].values                          # SNR值 (從 CSV 讀取)
    
    # 1.2 座標轉換
    theta = np.radians(deg_theta)
    phi = np.radians(deg_phi)
    N = len(theta)
    x = np.sin(theta) * np.cos(phi)
    y = np.sin(theta) * np.sin(phi)
    z = np.cos(theta)
    
    # ===========================================
    # ============ Step 2 二次擬合 ============
    
    # 數學模型說明：
    # 我們使用二次曲面方程來擬合 SNR 與方位角、仰角的關係
    # SNR = β₀ + β₁x + β₂y + β₃x² + β₄y² + β₅xy
    # 其中 x、y 是球座標轉換為笛卡爾座標系的投影點
    
    # =========================================== 
    # =========================================== 
    
    # 2.1 建立二次曲面模型設計矩陣
    # 每一行代表一個樣本點，每一列代表一個特徵 [1, x, y, x², y², xy]    
    X_quad = np.column_stack((np.ones(N), x, y, x**2, y**2, x*y))       # 建立設計矩陣
    
    # 2.2 添加正則化項避免過擬合
    # 正則化參數λ控制模型複雜度，較大的λ會得到更平滑的曲面    
    lambda_quad = 0.1                                                   # 正則化參數（Tikhonov regularization）
    
    # 2.3 求解正則化最小二乘問題
    # min ||X_quad·β - R_meas||² + λ||β||²
    # 解析解：β = (X'X + λI)⁻¹X'y    
    XTX = X_quad.T @ X_quad                                             # 計算 X'X
    reg_term = lambda_quad * np.eye(XTX.shape[0])                       # 正則化項 λI
    beta_quad = np.linalg.solve(XTX + reg_term, X_quad.T @ R_meas)      # 求解 β
    
    # 2.4 直接計算二次曲線的最高點
    # 由於仰角固定，我們只需要考慮方位角變化，這實際上是一個一維優化問題
    # 為了計算更精確，我們創建一個高精度的方位角採樣
    phi_dense = np.linspace(0, 2*np.pi, 3600)                           # 0.1度的精度，完整的360度圓周
    theta_fixed = np.radians(ANGLE_MODE)                                # 固定仰角
    
    # 2.5 將方位角轉換為笛卡爾坐標（固定仰角）
    x_dense = np.sin(theta_fixed) * np.cos(phi_dense)                   # x = sin(θ)cos(φ)
    y_dense = np.sin(theta_fixed) * np.sin(phi_dense)                   # y = sin(θ)sin(φ)
    
    # 2.6 使用擬合的模型預測每個方位角的SNR值
    SNR_dense = np.zeros_like(phi_dense)
    for i in range(len(phi_dense)):
        features = np.array([1, x_dense[i], y_dense[i], x_dense[i]**2, y_dense[i]**2, x_dense[i]*y_dense[i]])
        SNR_dense[i] = features @ beta_quad
    
    # 2.7 找出最佳方位角（最大SNR值對應的方位角）
    max_idx = np.argmax(SNR_dense)                                      # 找出最大值的索引
    max_snr_quad = SNR_dense[max_idx]                                   # 最大SNR值
    best_phi_quad = np.degrees(phi_dense[max_idx]) % 360                # 最佳方位角 (度)，確保在 0-360 範圍內
    best_theta_quad = ANGLE_MODE                                        # 固定仰角 (度)

    # 2.8 計算模型的擬合優度
    pred_quad = X_quad @ beta_quad                                      # 計算原始數據點的預測值
    res_quad = R_meas - pred_quad                                       # 計算殘差（實測值 - 預測值）
    rmse_quad = np.sqrt(np.mean(res_quad**2))                           # 計算均方根誤差(RMSE)
    ss_res = np.sum(res_quad**2)                                        # 殘差平方和
    ss_tot = np.sum((R_meas - np.mean(R_meas))**2)                      # 總平方和
    r2_quad = 1 - ss_res / ss_tot                                       # 判定係數 R²
    
    # 2.9 輸出分析結果
    print(f'最佳照明角度: 仰角={best_theta_quad}° (固定), 方位角={best_phi_quad:.2f}°')
    print(f'預測最大 SNR: {max_snr_quad:.4f} (R²={r2_quad:.4f})')
    print(f'RMSE: {rmse_quad:.4f}')
    
    # ===========================================
    # ============ Step 3 視覺化結果 ============
    # ===========================================
    
    # 3.1 創建圖表參數
    plt.figure(figsize=(10, 6), num=f'方位角SNR分析 (仰角{ANGLE_MODE}°)')              # 創建圖表
    phi_plot = np.linspace(0, 360, 360)                                              # 360個點，提供平滑的曲線
    theta_fixed = ANGLE_MODE                                                         # 固定仰角
    x_plot = np.sin(np.radians(theta_fixed)) * np.cos(np.radians(phi_plot))          # x = sin(θ)cos(φ)
    y_plot = np.sin(np.radians(theta_fixed)) * np.sin(np.radians(phi_plot))          # y = sin(θ)sin(φ)
    
    # 3.2 使用擬合模型計算每個方位角的預測SNR值
    SNR_plot = np.zeros_like(phi_plot)                                               # 初始化預測SNR數組
    for i in range(len(phi_plot)):
        features = np.array([1, x_plot[i], y_plot[i], x_plot[i]**2, y_plot[i]**2, x_plot[i]*y_plot[i]])
        SNR_plot[i] = features @ beta_quad
    
    # 3.3 繪製2D曲線和數據點
    plt.plot(phi_plot, SNR_plot, 'b-', linewidth=2, label='模型預測')                 # 繪製模型預測曲線
    plt.scatter(deg_phi, R_meas, s=100, c='black', edgecolors='k', label='實測數據')  # 繪製實測數據點
    
    # 3.4 標記最佳角度（從網格搜索中得出）
    plt.scatter(best_phi_quad, max_snr_quad, s=150, marker='p', c='y', 
                edgecolors='k', label='最佳角度')
    plt.vlines(best_phi_quad, 0, max_snr_quad, linestyles='--', colors='k')
    plt.text(best_phi_quad+10, max_snr_quad-0.3, 
             f'最佳角度: {best_phi_quad:.0f}°\nSNR={max_snr_quad:.2f}', 
             fontsize=12)
    
    # 3.5 設置其他屬性
    plt.xlabel('方位角 (度)', fontsize=12)
    plt.ylabel('SNR', fontsize=12)
    plt.title(f'SNR vs 方位角 (仰角{ANGLE_MODE}°) - 最佳角度: {best_phi_quad:.0f}°', fontsize=14)
    plt.grid(True)
    plt.xlim(0, 360)
    plt.xticks(np.arange(0, 361, 45))
    plt.legend(loc='best')
    

    plt.annotate(f'模型資訊:\nR² = {r2_quad:.3f}\nRMSE = {rmse_quad:.3f}',
                xy=(0.7, 0.02), xycoords='figure fraction',
                bbox=dict(boxstyle="round,pad=0.3", facecolor='lightgray', alpha=0.5),
                fontsize=10)
    
    plt.tight_layout()
    
    # 保存圖片
    output_file = os.path.join(PROJECT_ROOT, f'snr_angle_{ANGLE_MODE}_analysis.png')
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"圖表已保存為: {output_file}")
    
    # 顯示圖表
    plt.show()

if __name__ == "__main__":
    main()