# 瑕疵檢測之光源角度最佳化專案

## 專案簡介 (Overview)

本專案旨在透過實驗與數據分析，找出檢測特定產品瑕疵的最佳光源照射角度。我們透過以下四個主要步驟來達成目標：

1. **資料採集**：使用 Intel RealSense 相機，在固定仰角下，環繞拍攝不同方位角的影像。
2. **區域選取 (ROI)**：在影像上手動標記出感興趣的瑕疵區域。
3. **訊號雜訊比 (SNR) 計算**：量化在不同角度下，瑕疵特徵的清晰度。
4. **模型擬合與最佳化**：建立數學模型，預測並找出理論上的最佳照射角度。

## 環境需求 (Requirements)

本專案基於 Python 3。建議建立一個虛擬環境，並安裝所需的相依套件。

首先，請透過以下指令安裝 `requirements.txt` 中定義的套件：

```bash
pip install -r requirements.txt
```

## 設定 (Configuration)

在執行任何腳本之前，請務必**統一設定分析的仰角**。

打開以下四個檔案，並將檔案最上方的 `ANGLE_MODE` 全域變數設定為您想要的仰角（例如 `45`, `60` 或 `90`）。**請確保四個檔案中的設定值完全相同**。

* `realsense_data_collector.py`
* `roi_selector.py`
* `snr.py`
* `snr_analysis.py`

**範例 (設定為 60 度仰角):**

```python
# 全域變數，用於切換分析角度 (可選: 45 或 60 或 90)
ANGLE_MODE = 60
```


## 執行步驟 (How to Run)

請嚴格依照以下順序執行腳本。假設您已將 `ANGLE_MODE` 設定為 `60`。

### 步驟 1：採集影像資料

執行資料採集程式。程式會開啟一個 GUI 介面，讓您為 8 個不同的方位角拍攝影像。

```bash
python realsense_data_collector.py
```

* **操作**：依照 GUI 上的指示，為 `Group_1` 到 `Group_8` 收集影像，每個組別收集 50 張。
* **輸出**：程式會自動建立一個名為 `angle_60_dataset` 的資料夾，並將影像儲存在對應的子資料夾中。

### 步驟 2：選取目標區域 (ROI)

執行 ROI 選取程式。程式會依序顯示 8 個角度的樣本影像，讓您手動框選瑕疵區域。

```bash
python roi_selector.py
```

* **操作**：在彈出的影像視窗中，用滑鼠左鍵點擊四個點以定義一個四邊形 ROI，然後點擊 "Confirm" 按鈕。為所有 8 組影像完成此操作。
* **輸出**：產生一個 `angle_60_roi_coordinates.json` 檔案，儲存所有 ROI 的座標。

### 步驟 3：計算訊號雜訊比 (SNR)

執行 SNR 分析程式。此腳本會自動讀取影像與 ROI 座標，並計算每個角度的 SNR 分數。

```bash
python snr.py
```

* **操作**：無須手動操作，程式會自動執行並在終端機顯示進度。
* **輸出**：產生一個 `snr_angle_60_results_summary.csv` 檔案，包含每個角度的詳細數據。同時會顯示一張視覺化圖表。

### 步驟 4：分析與最佳化

執行最後的擬合分析程式，找出理論上的最佳角度。

```bash
python snr_analysis.py
```

* **操作**：無須手動操作。
* **輸出**：
    1. 在終端機印出預測的**最佳方位角**與對應的 SNR 分數。
    2. 顯示一張圖表，其中包含原始數據點、擬合的二次曲線以及預測的最佳點。

## 腳本說明 (File Descriptions)

* `realsense_data_collector.py`: 使用 PyQt5 和 pyrealsense2 建立的圖形化資料採集工具。
* `roi_selector.py`: 使用 Matplotlib 的互動功能，手動選取並儲存 ROI 座標。
* `snr.py`: 核心演算法，用於計算影像的訊號、雜訊及最終的 SNR 分數。
* `snr_analysis.py`: 讀取 SNR 結果，進行二次曲面擬合，並找出最佳化角度。
