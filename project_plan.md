# DNN 推論測試專案執行計畫

## 1. 環境準備與初步分析 (目前階段)
*   **目標**：確認對專案需求的完整理解，並設定好初步的開發方向。
*   **行動**：
    *   已閱讀並分析 `README.md` 以了解專案目標、需求與評分標準。
    *   確認需要實作的 Python 檔案 (`nn_predict.py`) 和資料集位置 (`./data`)。
    *   識別出模型檔案最終需放置的資料夾 (`./model`)，此資料夾目前不存在，後續步驟中需要建立。
    *   了解測試將由 `model_test.py` 執行，且此檔案不可修改。

## 2. 核心函數實作
*   **目標**：在 `nn_predict.py` 中完成 `relu()` 和 `softmax()` 函數的實作。
*   **行動**：
    *   仔細閱讀 `nn_predict.py` 的現有程式碼，了解其架構和資料流程。
    *   根據 NumPy 的語法和神經網路的定義，實作 `relu(x)` 函數。
    *   根據 NumPy 的語法和神經網路的定義，實作 `softmax(x)` 函數，特別注意數值穩定性 (例如，減去最大值技巧)。
*   **產出**：修改後的 `nn_predict.py` 檔案，包含已實作的 `relu` 和 `softmax` 函數。

## 3. 神經網路模型設計與訓練
*   **目標**：設計、訓練並優化一個適用於 Fashion-MNIST 資料集的神經網路模型，使其達到 `README.md` 中定義的準確度要求。
*   **行動**：
    *   **資料準備**：使用 `./data/fashion/` 中的 Fashion-MNIST 資料集。可能需要使用 `utils/mnist_reader.py` 來載入和預處理資料 (例如，正規化圖像像素值、轉換標籤為 one-hot 編碼)。
    *   **模型架構設計 (TensorFlow/Keras)**：
        *   設計一個僅包含**全連接 (Dense)** 層、**ReLU** 活化函數和最後輸出層使用 **Softmax** 活化函數的神經網路。
        *   考慮網路的深度 (層數) 和寬度 (每層的神經元數量) 以平衡模型的表達能力和複雜度。
        *   範例架構：輸入層 -> Dense (ReLU) -> Dense (ReLU) -> Dense (Softmax)。
    *   **模型編譯與訓練**：
        *   選擇合適的優化器 (例如 Adam)、損失函數 (例如 categorical_crossentropy) 和評估指標 (例如 accuracy)。
        *   在 Fashion-MNIST 訓練集上訓練模型，並在驗證集上監控其性能，以避免過度擬合。
        *   調整超參數 (學習率、批次大小、訓練週期等) 以提升模型準確度。
    *   **模型評估**：在測試集上評估模型，確保其泛化能力，並以 `README.md` 中的準確度門檻為目標。
*   **工具**：Google Colab (建議，如 `README.md` 中連結所示) 或本地 TensorFlow 環境。
*   **產出**：
    *   訓練好的 TensorFlow/Keras 模型 (暫存)。
    *   記錄模型架構、訓練過程和最終準確度的筆記或腳本。

## 4. 模型轉換與儲存
*   **目標**：將訓練好的 TensorFlow/Keras 模型轉換為專案要求的 `fashion_mnist.json` (架構) 和 `fashion_mnist.npz` (權重) 格式。
*   **行動**：
    *   **儲存為 .h5 格式**：將訓練好的 TensorFlow/Keras 模型儲存為 `.h5` 檔案 (例如 `my_fashion_mnist_model.h5`)。
    *   **提取模型架構**：
        *   載入 `.h5` 模型。
        *   將模型架構轉換為 JSON 格式並儲存為 `fashion_mnist.json`。可以參考 `README.md` 中提供的 demo notebook 中的做法。
    *   **提取模型權重**：
        *   載入 `.h5` 模型。
        *   提取每一層的權重 (weights) 和偏置 (biases)。
        *   將這些權重和偏置儲存到 `fashion_mnist.npz` 檔案中，使用 NumPy 的 `savez` 或 `savez_compressed` 函數。鍵名 (key names) 需要與 `nn_predict.py` 中載入權重時使用的鍵名一致。
    *   **建立資料夾並放置檔案**：
        *   如果 `./model` 資料夾不存在，則建立它。
        *   將生成的 `fashion_mnist.json` 和 `fashion_mnist.npz` 檔案移動到 `./model` 資料夾中。
*   **工具**：Python、TensorFlow/Keras、NumPy。
*   **產出**：
    *   `model/fashion_mnist.json`
    *   `model/fashion_mnist.npz`

## 5. 本地測試與迭代 (建議)
*   **目標**：在提交之前，本地驗證實作的函數和轉換後的模型是否能正確執行推論。
*   **行動**：
    *   可以考慮在 `model_test.py` (注意不要修改並提交此檔案) 或一個新的測試腳本中，嘗試載入模型並對 `./data/fashion/` 中的一些樣本進行預測，以初步驗證流程的正確性。
    *   檢查預測結果的格式和大致的準確性。
    *   如果發現問題，回到步驟 2 (核心函數實作) 或步驟 3 (模型訓練)、步驟 4 (模型轉換) 進行修正。
*   **產出**：(可選) 本地測試腳本和測試結果。

## 6. 提交與最終測試
*   **目標**：提交所有必要的更改以觸發自動化測試，並根據反饋進行迭代，直到滿足所有要求。
*   **行動**：
    *   將修改後的 `nn_predict.py` 和新建立的 `./model` 資料夾 (包含 `fashion_mnist.json` 和 `fashion_mnist.npz`) 加入版本控制。
    *   提交更改。根據 `README.md`，這將觸發測試。
    *   查看測試結果和評分。
    *   如果未達到預期，根據反饋重複步驟 2 至 5，直到模型達到所需的性能和所有功能都已正確實作。
*   **產出**：通過所有測試並獲得滿意分數的專案版本。

## Mermaid 圖表：專案流程

```mermaid
graph TD
    A[開始] --> B{環境準備與分析};
    B --> C{核心函數實作: relu() & softmax()};
    C --> D{模型設計與訓練 (TensorFlow/Keras)};
    D -- .h5 --> E{模型轉換};
    E -- fashion_mnist.json & fashion_mnist.npz --> F[放置模型檔案至 ./model];
    F --> G{本地測試與迭代 (可選)};
    G --> H{提交與最終測試};
    H -- 未通過 --> C;
    H -- 通過 --> I[結束];