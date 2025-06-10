# load_data_example.py
import numpy as np
from utils.mnist_reader import load_mnist
# 如果需要 One-Hot 編碼，取消以下註解 (需要安裝 tensorflow 或 scikit-learn)
# from tensorflow.keras.utils import to_categorical
# from sklearn.preprocessing import OneHotEncoder

# --- 1. 載入資料 ---
# 指定 Fashion-MNIST 資料集的路徑
fashion_data_path = './data/fashion/'

# 載入訓練資料
# 'kind' 參數對於 Fashion-MNIST 訓練集通常是 'train'，
# 但根據您提供的檔案名稱 (t10k-images-idx3-ubyte.gz, t10k-labels-idx1-ubyte.gz)，
# 似乎您目前只有測試集檔案。
# 如果您有 'train-images-idx3-ubyte.gz' 和 'train-labels-idx1-ubyte.gz'，
# 請將下面的 't10k' 改為 'train' 來載入訓練集。
# 假設我們先載入 README 中提到的測試集檔案作為範例
# 如果您有訓練集，請取消註解下一行並修改 kind
# x_train, y_train = load_mnist(fashion_data_path, kind='train')

# 根據您 workspace 中的檔案，我們載入 t10k 資料 (通常作為測試集)
x_test, y_test = load_mnist(fashion_data_path, kind='t10k')

# print(f"原始訓練圖像形狀: {x_train.shape}")
# print(f"原始訓練標籤形狀: {y_train.shape}")
print(f"原始測試圖像形狀: {x_test.shape}")
print(f"原始測試標籤形狀: {y_test.shape}")
print(f"第一個測試圖像的前10個像素: {x_test[0, :10]}")
print(f"第一個測試標籤: {y_test[0]}")

# --- 2. 預處理圖像資料 ---
# 將像素值從 uint8 (0-255) 轉換為 float32 並正規化到 0-1
# x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0

# 將圖像資料從扁平的 784 維向量重塑為 28x28 的圖像格式
# (對於全連接層，扁平格式也可以直接使用)
# x_train = x_train.reshape(x_train.shape[0], 28, 28)
x_test = x_test.reshape(x_test.shape[0], 28, 28)

# 如果您的模型期望輸入是 (num_samples, 28, 28, 1) (例如用於卷積層)
# x_train = np.expand_dims(x_train, -1)
# x_test = np.expand_dims(x_test, -1)

# 如果您的模型期望輸入是扁平的 (num_samples, 784) (用於全連接層)
# 則在 load_mnist 之後不需要 reshape 成 28x28，或者在這裡再 reshape 回去
# x_train = x_train.reshape(x_train.shape[0], 784) # 如果之前 reshape 成了 28x28
x_test = x_test.reshape(x_test.shape[0], 784) # 確保是扁平的

# print(f"正規化後的訓練圖像形狀: {x_train.shape}")
print(f"正規化後的測試圖像形狀: {x_test.shape}")
print(f"正規化後的第一個測試圖像的前10個像素: {x_test[0, :10]}")


# --- 3. 預處理標籤資料 (One-Hot 編碼) ---
# Fashion-MNIST 有 10 個類別 (0-9)
num_classes = 10

# 使用 TensorFlow 的 to_categorical
# y_train_one_hot = to_categorical(y_train, num_classes)
# y_test_one_hot = to_categorical(y_test, num_classes)

# 或者使用 scikit-learn 的 OneHotEncoder
# encoder = OneHotEncoder(sparse_output=False, categories='auto')
# y_train_one_hot = encoder.fit_transform(y_train.reshape(-1, 1))
# y_test_one_hot = encoder.transform(y_test.reshape(-1, 1))

# print(f"One-Hot 編碼後的訓練標籤形狀: {y_train_one_hot.shape}")
# print(f"One-Hot 編碼後的第一個訓練標籤: {y_train_one_hot[0]}")
# print(f"One-Hot 編碼後的測試標籤形狀: {y_test_one_hot.shape}")
# print(f"One-Hot 編碼後的第一個測試標籤: {y_test_one_hot[0]}")

print("\n資料載入與預處理範例完成。")
print("請注意：")
print("1. 上述程式碼假設您在 './data/fashion/' 路徑下有 't10k-labels-idx1-ubyte.gz' 和 't10k-images-idx3-ubyte.gz'。")
print("2. 如果您有訓練集檔案 (例如 'train-labels-idx1-ubyte.gz'), 請修改 'kind' 參數並取消註解相關訓練集處理程式碼。")
print("3. One-Hot 編碼部分已註解，因為它需要額外的套件 (TensorFlow 或 scikit-learn)。")
print("   如果您的模型需要 One-Hot 編碼的標籤，請先安裝相應套件並取消註解。")
print("4. 圖像 reshape 的部分，請根據您的模型輸入層期望的形狀進行調整。對於全連接層，通常使用扁平的 (num_samples, 784) 輸入。")