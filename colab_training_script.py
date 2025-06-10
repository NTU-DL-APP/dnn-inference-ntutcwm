# colab_training_script.py
import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt # For optional plotting

print(f"TensorFlow version: {tf.__version__}")

# --- 1. 載入 Fashion-MNIST 資料集 ---
(x_train_orig, y_train_orig), (x_test_orig, y_test_orig) = keras.datasets.fashion_mnist.load_data()

print(f"Original training images shape: {x_train_orig.shape}") # (60000, 28, 28)
print(f"Original training labels shape: {y_train_orig.shape}") # (60000,)
print(f"Original test images shape: {x_test_orig.shape}")     # (10000, 28, 28)
print(f"Original test labels shape: {y_test_orig.shape}")     # (10000,)

# --- 2. 預處理圖像資料 ---
# 將像素值從 uint8 (0-255) 轉換為 float32 並正規化到 0-1
x_train = x_train_orig.astype('float32') / 255.0
x_test = x_test_orig.astype('float32') / 255.0

# 將圖像資料從 (28, 28) 扁平化為 (784,) 以便輸入全連接層
x_train = x_train.reshape(x_train.shape[0], 784)
x_test = x_test.reshape(x_test.shape[0], 784)

print(f"Processed training images shape: {x_train.shape}") # (60000, 784)
print(f"Processed test images shape: {x_test.shape}")     # (10000, 784)

# --- 3. 預處理標籤資料 (One-Hot 編碼) ---
num_classes = 10
y_train = keras.utils.to_categorical(y_train_orig, num_classes)
y_test = keras.utils.to_categorical(y_test_orig, num_classes)

print(f"One-hot encoded training labels shape: {y_train.shape}") # (60000, 10)
print(f"One-hot encoded test labels shape: {y_test.shape}")     # (10000, 10)
print(f"Example original label: {y_train_orig[0]}, One-hot: {y_train[0]}")

# --- 4. 建立模型 ---
# 符合專案要求的模型：僅使用 Dense 層、ReLU 活化，輸出層 Softmax
model = keras.Sequential([
    keras.layers.InputLayer(input_shape=(784,)), # 輸入層，784 個特徵 (28*28 pixels)
    keras.layers.Dense(256, activation='relu', name='dense_1'),
    keras.layers.Dense(128, activation='relu', name='dense_2'),
    keras.layers.Dense(64, activation='relu', name='dense_3'),
    keras.layers.Dense(num_classes, activation='softmax', name='output_softmax') # 輸出層
])

model.summary()

# --- 5. 編譯模型 ---
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# --- 6. 訓練模型 ---
epochs = 30 # 可以調整
batch_size = 128 # 可以調整
validation_split = 0.1 # 使用 10% 的訓練資料作為驗證集

print("\nStarting model training...")
history = model.fit(x_train, y_train,
                    epochs=epochs,
                    batch_size=batch_size,
                    validation_split=validation_split,
                    verbose=1) # verbose=1 會顯示進度條

print("Model training finished.")

# --- (可選) 繪製訓練過程中的損失和準確率 ---
# plt.figure(figsize=(12, 4))
# plt.subplot(1, 2, 1)
# plt.plot(history.history['loss'], label='Loss')
# plt.plot(history.history['val_loss'], label='Validation Loss')
# plt.title('Model Loss')
# plt.xlabel('Epoch')
# plt.ylabel('Loss')
# plt.legend()

# plt.subplot(1, 2, 2)
# plt.plot(history.history['accuracy'], label='Accuracy')
# plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
# plt.title('Model Accuracy')
# plt.xlabel('Epoch')
# plt.ylabel('Accuracy')
# plt.legend()
# plt.show()

# --- 7. 在測試集上評估模型 ---
print("\nEvaluating model on test data...")
test_loss, test_accuracy = model.evaluate(x_test, y_test, verbose=0)
print(f"Test Loss: {test_loss:.4f}")
print(f"Test Accuracy: {test_accuracy:.4f}")

# --- 8. 儲存模型為 .h5 格式 ---
model_filename = 'fashion_mnist_model.h5'
model.save(model_filename)
print(f"\nModel saved as {model_filename}")
print("You can download this file from Colab's file browser (usually on the left panel).")

# 提示：在 Colab 中，可以使用以下程式碼來觸發檔案下載
# from google.colab import files
# files.download(model_filename)