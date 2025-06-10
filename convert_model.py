import tensorflow as tf
from tensorflow import keras
import numpy as np
import json
import os

# --- 配置 ---
H5_MODEL_PATH = 'fashion_mnist_model.h5' # 您的 .h5 模型檔案名稱
OUTPUT_MODEL_DIR = 'model'
OUTPUT_JSON_PATH = os.path.join(OUTPUT_MODEL_DIR, 'fashion_mnist.json')
OUTPUT_WEIGHTS_PATH = os.path.join(OUTPUT_MODEL_DIR, 'fashion_mnist.npz')

def convert_h5_to_json_npz(h5_path, output_json_path, output_weights_path):
    """
    將 Keras .h5 模型轉換為 nn_predict.py 所需的 .json (架構) 和 .npz (權重) 格式。
    此版本根據用戶提供的 numpy_nn_test.ipynb 中的邏輯進行修改。
    """
    print(f"正在載入模型: {h5_path}")
    try:
        model = keras.models.load_model(h5_path)
        print("模型載入成功。")
        model.summary()
    except Exception as e:
        print(f"載入模型失敗: {e}")
        return

    # --- 1. 提取並儲存模型權重 (fashion_mnist.npz) ---
    # 參考 numpy_nn_test.ipynb 中的 Step 2 & 3
    model_weights_dict = {}
    print("\n🔍 正在從模型中提取權重...")
    for layer in model.layers:
        layer_weights = layer.get_weights()
        if layer_weights: # 只處理有權重的層
            # print(f"層: {layer.name}") # 可選的詳細輸出
            for i, w_array in enumerate(layer_weights):
                param_name = f"{layer.name}_{i}" # 例如 dense_1_0, dense_1_1
                # print(f"  {param_name}: shape={w_array.shape}") # 可選的詳細輸出
                model_weights_dict[param_name] = w_array
    
    # 確保輸出目錄存在
    if not os.path.exists(OUTPUT_MODEL_DIR):
        os.makedirs(OUTPUT_MODEL_DIR)
        print(f"已建立目錄: {OUTPUT_MODEL_DIR}")

    print(f"\n正在儲存模型權重到: {output_weights_path}")
    try:
        np.savez(output_weights_path, **model_weights_dict)
        print("✅ 模型權重儲存成功。")
    except Exception as e:
        print(f"儲存模型權重失敗: {e}")
        return

    # --- 2. 提取並儲存模型架構 (fashion_mnist.json) ---
    # 參考 numpy_nn_test.ipynb 中的 Step 6
    model_arch_list = []
    print("\n📜 正在提取模型架構...")
    for layer in model.layers:
        if layer.__class__.__name__ == 'InputLayer': # 跳過 InputLayer
            continue

        layer_config = layer.get_config()
        layer_info = {
            "name": layer.name,
            "type": layer.__class__.__name__, # 例如 "Dense", "Flatten"
            "config": layer_config, # 儲存完整的 Keras 層配置
            "weights": [f"{layer.name}_{i}" for i in range(len(layer.get_weights()))]
        }
        model_arch_list.append(layer_info)

    print(f"\n正在儲存模型架構到: {output_json_path}")
    try:
        with open(output_json_path, 'w') as f:
            json.dump(model_arch_list, f, indent=2) # 使用 indent=2 與筆記本一致
        print("✅ 模型架構儲存成功。")
    except Exception as e:
        print(f"儲存模型架構失敗: {e}")
        return

    print("\n模型轉換完成！")
    print(f"架構檔案: {output_json_path}")
    print(f"權重檔案: {output_weights_path}")

if __name__ == '__main__':
    if not os.path.exists(H5_MODEL_PATH):
        print(f"錯誤: 找不到 H5 模型檔案 '{H5_MODEL_PATH}'。")
        print(f"請確認名為 '{H5_MODEL_PATH}' 的檔案存在於專案根目錄，或者修改 H5_MODEL_PATH 變數。")
    else:
        # 確保 TensorFlow 不要佔用所有 GPU 記憶體 (如果有的話)，雖然對於轉換可能不是大問題
        try:
            gpus = tf.config.experimental.list_physical_devices('GPU')
            if gpus:
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
                print("已為 GPU 設定記憶體增長模式。")
        except RuntimeError as e:
            print(f"設定 GPU 記憶體時發生錯誤 (可忽略): {e}")
            
        convert_h5_to_json_npz(H5_MODEL_PATH, OUTPUT_JSON_PATH, OUTPUT_WEIGHTS_PATH)