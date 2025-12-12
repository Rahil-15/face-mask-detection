import os
import cv2
import numpy as np
import yaml
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import img_to_array, load_img

def load_config(config_path="src/config/config.yaml"):
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config

def load_data(config):
    raw_data_path = config['paths']['raw_data']
    categories = config['classes']
    input_shape = config['model']['input_shape'][:2] # (224, 224)

    data = []
    labels = []

    for category in categories:
        path = os.path.join(raw_data_path, category)
        if not os.path.exists(path):
            print(f"Warning: Path not found: {path}")
            continue
            
        class_num = categories.index(category)
        
        for img_name in os.listdir(path):
            try:
                img_path = os.path.join(path, img_name)
                image = load_img(img_path, target_size=input_shape)
                image = img_to_array(image)
                
                data.append(image)
                labels.append(class_num)
            except Exception as e:
                print(f"Error loading {img_name}: {e}")

    data = np.array(data, dtype="float32")
    labels = np.array(labels)
    
    # Preprocessing: Scale to [-1, 1] for MobileNetV2
    data = (data / 127.5) - 1.0 
    
    labels = to_categorical(labels, num_classes=len(categories))

    return data, labels

def get_train_test_data(config_path="src/config/config.yaml"):
    config = load_config(config_path)
    data, labels = load_data(config)
    
    (trainX, testX, trainY, testY) = train_test_split(data, labels, test_size=0.20, stratify=labels, random_state=42)
    
    return trainX, testX, trainY, testY

if __name__ == "__main__":
    trainX, testX, trainY, testY = get_train_test_data()
    print(f"Train data shape: {trainX.shape}")
    print(f"Test data shape: {testX.shape}")
