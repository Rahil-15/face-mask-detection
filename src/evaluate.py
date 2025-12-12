from sklearn.metrics import classification_report
from tensorflow.keras.models import load_model
import numpy as np
import os
import sys

# Add project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.data.data_loader import get_train_test_data, load_config

def evaluate():
    config = load_config()
    
    print("[INFO] loading data...")
    _, testX, _, testY = get_train_test_data()
    
    print("[INFO] loading model...")
    model_path = os.path.join(config['paths']['models'], "mask_detector.model.h5")
    model = load_model(model_path)
    
    print("[INFO] evaluating network...")
    predIdxs = model.predict(testX, batch_size=config['model']['batch_size'])
    
    # For each image in the testing set, we find the index of the
    # label with corresponding largest predicted probability
    predIdxs = np.argmax(predIdxs, axis=1)
    
    print(classification_report(testY.argmax(axis=1), predIdxs, target_names=config['classes']))

if __name__ == "__main__":
    evaluate()
