# src/models/evaluate.py
import yaml
from pathlib import Path
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import tensorflow as tf
from src.data.data_loader import get_image_datasets
from src.data.data_preprocessing import preprocess_input, IMG_SIZE

def load_config():
    cfg_path = Path(__file__).resolve().parents[1] / "config" / "config.yaml"
    with open(cfg_path, "r") as f:
        return yaml.safe_load(f)

def evaluate():
    cfg = load_config()
    _, val_ds, class_names = get_image_datasets()
    num_classes = len(class_names)
    best_path = Path(__file__).resolve().parents[2] / cfg["paths"]["best_model_path"]

    model = tf.keras.models.load_model(best_path)

    # Collect true labels and predictions
    y_true = []
    y_pred = []
    for batch_images, batch_labels in val_ds:
        batch_images = tf.image.resize(batch_images, tuple(cfg["training"]["image_size"]))
        batch_images = preprocess_input(batch_images)
        preds = model.predict(batch_images)
        preds = np.argmax(preds, axis=1)
        y_true.extend(batch_labels.numpy().tolist())
        y_pred.extend(preds.tolist())

    print("Classification report:")
    print(classification_report(y_true, y_pred, target_names=class_names, digits=4))
    print("Confusion Matrix:")
    print(confusion_matrix(y_true, y_pred))

if __name__ == "__main__":
    evaluate()
