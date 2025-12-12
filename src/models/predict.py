# src/models/predict.py
import cv2
import sys
import argparse
import numpy as np
from pathlib import Path
import tensorflow as tf
from src.data.data_preprocessing import preprocess_input
from src.data.data_loader import get_image_datasets
import yaml

def load_config():
    cfg_path = Path(__file__).resolve().parents[1] / "config" / "config.yaml"
    with open(cfg_path, "r") as f:
        return yaml.safe_load(f)

def preprocess_frame(frame, img_size):
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame = cv2.resize(frame, img_size)
    frame = np.expand_dims(frame, axis=0)
    frame = preprocess_input(frame)
    return frame

def main():
    cfg = load_config()
    img_size = tuple(cfg["training"]["image_size"])
    best_path = Path(__file__).resolve().parents[2] / cfg["paths"]["best_model_path"]

    _, _, class_names = get_image_datasets()
    model = tf.keras.models.load_model(best_path)

    parser = argparse.ArgumentParser()
    parser.add_argument("--image", help="Path to image file", default=None)
    parser.add_argument("--video", help="Path to video file", default=None)
    parser.add_argument("--webcam", action="store_true", help="Use webcam for live predictions")
    args = parser.parse_args()

    if args.image:
        img = cv2.imread(args.image)
        inp = preprocess_frame(img, img_size)
        preds = model.predict(inp)
        label_idx = int(np.argmax(preds, axis=1)[0])
        label = class_names[label_idx]
        prob = float(np.max(preds))
        print(f"Prediction: {label} ({prob:.4f})")
    else:
        # video or webcam
        if args.video:
            cap = cv2.VideoCapture(args.video)
        else:
            cap = cv2.VideoCapture(0)

        if not cap.isOpened():
            print("Error: could not open video source")
            return

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            inp = preprocess_frame(frame, img_size)
            preds = model.predict(inp)
            label_idx = int(np.argmax(preds, axis=1)[0])
            label = class_names[label_idx]
            prob = float(np.max(preds))

            # overlay
            text = f"{label}: {prob:.2f}"
            cv2.putText(frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
            cv2.imshow("Mask Detection", frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
