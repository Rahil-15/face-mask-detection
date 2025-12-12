# src/models/train.py
import os
from pathlib import Path
import yaml
import tensorflow as tf
import numpy as np

from src.data.data_loader import get_image_datasets
from src.data.data_preprocessing import make_augmentation_layer, preprocess_input, IMG_SIZE
from src.features.feature_builder import build_model

def load_config():
    cfg_path = Path(__file__).resolve().parents[1] / "config" / "config.yaml"
    with open(cfg_path, "r") as f:
        return yaml.safe_load(f)

def adapt_dataset(ds):
    # map preprocessing + augmentation
    augmentation = make_augmentation_layer()
    def _process(images, labels):
        images = tf.image.resize(images, IMG_SIZE)
        images = preprocess_input(images)
        images = augmentation(images)
        return images, labels
    return ds.map(_process, num_parallel_calls=tf.data.AUTOTUNE)

def main():
    cfg = load_config()
    train_ds, val_ds, class_names = get_image_datasets()
    num_classes = len(class_names)

    train_ds = adapt_dataset(train_ds)
    val_ds = adapt_dataset(val_ds)

    model = build_model(num_classes)

    models_dir = Path(__file__).resolve().parents[2] / cfg["paths"]["models_dir"]
    models_dir.mkdir(parents=True, exist_ok=True)
    best_path = Path(__file__).resolve().parents[2] / cfg["paths"]["best_model_path"]

    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(best_path),
            monitor="val_accuracy",
            save_best_only=True,
            save_weights_only=False,
            verbose=1
        ),
        tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)
    ]

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=cfg["training"]["epochs"],
        callbacks=callbacks
    )

    # Save final model (again)
    model.save(best_path)
    print(f"Saved model to {best_path}")

if __name__ == "__main__":
    main()
