import os
import sys
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.keras.optimizers import Adam

# Add project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.data.data_loader import get_train_test_data, load_config
from src.data.data_preprocessing import get_data_augmentation
from src.features.feature_builder import build_model

def train():
    config = load_config()
    
    # Load data
    print("[INFO] loading data...")
    trainX, testX, trainY, testY = get_train_test_data()
    
    # Data Augmentation
    aug = get_data_augmentation()
    
    # Build Model
    print("[INFO] compiling model...")
    model = build_model(config['model']['input_shape'], config['model']['num_classes'])
    
    opt = Adam(learning_rate=config['model']['learning_rate']) 
    model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"])
    
    # Train
    print("[INFO] training head...")
    H = model.fit(
        aug.flow(trainX, trainY, batch_size=config['model']['batch_size']),
        steps_per_epoch=len(trainX) // config['model']['batch_size'],
        validation_data=(testX, testY),
        validation_steps=len(testX) // config['model']['batch_size'],
        epochs=config['model']['epochs']
    )
    
    # Save Model
    print("[INFO] saving mask detector model...")
    model_save_path = os.path.join(config['paths']['models'], "mask_detector.model.h5")
    model.save(model_save_path)
    
    # Plot training loss and accuracy
    print("[INFO] saving training plot...")
    plt.style.use("ggplot")
    plt.figure()
    plt.plot(np.arange(0, config['model']['epochs']), H.history["loss"], label="train_loss")
    plt.plot(np.arange(0, config['model']['epochs']), H.history["val_loss"], label="val_loss")
    plt.plot(np.arange(0, config['model']['epochs']), H.history["accuracy"], label="train_acc")
    plt.plot(np.arange(0, config['model']['epochs']), H.history["val_accuracy"], label="val_acc")
    plt.title("Training Loss and Accuracy")
    plt.xlabel("Epoch #")
    plt.ylabel("Loss/Accuracy")
    plt.legend(loc="lower left")
    plot_path = os.path.join(config['paths']['models'], "plot.png")
    plt.savefig(plot_path)

if __name__ == "__main__":
    train()
