import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split
from .utils import load_configs


train_config = load_configs(r"./config/train.yaml")
print("Train config loaded successfully.")

DATA_DIR = train_config['dir']['train']  # Path to your dataset
IMG_SIZE = train_config['image']['img_size']  # Size of the images (height, width)
FRAMES_PER_SAMPLE = train_config['image']['frames_per_sample']  # Number of frames per sample
NUM_CLASSES = train_config['model']['num_classes']  # Number of classes
CHANNELS_PER_FRAME = train_config['image']["channels"]  # Grayscale
TOTAL_CHANNELS = FRAMES_PER_SAMPLE * CHANNELS_PER_FRAME
EPOCHS = train_config['model']['epochs']  # Number of epochs for training
BATCH_SIZE = train_config['model']['batch_size']  # Batch size for training
MODEL_DIR = train_config['dir']['save_dir']  # Directory to save the model
MODEL_NAME = train_config['model']['name']  # Name of the model
CPU_MODEL = os.path.join(MODEL_DIR, MODEL_NAME + '.h5')  # Full path to save the model
TFLITE_MODEL = os.path.join(MODEL_DIR, MODEL_NAME + '.tflite')  # Full path to save the TFLite model
TPU_MODEL = os.path.join(MODEL_DIR, MODEL_NAME + '_edgetpu.tflite')  # Full path to save the TPU mode

model_config = load_configs(r"./config/model.yaml")
print("Model config loaded successfully.")
OPTIMIZER = model_config['optimizer']  # Optimizer for the model
LOSS = model_config['loss']  # Loss function for the model
METRICES = model_config['metrics']  # Metrics for the model
LOG_DIR = model_config['log']  # Directory to save the logs
