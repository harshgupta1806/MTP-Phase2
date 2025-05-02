import os
import re
import cv2
import time
import csv
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from .utils import load_configs
from Scripts.imports import *

test_config = load_configs(r"./config/test.yaml")
print("Loaded test config file")

# === CONFIG ===
MODEL_DIR = test_config['model']['model_dir'] 
MODEL_NAME = test_config['model']['model_name'] # Change to your model path
MODEL_PATH = os.path.join(MODEL_DIR, MODEL_NAME + test_config['extantion'][test_config['device']])  # Change to your model path
IMG_SIZE = test_config['image']['img_size']  # Size of the images (height, width)
FRAMES_PER_SAMPLE = test_config['image']['frames_per_sample']  # Number of frames per sample
CHANNELS_PER_FRAME = test_config['image']["channels"]  # Grayscale
NUM_CLASSES = test_config['num_classes']  # Number of classes
CLASS_NAMES = ['lying', 'sitting', 'standing', 'walking']
DEVICE = test_config['device']  # Device type (cpu or tpu)
TEST_DIR = test_config['data']['test_dir']  # Path to your test dataset

timeline = [(0, 0)]  # Initialize timeline for performance tracking
initial_time = time.time()

# === Detect TPU or CPU ===
is_tpu = (DEVICE == "tpu")

# Try importing EdgeTPU interpreter
try:
    from pycoral.utils.edgetpu import make_interpreter as coral_make_interpreter
    from pycoral.adapters import common as coral_common
    PY_CORAL_AVAILABLE = True
except ImportError:
    PY_CORAL_AVAILABLE = False

# === Setup result log ===
RESULT_DIR = test_config['results']['save_dir']  # Directory to save the results
if not os.path.exists(RESULT_DIR):  # Create the directory if it doesn't exist  
    os.makedirs(RESULT_DIR, exist_ok=True)
csv_path = os.path.join(RESULT_DIR, f"test_data_results.csv")

# === Load model ===
print(f"[INFO] Loading model for {'Edge TPU' if is_tpu else 'CPU'}...")

if is_tpu and PY_CORAL_AVAILABLE:
    interpreter = coral_make_interpreter(MODEL_PATH)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()[0]
    output_details = interpreter.get_output_details()[0]
    input_index = input_details['index']
    output_index = output_details['index']
else:
    model = load_model(MODEL_PATH)

print("[INFO] Model loaded successfully.")

# === Helpers ===
def natural_sort_key(s):
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r'(\d+)', s)]

def load_sorted_grayscale_frames(folder_path):
    images = sorted(os.listdir(folder_path), key=natural_sort_key)
    frames = []
    for fname in images:
        img_path = os.path.join(folder_path, fname)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))  # Resize to match model input size
        img = np.expand_dims(img, axis=-1)  # shape: (224, 224, 1)
        frames.append(img)
    return frames

# === Evaluation ===
total = 0
correct = 0
inference_times = []

# For per-class accuracy
class_totals = {cls: 0 for cls in CLASS_NAMES}
class_correct = {cls: 0 for cls in CLASS_NAMES}

for label_name in os.listdir(TEST_DIR):
    label_folder = os.path.join(TEST_DIR, label_name)
    if not os.path.isdir(label_folder):
        continue

    true_label = label_name.lower()
    frames = load_sorted_grayscale_frames(label_folder)
    num_samples = len(frames) - FRAMES_PER_SAMPLE + 1

    print(f"\n[{true_label.upper()}] Total Frames: {len(frames)}, Samples: {num_samples}")

    for i in range(0, num_samples, 1):
        sample = frames[i:i + FRAMES_PER_SAMPLE]
        if len(sample) < FRAMES_PER_SAMPLE:
            continue

        stacked = np.concatenate(sample, axis=-1)  # shape: (224, 224, 10)
        input_tensor = np.expand_dims(stacked.astype(np.uint8), axis=0)  # shape: (1, 224, 224, 10)
        # Inference
        start = time.time()
        if is_tpu and PY_CORAL_AVAILABLE:
            interpreter.set_tensor(input_index, input_tensor)
            interpreter.invoke()
            output_data = interpreter.get_tensor(output_index)
            if 'quantization' in output_details and output_details['quantization'] != (0.0, 0):
                scale, zero_point = output_details['quantization']
                output_data = scale * (output_data.astype(np.float32) - zero_point)
            output_data = output_data.flatten()
        else:
            output_data = model.predict(input_tensor, verbose=0)[0]

        end = time.time()
        timeline.append((start - initial_time, end - initial_time))
        pred_idx = np.argmax(output_data)
        pred_label = CLASS_NAMES[pred_idx]
        # if(true_label == 'standing'):
        #     print(pred_label)
        # print(true_label, pred_label)
        if pred_label == true_label:
            correct += 1
            class_correct[true_label] += 1
        class_totals[true_label] += 1
        total += 1
        inference_times.append(end - start)

# === Metrics ===
accuracy = 100 * correct / total if total else 0
avg_inf_time = np.mean(inference_times) if inference_times else 0
total_duration = sum(inference_times) if inference_times else 0
fps = total / total_duration if total_duration > 0 else 0
print("\n=== TEST EVALUATION ===")
print(f"Total Samples        : {total}")
print(f"Correct Predictions  : {correct}")
print(f"Accuracy (%)         : {accuracy:.2f}")
print(f"Total Time (s)       : {total_duration:.2f}")
print(f"Avg Inference Time   : {avg_inf_time:.4f} sec")
print(f"Approximate FPS      : {total / total_duration:.2f}")
print(f"Device               : {DEVICE}")

print("\n=== PER-CLASS ACCURACY ===")
for cls in CLASS_NAMES:
    if class_totals[cls] > 0:
        acc = 100 * class_correct[cls] / class_totals[cls]
        print(f"{cls:10s}: {acc:.2f}% ({class_correct[cls]}/{class_totals[cls]})")
    else:
        print(f"{cls:10s}: No samples found.")

# === Save CSV ===
csv_header = ["Total Samples", "Correct", "Accuracy (%)", "Total Time (s)", "Avg Inference Time (s)", "FPS", "Device"]
csv_row = [total, correct, round(accuracy, 2), round(total_duration, 4), round(avg_inf_time, 4), round(fps, 2), DEVICE]
file_exists = os.path.exists(csv_path)

with open(csv_path, mode='a', newline='') as f:
    writer = csv.writer(f)
    if not file_exists:
        writer.writerow(csv_header)
    writer.writerow(csv_row)
    print(f"[INFO] Evaluation results saved to: {csv_path}")

inference_csv_path = os.path.join(RESULT_DIR, f"inference_times_{DEVICE}_image.csv")
timeline_csv_path = os.path.join(RESULT_DIR, f"timeline_{DEVICE}_image.csv")

# Save inference times
with open(inference_csv_path, mode='w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(["Frame Index", "Inference Time (s)", "Device", "Input Type"])
    for idx, t in enumerate(inference_times):
        writer.writerow([idx, round(t, 10), DEVICE, "image"])
    print(f"[INFO] Inference times saved to: {inference_csv_path}")

# Save timeline (start and end timestamps)
with open(timeline_csv_path, mode='w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(["Index", "Start Time (s)", "End Time (s)", "Device", "Input Type"])
    for idx, (start, end) in enumerate(timeline[1:]):  # Skip first dummy (0, 0)
        writer.writerow([idx, round(start, 10), round(end, 10), DEVICE, "image"])
    
    print(f"[INFO] Timeline saved to: {timeline_csv_path}")
    


print(f"[INFO] Evaluation results saved to: {csv_path}")