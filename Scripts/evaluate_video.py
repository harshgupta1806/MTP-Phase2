import cv2
import os
import numpy as np
import time
from tensorflow.keras.models import load_model
from collections import defaultdict
from .utils import load_configs  # To load configuration files (like stream settings)

# Load configuration settings from YAML file
configs = load_configs('./config/test.yaml')

# Extract stream type, resolution, and model path from the config
VIDEO_PATH = configs['data']['video_path']
MODEL_DIR = configs['model']['model_dir'] 
MODEL_NAME = configs['model']['model_name'] # Change to your model path
MODEL_PATH = os.path.join(MODEL_DIR, MODEL_NAME + configs['extantion'][configs['device']])  # Change to your model path
DEVICE = configs['device']  # Device type (cpu or tpu)

IMG_SIZE = (configs['image']['img_size'], configs['image']['img_size'])
FRAMES_PER_SAMPLE = 10
CLASS_NAMES = ['lying', 'sitting', 'standing', 'walking']

# === Detect TPU or CPU ===
is_tpu = (DEVICE == "tpu")

# Try importing EdgeTPU interpreter
try:
    from pycoral.utils.edgetpu import make_interpreter as coral_make_interpreter
    from pycoral.adapters import common as coral_common
    PY_CORAL_AVAILABLE = True
except ImportError:
    PY_CORAL_AVAILABLE = False

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

# === Open Video ===
print(f"[INFO] Opening video: {VIDEO_PATH}")
cap = cv2.VideoCapture(VIDEO_PATH)
if not cap.isOpened():
    print(f"[ERROR] Could not open video file: {VIDEO_PATH}")
    exit()

# === Initialize Variables ===
frame_buffer = []
cntr = 0
total_per_class = defaultdict(int)
correct_per_class = defaultdict(int)
total_frames = 0

# === Process Video Frames ===
while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Resize and convert to grayscale
    resized_frame = cv2.resize(frame, IMG_SIZE)
    gray_frame = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2GRAY)
    gray_frame = np.expand_dims(gray_frame, axis=-1)
    frame_buffer.append(gray_frame)

    if len(frame_buffer) > FRAMES_PER_SAMPLE:
        frame_buffer.pop(0)

    # Update TRUE_LABEL dynamically based on frame number
    if cntr < 50:
        TRUE_LABEL = "lying"
    elif cntr < 100:
        TRUE_LABEL = "sitting"
    elif cntr < 150:
        TRUE_LABEL = "standing"
    elif cntr < 200:
        TRUE_LABEL = "walking"
    else:
        TRUE_LABEL = "unknown"

    # Run inference when buffer is ready
    if len(frame_buffer) == FRAMES_PER_SAMPLE:
        stacked = np.concatenate(frame_buffer, axis=-1)
        input_tensor = np.expand_dims(stacked, axis=0)

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
        inference_time = time.time() - start

        pred_label = CLASS_NAMES[np.argmax(output_data)]
        total_frames += 1

        # Update class counts
        if TRUE_LABEL in CLASS_NAMES:
            total_per_class[TRUE_LABEL] += 1
            if pred_label == TRUE_LABEL:
                correct_per_class[TRUE_LABEL] += 1

        # Print prediction info
        print(f"[Frame {total_frames}] True: {TRUE_LABEL}, Pred: {pred_label}, Time: {inference_time:.2f}s")

        # Overlay predictions
        cv2.putText(frame, f"True     : {TRUE_LABEL}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        cv2.putText(frame, f"Predicted: {pred_label}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.putText(frame, f"InferTime: {inference_time:.2f}s", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)

    # Show frame
    cv2.imshow("Live Video Inference", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    cntr += 1

# === Accuracy Summary ===
print("\n[INFO] Total Frames Processed:", total_frames)
print("[INFO] Class-wise Accuracy:")

for cls in CLASS_NAMES:
    total = total_per_class[cls]
    correct = correct_per_class[cls]
    acc = (correct / total * 100) if total > 0 else 0
    print(f"  - {cls:9s}: {correct}/{total} correct — Accuracy: {acc:.2f}%")

# === Clean Up ===
cap.release()
cv2.destroyAllWindows()
print("[INFO] Video processing completed.")
