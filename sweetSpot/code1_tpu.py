import tensorflow as tf
import numpy as np
import time
import os
import csv
from pycoral.utils.edgetpu import make_interpreter
import tflite_runtime.interpreter as tflite
import sys
import io

# Ensure the output is in UTF-8 encoding
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')


# Ensure the script's folder exists (optional)
script_dir = os.path.dirname(os.path.realpath(__file__))
results_file = os.path.join(script_dir, "results_tpu_models.csv")  # Save the results CSV in the same directory

# Ensure the folder for the results file exists
os.makedirs(script_dir, exist_ok=True)

# Load MNIST dataset (test set only)
(_, _), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

# Normalize and reshape test images
x_test = x_test.reshape(-1, 28, 28, 1).astype(np.uint8)

# Function to count model parameters
def count_tflite_parameters(interpreter):
    total_params = 0
    for detail in interpreter.get_tensor_details():
        shape = detail['shape']
        if shape is not None and len(shape) > 0:
            total_params += np.prod(shape)
    return total_params

# Function to run inference on one image
def predict(interpreter, image, input_details, output_details):
    interpreter.set_tensor(input_details[0]['index'], np.expand_dims(image, axis=0))  # Add batch dimension
    start = time.time()
    interpreter.invoke()
    end = time.time()
    time_taken = end - start    
    output = interpreter.get_tensor(output_details[0]['index'])
    return np.argmax(output), time_taken  # Return predicted label

# Function to evaluate a model
def evaluate_model(model_path, model_name, device):
    print(f"\n🔍 Evaluating: {model_path} on {device}")

    # Load the TFLite model
    interpreter = make_interpreter(model_path)
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    model_size_kb = os.path.getsize(model_path) / 1024
    model_size_mb = model_size_kb / 1024
    total_params = count_tflite_parameters(interpreter)

    correct_predictions = 0
    total_time = 0
    for i in range(len(x_test)):
        if i % 1000 == 0:
            print(f"Processing image {i}...")
        pred_label, time_taken = predict(interpreter, x_test[i], input_details, output_details)
        total_time += time_taken
        if pred_label == y_test[i]:  # Compare with ground truth
            correct_predictions += 1

    # Calculate metrics
    accuracy = correct_predictions / len(x_test) * 100
    avg_time_per_image = total_time / len(x_test) * 1000

    # Prepare the results row
    row = [
        model_name,
        f"{model_size_kb:.2f}", f"{model_size_mb:.2f}",
        total_params,
        f"{total_time:.4f}", f"{avg_time_per_image:.2f}",
        f"{accuracy:.2f}", device
    ]

    # Save results to CSV
    with open(results_file, mode="a", newline='') as f:
        writer = csv.writer(f)
        writer.writerow(row)

    print(f"✅ Results for {model_name} on {device} saved.")

# Header for the results CSV
header = ["Model_Name", "Model_Size(KB)", "Model_Size(MB)",
          "Total_Parameters", "Total_Time(s)", "Avg_Time_per_Image(ms)",
          "Accuracy(%)", "Device"]

# Overwrite the results file and add the header
with open(results_file, mode="a", newline='') as f:
    writer = csv.writer(f)
    writer.writerow(header)

# Evaluate each of the 5 models
for i in range(1, 6):
    model_path = f"E:\MTP_DATA\sweetSpot\model_{i}_edgetpu.tflite"
    if os.path.exists(model_path):
        evaluate_model(model_path, f"model_{i}", "Edge TPU")
    else:
        print(f"❌ Model {i} missing: {model_path}")

print(f"\n📁 All results saved to: {results_file}")
