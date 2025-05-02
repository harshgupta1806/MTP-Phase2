import tensorflow as tf
import numpy as np
import time
import csv
import os
import sys
import io

# Ensure the output is in UTF-8 encoding
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# ---------- Configuration ----------
models_dir = r"E:\MTP_DATA\sweetSpot"

script_dir = os.path.dirname(os.path.realpath(__file__))
results_file = os.path.join(script_dir, "results_cpu_models.csv")  # Save the results CSV in the same directory
device = "CPU"

# ---------- Load test dataset ----------
(_, _), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
x_test = x_test.reshape(-1, 28, 28, 1).astype('uint8')  # Expected input dtype

# ---------- Helper to count total model parameters ----------
def count_tflite_parameters(interpreter):
    total_params = 0
    for detail in interpreter.get_tensor_details():
        shape = detail['shape']
        if shape is not None and len(shape) > 0:
            total_params += np.prod(shape)
    return total_params

# ---------- Inference function ----------
def predict(interpreter, image, input_details, output_details):
    interpreter.set_tensor(input_details[0]['index'], np.expand_dims(image, axis=0))
    start = time.time()
    interpreter.invoke()
    end = time.time()
    output = interpreter.get_tensor(output_details[0]['index'])
    return np.argmax(output), end - start

# ---------- Write fresh CSV header ----------
header = ["Model_Name", "Model_Size", "Total_Parameters", "Total_Time(s)", "Avg_Time_per_Image(ms)", "Accuracy(%)", "Device"]
with open(results_file, mode="w", newline='') as f:
    writer = csv.writer(f)
    writer.writerow(header)

# ---------- Loop over 5 models ----------
for i in range(1, 6):
    model_name = f"model_{i}"
    model_path = os.path.join(models_dir, f"{model_name}.tflite")

    if not os.path.exists(model_path):
        print(f"❌ Model not found: {model_path}")
        continue

    print(f"\n🔍 Evaluating {model_name}...")

    model_size = os.path.getsize(model_path) / (1024 * 1024)

    # Load TFLite model
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    total_params = count_tflite_parameters(interpreter)

    # Run inference
    correct_predictions = 0
    total_time = 0
    for j in range(len(x_test)):
        if j % 1000 == 0:
            print(f"Processing image {j}...")
        pred_label, time_taken = predict(interpreter, x_test[j], input_details, output_details)
        total_time += time_taken
        if pred_label == y_test[j]:
            correct_predictions += 1

    accuracy = correct_predictions / len(x_test) * 100
    avg_time_per_image = total_time / len(x_test) * 1000  # in ms

    print(f"✔️  Accuracy: {accuracy:.2f}% | Total Time: {total_time:.4f}s | Avg Time: {avg_time_per_image:.2f}ms")

    # Save result to CSV
    row = [model_name, f"{model_size:.2f}", total_params, f"{total_time:.4f}", f"{avg_time_per_image:.2f}", f"{accuracy:.2f}", device]
    with open(results_file, mode="a", newline='') as f:
        writer = csv.writer(f)
        writer.writerow(row)

print(f"\n✅ All results saved to '{results_file}'")
