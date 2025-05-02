import os
import time
import warnings
import numpy as np
from tqdm import tqdm
import tensorflow as tf
import tflite_runtime.interpreter as tflite
from PIL import Image

warnings.filterwarnings('ignore', category=UserWarning, module='numpy')


def is_edgetpu_available():
    try:
        tflite.load_delegate("libedgetpu.so.1.0")
        return True
    except (ValueError, OSError):
        return False


def load_model():
    if is_edgetpu_available():
        print("Using Edge TPU model.")
        return tflite.Interpreter(
            model_path="/home/raghav/mtp/phase2/models/model_cropped_edgetpu.tflite",
            experimental_delegates=[tflite.load_delegate("libedgetpu.so.1.0")])
    else:
        print("Using CPU model.")
        return tf.lite.Interpreter(model_path="/home/raghav/mtp/phase2/models/model_cropped.tflite")

def load_sequence_as_input(image_paths, target_size=(224, 224), expected_channels=10):
    """
    Load a sequence of images and stack them along the channel dimension.
    If more than 10 channels (e.g. 30 from 10 RGB), slice or adjust accordingly.
    """
    frames = []
    for img_path in image_paths:
        image = Image.open(img_path).convert('RGB').resize(target_size)
        image_np = np.array(image)  # Shape: (H, W, 3)
        frames.append(image_np)

    stacked = np.concatenate(frames, axis=-1)  # (H, W, 3*10) = (224, 224, 30)
    
    # Slice to expected channel count if needed
    if stacked.shape[-1] > expected_channels:
        stacked = stacked[:, :, :expected_channels]

    input_data = np.expand_dims(stacked, axis=0)  # Shape: (1, H, W, C)
    return input_data.astype(np.uint8)


def run_inference(test_root_path, interpreter, image_shape=(224, 224), sequence_length=10):
    """
    Run inference using sequences of frames (e.g., 10 stacked images).
    Assumes each class folder has multiple sequential images.
    """
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    expected_channels = input_details[0]['shape'][3]

    class_names = sorted([d for d in os.listdir(test_root_path) if os.path.isdir(os.path.join(test_root_path, d))])
    class_counts = {cls: {"correct": 0, "total": 0} for cls in class_names}

    total_images = 0
    total_correct = 0
    total_inference_time = 0

    for class_name in class_names:
        class_dir = os.path.join(test_root_path, class_name)
        image_files = sorted([f for f in os.listdir(class_dir) if f.lower().endswith(('png', 'jpg', 'jpeg'))])

        # Slide over the image sequence in windows of 10
        for i in tqdm(range(len(image_files) - sequence_length + 1), desc=f"Processing {class_name}"):
            sequence_files = image_files[i:i+sequence_length]
            sequence_paths = [os.path.join(class_dir, f) for f in sequence_files]
            input_data = load_sequence_as_input(sequence_paths, image_shape, expected_channels)

            # Convert to float if model expects it
            if input_details[0]['dtype'] == np.float32:
                input_data = input_data.astype(np.float32) / 255.0

            interpreter.set_tensor(input_details[0]['index'], input_data)

            start_time = time.time()
            interpreter.invoke()
            end_time = time.time()

            output_data = interpreter.get_tensor(output_details[0]['index'])
            predicted_idx = np.argmax(output_data)
            predicted_label = class_names[predicted_idx]

            is_correct = (predicted_label == class_name)

            total_images += 1
            total_correct += int(is_correct)
            total_inference_time += (end_time - start_time)

            class_counts[class_name]["total"] += 1
            class_counts[class_name]["correct"] += int(is_correct)

            print(f"Seq: {sequence_files[0]}-{sequence_files[-1]}, True: {class_name}, Predicted: {predicted_label}, {'✓' if is_correct else '✗'}")

    # Summary
    accuracy = total_correct / total_images * 100
    avg_time = total_inference_time / total_images
    print(f"\nTotal sequences: {total_images}")
    print(f"Overall Accuracy: {accuracy:.2f}%")
    print(f"Avg inference time: {avg_time:.4f} seconds")

    print("\nClass-wise Accuracy:")
    for cls, stats in class_counts.items():
        total = stats["total"]
        correct = stats["correct"]
        acc = (correct / total * 100) if total > 0 else 0.0
        print(f"{cls:<10} - Accuracy: {acc:.2f}% ({correct}/{total})")



if __name__ == "__main__":
    test_data_path = "/home/raghav/mtp/phase2/test"  # Change as needed
    interpreter = load_model()
    interpreter.allocate_tensors()
    run_inference(test_data_path, interpreter)
