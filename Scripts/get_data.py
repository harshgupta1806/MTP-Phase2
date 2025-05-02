
from Scripts.imports import *


def load_sequence_images(label_folder, label_index):
    images = []
    image_files = sorted(os.listdir(label_folder))
    for i in range(0, len(image_files) - FRAMES_PER_SAMPLE + 1):
        frame_stack = []
        for j in range(FRAMES_PER_SAMPLE):
            img_path = os.path.join(label_folder, image_files[i + j])
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
            frame_stack.append(img)
        stacked = np.stack(frame_stack, axis=-1)  # shape: (224, 224, 10) 
        images.append(stacked)
    labels = [label_index] * len(images)
    return images, labels

def load_data():
    X, y = [], []
    class_names = sorted(os.listdir(DATA_DIR))
    for idx, label in enumerate(class_names):
        print(f"[info] Loading... {label}")
        folder_path = os.path.join(DATA_DIR, label)
        if os.path.isdir(folder_path):
            imgs, lbls = load_sequence_images(folder_path, idx)
            X.extend(imgs)
            y.extend(lbls)

    X = np.array(X, dtype=np.uint8)  # Required for uint8 quantization
    X_normalized = X.astype(np.float32) / 255.0  # Used for training
    y = tf.keras.utils.to_categorical(y, num_classes=NUM_CLASSES)

    # === SPLIT ===
    X_train, X_val, y_train, y_val = train_test_split(X_normalized, y, test_size=0.2, random_state=100)

    return X_train, y_train, X_val, y_val, class_names