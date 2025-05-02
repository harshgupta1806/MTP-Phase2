from Scripts.imports import *
from .utils import log_model_info
from Scripts.get_data import load_data
from Scripts.model import create_model
from .utils import load_configs

def train():
    model = create_model()
    X_train, y_train, X_val, y_val, class_names = load_data()
    model.fit(X_train, y_train, epochs=EPOCHS, batch_size=BATCH_SIZE, validation_data=(X_val, y_val), verbose=1)
    log_model_info(MODEL_NAME, model, OPTIMIZER, EPOCHS, BATCH_SIZE, LOG_DIR)
    model.save(CPU_MODEL)
    print(f"[info] Model saved at {CPU_MODEL}")
    return model, X_train

def convert_to_tflite(X):
    model = tf.keras.models.load_model(CPU_MODEL)
    def representative_data_gen():
        for i in range(100):
            img = X[i].astype(np.float32) / 255.0  # Normalize to float32
            img = np.expand_dims(img, axis=0)      # Add batch dim
            yield [img]

    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = representative_data_gen
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.uint8                # For Edge TPU
    converter.inference_output_type = tf.float32          # For Edge TPU            

    tflite_quant_model = converter.convert()

    # === SAVE MODEL ===
    with open(TFLITE_MODEL, "wb") as f:
        f.write(tflite_quant_model)

    print("[info] Quantized model saved at: {}".format(TFLITE_MODEL))

if __name__ == "__main__":
    model, X = train()
    convert_to_tflite(X)
    print("[info] Training and conversion completed successfully.")


