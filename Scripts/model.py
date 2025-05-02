from Scripts.imports import *

def create_edgetpu_model_v2():
    inputs = tf.keras.Input(shape=(IMG_SIZE, IMG_SIZE, TOTAL_CHANNELS))
    
    # First Conv block
    x = layers.Conv2D(64, 3, padding='same', activation='relu')(inputs)
    x = layers.Conv2D(64, 3, padding='same', activation='relu')(x)  # Extra layer for more features
    x = layers.MaxPooling2D(2)(x)
    
    # Second Conv block
    x = layers.Conv2D(128, 3, padding='same', activation='relu')(x)
    x = layers.Conv2D(128, 3, padding='same', activation='relu')(x)
    x = layers.MaxPooling2D(2)(x)

    # Global Average Pooling
    x = layers.AveragePooling2D(pool_size=(7, 7))(x)
    
    # Flatten + Dense
    x = layers.Flatten()(x)
    x = layers.Dense(128, activation='relu')(x)  # Increased dense size
    x = layers.Dropout(0.5)(x)  # Helps prevent overfitting, makes features more robust

    outputs = layers.Dense(NUM_CLASSES, activation='softmax')(x)
    
    return tf.keras.Model(inputs, outputs)


def create_model():
    model = create_edgetpu_model_v2()
    model.compile(optimizer=OPTIMIZER, loss=LOSS, metrics=METRICES)
    print(f"[info] Model compiled with {OPTIMIZER} optimizer, {LOSS} loss function, and {METRICES} metrics.")
    return model