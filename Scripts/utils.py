import time
import numpy as np
from PIL import Image
from tqdm import tqdm
import yaml
import os   
import io
from datetime import datetime

def load_configs(*file_paths):
    merged_config = {}
    for file_path in file_paths:
        with open(file_path, 'r') as file:
            config = yaml.safe_load(file) or {}
            merged_config.update(config)
    return merged_config


def log_model_info(model_name, model, optimizer = 'adam', epochs = '10', batch_size = '32', file_path="model_info.txt"):
    """
    Appends model name, summary, and total parameters to a text file.
    """
    # Capture model.summary()
    stream = io.StringIO()
    model.summary(print_fn=lambda x: stream.write(x + "\n"))
    summary_str = stream.getvalue()
    stream.close()

    # Total trainable parameters
    total_params = model.count_params()

    # Timestamp
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Log content
    log_text  = f"\n{'='*60}\n"
    log_text += f"Timestamp     : {timestamp}\n"
    log_text += f"Model Name    : {model_name}\n"
    log_text += f"Optimizer     : {optimizer}\n"
    log_text += f"Epochs        : {epochs}\n"
    log_text += f"Batch Size    : {batch_size}\n"
    log_text += f"Total Params  : {total_params}\n"
    log_text += f"\nModel Summary:\n{summary_str}\n"

    # Append to file
    with open(file_path, "a", encoding="utf-8") as f:
        f.write(log_text)

    print(f"✅ Appended model info for '{model_name}' to {file_path}")

