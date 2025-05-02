# Benchmarking & Optimizing Inference Time on Hardware Accelerator

## Project Description 

This project investigates deep learning model optimization and deployment on resource-constrained devices for **real-time human activity recognition** using depth images captured from a **LiDAR CS20 camera**. A variety of models are trained and evaluated to classify activities such as walking, jumping, standing, and lying.

The trained models are benchmarked across **CPU**, **Edge TPU**, and **Raspberry Pi** platforms to measure performance differences, with an emphasis on **reducing inference latency** and maximizing **hardware efficiency**. Additionally, the project explores how **inference time behavior changes with increasing model complexity**, while keeping the input size constant. This enables a fair comparison of models and provides insights into optimal model selection for embedded deployment. 

---

## Hardware Stack
- Coral Edge TPU USB Accelerator
- Raspberry Pi 4
- CS20 LiDAR Depth Camera
---

## System Setup

Follow the steps below to set up the environment and run the project on local machine or edge device.

---

### 1. Clone the Repository

```bash
git clone https://github.com/harshgupta1806/MTP-Phase2.git
cd MTP-Phase2
```

---

### 2. Create a Virtual Environment

Using `venv`:

```bash
python3 -m venv qat-env
qat-env\Scripts\activate  # On Linux: source qat-env\Scripts\activate
```
---

### 3. Install Python Dependencies

Install all required packages from `requirements.txt`:

```bash
pip install -r requirements.txt
```

---

### 4. Edge TPU Setup

For Windows:

``` bash
# Install Edge TPU runtime
1. Download a edgetpu_runtime.zip from link below: 
    "https://github.com/google-coral/libedgetpu/releases/download/release-grouper/edgetpu_runtime_20221024.zip"
2. Extract the ZIP files and double-click the install.bat file inside.
    A console opens to run the install script and it asks whether you want to enable the maximum operating frequency. 
    Running at the maximum operating frequency increases the inferencing speed but also increases power consumption and causes the USB Accelerator to become very hot. 
    If you're not certain your application requires increased performance, you should type "N" to use the reduced operating frequency. You can change this later by re-running this script.
3. Now connect the USB Accelerator to your computer using the provided USB 3.0 cable.

#  Install the PyCoral library
python3 -m pip install --extra-index-url https://google-coral.github.io/py-repo/ pycoral~=2.0
```

For Linux:
```bash
# Install Edge TPU runtime
sudo apt-get update
sudo apt-get install libedgetpu1-std

# Install PyCoral
sudo apt-get install python3-pycoral
```
---

### 5. Raspberry Pi Setup

If deploying on Raspberry Pi:
```bash
Flash your SD card with the "AIY Maker Kit system image", which includes everything you need to use the USB Accelerator.
Link: "https://github.com/google-coral/aiy-maker-kit-tools/releases/download/v20220518/aiy-maker-kit-2022-05-18.img.xz"

Reference-Document: https://aiyprojects.withgoogle.com/maker/
```
---

### 6. Verify Installation

Run a sample evaluation to test your setup:

```bash
python -m Scripts.evaluate

We can see inference results and performance metrics in the terminal. Results are also saved in result folder.
```
---

## Folder Structure
```bash
MTP-Phase2/
├── requirements.txt                # Python dependencies
├── collected_data/                 # Raw input data collected from depth camera
│       ├── frames/                 # Depth frames (.png format)
│       └── pcd/                    # Point Cloud Data (.pcd files)
├── config/                         # YAML configuration files
│       ├── lidar.yaml              # LiDAR sensor configuration
│       ├── model.yaml              # Model parameters
|       ├── test.yaml               # Test config
│       └── train.yaml              # Training config
│
├── log/                            # Output logs
│   ├── mnist.txt
│   ├── model.txt
│   ├── requirements.txt
│   └── SYSDKLog.txt
│
├── Models/                         # All trained models (.h5, .tflite, EdgeTPU versions)
│   ├── complex_mnist_model.*
│   ├── final_lidar.*
│   ├── lidar_model_*.*
│   ├── test_*.*                    # Includes edgeTPU logs and variants
│
├── plots/                      # Visualizations of model accuracy, timing, etc.
│   ├── accuracy_comparison.png
│   ├── inference_time_heatmap.png
│   └── ... (more plots)
│
├── results/                    # CSV files containing evaluation results and inference times
│   ├── *_results.csv
│   ├── inference_times_*.csv
│   └── timeline_*.csv
│
├── Scripts/                    # All Python scripts and notebooks
│   ├── collect_data.py
│   ├── evaluate.py
│   ├── ...
│   └── Data_Collection/        # SDK and utilities for collecting LiDAR data
│       ├── *.h5, *.py
│       ├── dll/
│       ├── log/
│       ├── parameters/
│       ├── pcd/
│       └── so/
│
└── sweetSpot/                  # Models and results for CPU vs TPU performance comparison
    ├── code*_cpu.py
    ├── code*_tpu.py
    ├── model_*.tflite
    └── results_*.csv
```
---

## Dataset Collection

<!-- - **Source**: CS20 LiDAR camera
- **Format**: Depth image sequences (PNG), 640x480 resolution
- **Activities**: Walking, Sitting, Standing, Lying
- **Preprocessing**:
  - Resizing to 92x92
  - Grayscale normalization
  - Background subtraction
  - Frame sequencing (10 frames per sample) -->

- **Hardware Required**
    - Hard Disk
    - Lidar CS20 Camera
    - USB Type - C
- **Running Data Collection Script**
    - Open lidar.yaml in config folder and set following parameters:
    ```bash
        frame_dir : ./collected_data/frames
        pcd_dir : ./collected_data/pcd
        init_frame_number : 0
        streamtype : SYStreamTypeEnum.SYSTREAMTYPE_DEPTH  # SYStreamTypeEnum.SYSTREAMTYPE_DEPTH
        resolution : SYResolutionEnum.SYRESOLUTION_640_480 # SYResolutionEnum.SYRESOLUTION_640_480
        save_pcd : True
        save_frame : True 
        is_live : False
        model_path : "./Models/lidar_model_1.h5"
    ```
    - Run Following Command: 
        ```bash
        # Ensure current dir is MTP-Phase2
            python -m Scripts.collect_data
        ```
- Depth Frames of Resolution 480x640 saved in ./collected_data/frames folder and pcd in ./collected_data/pcd
---

## Config Overview

All configuration files are stored in the `config/` directory. These YAML files allow you to adjust paths, hyperparameters, model types, and camera settings without modifying the code.

| Config File        | Description |
|--------------------|-------------|
| `train.yaml`       | Training configuration (epochs, batch size, model name, data_path, input image configs etc.). |
| `test.yaml`        | Evaluation and test parameters (test_data path, model_name, model_ directoy, input image information, result directory, device, etc.). |
| `model.yaml`       | Model configuration (input name, optimizer, loss function, metrics, and path where model summary and other informantion saved after training). |
| `lidar.yaml`       | LiDAR camera settings (streamType, Resolution). Path to save frames and pcd Data. |

You can easily edit these files before running any script to customize your workflow.

---

## Scripts Overview

The `Scripts/` folder contains all the core Python scripts and notebooks used throughout the project—from data collection and model training Conversion to tflite to evaluation and visualization.

### Contents

| Script / Folder                 | Description |
|--------------------------------|-------------|
| `collect_data.py`              | Collects depth and point cloud data using the LiDAR sensor and save it in specific folder. |
| `train.py`                     | Trains a model on the collected depth data. and convert it in tflite format |
| `evaluate.py`                  | Evaluates the trained model and outputs performance metrics in cmd and save results in csv file present in `/results` folder (For Inference on CPU / CPU + TPU)|
| `evaluate_rpi.py`              | Evaluates the trained model and outputs performance metrics in cmd (For Inference on Rpi / Rpi + TPU) |   
| `predict_live.py`              | Performs real-time inference using model specified in `./config/lidar.yaml` using live depth frames captured by lidar CS20 Camera. |
| `model.py`                     | Contains the architecture definition. This defination of model is used when we train the model using `train.py`|
| `imports.py`                   | Centralized common imports for the project. |
| `Data_Collection/`             | Contains SDK scripts, DLLs, `.pcd` samples, and helpers for data acquisition. |

---

## How to Run the Scripts

Before running, Ensure base dir is `MTP-Phase2` and then activate the virtual environment using following command:

```bash
# Windows
.\qat-env\Scripts\activate

# Linux/Mac
source qat-env/bin/activate
```

#### 1. Collect Data
```bash
python -m Scripts.collect_data
```

#### 2. Train the Model
```bash
python -m Scripts.train
```

#### 3. Evaluate the Model on CPU/CPU + TPU
```bash
python -m Scripts.evaluate
```

#### 3. Evaluate the Model on Respberry Pi /Respberry Pi + TPU
```bash
python -m Scripts.evaluate_rpi
```

#### 4. Run Real-Time Inference 
* Before running script, SET is_live: True. save_frame: False, save_pcd: False and set model path in `./config/lidar.yaml`

```bash
python -m Scripts.predict_live.py
```

---

### Notes
- Before Running any script make sure current dir is MTP-Phase2
- Make sure your depth camera is connected before running data collection or live inference.
- If inference is on Respberry Pi make sure to change model_path and test_dir as needed in `evaluate_rpi.py`
- Modify YAML files in the `config/` directory to change paths, model parameters, or sensor settings.
- Trained models are stored in the `Models/` directory.
- Logs can be found in the `log/`.
- Evaluation results are saved in the `results/` directory.
---


## Results Summary

| Metric                | CPU         | Edge TPU     |
|-----------------------|-------------|--------------|
| Accuracy              | 76.35%      | 71.92%       |
| Inference Time (avg)  | 137 ms      | 14.6 ms      |
| Approximate FPS       | 7.29        | 68.34        |

> TPU achieved ~9.3× faster inference with slight accuracy trade-off.

---

<!-- ### Harsh Gupta & Raghav Gupta 
*M.Tech Project*  
*Department of Computer Science and Engineering*  
*Indian Institute of Technology Madras*  
*Advisor: Dr. Ayon Chakraborty*  

--- -->


