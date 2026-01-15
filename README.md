# ReMaskNet: Regenerate Mask Network for Industrial Anomaly Detection and Segmentation

[![CASE 2025](https://img.shields.io/badge/CASE-2025-4b44ce.svg)](https://ieeexplore.ieee.org/)

This repository contains the official PyTorch implementation of **ReMaskNet**, as presented in the paper **"ReMaskNet: Regenerate Mask Network for Industrial Anomaly Detection and Segmentation"**, which has been accepted by **IEEE CASE 2025**.

ReMaskNet is a novel unsupervised anomaly detection framework that synergizes the strengths of **reconstruction-based, synthesis-based, and embedding-based** methods. To address the challenges of subtle defect detection and unstable masking in existing approaches, we propose a **Dual-level Anomaly Synthesis** strategy (incorporating both image-level and feature-level synthesis) and an **Iterative Feature Inpainting** module.

Unlike traditional methods that rely on random masks, ReMaskNet utilizes a reconstruction-guided segmentation map as an initial mask and refines features through an iterative optimization process. Extensive experiments on the **MVTec AD** dataset demonstrate that ReMaskNet outperforms existing state-of-the-art methods, achieving **99.6% Image-AUROC** and **98.4% Pixel-AUROC**.

## 📂 Project Structure

    ├── main.py           # Main entry point (Training & Testing)
    ├── train.py          # Training logic
    ├── test.py           # Testing logic & Visualization
    ├── run.sh            # Shell script for batch execution
    ├── mvtec.py          # Dataset loader
    ├── ReMaskNet.py      # Model architecture
    ├── utils.py          # Utilities
    └── utils_eval.py     # Evaluation metrics

## 🛠️ Requirements

* Python 3.8+
* PyTorch (CUDA supported)
* Torchvision, Numpy, Scikit-learn, SciPy, OpenCV, Matplotlib, tqdm

Install dependencies via pip:

    pip install torch torchvision numpy scikit-learn scipy opencv-python matplotlib tqdm

## 💾 Data Preparation

The project expects the following directory structure by default. You can change these paths via command-line arguments.

1. **MVTec AD Dataset**: [Download Here](https://www.mvtec.com/company/research/datasets/mvtec-ad)
2. **DTD (Describable Textures Dataset)**: [Download Here](https://www.robots.ox.ac.uk/~vgg/data/dtd/)

Default expected paths:
* MVTec AD: `./MVTec_ad`
* DTD: `./dtd/images`

## 🚀 Usage

The `main.py` script handles both training and testing. You can run it directly using Python or use the provided shell script `run.sh` for batch management.

### 1. Training
Train on specific classes (e.g., `bottle`). Results are saved to `./saved_models`.

    python main.py --mode train --classes bottle --device cuda:0

To train on multiple classes at once:

    python main.py --mode train --classes screw tile toothbrush --batch_size 4

### 2. Testing
Evaluate the model. This will generate a summary CSV and visualizations in `./results`.

    python main.py --mode test --classes bottle --device cuda:0

### 3. End-to-End (Train + Test)
Run training immediately followed by testing:

    python main.py --mode all --classes bottle --epochs 200

### 4. Using the Shell Script (`run.sh`)
For easier configuration management and batch execution, use the provided `run.sh` script.

**Step 1: Modify the script**
Open `run.sh` in a text editor to adjust the parameters:
* **Paths**: Update `SAVE_ROOT` and `RESULT_ROOT` to your preferred directories.
* **Classes**: Edit the `--classes` line to select the objects you want to run (e.g., `bottle cable capsule`).
* **Mode**: Change `--mode` to `train`, `test`, or `all`.

**Step 2: Run the script**

    bash run.sh

## ⚙️ Arguments

You can configure the model using the following arguments in `main.py`:

| Argument | Default | Description |
| :--- | :--- | :--- |
| **General** | | |
| `--mode` | `train` | Execution mode: `train`, `test`, or `all` |
| `--classes` | `['bottle']` | List of object classes to process |
| `--device` | `cuda:0` | Compute device (e.g., `cuda:0`, `cpu`) |
| `--seed` | `0` | Random seed for reproducibility |
| **Paths** | | |
| `--data_root` | `./MVTec_ad` | Path to MVTec AD dataset |
| `--dtd_path` | `./dtd/images` | Path to DTD dataset (for augmentation) |
| `--save_root` | `./saved_models` | Directory to save model checkpoints |
| `--result_root` | `./results` | Directory to save visualizations and CSV results |
| **Hyperparameters** | | |
| `--epochs` | `200` | Number of training epochs |
| `--batch_size` | `4` | Batch size |
| `--lr` | `0.0003` | Learning rate |
| `--image_size` | `288` | Input image size (CenterCropped) |
| `--resize` | `329` | Image resize dimension before cropping |
| `--feature_layers`| `2 3` | Layers of backbone to extract features from |
| `--target_embed_dim`| `1536` | Dimension of the target embedding |
| **Model Specifics** | | |
| `--iteration` | `3` | Number of iterations for the Inpainting training phase |
| `--eval_epoch` | `2` | Interval (in epochs) to run evaluation during training |
| `--eval_iteration`| `3` | Number of inference iterations during testing |

## 📊 Outputs

### Test Results
After running testing, a summary file is generated at:
`./results/test_results_summary.csv`

It contains the AUROC and AP metrics for both Image-level and Pixel-level detection:

| Class | Img_AUROC | Img_AP | Pix_AUROC | Pix_AP |
| :--- | :--- | :--- | :--- | :--- |
| bottle | 0.xxxx | 0.xxxx | 0.xxxx | 0.xxxx |
| ... | ... | ... | ... | ... |
| **AVERAGE**| **0.xxxx** | **0.xxxx**| **0.xxxx** | **0.xxxx**|

### Visualizations
Visual results are saved in:
`./results/visualization/<class_name>/`
* Includes original image, ground truth mask, anomaly heatmap, and segmentation result.
