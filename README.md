# ReMaskNet: Reconstruction & Masking Network for Anomaly Detection

This repository contains the PyTorch implementation of **ReMaskNet**, a framework for unsupervised anomaly detection and localization using the MVTec AD dataset. The model utilizes a multi-stage approach involving feature reconstruction and adaptive masked autoencoding (MAE) to detect anomalies.

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

The `main.py` script handles both training and testing.

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
