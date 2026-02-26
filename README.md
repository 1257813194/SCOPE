# SCOPE

Official implementation for the paper **"SCOPE: [A generative model for decoding cellular evolution from unpaired single-cell transcriptomic snapshots via Schrodinger Bridge]"**.

This repository contains the source code for the SCOPE model, data preprocessing tools (incorporating STAGATE), and Jupyter notebooks to reproduce the results presented in the paper.

## 📂 Repository Structure

*   **`scope/`**: Core library containing the model architecture, dataset configurations, and training logic.
*   **`STAGATE_pyG/`**: A Graph Attention Auto-Encoder framework used for spatial transcriptomics data preprocessing.
*   **`*.ipynb`**: Jupyter notebooks for reproducing experiments and figures.

## 🛠️ Environment Setup

This project depends on `PyTorch`, `Scanpy`, and `PyG` (PyTorch Geometric). We recommend using Anaconda to manage the environment.

```bash
# Clone the repository
git clone https://github.com/your_username/scope.git
cd scope

# Create a virtual environment (Python 3.8+ recommended)
conda create -n scope_env python=3.9
conda activate scope_env

# Install standard dependencies
pip install numpy pandas scipy matplotlib scanpy tqdm

# Install PyTorch
# Please verify your CUDA version and install the appropriate PyTorch version from https://pytorch.org/
# Example for CUDA 11.8:
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

> **Note**: The `STAGATE_pyG` package is included in this repository to ensure compatibility. You do not need to install it separately via pip; simply ensure the folder is present in your working directory.

## 💾 Data Preparation

The datasets required to reproduce the results are not included in this repository due to size constraints.

1.  **Download Data**: Please download the dataset archive `Data.rar` from our Google Drive:
    *   https://drive.google.com/file/d/1PLaDfJolvY1gS23UMKHR4MQxLJFCmQ59/view?usp=sharing
2.  **Extract**: Unzip `Data.rar` into the root directory of this project.

The directory structure should look like this:
```
project_root/
├── Data/                 # Extracted data folder
├── scope/                # Source code
├── STAGATE_pyG/          # Preprocessing tools
├── 2.2 SCOPE_...ipynb    # Reproduction notebooks
└── ...
```

## 🚀 Usage & Reproduction

The notebooks are numbered according to the sections in the paper. Models are trained and evaluated in the following order:

### 1. Simulated / Time-Series Data
*   **`2.2 SCOPE_trained_on_day2_to_day6.ipynb`**: Main training pipeline for time-series data.
*   **`2.2 SCOPE_trained_on_day2_and_day6.ipynb`**: Ablation study using only start and end timepoints.
*   **`2.2 SCOPE_...with_recorders.ipynb`**: Experiments incorporating lineage recorders.

### 2. Human Embryonic Development
*   **`2.3 SCOPE_human embryonic development_prematched.ipynb`**: Analysis on pre-matched embryonic data.
*   **`2.3 SCOPE_human embryonic development_no_prematch.ipynb`**: Analysis without pre-matching.

### 3. Other Biological Applications
*   **`2.4 SCOPE_cross_sectional_data_with_pseudotime.ipynb`**: Handling cross-sectional data with inferred pseudotime.
*   **`2.5 SCOPE_CRISPR_screens.ipynb`**: Application to CRISPR screen data.
*   **`2.6 SCOPE_human_dorsolateral_prefrontal_cortex.ipynb`**: Analysis of Human DLPFC spatial transcriptomics data.

### 4. Preprocessing
*   **`2.6 data_preprocess_human_dorsolateral_prefrontal_cortex.ipynb`**: Preprocessing pipeline using STAGATE for the DLPFC dataset.

## 🔗 Citation

If you find this code or data useful for your research, please cite our paper:

```bibtex
@article{YourPaper202X,
  title={SCOPE: [Your Paper Title]},
  author={Your Name and Co-authors},
  journal={Journal Name},
  year={202X},
  publisher={Publisher}
}
```

## 🙏 Acknowledgements

This codebase incorporates components from [STAGATE](https://github.com/RucDongLab/STAGATE_pyG) for spatial data processing.

