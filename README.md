# D-Blur: Diffusion Coefficient Prediction from Single PSFs

D-Blur is a deep learning framework that extracts diffusion coefficients directly from single-molecule point spread functions (PSFs) in single-molecule microscopy.

This approach allows accurate estimation of molecular dynamics from blurred images using a U-Net-based architecture.

---

## 📂 Project Structure

- Model Training 
- Simulation 
  - `train_locD_padding.ipynb` – Notebook for training the U-Net on simulated or experimental PSF data with padding.
  - `Evaluation.ipynb` – Performs localization evaluation, R² scoring, KNN-based matching, error histogram analysis.
  - `Experiment_data.ipynb` – For checking individual files and aggregating multiple experimental datasets.
 
 
- `requirements.txt` – Python dependency list.

---

## 🚀 Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
