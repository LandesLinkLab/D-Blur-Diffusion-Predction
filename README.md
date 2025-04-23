# D-Blur: Diffusion Coefficient Prediction from Single PSFs

D-Blur is a deep learning framework that extracts diffusion coefficients directly from single-molecule point spread functions (PSFs) in single-molecule microscopy.

This approach allows accurate estimation of molecular dynamics from blurred images using a U-Net-based architecture.

---

This repository accompanies the work presented in our 2025 paper:  

**D-Blur: A Deep Learning-Enhanced Approach for Resolving Fast Diffusion Dynamics in Single Molecule Microscopy with Motion Blur.**


## Project Structure

- Model Training
  Generate simulated data for model training. 
  - `Step1_Brownian_trajectories.m` - This file generates a set of Brownian‐motion trajectories, each with a random diffusion coefficient within the given range.
  - `Step2_Movie_Simulation.m` - This file generates PSF movies from Brownian trajectories simulated in step 1.
  - `Step3_Post_overlay.m` - This file combines the simulated movies to generate motion-blurred images, preparing the data for model training.
  
- Simulation 
  - `training.ipynb` – This notebook is used for training the U-Net on simulated PSF data.
  - `Evaluation.ipynb` – This notebook performs localization evaluation, R² scoring, KNN-based matching, and error histogram analysis on the trained model.
  - `Experiment_data.ipynb` – This notebook gives an example of applying the model to experimental data, checking individual files, and aggregating multiple experimental datasets.
 
 
- `requirements.txt` – Python dependency list.

## Model Architecture

The model used in this project is adapted from the U-Net architecture introduced in:

> Ronneberger, O., Fischer, P., & Brox, T. (2015). U-Net: Convolutional Networks for Biomedical Image Segmentation. *arXiv preprint arXiv:1505.04597*.

Our adaptation modifies the U-Net to predict both particle localization and diffusion coefficient from motion-blurred point spread functions (PSFs).


## Citation

This repository supports the methods and results presented in the following paper:

> **D-Blur: A Deep Learning-Enhanced Approach for Resolving Fast Diffusion Dynamics in Single Molecule Microscopy with Motion Blur**  
> Dongyu Fan, Nikita Kovalenko, Jagriti Chatterjee, Subhojyoti Chatterjee, Cong Xu, and Emil N. Gillett  
> *Submitted to Chemical & Biomolecule Imaging, 2025.*

If you find this work useful, please consider citing our paper.

## Contact

If you have any questions, feel free to email us: 

**Dongyu Fan**  
University of Illinois Urbana-Champaign  
dongyuf2@illinois.edu  


