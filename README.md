# Clinical Predictive Modeling with CoI and TFCAM

This repository contains the implementation and experiments for two advanced interpretable deep learning frameworks for clinical time-series prediction:

- **Chain-of-Influence (CoI)** – Models and traces how clinical features influence each other across time, creating a transparent "audit trail" of the prediction process.
- **Temporal-Feature Cross Attention Mechanism (TFCAM)** – Captures dynamic interactions between clinical features across time, providing multi-level interpretability.

Both models are designed to address the "black box" limitation of deep learning in high-stakes healthcare settings, offering state-of-the-art predictive performance with clinically meaningful explanations.

## 📄 Research Papers

| Model | Paper Link |
|-------|-------------|
| Chain-of-Influence (CoI) | [arXiv:2510.09895](https://arxiv.org/html/2510.09895) |
| TFCAM | [arXiv:2503.19285](https://arxiv.org/pdf/2503.19285) |
| CoI GitHub Repository | [yubol-bobo/Chain-of-Dynamics](https://github.com/yubol-bobo/Chain-of-Dynamics) |

## 📁 Repository Structure

| File | Description |
|------|-------------|
| `DataPreprocessing_1.ipynb` | Extracts raw data from BigQuery, performs outlier detection, aggregation, and initial processing for model input. |
| `Model_GPT.ipynb` | Trains and evaluates the CoI model on preprocessed CKD data. Contains full training and testing pipeline. |
| `Plausibility_GPT.ipynb` | Uses SHAP to check clinical plausibility of model predictions and ensure alignment with medical knowledge. |
| `TFCAM.ipynb` | Trains and evaluates the TFCAM model on the same dataset, with interpretability analysis. |
| `enhanced_preprocessing.py` | Advanced preprocessing pipeline (taken from the official CoI repository) for temporal-aware imputation, feature engineering, and resampling. |
| `model.py` | PyTorch implementation of the CoI model, adapted from the official GitHub repository. |
| `tfcam.py` | PyTorch implementation of the TFCAM model with cross-temporal attention mechanism. |
| `train_coi.py` | Training utilities for CoI, including hyperparameter search, early stopping, and evaluation metrics. |
| `requirements.txt` | Python dependencies for the project. |
