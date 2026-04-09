Clinical Predictive Modeling with CoI and TFCAM
This repository contains the implementation and experiments for two advanced interpretable deep learning frameworks for clinical time-series prediction:

Chain-of-Influence (CoI) – Models and traces how clinical features influence each other across time, creating a transparent "audit trail" of the prediction process.

Temporal-Feature Cross Attention Mechanism (TFCAM) – Captures dynamic interactions between clinical features across time, providing multi-level interpretability.

Both models are designed to address the "black box" limitation of deep learning in high-stakes healthcare settings, offering state-of-the-art predictive performance with clinically meaningful explanations.

📄 Research Papers
Model	Paper Link
Chain-of-Influence (CoI)	arXiv:2510.09895
TFCAM	arXiv:2503.19285
CoI GitHub Repository	yubol-bobo/Chain-of-Dynamics
📁 Repository Structure
File	Description
DataPreprocessing_1.ipynb	Extracts raw data from BigQuery, performs outlier detection, aggregation, and initial processing for model input.
Model_GPT.ipynb	Trains and evaluates the CoI model on preprocessed CKD data. Contains full training and testing pipeline.
Plausibility_GPT.ipynb	Uses SHAP to check clinical plausibility of model predictions and ensure alignment with medical knowledge.
TFCAM.ipynb	Trains and evaluates the TFCAM model on the same dataset, with interpretability analysis.
enhanced_preprocessing.py	Advanced preprocessing pipeline (taken from the official CoI repository) for temporal-aware imputation, feature engineering, and resampling.
model.py	PyTorch implementation of the CoI model, adapted from the official GitHub repository.
tfcam.py	PyTorch implementation of the TFCAM model with cross-temporal attention mechanism.
train_coi.py	Training utilities for CoI, including hyperparameter search, early stopping, and evaluation metrics.
requirements.txt	Python dependencies for the project.
🚀 Quick Start
1. Clone the Repository
bash
git clone https://github.com/your-username/your-repo-name.git
cd your-repo-name
2. Install Dependencies
bash
pip install -r requirements.txt
3. Data Preparation
Use DataPreprocessing_1.ipynb to extract and prepare your clinical time-series data.

The notebook handles BigQuery extraction, outlier detection, and temporal aggregation.

Output is formatted for input to CoI or TFCAM.

4. Run Model Training
CoI Model
Open and run Model_GPT.ipynb. This notebook:

Loads preprocessed data

Initializes the CoI model from model.py

Trains using utilities from train_coi.py

Evaluates predictions and generates interpretability outputs (attention weights, influence chains)

TFCAM Model
Open and run TFCAM.ipynb. This notebook:

Loads the same preprocessed data

Trains the TFCAM model from tfcam.py

Evaluates performance and visualizes cross-temporal feature interactions

5. Plausibility Check
Run Plausibility_GPT.ipynb to:

Compute SHAP values for model predictions

Validate clinical consistency of feature importance rankings

Generate human-readable explanations for domain experts

🧠 Model Overview
Chain-of-Influence (CoI)
CoI explicitly models how feature A at time t affects feature B at time t+k. It combines:

Temporal attention – Identifies critical time periods

Feature-level attention – Ranks variable importance

Cross-feature transformer layers – Captures complex inter-feature dependencies

DyT (Dynamic Tanh) normalization – Adaptive scaling for non-stationary clinical data

Interpretability outputs: Influence chain graphs, temporal attention maps, feature contribution matrices.

TFCAM
TFCAM extends dual attention (temporal + feature) with a cross-feature attention mechanism to quantify how features influence each other across time. It provides:

Temporal-level explainability – Critical time windows

Feature-level explainability – Variable importance ranking

Cross-temporal feature influence – Quantifies propagation of influence across time steps

📊 Experimental Results (from Papers)
Model	Dataset	AUROC	F1	Accuracy
CoI	CKD (ESRD progression)	0.960	0.721	0.950
CoI	MIMIC-IV (ICU mortality)	0.950	0.865	0.950
TFCAM	CKD (ESRD progression)	0.950	0.690	0.940
RETAIN (baseline)	CKD	0.930	0.671	0.920
Both models outperform LSTM and RETAIN baselines while providing richer interpretability.

📖 Usage Notes
Enhanced Preprocessing: enhanced_preprocessing.py contains temporal-aware imputation (MICE, forward/backward fill), feature engineering (trends, variability, time since last), and ADASYN resampling. It is adapted from the official CoI repository.

Training Scripts: train_coi.py includes hyperparameter grid search, early stopping, and validation AUROC/F1 tracking.

Mask Support: The CoI implementation supports variable-length sequences via mask tensors (passed through attention layers).

Reproducibility: All notebooks use fixed random seeds. Results are averaged over 5 runs.

🤝 Acknowledgments
CoI Paper & Code: Yubo Li, Rema Padman (Carnegie Mellon University) – Chain-of-Dynamics GitHub

TFCAM Paper: Yubo Li, Xinyu Yao, Rema Padman (Carnegie Mellon University)

Data Providers: CKD cohort (proprietary), MIMIC-IV v3.1 (PhysioNet credentialed access required)

📝 License
This project is for research purposes. Please cite the original papers if you use this code in your work.

For questions or issues, please open a GitHub issue or contact the repository maintainer.

