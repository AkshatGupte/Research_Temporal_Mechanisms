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
| `DataPreprocessing_1.ipynb` | Extracts and preprocesses raw ICU data for downstream modeling. |
| `Model_GPT.ipynb` | Main CoI training and evaluation notebook. |
| `Model_DP.ipynb` | Alternate CoI experimentation notebook for data-processing variants. |
| `Faithfulness_Testing.ipynb` | Faithfulness experiments to validate temporal attention behavior. |
| `Plausibility_GPT.ipynb` | SHAP-based plausibility checks against clinical expectations. |
| `TFCAM.ipynb` | TFCAM training, evaluation, and interpretability analysis. |
| `requirements.txt` | Python dependencies for the project. |

### `Scripts/`

| File | Description |
|------|-------------|
| `Scripts/model.py` | PyTorch CoI model implementation. |
| `Scripts/model-gpt.py` | Extended/alternate CoI implementation variant. |
| `Scripts/tfcam.py` | PyTorch TFCAM model implementation. |
| `Scripts/train_coi.py` | Training utilities for CoI. |
| `Scripts/train_tfcam.py` | Training utilities for TFCAM. |
| `Scripts/evaluate.py` | Evaluation helpers and detailed metric reporting. |
| `Scripts/faithfulness_tests.py` | Erasure-based faithfulness testing utilities. |
