
# Detecting and Preventing Data Leakage in Deep Learning Pipelines 

## Overview

Data leakage is a critical but often overlooked issue in medical machine learning, where improper dataset splitting can lead to overly optimistic performance estimates and unreliable models. This project investigates the impact of **patient-level data leakage** in chest X-ray classification and demonstrates how enforcing **group-aware, leak-free splits** leads to more trustworthy model evaluation and deployment.

The project progresses from classical baselines to deep convolutional neural networks, culminating in a **production-ready, leak-free ResNet18 model** deployed via a Dockerized REST API.

---

## Problem Statement

Chest X-ray pneumonia detection is a high-impact medical imaging task. However, many public benchmarks unintentionally allow **patient overlap between training and evaluation sets**, enabling models to memorize patient-specific features rather than learning disease-relevant patterns.

**Objective:**
Build a deep learning pipeline that:

* Detects and demonstrates the effects of data leakage
* Enforces strict patient-level separation
* Produces a reproducible, deployable, leak-free model

---

## Dataset

* **Source:** **Dataset source (clickable):**
🔗 [https://activeloop.ai/datasets/chest-x-ray-pneumonia](https://activeloop.ai/datasets/chest-x-ray-pneumonia)
* **Modality:** Grayscale chest X-ray images
* **Labels:** Normal / Pneumonia
* **Key Challenge:** Multiple images per patient → high risk of leakage if not split correctly

All deep learning experiments use **patient-level splits**, ensuring no individual appears in more than one dataset partition.

### Key Characteristics

* Medical chest X-ray images
* Multiple images per patient (critical for leakage analysis)
* Binary classification task:

  * **0** → Normal
  * **1** → Bacterial Pneumonia
  * **2** → Viral Pneumonia
* Loaded programmatically using Deep Lake:

```python
deeplake.load("hub://activeloop/chest-xray-pneumonia")
```
---

## Project Structure

```
data-leakage-capstone/
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_leakage_analysis.ipynb
│   ├── 02_model_with_leakage.ipynb
│   └── 03_model_without_leakage.ipynb
├── src/
│   ├── train.py
│   ├── predict.py
│   └── serve.py
├── artifacts/
│   ├── resnet18_leakfree.pth
│   └── preprocessing.json
├── data/
│   └── sample_xray.png
├── requirements.txt
├── Dockerfile
├── test_api.py
└── README.md
```

---

## Notebook Walkthrough & Findings

The project is organized into a series of notebooks, each corresponding to a key phase in the pipeline.

---

### 01_data_exploration.ipynb — Dataset Exploration & Leakage Risk Identification

This notebook focuses on understanding the dataset structure and identifying potential leakage risks.

**Key analyses performed:**

* Dataset shape and class distribution
* Patient ID frequency analysis
* Number of images per patient
* Verification that multiple samples exist for individual patients
* Initial visualization of X-ray images
<img width="894" height="612" alt="image" src="https://github.com/user-attachments/assets/b0386b18-cd7f-43a8-adad-776bf186bcfc" />
<img width="1348" height="361" alt="image" src="https://github.com/user-attachments/assets/1d4e9515-a6c6-49b4-a872-9e21a0613813" />

**Key findings:**

* Many patients have **multiple X-ray images**
* A naive image-level split would allow the same patient to appear in both training and evaluation sets
* This confirms a **high risk of data leakage** if not handled properly

 **Conclusion:** Patient-level splitting is mandatory for valid evaluation.

---

### 02_model_with_leakage.ipynb — CNN with Image-Level Splits (Leaky Baseline)

This notebook intentionally demonstrates **what not to do**.

**Approach:**

* Standard random image-level train/validation/test split
* Simple CNN trained without grouping by patient ID

**Results (invalid due to leakage):**

| Metric                | Value |
| --------------------- | ----- |
| Test Accuracy         | 0.784 |
| Test ROC-AUC (macro)  | 0.913 |
| Test F1-score (macro) | 0.744 |

**Leakage analysis:**

* **435 patients** appeared in both training and test sets
* The model partially memorized patient-specific features rather than learning disease patterns

A baseline CNN was trained for **5 epochs** on **4,172 training images** using a **batch size of 8** under an **image-level (leaky) split**. Training loss decreased steadily across epochs, indicating successful optimization.

However, this training setup required **~3–4 hours on standard hardware**, and the resulting metrics are **not reliable indicators of true generalization** due to patient-level data leakage.

**Key observations:**

* Rapid loss reduction and very low batch losses suggest memorization of patient-specific features.
* Loss stabilization in later epochs reflects convergence, but leakage undermines interpretability.
* These findings motivated the transition to **strict patient-level splits** in subsequent experiments.

 **Conclusion:** These metrics are **overly optimistic and unreliable**.

---

### 03_model_without_leakage.ipynb — Leak-Free Classical Models

This notebook establishes a **fair baseline** using strict patient-level splits.

**Approach:**

* Group-aware splitting using patient IDs
* Classical machine learning models trained on flattened image features

**Models trained:**

* Logistic Regression (with hyperparameter tuning)
* Random Forest
* XGBoost (with hyperparameter tuning)

**Validation Results (Leak-Free):**

| Model               | Validation F1 |
| ------------------- | ------------- |
| Logistic Regression | 0.728         |
| Random Forest       | 0.696         |
| XGBoost             | 0.736         |


---

## Hyperparameter Tuning (Leak-Free Baselines)

The two strongest classical baseline models were selected for controlled hyperparameter tuning under strict patient-level (no-leakage) splits:

* **Logistic Regression** — baseline accuracy: **77.2%**
* **XGBoost** — baseline accuracy: **76.3%**

Hyperparameter optimization was performed using **RandomizedSearchCV**, with **macro-averaged F1-score** as the primary evaluation metric to account for class imbalance.

### Logistic Regression (Tuned)

* **Pipeline:** `StandardScaler → LogisticRegression`
* **Tuned parameters:** `C` (regularization strength), `solver`
* **Cross-validation:** 3-fold
* **Best CV F1-score:** 0.675
* **Best parameters:**

  ```text
  solver = liblinear
  C = 1.0
  ```

### XGBoost (Tuned)

* **Pipeline:** `StandardScaler → XGBClassifier`
* **Tuned parameters:** `max_depth`, `learning_rate`
* **Cross-validation:** 2-fold
* **Best CV F1-score:** 0.649
* **Best parameters:**

  ```text
  max_depth = 5
  learning_rate = 0.1
  ```

### Runtime Note

Hyperparameter tuning was **computationally intensive** and resulted in long runtimes, particularly for Logistic Regression. Users running this phase should expect extended execution times unless parallel processing or GPU acceleration is available.

---

## Transition to Deep Learning Models

The results above confirm that meaningful predictive signal exists in the dataset under a fully leak-free, patient-level split. Classical machine learning models achieved reasonable performance, with XGBoost emerging as the strongest baseline after controlled hyperparameter tuning. However, improvements from tuning were modest, and performance began to plateau across models.

This behavior reflects a fundamental limitation of classical approaches when applied to medical imaging. All models in this phase operate on flattened pixel representations, which discard spatial structure and pixel-to-pixel relationships. As a result, these models are unable to learn anatomical patterns, textures, or hierarchical features that are critical for interpreting chest X-ray images.

Consequently, the classical models done served as **lower-bound baselines and validation checks**. They confirm that the leak-free preprocessing pipeline preserves meaningful signal, while also highlighting the need for spatially aware models to fully exploit the data.

In the next phase, the pipeline transitions to convolutional neural networks (CNNs) trained on the same strict patient-disjoint splits. CNNs are specifically designed to learn hierarchical spatial features and represent the standard modeling approach for medical imaging. This transition enables a fair and methodologically sound comparison between classical, non-spatial baselines and deep learning models that operate directly on image structure.

---

### Quantifying the Impact of Leakage

I compared **leaky vs leak-free setups**.

**Key analyses:**

* Patient overlap counts across splits
* Metric deltas caused by leakage
* Side-by-side comparison of leaky and clean models

**Key takeaway:**

| Setup                           | F1-score |
| ------------------------------- | -------- |
| CNN with leakage                | 0.744    |
| Logistic Regression (leak-free) | 0.728    |
| XGBoost (leak-free)             | 0.736    |

 **Conclusion:**
Data leakage alone can account for **large apparent performance gains**, highlighting why leakage detection is critical in healthcare ML.

---

## Deep Learning Models (Leak-Free)

After establishing clean baselines, deep learning models were trained under **strict patient-level separation**.

### Models Evaluated

| Model                     | Validation F1 |
| ------------------------- | ------------- |
| SimpleCNN                 | 0.9484        |
| **ResNet18 (pretrained)** | **0.9894**    |

**Why ResNet18 was selected:**

* Superior generalization performance
* Transfer learning benefits from ImageNet
* Stable convergence with minimal tuning
* Clear performance gap vs SimpleCNN

Hyperparameter tuning was intentionally limited due to:

* Near-saturated performance
* Diminishing returns
* Emphasis on reproducibility and clean methodology

---

## Final Results Summary

| Setup                           | F1-Score   |
| ------------------------------- | ---------- |
| Leaky CNN (Invalid)             | 0.744      |
| Logistic Regression (Leak-Free) | 0.728      |
| XGBoost (Leak-Free, Tuned)      | 0.736      |
| **ResNet18 (Leak-Free)**        | **0.9894** |

This demonstrates that **proper experimental design**, not leakage, is the true driver of high performance.

---

## Production Pipeline

The project includes a fully aligned production pipeline:

* `train.py` — leak-free ResNet18 training
* `predict.py` — single-image inference
* `serve.py` — Flask REST API
* `test_api.py` — API validation
* `artifacts/` – Saved model & preprocessing config
* `Dockerfile` – Containerized deployment
* `requirements.txt` – Dependency management
* Validation checks for input data
* Consistent preprocessing across training and inference
* Structured logging

---

## Artifacts & Reproducibility

The final ResNet18 model was trained using strict patient-level (no-leakage) splits and saved as a reusable artifact:

* `artifacts/resnet18_leakfree.pth` — trained model weights
* `artifacts/preprocessing.json` — image size and normalization parameters
* Training the model is computationally expensive (≈3–4 hours on standard hardware). For this reason, the trained artifacts are included and used directly by the inference and API pipelines.
These artifacts are used consistently across:

* Training
* Inference
* API serving
* Docker deployment

---

## Reproducibility and Usage

### 1. Clone the Repository

```bash
git clone https://github.com/YohanneNoume/data-leakage-capstone.git
cd data-leakage-capstone
```

---

### 2. Create a Virtual Environment

This project was developed using a **Conda environment**, but it can also be run using a standard Python virtual environment (`venv`).

#### Option 1: Using Conda (Recommended)

```bash
conda create -n leakage-dl python=3.11
conda activate leakage-dl
pip install -r requirements.txt
```

#### Option 2: Using Python Virtual Environment

```bash
python -m venv venv
```

Activate the environment:

* **Windows**

```bash
venv\Scripts\activate
```

* **Linux / macOS**

```bash
source venv/bin/activate
```

Then install dependencies:

```bash
pip install -r requirements.txt
```

---

### 3. Model Training (Optional – Reproducibility)

The final ResNet18 model was trained under **strict patient-level (no-leakage) splits** and saved as a reusable artifact:

```
artifacts/resnet18_leakfree.pth
artifacts/preprocessing.json
```

Training is computationally expensive (≈3–4 hours on standard hardware).
For this reason, **trained artifacts are included and used directly for inference and deployment**.

#### Re-training from Scratch (Optional)

Reviewers or users who wish to reproduce training can run:

```bash
python src/train.py
```

This will:

* Reload the dataset from Deep Lake
* Apply patient-level group splits
* Train ResNet18 under leak-free conditions
* Save a new model artifact to `artifacts/`

 **Runtime note:** Expect long runtimes without GPU acceleration.
 The pretrained model artifact (`artifacts/resnet18_leakfree.pth`) included in this repository was trained using **strict patient-level splits**, as implemented in the notebook **`03_model_without_leakage.ipynb`**. All reported performance metrics and conclusions are based exclusively on this leak-free training pipeline.

The script **`src/train.py`** provides a fully reproducible, end-to-end training pipeline extracted from the notebook for production and experimentation purposes. For simplicity and runtime considerations, the standalone script performs **index-level data splitting** rather than patient-grouped splitting.

This design allows reviewers and users to:

* Reproduce the complete training process end-to-end
* Verify model architecture, preprocessing steps, and optimization logic
* Optionally extend or modify the script to reintroduce patient-level grouping if desired

Due to the computational cost of training convolutional neural networks on medical imaging data, pretrained artifacts are provided for convenience. Retraining the model is optional and not required to reproduce inference, API serving, or deployment workflows.

---

### 4. Single Image Inference (Local Test)

Run inference on a single chest X-ray image:

```bash
python src/predict.py data/sample_xray.png
```

This loads the saved leak-free model and applies the same preprocessing used during training.

---

### 5. Model Deployment (Flask API)

The trained CNN is served via a **Flask REST API** for real-time inference.

#### Run the API locally

```bash
python src/serve.py
```

The API will be available at:

```
http://localhost:9696
```

Health check:

```bash
GET /health
```

Prediction endpoint:

```bash
POST /predict
```

---

### 6. API Testing

A lightweight test client is provided to validate the API.

```bash
python test_api.py
```
<img width="971" height="77" alt="image" src="https://github.com/user-attachments/assets/83b686af-c02c-4226-ae07-7ec1c05150f5" />

This script:

* Encodes a sample X-ray image in base64
* Sends a request to `/predict`
* Prints the predicted class and confidence score

---

###  Docker Containerization

Build the Docker image:

```bash
docker build -t leakage-cnn .
```

Run the container:

```bash
docker run -p 9696:9696 leakage-cnn
```
<img width="1116" height="1033" alt="image" src="https://github.com/user-attachments/assets/8af80db1-bb05-4bf0-9b07-505cfddeba81" />
The API will be accessible at:

```
http://localhost:9696
```

---

## Key Takeaways

* Patient-level leakage can severely inflate evaluation metrics
* Leak-free splits provide realistic estimates of generalization
* Transfer learning outperforms training from scratch
* Sound experimental design is as important as model choice
* Production-ready pipelines require aligned training and inference logic

---

## Final Notes

This project emphasizes **methodological rigor**, **reproducibility**, and **deployment readiness** over inflated metrics, reflecting real-world best practices in applied machine learning for healthcare.

---
