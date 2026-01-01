
# 🧠 **FineFormer**

## *Transformer-Based Differential Diagnosis of Bipolar Disorder and Schizophrenia from rs-fMRI*

<p align="center">
  <a href="https://www.python.org/">
    <img src="https://img.shields.io/badge/Python-3.8%2B-blue?logo=python&logoColor=white">
  </a>
  <a href="https://pytorch.org/">
    <img src="https://img.shields.io/badge/PyTorch-2.0+-red?logo=pytorch&logoColor=white">
  </a>
  <a href="https://numpy.org/">
    <img src="https://img.shields.io/badge/NumPy-Latest-blue?logo=numpy&logoColor=white">
  </a>
  <a href="https://scipy.org/">
    <img src="https://img.shields.io/badge/SciPy-Latest-lightgrey?logo=scipy">
  </a>
  <a href="https://scikit-learn.org/">
    <img src="https://img.shields.io/badge/scikit--learn-Latest-orange?logo=scikit-learn">
  </a>
</p>


---

## 📌 Overview

This repository provides the official implementation of **FineFormer**, a Transformer-based deep learning framework for the **differential diagnosis of Schizophrenia (SZ) and Bipolar Disorder (BD)** using **resting-state functional magnetic resonance imaging (rs-fMRI)**.

The proposed approach integrates:

* **Attention-based Transformer architectures**
* A **cyclic transfer learning strategy**

to address two fundamental challenges in psychiatric neuroimaging:

1. **Limited availability of labeled rs-fMRI data**
2. **Substantial clinical and neurobiological overlap between SZ and BD**

### Framework Objectives

The framework is designed to:

* Learn compact **spatiotemporal representations** of whole-brain rs-fMRI signals
* Leverage **self-attention mechanisms** for interpretability
* Improve generalization via **sequential knowledge transfer** across related diagnostic tasks

---

## 🎯 Motivation

Schizophrenia and Bipolar Disorder frequently present overlapping symptoms—particularly during manic or psychotic episodes—making accurate differential diagnosis based solely on clinical assessments highly challenging. Misdiagnosis can result in inappropriate treatment strategies and adverse patient outcomes.

While **resting-state fMRI** provides a non-invasive window into intrinsic brain dynamics and functional connectivity, its application is hindered by:

* High dimensionality
* Temporal complexity
* Limited sample sizes

This work addresses these challenges through:

* Compact and task-aware Transformer architectures
* Cyclic and sequential transfer learning
* Neuroimaging-specific data augmentation strategies
* Rigorous cross-validation protocols to prevent data leakage

---

## 📊 Dataset Description

Experiments were conducted using two publicly available neuroimaging datasets:

* **UCLA Consortium for Neuropsychiatric Phenomics**
* **COBRE (Center for Biomedical Research Excellence)**

### 🧩 Participants

A total of **308 subjects** were included:

| Group                 | Count |
| --------------------- | ----: |
| Healthy Controls (HC) |   139 |
| Schizophrenia (SZ)    |   120 |
| Bipolar Disorder (BD) |    49 |

---

### 🧠 Input Representation

Each subject is represented by a **spatiotemporal matrix** of size:

```
(T × R) = 142 × 118
```

where:

* 🕒 **T = 142** time points (TRs)
* 🧠 **R = 118** brain regions of interest (ROIs)

Each time point is treated as a **token encoding whole-brain activity**, enabling attention-based modeling of long-range dependencies.

---

## 🧪 Classification Tasks

The diagnostic problem is decomposed into three binary classification tasks:

|  Task  | Description                          |
| :----: | ------------------------------------ |
| **HS** | Healthy Control vs. Schizophrenia    |
| **HB** | Healthy Control vs. Bipolar Disorder |
| **BS** | Bipolar Disorder vs. Schizophrenia   |

**Label convention (used consistently):**

* `0` → patient group
* `1` → control or comparison group

---

## 🏗️ Model Architectures

Three Transformer-based architectures are investigated:

### ⏱️ 1. Time-Transformer

* Models **temporal dependencies** across the rs-fMRI time series
* Sequence length corresponds to time points
* Tokens encode whole-brain ROI activity

### 🌐 2. Region-Transformer

* Models **spatial dependencies** between brain regions
* Sequence length corresponds to ROIs
* Tokens encode temporal activity patterns per region

### 🔀 3. Hybrid-Transformer

* Sequentially combines:

  1. Temporal Transformer layers
  2. Region-based Transformer layers
* Enables joint modeling of **temporal dynamics** and **spatial connectivity**

All models employ multi-head self-attention, residual connections, layer normalization, and GELU activations.

---

## 🔁 Training Strategy: Cyclic Transfer Learning

To mitigate data scarcity and enhance representation learning, a **cyclic sequential transfer learning strategy** is adopted.

### 🔄 Training Procedure

1. Train the model on **HS (HC vs. SZ)**
2. Transfer encoder weights and retrain on **HB (HC vs. BD)**
3. Transfer encoder weights and retrain on **BS (BD vs. SZ)**
4. Repeat the entire cycle **twice**

During transfer:

* The Transformer encoder is preserved
* The classification head may be reinitialized
* Encoder freezing and fine-tuning are configurable

This cyclic exposure enables the encoder to learn increasingly **general and task-agnostic rs-fMRI representations**.

---

### 📐 Training Workflow Illustration

<p align="center">
  <img src="figures/training_strategy.png" width="80%">
</p>

**Figure:** Cyclic transfer learning strategy across HS, HB, and BS diagnostic tasks.

---

## 🧪 Cross-Validation and Evaluation

* **5-fold cross-validation** with fixed subject splits
* Identical folds used across all tasks to prevent data leakage
* Early stopping based on validation accuracy
* Final results reported as the mean across folds

### 📊 Evaluation Metrics

* Accuracy (ACC)
* Sensitivity (Sens)
* Specificity (Spec)
* Negative Predictive Value (NPV)
* Area Under the ROC Curve (AUC)
* Precision (Prec)
* F1-score (F1)

---

## 📈 Results

**Table 1** summarizes model performance after the second iteration of cyclic transfer learning.
**Best-performing values per task and metric are shown in bold.**

### Table 1. Performance of Transformer-Based Models (Iteration 2)
| Model | Task | ACC | Sens | Spec | NPV | AUC | Prec | F1 | | ------------------ | ---- | --------- | --------- | --------- | --------- | --------- | --------- | --------- | 
| Time-Transformer | HS | **0.884** | 0.833 | **0.928** | **0.911** | 0.860 | **0.911** | 0.870 | 
| Region-Transformer | HS | 0.780 | 0.742 | 0.814 | 0.789 | 0.737 | 0.787 | 0.756 | 
| Hybrid-Transformer | HS | **0.884** | **0.867** | 0.900 | 0.888 | **0.881** | 0.888 | **0.874** | 
| Time-Transformer | HB | **0.900** | 0.698 | **0.971** | 0.903 | 0.784 | 0.898 | 0.773 | 
| Region-Transformer | HB | 0.857 | 0.573 | 0.957 | 0.867 | 0.756 | 0.867 | 0.663 | | 
Hybrid-Transformer | HB | 0.899 | **0.738** | 0.957 | **0.914** | **0.820** | **0.900** | **0.790** | 
| Time-Transformer | BS | **0.923** | **1.000** | 0.739 | **1.000** | 0.850 | 0.904 | **0.950** | 
| Region-Transformer | BS | 0.870 | 0.983 | 0.591 | 0.949 | 0.718 | 0.858 | 0.916 | 
| Hybrid-Transformer | BS | 0.911 | 0.942 | **0.839** | 0.862 | **0.854** | **0.936** | 0.938 |

---

### 🔍 Key Observations

* The **Hybrid-Transformer** achieves the most balanced performance across tasks
* In the challenging **BS task**, both Time and Hybrid models exceed **90% accuracy**
* The **Time-Transformer** achieves **perfect sensitivity (1.00)** for distinguishing BD from SZ

---

## 🔎 Interpretability

FineFormer supports **attention weight extraction** at the subject level:

* Attention maps across time or ROIs
* Identification of salient temporal segments or brain regions
* Facilitates neurobiological interpretation of model decisions

---

## ♻️ Reproducibility

The repository includes:

* Fixed cross-validation fold indices
* Saved model checkpoints
* Training histories
* Per-fold and aggregated metrics
* Attention weight files

All experiments are fully reproducible given identical preprocessing and fold definitions.

---

## 📚 Citation

If you use this code, please cite the associated paper:

```bibtex
@article{FineFormer2025,
  title   = {FineFormer: Transformer-Based Differential Diagnosis of Bipolar Disorder and Schizophrenia from rs-fMRI},
  author  = {...},
  journal = {...},
  year    = {2025}
}
```

