# Multi-Agent AI System for ICU Diagnosis

> **A transparent, interpretable multi-agent framework for critical care diagnosis using MIMIC-III**


## 📋 Overview

This project implements a multi-agent AI system for diagnosing 9 critical ICU conditions using heterogeneous clinical data. Unlike traditional black-box approaches, our system achieves **0.886 AUC and 0.710 F1** while maintaining feature-level interpretability through explicit agent communication and XGBoost-based fusion.

**Key Innovation:** Three specialized agents (labs, clinical notes, vitals) communicate structured messages to a fusion coordinator that uses 105 engineered coordination features—enabling transparency impossible with neural attention mechanisms.

## 🔬 Predicted Conditions

1. Sepsis
2. Pneumonia
3. Respiratory Failure
4. Acute Kidney Injury (AKI)
5. Heart Failure
6. Atrial Fibrillation
7. Coronary Artery Disease
8. Anemia
9. Pancreatitis

## 🚀 Key Features

- **Explicit Multi-Agent Communication** - Structured message protocol with predictions, confidence, and data quality
- **Heterogeneous Agent Architectures** - XGBoost for structured data, BERT for text, optimized per modality
- **105 Coordination Features** - Interactions, disagreements, consensus, variance explicitly modeled
- **Feature-Level Interpretability** - SHAP explanations show exact contributions (impossible with neural attention)
- **Clinical Evidence Extraction** - Traceable to specific lab values, clinical phrases, vital patterns
- **Graceful Degradation** - Availability flags enable operation with missing modalities
- **Modular & Extensible** - Add new agents without retraining existing ones


## ⚙️ Setup & Installation

### Prerequisites

#### 1. MIMIC-III Access

- Complete CITI training: https://about.citiprogram.org/
- Request access: https://physionet.org/content/mimiciii/1.4/
- Download dataset to local directory

#### 2. Python Environment

```bash
python >= 3.8
pip install -r requirements.txt
```

### Required Libraries

Create a `requirements.txt` file with:

```txt
# Core ML
numpy>=1.21.0
pandas>=1.3.0
scikit-learn>=1.0.0
xgboost>=1.5.0
lightgbm>=3.3.0

# Deep Learning
torch>=1.10.0
transformers>=4.15.0

# Interpretability
shap>=0.40.0

# Visualization
matplotlib>=3.4.0
seaborn>=0.11.0

# Interface
streamlit>=1.10.0

# Medical NLP
scispacy>=0.5.0
```

### Installation

```bash
# Clone repository
git clone https://github.com/yourusername/multi-agent-icu-diagnosis.git
cd multi-agent-icu-diagnosis

# Create virtual environment
python -m venv mimic_env
source mimic_env/bin/activate  # On Windows: mimic_env\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install scispacy model
pip install https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.5.0/en_core_sci_sm-0.5.0.tar.gz

# Download BioClinicalBERT
# Model will auto-download from HuggingFace when running Agent 2 training
```

---

## 🔧 Usage

### 1. Data Preprocessing

**⚠️ CRITICAL:** MIMIC-III data is NOT included due to data use agreements. You must:

1. Obtain MIMIC-III access from PhysioNet
2. Place raw CSVs in `data/mimic-iii/` (not tracked by git)
3. Run preprocessing notebooks

```bash
# Navigate to data processing
cd notebooks/data_process/

# Run preprocessing (in order)
jupyter notebook lab_preprocessing.ipynb
jupyter notebook data_exploration.ipynb
```

**Output:** Preprocessed features saved to `data/processed/`

---

### 2. Train Individual Agents

#### Agent 1: Laboratory Values

```bash
cd notebooks/agent1/
jupyter notebook agent1_training.ipynb
```

- **Input:** 27 lab features (creatinine, lactate, WBC, etc.)
- **Model:** XGBoost classifier per disease
- **Output:** `models/agent1_*.pkl`

#### Agent 2: Clinical Notes

```bash
cd notebooks/agent2/
jupyter notebook Agent2_BERT_Training_Colab.ipynb  # Requires GPU
```

- **Input:** Discharge summaries
- **Model:** BioClinicalBERT fine-tuned
- **Output:** `models/agent2_bert/`

#### Agent 3: Vital Signs

```bash
cd notebooks/agent3/
jupyter notebook agent3_training.ipynb
```

- **Input:** 42 vital statistics (mean, std, min, max of HR, BP, SpO2, etc.)
- **Model:** XGBoost classifier
- **Output:** `models/agent3_*.pkl`

---

### 3. Generate Agent Messages

```bash
cd notebooks/fusion/
jupyter notebook generate_agent_messages.ipynb
```

This creates the structured communication protocol:

- Collects predictions from all 3 agents
- Adds confidence scores (max probability per agent)
- Adds availability flags (which modalities present)
- **Output:** `results/agent_messages.pkl`

---

### 4. Train Fusion Coordinator

```bash
jupyter notebook train_fusion_coordinator.ipynb
```

**Output:** `models/fusion_*.pkl`

---

### 5. Run Evaluation

```bash
cd notebooks/evaluation/

# Baseline comparisons
jupyter notebook attention_based.ipynb  # Neural attention fusion

# Ablation studies
jupyter notebook ablation_study.ipynb  # Remove agents systematically

# Error analysis
jupyter notebook error_analysis.ipynb  # Where system fails
```

---

### 6. Clinical Interface

```bash
cd src/communication/
streamlit run app.py
```

**Interface Features:**

- Conversational agent collaboration view
- Detailed dashboard with evidence extraction
- SHAP feature importance breakdown

---

## 📚 Key References

1. **MIMIC-III Database:**  
   Johnson, A.E.W., et al. (2016). MIMIC-III, a freely accessible critical care database. *Scientific Data*, 3, 160035.  
   https://doi.org/10.13026/C2XW26

2. **BioClinicalBERT:**  
   Alsentzer, E., et al. (2019). Publicly Available Clinical BERT Embeddings. *Clinical NLP Workshop*, 72-78.

3. **MIMIC-III Benchmarks:**  
   Harutyunyan, H., et al. (2019). Multitask learning and benchmarking with clinical time series data. *Scientific Data*, 6, 96.

4. **Multimodal Healthcare AI (HAIM):**  
   Soenksen, L.R., et al. (2022). Integrated multimodal artificial intelligence framework for healthcare applications. *npj Digital Medicine*, 5, 149.
