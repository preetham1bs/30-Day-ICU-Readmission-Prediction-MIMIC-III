# 30-Day ICU Readmission Prediction — MIMIC-III

> **Predicting which ICU patients are at risk of hospital readmission within 30 days using clinical data and machine learning.**

[![Python](https://img.shields.io/badge/Python-3.9+-3776AB?style=flat&logo=python&logoColor=white)](https://python.org)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-F7931E?style=flat&logo=scikit-learn&logoColor=white)](https://scikit-learn.org)
[![Tableau](https://img.shields.io/badge/Tableau-Dashboard-E97627?style=flat&logo=tableau&logoColor=white)](https://public.tableau.com)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

---

## Table of Contents
- [Background]
- [Research Question]
- [Dataset]
- [Methodology]
- [Key Findings]
- [Results]
- [Dashboard]
- [Project Structure]
- [How to Run]
- [Limitations & Next Steps]
- [Author]

---

## Background

Hospital readmissions within 30 days represent one of the most significant challenges in clinical operations — they indicate potential gaps in discharge planning, care quality, and post-discharge support. In India, readmission risk stratification is increasingly critical as hospital systems scale under Ayushman Bharat and other national health programmes.

This project builds an end-to-end pipeline — from raw ICU data extraction to a machine learning classifier and interactive dashboard — to identify high-risk patients before discharge.

---

## Research Question

> **"Can routinely collected ICU clinical data (vitals, lab values, demographics) predict whether a patient will be readmitted to hospital within 30 days of discharge?"**

---

## Dataset

**MIMIC-III Clinical Database** (Medical Information Mart for Intensive Care)
- Source: MIT PhysioNet — [physionet.org/content/mimiciii](https://physionet.org/content/mimiciii/1.4/)
- Coverage: Beth Israel Deaconess Medical Center, Boston — 2001 to 2012
- Size: 40,000+ ICU patients
- Access: Requires CITI human research training + Credentialing + Data use agreement

> ⚠️ **Note:** Raw MIMIC-III data files are not included in this repository in compliance with the PhysioNet data use agreement. See [How to Run](#how-to-run) for access instructions.

**Tables used:**

| Table | Purpose |
|---|---|
| `ADMISSIONS` | Hospital visits, admit/discharge times, mortality flag |
| `PATIENTS` | Demographics — age, gender |
| `ICUSTAYS` | ICU unit, length of stay |
| `LABEVENTS` | Blood test results (creatinine, WBC, glucose, haemoglobin) |
| `CHARTEVENTS` | (Explored but not included in final model due to high dimensionality) |

---

## Methodology

```
Raw MIMIC-III CSVs
       │
       ▼
  SQLite / MySQL
  (data loading)
       │
       ▼
  SQL Extraction ──► Target variable: 30-day readmission flag
       │
       ▼
  Python EDA ──────► Missing data analysis, distributions, group comparisons
       │
       ▼
  Feature Engineering
  - Demographics (age, gender)
  - ICU stay metrics (LOS, unit type)
  - Lab values (creatinine_max, WBC_max, glucose_mean, haemoglobin_min)
  - Comorbidities (diabetes, heart failure flags)
       │
       ▼
  ML Modelling
  - Logistic Regression
  - Random Forest
  - Gradient Boosting ◄── Best performer
       │
       ▼
  Evaluation ──────► ROC-AUC, Confusion Matrix, Feature Importance
       │
       ▼
  Tableau Dashboard
```

### Target Variable Definition

A patient is flagged as **readmitted (1)** if they have a subsequent hospital admission within 30 days of their previous discharge date. Patients who died during their index admission are excluded (discharge_location = 'DEAD/EXPIRED').

### Class Imbalance Handling

The dataset is imbalanced (~5.60% readmission rate). Addressed using `class_weight='balanced'` in all classifiers to prevent the model from defaulting to predicting the majority class.

---

## Key Findings

- **5.60%** of ICU patients were readmitted within 30 days
- **ICU length of stay** and **Hemoglobin (mean)** were the strongest predictors of readmission
- Patients aged **60–75** had the highest readmission rate across all age groups
- **MICU** (Medical ICU) had the highest proportion of patients with prolonged stays
-  Elevated WBC and creatinine levels indicate infection and organ dysfunction, increasing readmission risk
- Emergency admissions accounted for **71%** of total admissions.
- Repeat admissions and recent discharge history significantly increased readmission probability.

---

## Results

### Model Performance

| Model | AUC-ROC | Notes |
|---|---|---|
| Logistic Regression | 0.668 | Similar to RF, slower training |
| Random Forest | 0.640 | Baseline, interpretable |
| **Gradient Boosting** | **0.692** | Best overall performance |

> AUC of 0.64–0.69 is consistent with published clinical readmission prediction literature. Higher values would suggest overfitting on this dataset size.
> The relatively modest AUC reflects the absence of diagnosis codes (ICD), medication data, and temporal trends, which are known to significantly improve readmission prediction performance.

### ROC Curve

![ROC Curve](https://github.com/preetham1bs/30-Day-ICU-Readmission-Prediction-MIMIC-III/blob/8402626ed302e73731b633c477c4f8e1f95e57ec/roc_curve.png)

### Top Predictive Features

![Feature Importance](https://github.com/preetham1bs/30-Day-ICU-Readmission-Prediction-MIMIC-III/blob/25ed5620009689fae896a0200aecfa3369179a3b/feature_importance.png)

> **Clinical interpretation:** Low hemoglobin (anemia/chronic disease) emerges as the strongest predictor of readmission risk, followed by WBC as an infection marker and creatinine indicating renal dysfunction. Longer ICU stays and age further reflect disease severity, aligning with established clinical risk factors.

---

## Dashboard

Interactive Tableau dashboard visualising:
- KPI summary (readmission rate, avg ICU stay, total patients)
- Readmission rate by age group
- Top feature importances
- Patient risk scatter plot (ICU stay vs predicted risk, coloured by outcome)

🔗 **[View live dashboard on Tableau Public](https://public.tableau.com/views/mimicreadmission/30-DayReadmissionAnalysis?:language=en-US&:sid=&:redirect=auth&:display_count=n&:origin=viz_share_link)(#)**
![Dashboard Preview](https://github.com/preetham1bs/30-Day-ICU-Readmission-Prediction-MIMIC-III/blob/80351d77e0b70c6a2445ba29699a6c5baacf9b7d/30-Day%20Readmission%20Analysis.png)

---

## Project Structure

```
mimic-readmission/
│
├── README.md
│
├── notebooks/
│   ├── 01_data_loading.ipynb          # Load MIMIC CSVs into SQLite/MySQL
│   ├── 02_eda.ipynb                   # Exploratory data analysis
│   ├── 03_feature_engineering.ipynb   # Build feature matrix
│   └── 04_modelling.ipynb             # Train, evaluate, visualise models
│
├── sql/
│   └── readmission_target.sql         # SQL query to create 30-day flag
│
├── outputs/
│   ├── roc_curve.png
│   ├── feature_importance.png
│   ├── missing_data.png
│   └── dashboard_screenshot.png
│
└── requirements.txt
```

---

## How to Run

### 1. Get MIMIC-III Access

```
# Option A — Demo (100 patients, instant, no approval)
# Download from: physionet.org/content/mimiciii-demo/1.4/

# Option B — Full dataset (40,000+ patients, ~1 week approval)
# 1. Register at physionet.org
# 2. Complete CITI "Data or Specimens Only Research" course (~2 hrs, free)
# 3. Sign data use agreement → await approval email
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

**requirements.txt:**
```
pandas>=1.5.0
numpy>=1.23.0
matplotlib>=3.6.0
seaborn>=0.12.0
scikit-learn>=1.1.0
missingno>=0.5.0
plotly>=5.10.0
jupyter>=1.0.0
sqlalchemy>=1.4.0
```

### 3. Load Data & Run Notebooks

```bash
# Place MIMIC CSV files in ./data/ folder
# Then run notebooks in order:

jupyter notebook notebooks/01_data_loading.ipynb
jupyter notebook notebooks/02_eda.ipynb
jupyter notebook notebooks/03_feature_engineering.ipynb
jupyter notebook notebooks/04_modelling.ipynb
```

---

## Limitations & Next Steps

### Current Limitations

| Limitation | Impact |
|---|---|
| Demo dataset (100 patients) | Wide confidence intervals, lower statistical power |
| Single-centre data (BIDMC, Boston) | May not generalise to Indian hospital populations |
| Date-shifted records | Age calculation requires correction for MIMIC's privacy masking |
| No clinical notes | NLP on discharge summaries could significantly improve predictions |
| Observational data | Cannot establish causality — only association |

### Next Steps

- [ ] Re-run analysis on full MIMIC-III dataset (40,000+ patients) for more reliable results
- [ ] Add NOTEEVENTS table — apply NLP to discharge summaries as additional features
- [ ] Validate against published LACE and HOSPITAL readmission scores
- [ ] Explore MIMIC-IV for more recent data (2008–2019)
- [ ] Build a simple risk scoring tool for clinical use

---

## Author

**Preetham B S** |
B.E. Medical Electronics — M.S. Ramaiah Institute of Technology, Bengaluru (2025) |
Co-author, 3× IEEE International Conference Papers (CompSIF 2025)

📧 bs1preetham2002@gmail.com
🔗 [linkedin.com/in/preetham1bs](https://linkedin.com/in/preetham1bs)
🐙 [github.com/preetham1bs](https://github.com/preetham1bs)

---

## Acknowledgements

- **MIMIC-III Clinical Database** — Johnson AEW, Pollard TJ, Shen L, et al. *Scientific Data* 2016. [doi:10.1038/sdata.2016.35](https://doi.org/10.1038/sdata.2016.35)
- PhysioNet for providing open access to critical care research data

---

*This project was completed as part of a healthcare data analyst portfolio. All analysis is for educational and portfolio purposes only.*
