# 🎓 Predicting University Student Attrition
### A Machine Learning Approach to Identify At-Risk Students

**UFCEKP-30-3 — Data Science and AI Individual Project**  
**Ben Kefa ** · 25****53 · UWE Bristol**  
**Supervisor: Dr. Ethan**

---

## 📋 Overview

Student dropout is a costly problem for universities and a devastating one for students — leaving them with debt, stalled careers, and lost confidence. Most institutional interventions only kick in *after* a student has already disengaged. This project addresses that gap.

Using the **UCI "Predict Students' Dropout and Academic Success" dataset**, this project builds and compares multiple supervised machine learning classifiers to predict whether a student will **drop out**, remain **enrolled**, or **graduate** — with particular focus on identifying dropout risk as early as possible.

The goal: a proof-of-concept **early warning system** that could help university support teams intervene proactively, before a student makes the decision to leave.

---

## 🗂️ Repository Structure

```
├── Student_Predictor_Model.ipynb       # Model 1 — baseline pipeline
├── 2nd_Model_Student_Predictor.ipynb   # Model 2 — refined preprocessing
├── 3rd_Model.ipynb                     # Model 3 — full ML pipeline
├── 4th_Model.ipynb                     # Model 4 — ensembles + binary classifier
├── 5th_Model.ipynb                     # Model 5 — ensembles + binary classifier Higher Education Vs A Level  
│
├── Portugese_Higher_Education.csv      # Primary dataset (UCI, 4,424 students)
├── iti_student_dropout_dataset.csv     # Secondary dataset (ITI, 7,000 students)
├── studentInfo.csv                     # OU student info (32,593 records)
├── final_training_data_v4.csv          # Processed training data (v4)
├── HE_model_readyCopy1.csv             # HE-qualified student subset (5,043)
├── ALevel_model_readyCopy1.csv         # A-Level student subset (27,550)
│
```

---

## 📊 Datasets

| Dataset | Rows | Columns | Source |
|---|---|---|---|
| Portuguese Higher Education (UCI) | 4,424 | 37 | UCI ML Repository (Valentim et al., 2021) |
| ITI Student Dropout | 7,000 | 23 | Kaggle |
| Open University Student Info | 32,593 | 12 | Open University Learning Analytics |

The **primary dataset** (Portuguese HE) includes demographic, socioeconomic, and academic variables such as age at enrolment, gender, scholarship status, tuition payment, and semester grades and credits. The target variable has three classes: `Dropout`, `Enrolled`, and `Graduate`.

---

## 🤖 Models

Four iterative notebook versions implement and refine the following classifiers:

- **Logistic Regression** — interpretable baseline; coefficients linked to Tinto's integration theory
- **Decision Tree** — rule-based and explainable; suited for educator review
- **Random Forest** — ensemble method; consistently strong on tabular educational data
- **Gradient Boosting** — high-performance boosting; best cross-validated F1
- **Voting Classifier** (RF + GB) — soft-vote ensemble
- **Stacking Classifier** (RF + GB + LR) — meta-learner ensemble
- **Binary Classifier** (Dropout vs. Rest) — best ROC-AUC overall

---

## 📈 Results (Best Model — 4th Notebook)

| Model | Accuracy | F1 (Weighted) | ROC-AUC | F1 (Dropout) |
|---|---|---|---|---|
| Logistic Regression | 0.7333 | 0.7480 | 0.8763 | 0.7660 |
| Decision Tree | 0.7333 | 0.7438 | 0.8395 | 0.7583 |
| Random Forest | **0.7548** | **0.7622** | **0.8855** | **0.7710** |
| Gradient Boosting | 0.7492 | 0.7531 | 0.8790 | 0.7628 |
| Voting (RF+GB) | 0.7514 | 0.7564 | 0.8863 | 0.7669 |
| Stacking (RF+GB+LR) | 0.7548 | 0.7546 | 0.8561 | 0.7669 |
| **Binary (Dropout vs Rest)** | **0.8859** | — | **0.9389** | **0.8237** |

> **Random Forest** is the best multiclass model. The **Binary Dropout Classifier** achieves the highest overall ROC-AUC (0.9389), making it most suitable for a real-world early warning system.

---

## 🔧 Setup & Usage

### Requirements

```bash
pip install pandas numpy scikit-learn matplotlib seaborn
```

### Running the Notebooks

Open any notebook in Jupyter and run all cells in order. The notebooks are self-contained — they load the CSV data, preprocess it, train the models, and output evaluation metrics and plots.

```bash
jupyter notebook 4th_Model.ipynb
```

Plots are saved locally:
- `plot_eda.png` — exploratory data analysis
- `plot_metrics.png` — model comparison
- `plot_confusion.png` — confusion matrices
- `plot_roc.png` — ROC curves
- `plot_importance.png` — feature importance

---

## 🧪 Methodology

1. **Data Cleaning & Preprocessing** — handling missing values, encoding categorical variables, feature scaling
2. **Exploratory Data Analysis** — class distribution, correlation analysis, feature distributions
3. **Feature Engineering** — aggregating semester performance, creating dropout-focused features
4. **Model Training** — 80/20 stratified train-test split + 5-fold cross-validation
5. **Imbalance Handling** — resampling techniques and class-weighted loss functions
6. **Evaluation** — confusion matrix, precision, recall, F1-score, ROC-AUC (accuracy alone is insufficient for imbalanced classes)

---

## 💡 Key Findings

- **Semester grades and approved credit units** are the strongest predictors of dropout, consistent with Tinto's theory of academic integration.
- **Tuition fee status** and **scholarship holder** flag are highly informative socioeconomic features.
- The **"Enrolled" class is the hardest to classify**, often confused with both Dropout and Graduate — a known challenge in the literature.
- A **binary framing** (Dropout vs. Not Dropout) significantly improves performance and is more actionable for real-world deployment.

---

## ⚖️ Ethics & Limitations

Predicting student outcomes raises important ethical questions. Predictions should:
- Be used to **trigger support**, not penalise students
- Be **transparent and explainable** to affected individuals
- Be **regularly audited for bias**, particularly across demographic groups (gender, nationality, disability)
- Never replace human judgement in student welfare decisions

This project is a research prototype only and does not involve primary data collection or a production system.

---

## 📚 Key References

- Valentim et al. (2021) — UCI Dropout Dataset. https://doi.org/10.24432/C5MC89
- Tinto, V. (1975) — Dropout from Higher Education: A Theoretical Synthesis
- Kotsiantis et al. (2003) — Preventing Student Dropout in Distance Learning Using ML
- Lykourentzou et al. (2009) — Dropout Prediction in E-Learning Using ML Techniques
- Pal, A.K. (2012) — Classification Model of Prediction for Placement of Students

---

## 📄 License

This project was produced as part of an undergraduate academic assessment at UWE Bristol. The datasets used are publicly available and anonymised. The code is available for educational and research use.
