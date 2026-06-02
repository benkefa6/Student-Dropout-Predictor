# Predicting University Student Attrition: A Machine Learning Approach

**Module:** UFCEKP-30-3 — Data Science and AI Individual Project  
**Institution:** University of the West of England (UWE Bristol)  
**Student:** Benedict Kefa Purnomo (25049053)  
**Supervisor:** Dr. Ethan Lee  
**Dataset:** [Open University Learning Analytics Dataset (OULAD)](https://analyse.kmi.open.ac.uk/open_dataset) — Kuzilek, Hlosta & Zdrahal (2017)

---

## Overview

This project builds and evaluates a machine learning pipeline to predict student dropout risk at the Open University, using only administrative enrolment-time data available before any course activity begins.

The key novelty is a **subgroup comparison** by prior qualification level — a stratification not previously reported in the OULAD literature:

| Subgroup | n | Dropout Rate |
|---|---|---|
| Higher Education (HE) | 5,043 | 26.9% |
| A Level or Below (AL) | 27,550 | 31.9% |

Each subgroup is modelled independently through the same pipeline, enabling a direct comparison of predictability by prior qualification background.

---

## Results Summary

| Model | HE ROC-AUC | AL ROC-AUC |
|---|---|---|
| Logistic Regression | 0.6163 | 0.6573 |
| Decision Tree | 0.5850 | 0.6529 |
| Random Forest | 0.6376 | 0.6712 |
| Gradient Boosting | 0.6364 | 0.6760 |
| **Voting (RF+GB)** | 0.6424 | **0.6766 ← best** |
| Stacking (RF+GB+LR) | 0.6389 | 0.6719 |
| Binary Voting (RF+GB) | 0.6419 | 0.6737 |

- A Level or Below students are consistently more predictable across every model
- Ensemble methods outperform all individual base classifiers
- 10-fold CV reveals Logistic Regression generalises best (HE train-test gap: 0.006); RF and GB overfit (gaps: 0.203 and 0.211)

---

## Repository Structure

```
├── 5th_Model_HE_vs_ALevel.ipynb     # Main notebook — full pipeline (HE vs AL)
├── 4th_Model.ipynb                  # Previous iteration
├── 3rd_Model.ipynb                  # Earlier iteration
├── 2nd_Model_Student_Predictor.ipynb
├── Student_Predictor_Model.ipynb    # Initial baseline
│
├── HE_model_readyCopy1.csv          # Preprocessed Higher Education subgroup
├── ALevel_model_readyCopy1.csv      # Preprocessed A Level or Below subgroup
├── final_training_data_v4.csv       # Full merged training data
├── studentInfo.csv                  # Raw OULAD studentInfo table
│
├── UFCEKP_DS_and_AI_Project_report_Revised.docx   # Final report
├── UFCEKP_Project_Proposal.docx                   # Approved project proposal
└── README.md
```

---

## Pipeline

```
Raw OULAD Data (studentInfo + studentVle)
        │
        ▼
Subgroup Split (HE vs A Level or Below)
        │
        ▼
Preprocessing
  • StandardScaler (fit on train only)
  • Stratified 80/20 train-test split
  • Random oversampling of minority class (sklearn.utils.resample)
        │
        ▼
Base Model Training (5-fold CV on balanced training set)
  • Logistic Regression
  • Decision Tree
  • Random Forest
  • Gradient Boosting
        │
        ▼
Ensemble Methods
  • Soft Voting Classifier (RF + GB)
  • Stacking Classifier (RF + GB → LR meta-learner)
  • Binary Voting Classifier (deeper RF + GB)
        │
        ▼
Evaluation
  • Accuracy, Weighted F1, Macro F1, ROC-AUC, Dropout-class F1
  • Full 10-fold cross-validation
  • Learning curve analysis
  • Cross-group ROC-AUC heatmap
  • Final summary table and plots
```

---

## Features Used

All features are sourced from the OULAD `studentInfo` table — available at enrolment time, before any course activity:

| Feature | Description |
|---|---|
| `code_module` | Module identifier |
| `code_presentation` | Presentation/year |
| `gender` | Student gender |
| `region` | UK region |
| `highest_education` | Prior qualification level |
| `imd_band` | Index of Multiple Deprivation band |
| `age_band` | Age group at enrolment |
| `num_of_prev_attempts` | Number of previous module attempts |
| `studied_credits` | Credits registered for |
| `disability` | Declared disability status |
| `total_clicks` | Total VLE interaction count (derived) |

**Target:** `dropout` — 1 if `final_result == Withdrawn`, 0 otherwise

---

## Requirements

```bash
pip install pandas numpy matplotlib scikit-learn
```

> No additional dependencies required. Oversampling is implemented via `sklearn.utils.resample` (no imbalanced-learn needed).

### Versions used

| Library | Version |
|---|---|
| Python | 3.10+ |
| pandas | 2.x |
| numpy | 1.x |
| scikit-learn | 1.x |
| matplotlib | 3.x |

---

## How to Run

1. Clone the repository
2. Place `HE_model_readyCopy1.csv` and `ALevel_model_readyCopy1.csv` in the same directory as the notebook
3. Open `5th_Model_HE_vs_ALevel.ipynb` in Jupyter
4. Run all cells — plots are saved as PNG files in the working directory

All random seeds are fixed (`RANDOM_SEED = 42`) for full reproducibility.

---

## Key Findings

1. **Subgroup matters.** AL students are consistently more predictable than HE students across every model and every metric. Dropout risk and model performance are moderated by prior qualification level.

2. **Ensembles win.** Soft Voting (RF+GB) achieves the best ROC-AUC in the study for both groups (HE: 0.6424, AL: 0.6766), confirming findings from the ensemble learning literature.

3. **Overfitting is a real risk.** RF and GB show large train-test gaps under 10-fold CV (HE: 0.203 and 0.211), caused by sensitivity to oversampled (duplicated) training instances. Logistic Regression generalises far better (gap: 0.006).

4. **Enrolment-time features have a ceiling.** The best ROC-AUC of 0.6766 is at the upper end of published demographic-only OULAD baselines (Adnan et al., 2021; Jha et al., 2019). Adding temporal VLE clickstream data would likely push performance toward 0.90+.

---

## Ethical Considerations

- The OULAD is fully anonymised and released under CC-BY 4.0 — no individual student can be re-identified
- IMD band and disability are protected characteristics; any deployed system must be audited for algorithmic bias
- Model outputs should be treated as **decision-support for human advisers**, not automated labels
- SHAP values are recommended before any real-world deployment to provide per-student explanations

---

## References

- Adnan, M. et al. (2021) *IEEE Access*, 9, pp. 7519–7539.
- Andrade-Girón, D. et al. (2023) *EAI Endorsed Transactions on Scalable Information Systems*, 10(5).
- Jha, N., Ghergulescu, I. and Moldovan, A.-N. (2019) *CSEDU 2019*, pp. 154–164.
- Kuzilek, J., Hlosta, M. and Zdrahal, Z. (2017) *Scientific Data*, 4, 170171.
- Lee, S. and Chung, J.Y. (2019) *Applied Sciences*, 9(15), 3093.
- Niyogisubizo, J. et al. (2022) *Computers and Education: Artificial Intelligence*, 3, 100066.
- Tinto, V. (1975) *Review of Educational Research*, 45(1), pp. 89–125.
- Wongvorachan, T., He, S. and Bulut, O. (2023) *Information*, 14(1), 54.

---

## Licence

This project was submitted as assessed coursework at UWE Bristol (UFCEKP-30-3). The OULAD dataset is used under [CC-BY 4.0](https://creativecommons.org/licenses/by/4.0/).
