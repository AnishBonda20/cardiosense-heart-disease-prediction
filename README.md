# CardioSense — Predicting Heart Diseases Using Machine Learning and Different Data Classification Techniques

Predicts heart disease from 13 routine clinical measurements (UCI Cleveland data, 303 patients) and compares
three feature-selection methods — ANOVA, Chi-square and Mutual Information — across eleven classifiers,
with a leakage-free evaluation and a Flask web app.

| Method | Best classifier | 5-fold CV accuracy |
|--------|-----------------|-------------------:|
| ANOVA | KNN | 85.5% ± 2.3 |
| Chi-square | Naive Bayes | 84.6% ± 2.2 |
| Mutual Information | Naive Bayes | 83.8% ± 2.4 |

The three are within normal fold-to-fold variation of each other. A study and screening aid, not a diagnosis.

## What's here

| Folder | Contents |
|--------|----------|
| [`Predicting Heart Diseases Using …/`](Predicting%20Heart%20Diseases%20Using%20Machine%20Learning%20and%20Different%20Data%20Classification%20Techniques/) | The application, training script, models, dataset and tests — see its README for how to run it |
| [`reports/`](reports/) | Project report and project guide (Word and PDF), plus the design diagrams |
| [`presentation/`](presentation/) | Slide deck (PowerPoint and PDF) and app screenshots |

## Quick start

```bash
cd "Predicting Heart Diseases Using Machine Learning and Different Data Classification Techniques"
python -m venv .venv
.venv\Scripts\activate            # macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
python app.py                     # then open http://127.0.0.1:5000
```

Needs Python 3.10 or newer. Set `SECRET_KEY`, and `SMTP_USER` / `SMTP_PASS` (a Gmail app password) for sign-up emails.

## Data

UCI Machine Learning Repository, Heart Disease (Cleveland), Janosi et al. 1988, CC BY 4.0
(doi:10.24432/C52P4X). The labels in the commonly shared Kaggle copy were reversed; they were corrected
against the UCI original (the uncorrected file is kept as `archive/heart_kaggle_original.csv`).
