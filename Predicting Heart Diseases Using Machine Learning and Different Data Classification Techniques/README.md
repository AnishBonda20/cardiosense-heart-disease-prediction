# CardioSense — Predicting Heart Diseases with ML

A Flask web app that predicts heart disease from 13 clinical measurements. It compares three
feature-selection methods (ANOVA, Chi², Mutual Information), ten classifiers and a stacked ensemble.

## Run it

Python 3.10 or newer (tested on 3.13). In a terminal or Anaconda Prompt, from this folder:

```bash
python -m venv .venv                 # Anaconda: conda create -n heart python=3.13
.venv\Scripts\activate               # Anaconda: conda activate heart
pip install -r requirements.txt
```

Set these, then start the app (PowerShell shown; Command Prompt uses `set NAME=value`):

```powershell
$env:SECRET_KEY = "any-long-random-text"      # keeps people logged in across restarts
$env:SMTP_USER  = "you@gmail.com"             # sends the signup OTP
$env:SMTP_PASS  = "your-gmail-app-password"   # Google Account → Security → App passwords
python app.py
```

Open http://127.0.0.1:5000. Without `SMTP_USER`/`SMTP_PASS`, logging in works but new signups can't get an OTP.

## Pages

| Page | What it shows |
|------|---------------|
| Overview | The problem, the 5-step pipeline, the three methods |
| Predict | Enter a patient (or pick a random real one from the dataset) and get a risk score from one model. Every input has an (i) note explaining it, with a fact from the data. For a dataset patient the result also shows the real diagnosis and whether the model was right. "What stands out" marks each value as pointing toward or away from heart disease. Results can be printed or saved as PDF |
| Compare | One patient through all three models side by side. "Compare this patient" on a result carries the same patient over: a dataset patient is looked up in full; a typed-in one opens the form pre-filled. Each model shows ✓/✗ against the real diagnosis and its overall accuracy |
| Dataset | Class balance, averages by diagnosis, disease share per group, feature glossary |
| Insights | Feature-selection scores, 11-classifier leaderboard, ROC curve, confusion matrix, SMOTEENN comparison — all drawn from `models/metrics.json` |
| History | Every prediction the logged-in user has made, with a Clear history button |

Light and dark themes: the switch is in the top bar (it follows the system setting until you choose).

## Data

`archive/heart.csv` is the UCI Cleveland heart-disease data (303 patients). **`target`: 1 = heart disease, 0 = none.**

The Kaggle copy this project started from had every label reversed. All 303 rows were matched against the
original UCI file (`processed.cleveland.data`) on age, sex, blood pressure, cholesterol, fasting sugar,
max heart rate, angina and ST depression, and all 303 labels were the opposite of UCI's. The labels were
flipped to match UCI; the untouched Kaggle file is kept as `archive/heart_kaggle_original.csv`.
The original notebooks (`*.ipynb`) and their charts were made before this fix.

## Method (`train.py`)

1. **Clean**: 303 records → 302 after dropping one exact duplicate.
2. **Split first**: 61 patients (20%, stratified) are held back for testing before any resampling or selection.
3. **Balance with SMOTEENN** on the 241 training patients only. Tuned for this data: features are scaled
   first, SMOTENC keeps categorical codes as real values, and ENN uses `kind_sel='mode'`.
   (The default SMOTEENN removed about three quarters of the training set because raw cholesterol dominated
   its distance checks.) On this split SMOTE adds 21 synthetic patients with disease (110 → 131), then ENN
   removes 21 hard-to-classify ones (20 real, 1 synthetic), leaving 131 healthy + 110 with disease, 20 of
   them synthetic. `heart.csv` itself is never changed, which is why the Dataset page still shows 303 rows.
4. **Select features** with `SelectKBest`: ANOVA F-test (k=8), Chi² (k=11), mutual information (k=10).
5. **Train** 11 classifiers: Naive Bayes, SVM, XGBoost, AdaBoost, Bagging, Decision Tree, KNN, Random Forest,
   Logistic Regression, soft Voting, and Stacking (boosted decision stumps + extra trees → LightGBM).
6. **Pick the best** per method by 5-fold cross-validated accuracy on the training set (resampling and
   selection repeated inside every fold). Picking on training data keeps the test set a fair final check.
7. **Evaluate** the saved model on the 61 test patients.

### Results (saved model per method)

| Method | Features | Best classifier | 5-fold CV accuracy | Test accuracy | ROC AUC | Recall |
|--------|---------:|-----------------|-------------------:|--------------:|--------:|-------:|
| ANOVA | 8 | KNN | 85.5% ± 2.3 | 77.0% | 0.87 | 71.4% |
| Chi² | 11 | Naive Bayes | 84.6% ± 2.2 | 78.7% | 0.85 | 78.6% |
| MI | 10 | Naive Bayes | 83.8% ± 2.4 | 75.4% | 0.84 | 75.0% |

Cross-validation is the headline number: it averages five splits of the 241 training patients. The test set
is only 61 patients, so a single split swings by several points. Picking the best classifier instead of always
using Stacking raised CV accuracy by 2–5 points and recall (the share of disease cases caught) from 54–64% to
71–79%.

**The three methods are effectively tied.** The best models are 1.6 percentage points apart, while one model's
score moves about ±2.3 points between cross-validation folds, so no method is clearly better.
Three stronger stacking ensembles (different base models combined by logistic regression) were also tried on
all three methods with the same cross-validation; none beat the saved models (81.3–83.8%). Reproduce with
`python experiment_stacking.py` (about a minute; writes `models/stacking_experiment.json`, shown on Insights).

**SMOTEENN vs no resampling** (CV accuracy of the saved model): ANOVA 85.5% vs 84.6%, Chi² 84.6% vs 83.0%,
MI 83.8% vs 82.2%. A small, consistent gain; the dataset is already fairly balanced (164 healthy vs 138 with
disease). The default SMOTEENN does clearly worse (73–76%) because it throws most of the training data away.

**Risk scores**: Naive Bayes and KNN give very confident probabilities, so the app shows "<1%" or ">99%"
rather than 0% or 100%.

**Why not more data?** The other three UCI heart-disease sites (Hungary, Switzerland, VA Long Beach) would
bring the total to 920 patients, but they are missing most `ca` and `thal` values, which every model here
relies on, and use different codings. Merging them would add noise, not signal.

The original notebooks (`*.ipynb`) reported 100% for the stacking model. That came from resampling before
the train/test split, which let near-copies of test patients into training. `train.py` fixes this.

## Retrain

```bash
pip install -r requirements-train.txt
python train.py        # about 3 minutes; rewrites models/*.sav and models/metrics.json
```

The original model files are kept in `models_old/`.

## Presentation

An 18-slide deck in the app's style: https://claude.ai/artifact/SanBU5x9Pi9iaqMDRq77jW
(download it as PowerPoint or PDF from the deck's Share menu; fill in the [bracketed] names first).
Screenshots used in it are in `../presentation/screenshots/`.

## Test

```bash
python test_app.py
```

Runs signup → OTP → login → predict → compare → history against a throwaway database, and checks that
the saved models match the feature lists in `metrics.json`. Your `signup.db` isn't touched.
Full results, the browser checks and a manual checklist are in `TEST_REPORT.md`.

## Layout

| Path | What it is |
|------|------------|
| `app.py` | Routes, input rules (`FIELDS`), feature glossary, model setup |
| `train.py` | Reproducible training; writes `models/*.sav` and `models/metrics.json` |
| `experiment_stacking.py` | Extra stacking ensembles tried on all three methods; writes `models/stacking_experiment.json` |
| `templates/` | Pages. `base.html` is the frame, `_macros.html` holds shared pieces (risk ring, form fields, tabs) |
| `static/css/app.css` | The whole look, light and dark |
| `static/js/app.js` | Random patient button, theme switch, chart tooltips |
| `archive/heart.csv` | The dataset |
| `templates_old/`, `models_old/` | Previous versions, kept as backups. Safe to delete |
