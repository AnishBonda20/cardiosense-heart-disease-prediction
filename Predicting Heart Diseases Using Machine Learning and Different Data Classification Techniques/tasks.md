# tasks.md

## Now — resume point
Working on: nothing

## Open
- [ ] Revoke the old Gmail app password that was in app.py (report #1) — Anish, in the Google account
- [ ] Fill the [bracketed] placeholders in ../reports/*.docx (names, register numbers, guide, college, dates, project-plan weeks) and check each reference — Anish
- [x] Drop the unused tables in signup.db (accounts, details, detail, data, datas) (#2)
- [ ] Rate-limit OTP emails per address/IP (#7)
- [ ] Add CSRF protection (Flask-WTF CSRFProtect) now that sessions exist (#10)
- [ ] Add a database-level UNIQUE on info.user (the app checks it, the table doesn't) (#9)
- [ ] Delete unused theme assets: static/css (all but app.css), static/js (all but app.js), static/fonts, static/webfonts, static/sass, static/anova|chi2|mi (old notebook PNGs), static/images (all but form-img.png, UDMI_Cardiovascular-Disease.png, fav-icon.png), whitewolf-design.css, templates_old/, models_old/, ../.playwright-mcp/
- [x] Retrain the models on a current scikit-learn so the app runs on Python 3.11+ (train.py, Python 3.13)
- [x] Resample after the train/test split and compute AUC on test data (train.py; the old notebooks still show the leaky version)
- [x] Flip heart.csv labels and retrain (all 303 rows now agree with UCI processed.cleveland.data)
- [x] Clear the 3 History entries in signup.db made before the label fix
- [ ] Remove the old `heart` conda env (Python 3.10) once heart313 is confirmed working: `conda env remove -n heart`
- [x] Improve recall: picking the best classifier per method raised it to 71–79%
- [ ] Calibrate risk scores (Naive Bayes/KNN give near-0% or near-100%): wrap the saved model in CalibratedClassifierCV in train.py

## Recently landed
- **2026-10-27** — Saved the deck locally: ../presentation/CardioSense_Presentation.pptx (18 slides, speaker notes) and .pdf, plus slides/slide-01…18.png. Slides are full-resolution images; for editable text, export from the claude.ai deck (Share › Export).
  Learn: a picture-per-slide PowerPoint looks identical everywhere but can't be edited; the editable copy comes from the deck's own export.
- **2026-10-27** — Two Word reports in ../reports/ (with PDFs): CardioSense_Project_Report (35 pages, college template: abstract, 9 chapters, appendix, 6 UML/DFD diagrams, 10 screenshots, all numbers from metrics.json) and CardioSense_Project_Guide (15 pages: what it does, pitch + demo script + viva Q&A, run steps, troubleshooting).
  Learn: a table of contents in Word is a field built from the heading text, so chapter titles must be typed in capitals to appear in capitals there; styling them as caps isn't enough.
- **2026-10-27** — "Effectively tied" note on Insights and the results slide: methods are 1.6 points apart vs ±2.3 between folds. experiment_stacking.py tries 3 stronger stacking ensembles on all methods (81.3–83.8%, none beat the saved models); results shown on Insights.
  Learn: a difference smaller than the normal fold-to-fold wobble isn't evidence that one method is better.
- **2026-10-27** — Final version: "What stands out for this patient", print/save as PDF, Clear history. Full test: suite passes, 45/45 page × width checks, no console errors (TEST_REPORT.md). 18-slide deck in the CardioSense style: https://claude.ai/artifact/SanBU5x9Pi9iaqMDRq77jW
  Learn: describing what the *data* shows about each value is honest; claiming it's *why the model decided* would not be, because Naive Bayes and KNN don't expose their reasons that way.
- **2026-10-27** — (i) info icon on every input (meaning + a fact worked out from heart.csv). "Compare this patient with all 3 methods" carries the same patient from a result into Compare; dataset patients show the real diagnosis and ✓/✗ per model plus each model's accuracy.
  Learn: one patient only shows *how* models differ; accuracy is measured over many patients, which is why each card shows the model's overall score next to its ✓/✗.
- **2026-10-27** — Best-of-11 model per method (ANOVA→KNN, Chi²→Naive Bayes, MI→Naive Bayes; CV 84–86%). New inputs: clinical dropdown labels decoded against UCI, keyboard-friendly custom dropdown, yes/no toggles, numbers with unit + dataset range, inline errors. Dataset page explains SMOTEENN (21 synthetic added, 20 real + 1 synthetic removed, heart.csv untouched). Insights overflow fixed at 1280/820/375 px.
  Learn: choosing the "best" model must use only training data (cross-validation); if you pick by test score, the test score stops being an honest estimate.
- **2026-10-27** — Fixed `No module named 'numpy._core'`: the old `heart` conda env had Python 3.10 + numpy 1.23, but the retrained models need numpy 2.x. New conda env `heart313` (Python 3.13, requirements.txt) runs the app and passes test_app.py. Cleared 3 pre-fix History rows.
  Learn: a saved model remembers the library versions that wrote it, so the environment that runs the app must match requirements.txt, not just "have numpy".
- **2026-10-27** — Fixed reversed labels in heart.csv (1 = disease, verified row by row against UCI; original kept as archive/heart_kaggle_original.csv) and retrained. CV accuracy 80–83%, AUC 0.87–0.89. Insights now leads with CV accuracy. Previous models in models_old/before_label_fix/.
  Learn: flipping every label doesn't change how *accurate* a model is, only what its answers mean — which is why the bug was invisible in the accuracy scores and only showed up in the clinical sense-check.
- **2026-10-27** — Retrained with train.py: split first, tuned SMOTEENN on training data only, 5-fold CV, results in models/metrics.json. Runs on Python 3.13 (requirements.txt + requirements-train.txt). Insights charts are now drawn from the metrics (selection scores, leaderboard, ROC, confusion matrix, SMOTEENN comparison). Light/dark theme switch; animated pipeline icons.
  Learn: a test set only means something if it's set aside before anything learns from the data — resampling first let the old notebooks "see" the test patients, which is where the fake 100% came from.
- **2026-10-27** — New pages: Dataset (stats, group shares, feature glossary), Compare (one patient through all 3 models), History (saved predictions per user). Insights now has the 10-classifier leaderboard from the notebooks; Overview shows the 5-step pipeline.
  Learn: the History page needed a new `predictions` table; `create table if not exists` at startup adds it without touching existing data.
- **2026-10-27** — "Random patient from dataset" button on Predict: fills the form with a real row from archive/heart.csv and shows that patient's actual diagnosis. Replaced the fixed sample button.
  Learn: the page gets all 303 rows once (about 12 KB), so picking another patient is instant and needs no extra request to the server.
- **2026-10-27** — Dropped the 5 unused tables from signup.db and vacuumed it; only `info` (9 users) remains.
  Learn: deleting rows in SQLite leaves the old bytes in the file until VACUUM rewrites it, so "deleted" data can still be read without one.
- **2026-10-27** — New CardioSense UI (static/css/app.css, ECG motif, subtle motion), requirements.txt, README, server-side input rules, confidence ring, one route per page type.
  Learn: describing each input once (`FIELDS` in app.py) lets the same rule draw the form *and* check what comes back, so they can never disagree.
- **2026-10-27** — Fixed critical + high security findings: env-var secrets, hashed passwords, POST login, sessions, per-visitor signup state, 6-digit OTP with expiry and attempt cap. test_app.py covers them.
  Learn: a value stored in a global is shared by every visitor; anything per-person belongs in the session.
- **2026-10-26** — Security review written to SECURITY_REPORT.md (no code changed).
  Learn: "logging in" only means something if the server remembers it (a session); showing a page after a password check protects nothing.

## Frozen
- Dark mode for the new UI
