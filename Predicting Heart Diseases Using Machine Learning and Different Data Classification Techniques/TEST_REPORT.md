# Test Report — CardioSense (final version)

**Date:** 2026-10-27 · **Environment:** Windows 11, Python 3.13.17 (conda env `heart313`), packages from `requirements.txt`
**Result:** everything below passed. Nothing failed or was skipped.

## 1. Automated suite — `python test_app.py`

Runs against a throwaway database (your `signup.db` is never touched). Takes about 5 seconds.
Output: `all checks passed` / `model checks passed`.

| Area | What is checked |
|------|-----------------|
| Passwords | Old plain-text passwords are hashed at startup and still log in; new ones are stored hashed |
| Login | Protected pages redirect to login; credentials in the URL (GET) are refused (405) |
| Signup / OTP | 6-digit code; 5 wrong tries end the signup; a second visitor can't hijack someone else's signup; duplicate usernames refused |
| Pages | All 13 pages return 200 when logged in |
| Inputs | All 303 dataset rows pass the form rules for every model; text in a number field and out-of-range values are rejected without a crash; unknown model → 404 |
| Models | Each saved model expects exactly the features in `metrics.json`; the saved model is the best of 11 by cross-validation; SMOTEENN counts add up |
| UI | Dropdowns, toggles and (i) icons render; Dataset explains SMOTEENN; Insights draws the ROC chart |
| Same patient | A dataset patient carries from a result into Compare and all 3 models see the full record; editing a value breaks the link to the dataset row; a typed-in patient opens Compare pre-filled |
| Result extras | "What stands out" flags the expected risk signs for a sick patient (age, angina, vessels, heart rate); print button present |
| History | Predictions are saved per user; Clear history removes only your own entries |

## 2. Browser test — every page, three screen widths

Each page was loaded at **1280px (desktop), 820px (tablet) and 375px (phone)**, including a submitted
result and a submitted comparison: **45 of 45 passed**.

For every page and width it checked:
- the page never scrolls sideways
- nothing sticks out past its card
- every dropdown was built (if JavaScript had failed, they wouldn't be)
- no broken images

Also checked by hand:
- browser console: **no errors**
- light and dark theme on Overview, Insights and Result; the theme choice is remembered between pages
- dropdown by mouse and by keyboard (↑ ↓ Enter, Esc, typing a letter)
- the info (i) note opens on hover, keyboard focus and tap, and stays inside its field
- empty form submit: 11 field errors shown, focus moves to the first empty field
- full flow: Random patient → Predict → ✓ Correct → Compare this patient → "3 of 3 correct" (patients 131 and 166)

## 3. Model results (from `train.py`, test set never used for any choice)

| Method | Saved classifier | CV accuracy | Test accuracy | AUC | Recall |
|--------|------------------|------------:|--------------:|----:|-------:|
| ANOVA | KNN | 85.5% ± 2.3 | 77.0% | 0.87 | 71.4% |
| Chi² | Naive Bayes | 84.6% ± 2.2 | 78.7% | 0.85 | 78.6% |
| MI | Naive Bayes | 83.8% ± 2.4 | 75.4% | 0.84 | 75.0% |

## 4. Your own test — checklist

Start: `conda activate heart313`, then `python app.py`, then open http://127.0.0.1:5000.

- [ ] Overview loads; the heart logo beats and the ECG line draws in
- [ ] Register with a real email → 6-digit code arrives → enter it → account created (needs `SMTP_USER`/`SMTP_PASS`)
- [ ] Enter a wrong code 5 times → you're sent back to Register
- [ ] Log in; try `/predict/anova` after logging out → back to the login page
- [ ] Predict: open each dropdown; use the keyboard; hover each (i) icon
- [ ] Submit an empty form → red messages under every field
- [ ] Random patient → Predict → result shows ✓ Correct or ✗ Wrong and "What stands out"
- [ ] "Compare this patient with all 3 methods" → same patient number, 3 cards with ✓/✗
- [ ] Type a patient by hand on ANOVA → Compare this patient → the form opens pre-filled, asking for 4 more values
- [ ] Print / save as PDF on a result → only the content prints, on white
- [ ] Dataset page: SMOTEENN section shows 241 → 262 → 241
- [ ] Insights: switch ANOVA / Chi² / MI; hover the ROC curve and the bars
- [ ] History shows your predictions; Clear history asks first, then empties it
- [ ] Theme switch (sun/moon) → dark, reload → still dark
- [ ] Make the window narrow (phone size) → nothing spills off the screen
