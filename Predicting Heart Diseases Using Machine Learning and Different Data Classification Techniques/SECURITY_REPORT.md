# Security Report — Heart Disease Prediction (Flask app)

**Date:** 2026-10-26 · **Scope:** `app.py`, `templates/`, `static/js/`, `signup.db`, `models/`, notebooks, `zip/`
**Method:** read-only code review, plus a look at the database's structure (no passwords printed). Nothing was changed.

## What the project is

A Flask website that predicts heart disease. Users sign up (an OTP gets emailed to them), log in, then
fill in a form on `home1/2/3`. The values go to one of three saved scikit-learn models
(`model_anova.sav`, `model_chi.sav`, `model_mi.sav`) and the page shows POSITIVE or NEGATIVE.
The notebooks (`Anova/Chi2/MI.ipynb`) train those models from `archive/heart.csv`.

## Summary

| # | Severity | Issue | Where |
|---|----------|-------|-------|
| 1 | **Critical** | Gmail app password hard-coded in source | `app.py:76-77` |
| 2 | **Critical** | Passwords stored in plain text; the DB ships with real user data | `app.py:93`, `signup.db` |
| 3 | **High** | Login doesn't protect anything: no session, every page open to everyone | `app.py:98-112`, all routes |
| 4 | **High** | Signup state kept in global variables, shared by all visitors | `app.py:63-96` |
| 5 | **High** | Weak OTP: guessable, no expiry, no attempt limit, printed to console | `app.py:69-70, 87` |
| 6 | **High** | Username and password sent in the URL (GET forms) | `login.html:54`, `register.html:51` |
| 7 | Medium | `/signup` will email any address, as often as asked (spam via your account) | `app.py:61-81` |
| 8 | Medium | No server-side input checks → crashes (500 errors) | `app.py:87, 116-163` |
| 9 | Medium | Usernames aren't unique (`admin` appears 3 times) | `info` table |
| 10 | Low | No CSRF protection on POST forms | `val.html`, `home*.html` |
| 11 | Low | Models loaded with `joblib` (pickle) on every request | `app.py:119, 135, 151` |
| 12 | Low | Old/pre-release front-end libraries; CDN scripts without integrity checks | `static/js/`, notebook HTML |
| 13 | Info | `eval()` in `contact_form.js` | `static/js/contact_form.js` |

**What's already fine:** SQL queries use `?` placeholders, so no SQL injection. Jinja escapes `{{ output }}`
and there's no `|safe` anywhere, so no XSS found. `debug=False`, so no Werkzeug debugger exposed.
Flask 3.1.3 / Werkzeug 3.1.8 / Jinja2 3.1.6 are current.

---

## Findings in detail

### 1. Gmail app password hard-coded — Critical
`app.py:76-77` has the sender address and its Gmail **app password** in plain text
(not repeated here). Anyone who gets this code (GitHub, a zip sent to a friend, a project
submission) can send mail as that account.

**Fix now:** revoke that app password in the Google account (Security → App passwords) and make a new one.
Then read it from an environment variable:
```python
import os
SMTP_USER = os.environ["SMTP_USER"]
SMTP_PASS = os.environ["SMTP_PASS"]
```
If this folder was ever pushed to Git, the old password stays in the history, so revoking it is the real fix.

### 2. Plain-text passwords; real user data in the repo — Critical
`predict_lo` inserts the password exactly as typed, and `signin` compares it as text. All 9 rows in `info`
have unhashed passwords, and the short ones (5–6 characters) were never checked on the server.
`signup.db` also holds leftover tables (`accounts` 6 rows, `details` 10, `detail` 3, `datas` 1,
`data` 10,490) with names, emails, phone numbers and passwords.

**Fix:** Werkzeug is already installed, so hash with it:
```python
from werkzeug.security import generate_password_hash, check_password_hash
# signup:  store generate_password_hash(password)
# signin:  SELECT password WHERE user=?  then  check_password_hash(row[0], password1)
```
Don't share `signup.db`; ship an empty one (or have the app create the table) and add it to `.gitignore`.
Drop the unused tables.

### 3. Login protects nothing — High
`/signin` only decides which HTML page to show. There's no `session`, no cookie, no logout. Anyone can
open `/home1`, `/home2`, `/home3`, `/notebook1-3` or POST to `/predict*` directly without logging in.

**Fix:** set `app.secret_key` (from an env var), put `session["user"] = ...` in `signin` after a successful
login, and check `if "user" not in session: return redirect("/login")` in the protected routes
(a small decorator saves repeating it).

### 4. Signup state in global variables — High
`otp, username, name, email, number, password` are module-level globals. The server is one process
handling every visitor, so:
- If A starts signup and then B starts signup, B overwrites A's data. A's OTP no longer works, and A
  could end up creating **B's** account (with B's email and password) by typing B's OTP.
- An attacker can start a signup right before a victim types their OTP and take over the pending registration.

**Fix:** keep the pending signup in `session` (signed cookie) or a `pending` table keyed per user, not in globals.

### 5. Weak OTP — High
`random.randint(1000, 5000)` gives only 4,001 options. `random` isn't meant for security, the code never
expires, the server allows unlimited guesses, and `print(otp)` writes it to the console/logs.
Someone could register using anyone's email and brute-force the OTP in a few thousand requests,
which defeats email verification.

**Fix:** `secrets.randbelow(900000) + 100000` (6 digits), save it with a timestamp, reject after
~5 minutes or ~5 wrong tries, and remove the `print`.

### 6. Credentials in the URL — High
The login and register forms use `method="GET"`, so passwords appear in the address bar, browser
history, server access logs and `Referer` headers, e.g. `/signin?user=admin&password=...`.

**Fix:** `method="POST"` in both templates, `methods=["POST"]` on the routes, and `request.form` instead of `request.args`.

### 7. Email sending can be abused — Medium
Any GET to `/signup?email=anyone@x.com` sends an email from your Gmail account. There's no rate limit
or CAPTCHA, so a script could spam strangers and get the account suspended. An SMTP failure also
crashes the request with a 500 error.
**Fix:** POST only (see #6), limit sends per IP/email (even a simple timestamp check), wrap the SMTP call in `try/except`.
(Header injection through `email` isn't possible: Python's `EmailMessage` refuses newlines in headers.)

### 8. No server-side input checks — Medium
- `int(message)` in `predict_lo` → 500 error if the OTP field isn't a number.
- `float(x) for x in request.form.values()` → 500 error on text; a wrong number of fields → the model raises.
  It also depends on the order of the form fields.
- If the model ever returns something other than 0/1, `output` is never set → `UnboundLocalError`.
- The `pattern=` rules in `register.html` only run in the browser and are easy to skip.
  (Also a bug: the **Name** input has no `name=` attribute, so `name` is always saved empty.)

**Fix:** read fields by name (`request.form["0"] ... ["7"]`), wrap the conversion in `try/except ValueError`
and show a friendly error, and check the password rules again in Python.

### 9. Usernames aren't unique — Medium
The `info` table has no `UNIQUE` constraint, and `admin` is registered 3 times. Login takes whichever
row comes first. **Fix:** `user TEXT UNIQUE` (a database constraint is simpler than app code) and handle `sqlite3.IntegrityError`.

### 10. No CSRF protection — Low
POST forms carry no CSRF token. The impact is small today because there are no sessions (#3),
but once a login session exists, add `Flask-WTF`'s `CSRFProtect(app)`.

### 11. Pickle model files — Low
`joblib.load` runs pickle, which can execute code from the file. That's fine for your own files,
but never load a `.sav` from someone else. Also load the 3 models once at startup instead of on every request (faster too).

### 12. Front-end libraries — Low
- `bootstrap.min.js` is **v5.0.0-beta3** (a pre-release) → move to a stable 5.3.x.
- jQuery 3.6.0 has no known critical issues, but 3.7.1 is current.
- The exported notebook pages (`Anova/Chi2/MI.html`) load MathJax and require.js from cdnjs without
  `integrity=` attributes. Low risk; only matters if the CDN is compromised.

### 13. `eval()` in `contact_form.js` — Info
`eval(valid)` runs the `data-valid` attribute. Templates control that attribute, so it isn't exploitable
now, but don't ever let user input reach it.

---

## Suggested order of work
1. **Today:** revoke the Gmail app password (#1). Stop sharing `signup.db` (#2).
2. Hash passwords and switch login/register to POST (#2, #6).
3. Add `session` with a secret key, protect pages, and move signup state out of globals (#3, #4).
4. Stronger OTP with expiry and attempt limits; rate-limit emails (#5, #7).
5. Input validation, `UNIQUE` username, load models at startup (#8, #9, #11).
6. Update libraries and add CSRF once sessions exist (#10, #12).
