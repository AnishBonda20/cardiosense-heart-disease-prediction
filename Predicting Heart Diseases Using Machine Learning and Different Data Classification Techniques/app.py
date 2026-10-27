import os
import hmac
import secrets
import sqlite3
import smtplib
import time
import csv
import json
import warnings
from datetime import datetime
from email.message import EmailMessage
from functools import wraps

import joblib
import numpy as np
from flask import Flask, request, render_template, redirect, session, flash, url_for, abort, send_from_directory
from werkzeug.security import generate_password_hash, check_password_hash

warnings.filterwarnings('ignore')

# Secrets come from environment variables, never from the source code.
SMTP_USER = os.environ.get('SMTP_USER')
SMTP_PASS = os.environ.get('SMTP_PASS')
DB = os.environ.get('DB_PATH', 'signup.db')
OTP_TTL = 300        # seconds an OTP stays valid
OTP_MAX_TRIES = 5

app = Flask(__name__)
# Without SECRET_KEY set, a random key is used: logins reset every restart.
app.secret_key = os.environ.get('SECRET_KEY') or secrets.token_hex(32)

# Every input the models use: label, unit, and either (min, max, step) for a number or a list of
# (code, name, short explanation) choices. The form is drawn from this and the server checks
# submissions against it. The codes are the ones in heart.csv; their meanings were matched
# row by row against the original UCI Cleveland file (e.g. cp 0 here is UCI's "asymptomatic").
FIELDS = {
    'age':      ('Age', 'years', (1, 120, 1)),
    'sex':      ('Sex', '', [(0, 'Female', ''), (1, 'Male', '')]),
    'cp':       ('Chest pain type', '', [
                    (3, 'Typical angina', 'Chest pain brought on by effort, eased by rest'),
                    (1, 'Atypical angina', 'Chest pain that only partly fits that pattern'),
                    (2, 'Non-anginal pain', 'Chest pain not coming from the heart'),
                    (0, 'Asymptomatic', 'No chest pain')]),
    'trestbps': ('Resting blood pressure', 'mm Hg', (50, 250, 1)),
    'chol':     ('Serum cholesterol', 'mg/dl', (50, 700, 1)),
    'restecg':  ('Resting ECG result', '', [
                    (1, 'Normal', 'No abnormality on the resting ECG'),
                    (2, 'ST-T wave abnormality', 'T-wave inversion or ST change of more than 0.05 mV'),
                    (0, 'Left ventricular hypertrophy', 'Thickened heart wall (Estes criteria)')]),
    'thalach':  ('Max heart rate', 'bpm', (40, 250, 1)),
    'exang':    ('Exercise-induced angina', '', [(0, 'No', ''), (1, 'Yes', '')]),
    'oldpeak':  ('ST depression', 'mm', (0, 10, 0.1)),
    'slope':    ('ST segment slope', '', [
                    (2, 'Upsloping', 'Rises during peak exercise, usually normal'),
                    (1, 'Flat', 'Stays level during peak exercise'),
                    (0, 'Downsloping', 'Falls during peak exercise')]),
    'ca':       ('Major vessels coloured', '', [
                    (0, 'None', 'No major vessels seen on fluoroscopy'),
                    (1, '1 vessel', ''), (2, '2 vessels', ''), (3, '3 vessels', ''),
                    (4, 'Not recorded', 'Missing in the original data')]),
    'thal':     ('Thallium scan', '', [
                    (2, 'Normal', 'Normal blood flow to the heart'),
                    (1, 'Fixed defect', 'No blood flow in part of the heart, at rest or exercise'),
                    (3, 'Reversible defect', 'Reduced blood flow only during exercise'),
                    (0, 'Not recorded', 'Missing in the original data')]),
}
CHOICE_NAMES = {f: {code: name for code, name, _ in rule} for f, (_, _, rule) in FIELDS.items() if isinstance(rule, list)}

# What every column in heart.csv means. fbs is in the data but no selection method picked it.
GLOSSARY = {
    'age': ('Age', 'Age in years.'),
    'sex': ('Sex', 'Recorded as male or female.'),
    'cp': ('Chest pain type', 'Kind of chest pain the patient reports.'),
    'trestbps': ('Resting blood pressure', 'Blood pressure at rest on admission, in mm Hg. Above 130 counts as high.'),
    'chol': ('Serum cholesterol', 'Total cholesterol in the blood, in mg/dl. Above 240 counts as high.'),
    'fbs': ('Fasting blood sugar', 'Whether fasting blood sugar is above 120 mg/dl (1 = yes).'),
    'restecg': ('Resting ECG result', 'What the electrocardiogram (heart tracing) showed at rest.'),
    'thalach': ('Max heart rate', 'Highest heart rate reached during the exercise test.'),
    'exang': ('Exercise-induced angina', 'Chest pain brought on by exercise.'),
    'oldpeak': ('ST depression', 'How far the ECG ST segment drops during exercise compared with rest. Bigger drops suggest the heart muscle is short of blood during effort.'),
    'slope': ('ST segment slope', 'Shape of the ECG ST segment at peak exercise. Upsloping is usually normal; flat or downsloping can be a warning sign.'),
    'ca': ('Major vessels coloured', 'How many of the main heart arteries (0–3) showed up on fluoroscopy, an X-ray with dye.'),
    'thal': ('Thallium scan', 'Result of a thallium stress scan, which shows how well blood reaches the heart muscle.'),
}

# Each model's display text and files. Its feature list and test results come from
# models/metrics.json, which train.py writes when it retrains the models.
MODELS = {
    'anova': dict(name='ANOVA', long='Analysis of variance', file='models/model_anova.sav', notebook='Anova.html',
                  blurb='Keeps the features whose averages differ most between healthy and diseased patients.'),
    'chi2': dict(name='Chi²', long='Chi-square test', file='models/model_chi.sav', notebook='Chi2.html',
                 blurb='Keeps the features that depend most strongly on the diagnosis.'),
    'mi': dict(name='MI', long='Mutual information', file='models/model_mi.sav', notebook='MI.html',
               blurb='Keeps the features that share the most information with the diagnosis.'),
}
with open('models/metrics.json') as fh:
    METRICS = json.load(fh)
for key, m in MODELS.items():
    m.update(METRICS['methods'][key])  # adds fields-in-training-order, results, confusion, roc, ...
    m['fields'] = m.pop('features')
COMPARE_FIELDS = list(FIELDS)  # every field used by at least one model

# Extra stacking ensembles tried on all three methods (experiment_stacking.py); shown on Insights.
try:
    with open('models/stacking_experiment.json') as fh:
        STACKING_EXPERIMENT = json.load(fh)
except FileNotFoundError:
    STACKING_EXPERIMENT = []


def method_spread():
    """How far apart the three saved models are, next to how much one model's score moves between folds."""
    best = {k: next(r for r in m['results'] if r['name'] == m['saved']) for k, m in MODELS.items()}
    means = [r['cv_mean'] for r in best.values()]
    return dict(gap=max(means) - min(means), fold_spread=sum(r['cv_std'] for r in best.values()) / len(best))
# joblib uses pickle, which can run code: only load model files you trained yourself.
LOADED = {key: joblib.load(m['file']) for key, m in MODELS.items()}

# The 303 real patient rows the models were trained on, used by the "Random patient" button.
with open('archive/heart.csv', encoding='utf-8-sig') as fh:
    DATASET = [{k: float(v) for k, v in row.items()} for row in csv.DictReader(fh)]


def dataset_stats():
    """Summary numbers for the Dataset page, worked out once at startup."""
    pos = [r for r in DATASET if r['target'] == 1]
    neg = [r for r in DATASET if r['target'] == 0]
    mean = lambda rows, k: round(sum(r[k] for r in rows) / len(rows), 1)
    numeric = [(k, mean(pos, k), mean(neg, k), max(r[k] for r in DATASET))
               for k in ('age', 'trestbps', 'chol', 'thalach', 'oldpeak')]
    categorical = {}
    for k in ('sex', 'cp', 'exang', 'restecg', 'slope', 'ca', 'thal'):
        groups = []
        for value, label, _ in FIELDS[k][2]:
            rows = [r for r in DATASET if r[k] == value]
            if rows:
                groups.append((label, len(rows), sum(1 for r in rows if r['target'] == 1)))
        categorical[k] = groups
    return dict(n=len(DATASET), pos=len(pos), neg=len(neg), numeric=numeric, categorical=categorical)


STATS = dataset_stats()


def info_text(f):
    """What the info (i) icon next to an input says: what it means, plus what the data shows."""
    text = GLOSSARY[f][1]
    numeric = {k: (p, n) for k, p, n, _ in STATS['numeric']}
    if f in numeric:
        p, n = numeric[f]
        return f'{text} In this dataset the average is {p} for patients with heart disease and {n} without.'
    groups = [(npos / count, label) for label, count, npos in STATS['categorical'].get(f, []) if count >= 15]
    if groups:
        share, label = max(groups)
        return f'{text} In this dataset, {round(share * 100)}% of patients in the "{label}" group have heart disease.'
    return text


INFO = {f: info_text(f) for f in FIELDS}
DISEASE_SHARE = STATS['pos'] / STATS['n']


def explain(inputs):
    """For each of a patient's values, whether people in the dataset with that value tend to have heart
    disease. This describes the data, not how the model decided, and the page says so.
    Returns (field, value, direction, note), direction 'up' (points toward disease), 'down' or 'even',
    ordered up → down → even."""
    numeric = {k: (p, n) for k, p, n, _ in STATS['numeric']}
    out = []
    for f, v in inputs:
        if f in numeric:
            p, n = numeric[f]
            middle = (p + n) / 2
            # nearer the average of patients with disease → points toward disease
            direction = 'even' if v == middle else ('up' if (v - middle) * (p - n) > 0 else 'down')
            note = f'Nearer the average for patients {"with" if direction == "up" else "without"} heart disease ({p} with, {n} without).'
        else:
            group = next((g for g in STATS['categorical'].get(f, []) if g[0] == CHOICE_NAMES[f][int(v)]), None)
            if not group or group[1] < 15:
                continue  # too few patients with this value to say anything
            share = group[2] / group[1]
            direction = 'up' if share >= DISEASE_SHARE + .12 else ('down' if share <= DISEASE_SHARE - .12 else 'even')
            note = f'{round(share * 100)}% of the {group[1]} patients in this group have heart disease (overall {round(DISEASE_SHARE * 100)}%).'
        out.append((f, v, direction, note))
    return sorted(out, key=lambda row: ['up', 'down', 'even'].index(row[2]))

# For number inputs: (lowest, typical, highest) value in the dataset, shown as placeholder and hint.
RANGES = {}
for f, (_, _, rule) in FIELDS.items():
    if isinstance(rule, tuple):
        vals = sorted(r[f] for r in DATASET)
        median = vals[len(vals) // 2]
        RANGES[f] = tuple(int(v) if v == int(v) else v for v in (vals[0], median, vals[-1]))


def dataset_rows(fields):
    """Every dataset patient as [field values..., real diagnosis] for the "Random patient" button."""
    return [[r[f] for f in fields] + [r['target']] for r in DATASET]


def matching_row(form, values):
    """The heart.csv row index sent with the form, but only if every submitted value still matches that
    row. If someone picks a random patient and then edits a field, it's no longer that patient."""
    try:
        row = int(form.get('row', ''))
    except ValueError:
        return None
    if not 0 <= row < len(DATASET) or not values:
        return None
    return row if all(abs(DATASET[row][f] - v) < 1e-9 for f, v in values.items()) else None


def run_model(key, values):
    """Return (positive, risk) for one model; risk is the model's probability of heart disease."""
    est, x = LOADED[key], [np.array(values)]
    positive = int(est.predict(x)[0]) == 1
    risk = float(est.predict_proba(x)[0][1]) if hasattr(est, 'predict_proba') else float(positive)
    return positive, risk


def read_inputs(fields, form):
    """Return the form values as floats in model order, or raise ValueError with a readable message."""
    values = []
    for f in fields:
        label, _, rule = FIELDS[f]
        try:
            v = float(form[f])
        except (KeyError, ValueError):
            raise ValueError(f'{label} needs a number.')
        ok = (rule[0] <= v <= rule[1]) if isinstance(rule, tuple) else v in CHOICE_NAMES[f]
        if not ok:
            raise ValueError(f'{label} is out of range.')
        values.append(v)
    return values

# Signups waiting for their OTP, keyed by a random token kept in the visitor's session.
# ponytail: in-memory, so it only works with one server process; move to a DB table if you run several workers.
PENDING = {}


def db():
    return sqlite3.connect(DB)


def setup_db():
    """Create the tables if they're missing, and hash any plain-text password left in `info`."""
    con = db()
    con.execute("create table if not exists info(user varchar(250), email varchar(250), password varchar(250), mobile int, name varchar(250))")
    con.execute("create table if not exists predictions(user text, model text, inputs text, positive int, risk real, created text)")
    rows = con.execute("select rowid, password from info").fetchall()
    for rowid, pw in rows:
        if not str(pw).startswith(('scrypt:', 'pbkdf2:')):
            con.execute("update info set password = ? where rowid = ?", (generate_password_hash(str(pw)), rowid))
    con.commit()
    con.close()


setup_db()


def save_prediction(model, fields, values, positive, risk):
    con = db()
    con.execute("insert into predictions values (?, ?, ?, ?, ?, ?)",
                (session['user'], model, json.dumps(dict(zip(fields, values))), int(positive), risk,
                 datetime.now().strftime('%Y-%m-%d %H:%M')))
    con.commit()
    con.close()


def login_required(view):
    @wraps(view)
    def wrapped(*args, **kwargs):
        if 'user' not in session:
            return redirect(url_for('login'))
        return view(*args, **kwargs)
    return wrapped


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/logon')
def logon():
    return render_template('register.html')


@app.route('/login')
def login():
    return render_template('login.html')


@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('index'))


@app.route('/signup', methods=['POST'])
def signup():
    f = request.form
    username, name, email, number, password = (f.get(k, '').strip() for k in ('user', 'name', 'email', 'mobile', 'password'))
    if not all((username, name, email, number, password)):
        flash('Fill in every field.')
        return redirect(url_for('logon'))
    con = db()
    taken = con.execute("select 1 from info where `user` = ?", (username,)).fetchone()
    con.close()
    if taken:
        flash('That username is taken.')
        return redirect(url_for('logon'))
    if not (SMTP_USER and SMTP_PASS):
        flash('Email is not configured on the server (set SMTP_USER and SMTP_PASS).')
        return redirect(url_for('logon'))

    otp = f'{secrets.randbelow(10**6):06d}'
    msg = EmailMessage()
    msg.set_content('Your OTP is: ' + otp + '\nIt expires in 5 minutes.')
    msg['Subject'] = 'OTP'
    msg['From'] = SMTP_USER
    msg['To'] = email
    try:
        with smtplib.SMTP('smtp.gmail.com', 587) as s:
            s.starttls()
            s.login(SMTP_USER, SMTP_PASS)
            s.send_message(msg)
    except (smtplib.SMTPException, OSError, ValueError):
        flash('Could not send the OTP email. Check the address and try again.')
        return redirect(url_for('logon'))

    now = time.time()
    for t in [t for t, v in PENDING.items() if v['expires'] < now]:
        del PENDING[t]
    token = secrets.token_urlsafe(16)
    PENDING[token] = dict(user=username, name=name, email=email, mobile=number,
                          password=generate_password_hash(password),
                          otp=otp, expires=time.time() + OTP_TTL, tries=0)
    session['pending'] = token
    return render_template('val.html')


@app.route('/predict_lo', methods=['POST'])
def predict_lo():
    p = PENDING.get(session.get('pending', ''))
    if not p or time.time() > p['expires']:
        PENDING.pop(session.pop('pending', ''), None)
        flash('Your OTP expired. Register again.')
        return redirect(url_for('logon'))

    if not hmac.compare_digest(request.form.get('message', '').strip(), p['otp']):
        p['tries'] += 1
        if p['tries'] >= OTP_MAX_TRIES:
            PENDING.pop(session.pop('pending'), None)
            flash('Too many wrong OTPs. Register again.')
            return redirect(url_for('logon'))
        flash('Wrong OTP. Try again.')
        return render_template('val.html')

    PENDING.pop(session.pop('pending'))
    con = db()
    con.execute("insert into `info` (`user`,`email`, `password`,`mobile`,`name`) VALUES (?, ?, ?, ?, ?)",
                (p['user'], p['email'], p['password'], p['mobile'], p['name']))
    con.commit()
    con.close()
    flash('Account created. Log in.', 'ok')
    return redirect(url_for('login'))


@app.route('/signin', methods=['POST'])
def signin():
    user = request.form.get('user', '')
    con = db()
    row = con.execute("select password from info where `user` = ?", (user,)).fetchone()
    con.close()
    if row and check_password_hash(row[0], request.form.get('password', '')):
        session.clear()
        session['user'] = user
        return redirect(url_for('predict', model='anova'))
    flash('Wrong username or password.')
    return redirect(url_for('login'))


# ── pages behind the login ──
def get_model(key):
    if key not in MODELS:
        abort(404)
    return MODELS[key]


@app.context_processor
def shared():
    return dict(MODELS=MODELS, FIELDS=FIELDS, CHOICE_NAMES=CHOICE_NAMES, GLOSSARY=GLOSSARY, META=METRICS['meta'],
                RANGES=RANGES, INFO=INFO, explain=explain)


@app.context_processor
def comparison():
    return dict(STACKING_EXPERIMENT=STACKING_EXPERIMENT, SPREAD=method_spread())


@app.route('/predict/<model>', methods=['GET', 'POST'])
@login_required
def predict(model):
    m = get_model(model)
    if request.method == 'GET':
        return render_template('predict.html', key=model, m=m, rows=dataset_rows(m['fields']))
    try:
        values = read_inputs(m['fields'], request.form)
    except ValueError as e:
        flash(str(e))
        return redirect(url_for('predict', model=model))
    positive, risk = run_model(model, values)
    save_prediction(model, m['fields'], values, positive, risk)
    row = matching_row(request.form, dict(zip(m['fields'], values)))
    return render_template('result.html', key=model, m=m, positive=positive, risk=risk,
                           inputs=list(zip(m['fields'], values)), row=row,
                           actual=None if row is None else DATASET[row]['target'] == 1)


@app.route('/compare', methods=['GET', 'POST'])
@login_required
def compare():
    """Run one patient through all three models side by side.

    Arriving from a single-model result ("carry"), the same patient comes along: a dataset patient is
    looked up in full, so all three models see exactly the same person; a hand-entered patient has only
    the values that one model needed, so the form opens pre-filled and asks for the rest."""
    if request.method == 'GET':
        return render_template('compare.html', fields=COMPARE_FIELDS, rows=dataset_rows(COMPARE_FIELDS), values={})
    given = {}
    for f in COMPARE_FIELDS:  # whichever values arrived and are valid
        try:
            given[f] = read_inputs([f], request.form)[0]
        except ValueError:
            pass
    row = matching_row(request.form, given)
    if row is not None:
        patient = {f: DATASET[row][f] for f in COMPARE_FIELDS}
    elif len(given) == len(COMPARE_FIELDS):
        patient = given
    else:
        missing = [FIELDS[f][0] for f in COMPARE_FIELDS if f not in given]
        if request.form.get('carry'):
            flash(f'Same patient as before. Add the {len(missing)} values the other methods also use: {", ".join(missing)}.', 'ok')
        else:
            flash(f'{missing[0]} needs a valid value.')
        return render_template('compare.html', fields=COMPARE_FIELDS, rows=dataset_rows(COMPARE_FIELDS), values=given)
    results = []
    for key, m in MODELS.items():
        x = [patient[f] for f in m['fields']]
        positive, risk = run_model(key, x)
        save_prediction(key, m['fields'], x, positive, risk)
        results.append((key, m, positive, risk))
    return render_template('compare_result.html', results=results, inputs=list(patient.items()), row=row,
                           actual=None if row is None else DATASET[row]['target'] == 1)


@app.route('/history')
@login_required
def history():
    con = db()
    rows = con.execute("select model, inputs, positive, risk, created from predictions "
                       "where user = ? order by rowid desc limit 200", (session['user'],)).fetchall()
    con.close()
    entries = [dict(model=m, inputs=json.loads(i), positive=bool(p), risk=r, created=c)
               for m, i, p, r, c in rows if m in MODELS]
    return render_template('history.html', entries=entries)


@app.route('/history/clear', methods=['POST'])
@login_required
def clear_history():
    con = db()
    con.execute("delete from predictions where user = ?", (session['user'],))
    con.commit()
    con.close()
    flash('History cleared.', 'ok')
    return redirect(url_for('history'))


@app.route('/dataset')
@login_required
def dataset():
    return render_template('dataset.html', stats=STATS)


@app.route('/insights/<model>')
@login_required
def insights(model):
    return render_template('insights.html', key=model, m=get_model(model))


@app.route('/notebooks')
@login_required
def notebooks():
    return render_template('notebooks.html')


@app.route('/notebooks/<model>')
@login_required
def notebook(model):
    # Served as a plain file: the 1 MB notebook export doesn't need to go through the template engine.
    return send_from_directory('templates', get_model(model)['notebook'])


if __name__ == "__main__":
    app.run(debug=False)
