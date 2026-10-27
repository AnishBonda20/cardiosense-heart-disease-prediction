"""Smoke test for the login/signup security fixes. Run: python test_app.py
Uses a throwaway database, so signup.db is never touched."""
import os
import sqlite3
import tempfile

db_path = os.path.join(tempfile.mkdtemp(), 'test.db')
con = sqlite3.connect(db_path)
con.execute("CREATE TABLE info(user varchar(250), email varchar(250), password varchar(250), mobile int, name varchar(250))")
con.execute("INSERT INTO info VALUES ('olduser', 'o@x.com', 'plain123', 0, 'Old')")  # legacy plain-text row
con.commit()
con.close()
os.environ.update(DB_PATH=db_path, SMTP_USER='bot@x.com', SMTP_PASS='x')

import smtplib
sent = []


class FakeSMTP:
    def __init__(self, *a): pass
    def __enter__(self): return self
    def __exit__(self, *a): pass
    def starttls(self): pass
    def login(self, *a): pass
    def send_message(self, msg): sent.append(msg)


smtplib.SMTP = FakeSMTP
import app  # noqa: E402  (must import after the env vars are set)

c = app.app.test_client()
stored = lambda u: sqlite3.connect(db_path).execute("select password from info where user=?", (u,)).fetchone()[0]

# legacy plain-text password was hashed at startup, and still logs in
assert stored('olduser').startswith(('scrypt:', 'pbkdf2:'))
assert c.post('/signin', data={'user': 'olduser', 'password': 'plain123'}).headers['Location'].endswith('/predict/anova')
c.get('/logout')

# protected pages need a login; credentials can't go in the URL
assert c.get('/predict/anova').status_code == 302
assert c.post('/predict/anova').status_code == 302
assert c.get('/signin?user=olduser&password=plain123').status_code == 405

# signup: OTP is 6 digits, wrong guesses are capped
form = {'user': 'newuser1', 'name': 'New', 'email': 'n@x.com', 'mobile': '9999999999', 'password': 'Passw0rd!'}
c.post('/signup', data=form)
otp = sent[-1].get_content().split(': ')[1][:6]
assert len(otp) == 6 and otp.isdigit()
for _ in range(app.OTP_MAX_TRIES - 1):
    c.post('/predict_lo', data={'message': '000000' if otp != '000000' else '111111'})
assert c.post('/predict_lo', data={'message': 'nope'}).headers['Location'].endswith('/logon')  # 5th wrong try ends it
assert c.post('/predict_lo', data={'message': otp}).headers['Location'].endswith('/logon')     # right OTP now too late

# a second visitor's signup can't hijack the first one's
c.post('/signup', data=form)
otp = sent[-1].get_content().split(': ')[1][:6]
other = app.app.test_client()
other.post('/signup', data=dict(form, user='attacker', email='a@x.com'))
assert c.post('/predict_lo', data={'message': otp}).headers['Location'].endswith('/login')
assert stored('newuser1').startswith(('scrypt:', 'pbkdf2:'))
assert c.post('/signup', data=form).headers['Location'].endswith('/logon')  # username now taken

# login + every page + a prediction from each model
assert c.post('/signin', data={'user': 'newuser1', 'password': 'Passw0rd!'}).status_code == 302
for page in ['/', '/predict/anova', '/predict/chi2', '/predict/mi', '/insights/anova', '/insights/chi2', '/insights/mi', '/notebooks', '/notebooks/mi', '/dataset', '/compare', '/history', '/insights/chi2']:
    assert c.get(page).status_code == 200, page
row = lambda m, i=0: {f: app.DATASET[i][f] for f in m['fields']}
for model, m in app.MODELS.items():
    for i in range(len(app.DATASET)):  # every dataset row fits the form rules, so "Random patient" never fails
        app.read_inputs(m['fields'], row(m, i))
    assert b'data-random' in c.get(f'/predict/{model}').data
    r = c.post(f'/predict/{model}', data=row(m))
    assert r.status_code == 200 and (b'POSITIVE' in r.data or b'NEGATIVE' in r.data), model
assert c.post('/predict/anova', data={'sex': 'abc'}).status_code == 302  # bad input: no crash
bad = dict(row(app.MODELS['anova']), thalach=999)
assert c.post('/predict/anova', data=bad).status_code == 302  # out of range is rejected
assert c.get('/predict/nope').status_code == 404

# compare runs all three models; history lists this user's predictions only
r = c.post('/compare', data={f: app.DATASET[5][f] for f in app.COMPARE_FIELDS})
assert r.status_code == 200 and r.data.count(b'estimated risk') == 3
saved = sqlite3.connect(db_path).execute("select count(*) from predictions where user='newuser1'").fetchone()[0]
assert saved == 6, saved  # 3 single predictions + 3 from compare
assert b'POSITIVE' in c.get('/history').data or b'NEGATIVE' in c.get('/history').data
other.post('/signin', data={'user': 'olduser', 'password': 'plain123'})
assert b'No predictions yet' in other.get('/history').data
print('all checks passed')

# models were retrained by train.py: the app's feature lists match what each saved model expects
for key, m in app.MODELS.items():
    assert app.LOADED[key].n_features_in_ == len(m['fields']) == m['k'], key
    assert m['saved'] in m['roc'] and len(m['confusion']) == 2
    best = max(m['results'], key=lambda r: (round(r['cv_mean'], 3), -r['cv_std']))
    assert m['saved'] == best['name'], (key, m['saved'], best['name'])  # the best of the 11 is what's saved
sm = app.METRICS['meta']['smoteenn']
assert sum(sm['after_enn']) == sum(sm['before']) - sm['real_removed'] + sm['synthetic_kept']  # SMOTEENN counts add up
r = c.get('/predict/chi2').data
assert b'data-dd' in r and b'Non-anginal pain' in r and b'class="toggle"' in r  # dropdowns and toggles render
assert b'Why this page still shows' in c.get('/dataset').data
r = c.get('/insights/mi').data
assert b'<svg class="roc"' in r and b'Does SMOTEENN help?' in r

# the same patient carries from a single-model result into Compare
anova = app.MODELS['anova']
pick = dict(row(anova, 10), row='10')
r = c.post('/predict/anova', data=pick).data
assert b'Correct' in r or b'Wrong' in r, 'dataset patient shows its real diagnosis'
assert b'name="carry"' in r and b'name="row" value="10"' in r
carried = dict(row(anova, 10), row='10', carry='1')                 # only ANOVA's 8 values travel...
r = c.post('/compare', data=carried).data
assert r.count(b'estimated risk') == 3 and b'of 3 correct' in r     # ...but all 3 models see patient 11 in full
assert b'patient 11 from heart.csv' in r
edited = dict(pick, thalach=pick['thalach'] + 1)                     # edited after picking: no longer that patient
assert b'Correct' not in c.post('/predict/anova', data=edited).data and b'Wrong' not in c.post('/predict/anova', data=edited).data
typed = dict(row(anova, 10), row='', carry='1')                      # hand-entered: form opens pre-filled
r = c.post('/compare', data=typed).data
assert b'Same patient as before' in r and b'estimated risk' not in r
assert b'value="%d"' % int(app.DATASET[10]['thalach']) in r
assert b'class="info"' in c.get('/predict/mi').data                 # (i) icons render
r = c.post('/predict/anova', data=pick).data
assert b'What stands out for this patient' in r and b'data-print' in r     # explanation + print button
sick = {f: app.DATASET[165][f] for f in app.COMPARE_FIELDS}                  # 67, angina, 3 vessels
ups = [f for f, v, d, n in app.explain(list(sick.items())) if d == 'up']
assert {'age', 'exang', 'ca', 'thalach'} <= set(ups), ups                    # the obvious risk signs are flagged
assert b'Clear history' in c.get('/history').data
other.post('/predict/anova', data=row(anova, 3))                            # olduser makes one prediction
c.post('/history/clear')
count = lambda u: sqlite3.connect(db_path).execute("select count(*) from predictions where user=?", (u,)).fetchone()[0]
assert b'No predictions yet' in c.get('/history').data and count('newuser1') == 0
assert count('olduser') == 1                                                 # ...other users' history is untouched
r = c.get('/insights/chi2').data
assert b'effectively tied' in r and b'Stacking experiment' in r and r.count(b'extra ensemble tried') == len(app.STACKING_EXPERIMENT) == 3
saved_cv = {k: next(x for x in m['results'] if x['name'] == m['saved'])['cv_mean'] for k, m in app.MODELS.items()}
assert all(row[k] <= saved_cv[k] for row in app.STACKING_EXPERIMENT for k in app.MODELS)  # "none beat the saved models" is true
print('model checks passed')
