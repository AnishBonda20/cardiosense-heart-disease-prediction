"""Retrain the three models from archive/heart.csv.

Run:  python train.py        (needs requirements-train.txt)

Same idea as the notebooks (three feature-selection methods, ten classifiers,
a stacked ensemble), with three fixes:
  1. The test set is split off FIRST. SMOTEENN and feature selection only ever
     see training data, so the test score is honest (the notebooks resampled
     before splitting, which leaked test patients into training).
  2. AUC is measured on the test set, not the training set.
  3. For each method, the classifier with the best cross-validated accuracy is
     the one saved for the app. Choosing by cross-validation (training data only)
     keeps the test set untouched for the final, honest score.

Writes models/model_anova.sav, model_chi.sav, model_mi.sav and
models/metrics.json, which the app reads for its forms and charts.
"""
import csv
import json
import warnings
from functools import partial

import joblib
import numpy as np
import sklearn
import imblearn
from imblearn import FunctionSampler
from imblearn.combine import SMOTEENN
from imblearn.over_sampling import SMOTENC
from imblearn.pipeline import Pipeline
from imblearn.under_sampling import EditedNearestNeighbours
from lightgbm import LGBMClassifier
from sklearn.ensemble import (AdaBoostClassifier, BaggingClassifier, ExtraTreesClassifier,
                              RandomForestClassifier, StackingClassifier, VotingClassifier)
from sklearn.feature_selection import SelectKBest, chi2, f_classif, mutual_info_classif
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score, precision_score,
                             recall_score, roc_auc_score, roc_curve)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

warnings.filterwarnings('ignore')
SEED = 42
FEATURES = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach',
            'exang', 'oldpeak', 'slope', 'ca', 'thal']
# method key: (model file, scoring function, how many features to keep) — same k as the notebooks
METHODS = {
    'anova': ('models/model_anova.sav', f_classif, 8),
    'chi2': ('models/model_chi.sav', chi2, 11),
    'mi': ('models/model_mi.sav', partial(mutual_info_classif, random_state=SEED), 10),
}
STACKING = 'Stacking'
CATEGORICAL = [FEATURES.index(f) for f in ('sex', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'ca', 'thal')]


def _smoteenn(X, y, counts=None):
    """SMOTEENN tuned for this data, done as its two steps so we can report what each one did.
    Features are scaled first so cholesterol (hundreds) doesn't drown out yes/no features in the
    distance checks; SMOTENC keeps categories as real codes instead of inventing values like
    cp = 1.37; ENN 'mode' only removes a patient when most of its neighbours disagree.
    The default SMOTEENN threw away most of the training set.
    Pass a dict as `counts` to get the class sizes before, after SMOTE and after ENN."""
    scaler = StandardScaler().fit(X)
    X_s, y_s = SMOTENC(categorical_features=CATEGORICAL, random_state=SEED).fit_resample(scaler.transform(X), y)
    enn = EditedNearestNeighbours(kind_sel='mode')
    X_e, y_e = enn.fit_resample(X_s, y_s)
    if counts is not None:
        counts.update({stage: np.bincount(labels, minlength=2).tolist()
                       for stage, labels in (('before', y), ('after_smote', y_s), ('after_enn', y_e))})
        kept = set(enn.sample_indices_.tolist())  # SMOTENC appends synthetic rows after the len(y) real ones
        counts.update(synthetic_added=len(y_s) - len(y),
                      synthetic_kept=sum(i >= len(y) for i in kept),
                      real_removed=sum(i not in kept for i in range(len(y))))
    X_res = scaler.inverse_transform(X_e)
    X_res[:, CATEGORICAL] = np.round(X_res[:, CATEGORICAL])
    return np.clip(X_res, 0, None), y_e  # clip guards chi2, which needs non-negative inputs


def resampler():
    return FunctionSampler(func=_smoteenn, validate=False)


# For the comparison chart: the saved (best) model trained three ways.
RESAMPLING_VARIANTS = {
    'No resampling': lambda: None,
    'SMOTEENN (default)': lambda: SMOTEENN(random_state=SEED),
    'SMOTEENN (tuned, used)': resampler,
}


def classifiers():
    """The ten classifiers from the notebooks plus the stacked model. Fresh, unfitted copies each call.
    SVM, KNN and logistic regression get a scaler: they compare raw distances, and unscaled
    cholesterol (hundreds) would drown out yes/no features (0 or 1)."""
    base = {
        'Naive Bayes': GaussianNB(),
        'SVM': make_pipeline(StandardScaler(), SVC(probability=True, random_state=SEED)),
        'XGBoost': XGBClassifier(eval_metric='logloss', random_state=SEED, verbosity=0),
        'AdaBoost': AdaBoostClassifier(n_estimators=100, random_state=SEED),
        'Bagging': BaggingClassifier(AdaBoostClassifier(n_estimators=100, random_state=SEED),
                                     n_estimators=10, random_state=SEED),
        'Decision Tree': DecisionTreeClassifier(random_state=SEED),
        'KNN': make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=3)),
        'Random Forest': RandomForestClassifier(random_state=SEED),
        'Logistic Regression': make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000)),
    }
    base['Voting'] = VotingClassifier([(n.replace(' ', '_'), sklearn.base.clone(c)) for n, c in base.items()], voting='soft')
    base[STACKING] = StackingClassifier(
        estimators=[('boosted_dt', AdaBoostClassifier(DecisionTreeClassifier(max_depth=1), n_estimators=200, random_state=SEED)),
                    ('extra_trees', ExtraTreesClassifier(random_state=SEED))],
        final_estimator=LGBMClassifier(n_estimators=10, random_state=SEED, verbose=-1))
    return base


def load():
    with open('archive/heart.csv', encoding='utf-8-sig') as fh:
        rows = [tuple(float(v) for v in r.values()) for r in csv.DictReader(fh)]
    unique = list(dict.fromkeys(rows))  # drop exact duplicate patients, keep the first copy
    data = np.array(unique)
    return len(rows), data[:, :-1], data[:, -1].astype(int)


def main():
    n_raw, X, y = load()
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, stratify=y, random_state=SEED)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    smoteenn_counts = {}
    _smoteenn(X_tr, y_tr, smoteenn_counts)  # same training set for every method, so one summary covers all three
    report = dict(meta=dict(rows_raw=n_raw, rows=len(y), train=len(y_tr), test=len(y_te), seed=SEED,
                            class_counts=[int((y == 0).sum()), int((y == 1).sum())],
                            smoteenn=smoteenn_counts,
                            versions={'scikit-learn': sklearn.__version__, 'imbalanced-learn': imblearn.__version__}),
                  methods={})

    for key, (path, score_fn, k) in METHODS.items():
        # 1. clean up the training set: SMOTE adds synthetic patients, ENN then removes ones that sit among the other class
        X_res, y_res = resampler().fit_resample(X_tr, y_tr)
        # 2. pick the k best features using the resampled training data only
        selector = SelectKBest(score_fn, k=k).fit(X_res, y_res)
        keep = selector.get_support()
        Xs_res, Xs_te = X_res[:, keep], X_te[:, keep]

        # 3. train and score all eleven classifiers
        results, roc, fitted = [], {}, {}
        for name, clf in classifiers().items():
            clf.fit(Xs_res, y_res)
            fitted[name] = clf
            pred, prob = clf.predict(Xs_te), clf.predict_proba(Xs_te)[:, 1]
            # cross-validation repeats resampling + selection inside every fold, so no fold sees its own test rows
            cv_pipe = Pipeline([('resample', resampler()), ('select', SelectKBest(score_fn, k=k)),
                                ('model', classifiers()[name])])
            cv_acc = cross_val_score(cv_pipe, X_tr, y_tr, cv=cv, scoring='accuracy')
            results.append(dict(name=name, accuracy=accuracy_score(y_te, pred), precision=precision_score(y_te, pred),
                                recall=recall_score(y_te, pred), f1=f1_score(y_te, pred), auc=roc_auc_score(y_te, prob),
                                cv_mean=cv_acc.mean(), cv_std=cv_acc.std()))
            fpr, tpr, _ = roc_curve(y_te, prob)
            roc[name] = [[round(a, 4), round(b, 4)] for a, b in zip(fpr, tpr)]

        # 4. keep the best by cross-validated accuracy (ties: steadier, i.e. smaller spread across folds)
        best = max(results, key=lambda r: (round(r['cv_mean'], 3), -r['cv_std']))
        joblib.dump(fitted[best['name']], path)
        matrix = confusion_matrix(y_te, fitted[best['name']].predict(Xs_te)).tolist()

        # 5. does resampling help? cross-validate the saved model with each resampling choice
        ablation = []
        for label, make in RESAMPLING_VARIANTS.items():
            steps = [('resample', make())] if make() is not None else []
            pipe = Pipeline(steps + [('select', SelectKBest(score_fn, k=k)), ('model', classifiers()[best['name']])])
            acc = cross_val_score(pipe, X_tr, y_tr, cv=cv, scoring='accuracy')
            rows = len(make().fit_resample(X_tr, y_tr)[1]) if make() is not None else len(y_tr)
            ablation.append(dict(name=label, cv_mean=round(float(acc.mean()), 4), cv_std=round(float(acc.std()), 4), rows=rows))

        scores = np.nan_to_num(selector.scores_)
        report['methods'][key] = dict(
            k=k, features=[f for f, s in zip(FEATURES, keep) if s],
            scores={f: round(float(s), 4) for f, s in zip(FEATURES, scores)},
            results=[{k2: (round(float(v), 4) if k2 != 'name' else v) for k2, v in r.items()} for r in results],
            saved=best['name'], confusion=matrix, roc=roc, resampling=ablation)
        print(f"{key:6} features={report['methods'][key]['features']}")
        print(f"       saved {best['name']}: cv {best['cv_mean']:.3f} ± {best['cv_std']:.3f}, "
              f"test acc {best['accuracy']:.3f}, auc {best['auc']:.3f}, recall {best['recall']:.3f}")
        print('       resampling:', ', '.join(f"{a['name']} {a['cv_mean']:.3f} ({a['rows']} rows)" for a in ablation))

    print('SMOTEENN on the training set:', smoteenn_counts)
    with open('models/metrics.json', 'w') as fh:
        json.dump(report, fh, indent=1)
    print('wrote models/*.sav and models/metrics.json')


if __name__ == '__main__':
    main()
