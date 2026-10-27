"""Can a stronger stacking ensemble beat the saved models? Run:  python experiment_stacking.py

Tries three stacking ensembles on ALL three feature-selection methods, with exactly the same
5-fold cross-validation as train.py (training data only; the test set is never touched).
Writes models/stacking_experiment.json, which the Insights page shows. Takes about a minute.
"""
import json
import warnings

from imblearn.pipeline import Pipeline
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier, StackingClassifier
from sklearn.feature_selection import SelectKBest
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from train import METHODS, SEED, load, resampler

warnings.filterwarnings('ignore')


def scaled(model):
    return make_pipeline(StandardScaler(), model)


def lr():
    return LogisticRegression(max_iter=1000)


# Each candidate: several different base models, combined by a logistic regression on their probabilities.
CANDIDATES = {
    'Naive Bayes + KNN + Logistic Regression': lambda: StackingClassifier(
        [('nb', GaussianNB()), ('knn', scaled(KNeighborsClassifier(5))), ('lr', scaled(lr()))],
        final_estimator=lr(), cv=5),
    'Naive Bayes + KNN + LR + SVM + Random Forest': lambda: StackingClassifier(
        [('nb', GaussianNB()), ('knn', scaled(KNeighborsClassifier(5))), ('lr', scaled(lr())),
         ('svm', scaled(SVC(probability=True, random_state=SEED))), ('rf', RandomForestClassifier(random_state=SEED))],
        final_estimator=lr(), cv=5),
    'Naive Bayes + LR + Random Forest + Extra Trees': lambda: StackingClassifier(
        [('nb', GaussianNB()), ('lr', scaled(lr())),
         ('rf', RandomForestClassifier(random_state=SEED)), ('et', ExtraTreesClassifier(random_state=SEED))],
        final_estimator=lr(), cv=5),
}


def main():
    _, X, y = load()
    X_tr, _, y_tr, _ = train_test_split(X, y, test_size=0.2, stratify=y, random_state=SEED)  # same split as train.py
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    results = []
    for name, make in CANDIDATES.items():
        row = {'name': name}
        for key, (_, score_fn, k) in METHODS.items():
            pipe = Pipeline([('resample', resampler()), ('select', SelectKBest(score_fn, k=k)), ('model', make())])
            row[key] = round(float(cross_val_score(pipe, X_tr, y_tr, cv=cv, scoring='accuracy').mean()), 4)
        results.append(row)
        print(name, {k: v for k, v in row.items() if k != 'name'}, flush=True)
    with open('models/stacking_experiment.json', 'w') as fh:
        json.dump(results, fh, indent=1)
    print('wrote models/stacking_experiment.json')


if __name__ == '__main__':
    main()
