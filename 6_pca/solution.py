import sys
import pandas as pd

from sklearn import preprocessing
from sklearn.decomposition import PCA
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import f1_score
from sklearn.model_selection import RepeatedStratifiedKFold, GridSearchCV
from sklearn.pipeline import Pipeline

# Use cross-validation.
VALIDATING = False

# Find optimal model parameters using grid search.
GRIDSEARCH = False

if GRIDSEARCH:
    import numpy as np

# Imputation strategy, use one of the following.
IMPUTE = None
# IMPUTE = 'mean'
# IMPUTE = 'median'

FTR_YEAR = 'year'
FTR_AGE = 'age'
FTR_MARITL = 'maritl'
FTR_RACE = 'race'
FTR_EDU = 'education'
FTR_JOB = 'jobclass'
FTR_HEALTH = 'health'
FTR_INS = 'health_ins'
FTR_WAGE = 'wage'


def handle_missing_values(data):
    if not IMPUTE:
        return data.dropna()
    else:
        # Drop samples with missing label, instead of imputing it:
        data = data.loc[data[FTR_RACE].isin([0.0, 1.0, 2.0, 3.0])]

        imputer = preprocessing.Imputer(strategy=IMPUTE, axis=0)
        imputed = imputer.fit_transform(data)
        return pd.DataFrame(imputed, columns=data.columns)


def encode_data(data):
    data[FTR_MARITL] = data[FTR_MARITL].map(
        {'1. Never Married': 0, '2. Married': 1, '3. Widowed': 2, '4. Divorced': 3, '5. Separated': 4})
    data[FTR_RACE] = data[FTR_RACE].map(
        {'1. White': 0, '2. Black': 1, '3. Asian': 2, '4. Other': 3})
    data[FTR_EDU] = data[FTR_EDU].map(
        {'1. < HS Grad': 0, '2. HS Grad': 1, '3. Some College': 2, '4. College Grad': 3, '5. Advanced Degree': 4})
    data[FTR_JOB] = data[FTR_JOB].map(
        {'1. Industrial': 0, '2. Information': 1})
    data[FTR_HEALTH] = data[FTR_HEALTH].map(
        {'1. <=Good': 0, '2. >=Very Good': 1})
    data[FTR_INS] = data[FTR_INS].map(
        {'1. Yes': 0, '2. No': 1})

    return data


def solve(clf, x_train, y_train, x_test, y_test):
    clf.fit(x_train, y_train)
    y_pred = clf.predict(x_test)

    return f1_score(y_test, y_pred, average='micro')


def grid_search(clf, x, y, cv):
    params = {'reduce_dim__n_components': [5, 6, 7],
              'classify__max_depth': [3, 4, 5, 6, 7, 8, 9],
              'classify__min_samples_split': np.linspace(1e-4, 1e-1, 4),
              'classify__min_samples_leaf': np.linspace(1e-4, 1e-1, 4),
              'classify__subsample': np.linspace(0.6, 1.0, 5),
              'classify__max_features': np.linspace(0.1, 1.0, 5)}

    gs = GridSearchCV(estimator=clf, param_grid=params,
                      scoring='f1_micro', cv=cv, n_jobs=2, verbose=1)
    gs.fit(x, y)

    print(gs.best_params_)
    print(gs.best_score_)


def cross_validate(clf, x, y, cv):
    i = 0
    mean_score = 0

    for train_index, test_index in cv.split(x, y):
        i += 1
        x_train, y_train = x[train_index], y[train_index]
        x_test, y_test = x[test_index], y[test_index]
        score = solve(clf, x_train, y_train, x_test, y_test)

        if not GRIDSEARCH:
            print('(iter {0}) score={1}'.format(i, score))
        mean_score += score

    mean_score /= i
    print(mean_score)
    return mean_score


def main():
    train = pd.read_csv(sys.argv[1])
    train = encode_data(train)
    train = handle_missing_values(train)

    x_train = train.drop(FTR_RACE, axis=1).values
    y_train = train[FTR_RACE].values

    scaler = preprocessing.StandardScaler()
    pca = PCA(
        random_state=360, svd_solver='full', whiten=True, n_components=6)
    clf = GradientBoostingClassifier(
        random_state=360, n_estimators=200, warm_start=True,
        max_depth=2, max_features=0.1, min_samples_leaf=0.066,
        min_samples_split=0.0001, subsample=0.6)

    steps = [
        ('normalize', scaler),
        ('reduce_dim', pca),
        ('classify', clf)]

    pipe = Pipeline(steps)

    rskf = RepeatedStratifiedKFold(n_splits=4, n_repeats=1, random_state=360)
    split = rskf.split(x_train, y_train)

    if GRIDSEARCH:
        grid_search(pipe, x_train, y_train, split)
        return

    if VALIDATING:
        cross_validate(pipe, x_train, y_train, rskf)
    else:
        test = pd.read_csv(sys.argv[2])
        test = encode_data(test)

        x_test = test.drop(FTR_RACE, axis=1).values
        y_test = test[FTR_RACE].values

        score = solve(pipe, x_train, y_train, x_test, y_test)
        print(score)


if __name__ == '__main__':
    main()
