import sys
import pandas as pd

from collections import namedtuple
from sklearn import preprocessing
from sklearn.mixture import GaussianMixture
from sklearn.metrics import v_measure_score
from sklearn.model_selection import RepeatedStratifiedKFold

# Use cross-validation.
VALIDATING = False

# Find optimal model parameters using grid search.
GRIDSEARCH = False

# Remove or predict missing values.
REMOVE_MISSING = True

if GRIDSEARCH:
    import numpy as np
    Result = namedtuple('Result', 'score params')

FTR_INCOME = 'income'
FTR_INFANT = 'infant'
FTR_REGION = 'region'
FTR_OIL = 'oil'


def handle_missing_values(data):
    if REMOVE_MISSING:
        return data.dropna()
    else:
        imputer = preprocessing.Imputer(strategy='mean', axis=0)
        data[FTR_INFANT] = imputer.fit_transform(data[[FTR_INFANT]]).ravel()
        return data


def encode_data(data):
    encoder = preprocessing.LabelEncoder()
    data[FTR_OIL] = encoder.fit_transform(data[FTR_OIL])
    return data


def normalize(train, test):
    scaler = preprocessing.MinMaxScaler()

    train_scaled = scaler.fit_transform(train)
    test_scaled = scaler.transform(test)

    train = pd.DataFrame(train_scaled)
    test = pd.DataFrame(test_scaled)

    return train, test


def solve(clf, x_train, y_train, x_test, y_test):
    x_train, x_test = normalize(x_train, x_test)

    clf.fit(x_train, y_train)
    y_pred = clf.predict(x_test)

    return v_measure_score(y_test, y_pred)


def grid_search(clf, x, y, cv):
    i = 0
    results = []

    for covariance_type in ['full', 'diag', 'spherical']:
        clf.covariance_type = covariance_type

        for init_params in ['kmeans', 'random']:
            clf.init_params = init_params

            for max_iter in [int(x) for x in np.linspace(100, 100000, 21, endpoint=True)]:
                clf.max_iter = max_iter

                for tol in [1e-1, 1e-2, 1e-3, 1e-4, 1e-5]:
                    clf.tol = tol

                    for reg_covar in [1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7]:
                        clf.reg_covar = reg_covar

                        score = cross_validate(clf, x, y, cv)
                        params = "covariance_type='{}', init_params='{}', max_iter={}, tol={}, reg_covar={}".format(
                            covariance_type, init_params, max_iter, tol, reg_covar)
                        results.append(Result(score, params))

                        i += 1
                        print('iter {}'.format(i))
                        print(params)

    best = max(results, key=lambda result: result.score)
    print('Best:')
    print(best.params)
    print(best.score)


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
    train = handle_missing_values(train)
    train = encode_data(train)

    clf = GaussianMixture(n_components=4, n_init=10, random_state=360,
                          covariance_type='diag', init_params='kmeans',
                          max_iter=100, tol=0.1, reg_covar=0.001)

    x_train = train.drop(FTR_REGION, axis=1).values
    y_train = train[FTR_REGION].values

    rskf = RepeatedStratifiedKFold(n_splits=4, n_repeats=4, random_state=360)

    if GRIDSEARCH:
        grid_search(clf, x_train, y_train, rskf)
        return

    if VALIDATING:
        cross_validate(clf, x_train, y_train, rskf)
    else:
        test = pd.read_csv(sys.argv[2])
        test = encode_data(test)

        x_test = test.drop(FTR_REGION, axis=1).values
        y_test = test[FTR_REGION].values

        score = solve(clf, x_train, y_train, x_test, y_test)
        print(score)


if __name__ == '__main__':
    main()
