import sys
import warnings
import pandas as pd

from sklearn import preprocessing
from sklearn import ensemble
from sklearn.metrics import f1_score
from sklearn.model_selection import RepeatedStratifiedKFold

# Use cross-validation.
VALIDATING = False

FTR_DVCAT = 'dvcat'
FTR_WEIGHT = 'weight'
FTR_DEAD = 'dead'
FTR_AIRBAG = 'airbag'
FTR_SEATBELT = 'seatbelt'
FTR_FRONTAL = 'frontal'
FTR_SEX = 'sex'
FTR_AGEOFOCC = 'ageOFocc'
FTR_YEARACC = 'yearacc'
FTR_YEARVEH = 'yearVeh'
FTR_ABCAT = 'abcat'
FTR_OCCROLE = 'occRole'
FTR_DEPLOY = 'deploy'
FTR_INJSEVERITY = 'injSeverity'

CATEGORICAL_COLUMNS = [FTR_DVCAT, FTR_DEAD, FTR_AIRBAG,
                       FTR_SEATBELT, FTR_SEX, FTR_ABCAT, FTR_OCCROLE]


def handle_missing_values(data):
    return data.dropna()


def preprocess(data):
    data = encode_data(data)
    data = select_features(data)
    data = sort_injury_severity(data)

    return data


def encode_data(data):
    encoder = preprocessing.LabelEncoder()
    for category in CATEGORICAL_COLUMNS:
        data[category] = encoder.fit_transform(data[category])
    return data


def select_features(data):
    data = data.drop(FTR_DEAD, axis=1)  # Already in injSeverity.
    data = data.drop(FTR_AIRBAG, axis=1)  # Already in abcat.
    data = data.drop(FTR_DEPLOY, axis=1)  # Already in abcat.

    return data


def sort_injury_severity(data):
    data[FTR_INJSEVERITY] = data[FTR_INJSEVERITY].map(
        {0.0: 0.0, 1.0: 1.0, 2.0: 2.0, 3.0: 3.0, 4.0: 5.0, 5.0: -1.0, 6.0: 4.0})
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

    return f1_score(y_test, y_pred, average='micro')


def cross_validate(clf, x, y):
    rskf = RepeatedStratifiedKFold(n_splits=4, n_repeats=1, random_state=360)

    i = 0
    mean_score = 0

    for train_index, test_index in rskf.split(x, y):
        i += 1
        x_train, y_train = x[train_index], y[train_index]
        x_test, y_test = x[test_index], y[test_index]
        score = solve(clf, x_train, y_train, x_test, y_test)

        print('(iter {0}) score={1}'.format(i, score))
        mean_score += score

    print(mean_score / i)


def main():
    train = pd.read_csv(sys.argv[1])
    train = handle_missing_values(train)
    train = preprocess(train)

    # All parameters were optimized using grid search with cross-validation (GridSearchCV).

    rfc = ensemble.RandomForestClassifier(
        n_estimators=100, max_depth=9, max_features=3, min_samples_leaf=1e-5, min_samples_split=1e-5, criterion='entropy', random_state=360)

    gbc = ensemble.GradientBoostingClassifier(
        n_estimators=275, max_depth=6, max_features=9, min_samples_leaf=1e-9, min_samples_split=1e-9, subsample=0.9, random_state=360)

    clf = ensemble.VotingClassifier(
        estimators=[('rfc', rfc), ('gbc', gbc)], weights=[3.5, 6.5], voting='soft')

    x_train = train.drop(FTR_DVCAT, axis=1).values
    y_train = train[FTR_DVCAT].values

    if VALIDATING:
        cross_validate(clf, x_train, y_train)
    else:
        test = pd.read_csv(sys.argv[2])
        test = preprocess(test)

        x_test = test.drop(FTR_DVCAT, axis=1).values
        y_test = test[FTR_DVCAT].values

        score = solve(clf, x_train, y_train, x_test, y_test)
        print(score)


if __name__ == '__main__':
    warnings.simplefilter('ignore')
    main()
