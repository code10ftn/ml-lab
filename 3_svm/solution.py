import sys
import pandas as pd

from sklearn.preprocessing import StandardScaler, MinMaxScaler, Imputer
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.feature_selection import SelectFromModel
from sklearn.svm import LinearSVC, SVC
from sklearn.metrics import accuracy_score

# Visualize data and exit program.
VISUALIZE = False

# Use cross-validation.
VALIDATING = False

# Remove or predict missing values.
REMOVE_MISSING = True

# Use z-score or min-max normalization.
Z_SCORE = False

# Use additional model for feature selection.
SELECT_FEATURES = True


if VISUALIZE:
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D

FTR_HUMID = 'humid'
FTR_TEMP = 'temp'
FTR_TREES = 'trees'
FTR_XMIN = 'Xmin'
FTR_XMAX = 'Xmax'
FTR_YMIN = 'Ymin'
FTR_YMAX = 'Ymax'
FTR_NOYES = 'NoYes'


def visualize(data, feature1, feature2, feature3=None):
    no = data[data[FTR_NOYES] == 0]
    yes = data[data[FTR_NOYES] == 1]

    if feature3 is None:
        plt.plot(no[feature1], no[feature2], 'b')
        plt.plot(yes[feature1], yes[feature2], 'r')
    else:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        ax.scatter(no[feature1], no[feature2], no[feature3],
                   zdir='z', s=20, c='b', depthshade=True)
        ax.scatter(yes[feature1], yes[feature2], yes[feature3],
                   zdir='z', s=20, c='r', depthshade=True)

    plt.show()


def handle_missing_values(data):
    if REMOVE_MISSING:
        return data.dropna()
    else:
        imputer = Imputer(strategy='mean', axis=0)
        imputed = imputer.fit_transform(data)
        return pd.DataFrame(imputed, columns=data.columns)


def normalize(train, test):
    scaler = StandardScaler() if Z_SCORE else MinMaxScaler()

    train_scaled = scaler.fit_transform(train)
    test_scaled = scaler.transform(test)

    train = pd.DataFrame(train_scaled)
    test = pd.DataFrame(test_scaled)

    return train, test


def select_features(x_train, y_train, x_test):
    selector = LinearSVC(C=10, penalty='l1',
                         dual=False, random_state=360).fit(x_train, y_train)
    selector = SelectFromModel(selector, prefit=True)
    x_train = selector.transform(x_train)
    x_test = selector.transform(x_test)

    return x_train, y_train, x_test


def solve(clf, x_train, y_train, x_test, y_test):
    x_train, x_test = normalize(x_train, x_test)

    if SELECT_FEATURES:
        x_train, y_train, x_test = select_features(x_train, y_train, x_test)

    clf.fit(x_train, y_train)
    y_pred = clf.predict(x_test)

    return accuracy_score(y_test, y_pred)


def cross_validate(clf, x, y):
    rskf = RepeatedStratifiedKFold(n_splits=4, n_repeats=5, random_state=360)

    i = 0
    mean_prec = 0

    for train_index, test_index in rskf.split(x, y):
        i += 1
        x_train, y_train = x[train_index], y[train_index]
        x_test, y_test = x[test_index], y[test_index]
        prec = solve(clf, x_train, y_train, x_test, y_test)

        print('(iter {0}) prec={1}'.format(i, prec))
        mean_prec += prec

    print(mean_prec / i)


def main():
    train = pd.read_csv(sys.argv[1])
    train = handle_missing_values(train)

    clf = SVC(C=72, gamma=9.3, random_state=360)

    if VISUALIZE:
        visualize(train, FTR_HUMID, FTR_TEMP, FTR_TREES)
        return

    x_train = train.drop(FTR_NOYES, axis=1).values
    y_train = train[FTR_NOYES].values

    if VALIDATING:
        cross_validate(clf, x_train, y_train)
    else:
        test = pd.read_csv(sys.argv[2])

        x_test = test.drop(FTR_NOYES, axis=1).values
        y_test = test[FTR_NOYES].values

        precision = solve(clf, x_train, y_train, x_test, y_test)
        print(precision)


if __name__ == '__main__':
    main()
