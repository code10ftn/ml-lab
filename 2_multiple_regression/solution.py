import sys
import pandas as pd
import numpy as np

from functools import partial

# Use cross-validation.
VALIDATING = False

# Visualize data and exit program.
VISUALIZE = False

# Use z-score or min-max normalization.
Z_SCORE = True

# Use label encoding or one-hot encoding.
LABEL_ENCODING = False  # one-hot if False

if VISUALIZE:
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D

N = 5 if LABEL_ENCODING else 9  # Number of independent features.

FTR_RANK = 'rank'
FTR_DISCIPLINE = 'discipline'
FTR_PHD = 'yrs.since.phd'
FTR_SERVICE = 'yrs.service'
FTR_SEX = 'sex'
FTR_SALARY = 'salary'


class KernelRegression:
    def __init__(self):
        self._lambda = 0.25  # gaussian = 0.05

    def fit(self, x, y):
        self.x = x
        self.y = y

    def predict(self, x):
        return np.array(list(map(self._predict, x)))

    def _predict(self, x):
        partial_kernel = partial(self._kernel, x)
        pairs = list(zip(self.x, self.y))

        sum1 = sum(map(lambda pair: partial_kernel(pair[0]) * pair[1], pairs))
        sum2 = sum(map(lambda pair: partial_kernel(pair[0]), pairs))
        return sum1 / sum2

    def _kernel(self, x, y):
        return self._gaussian(euclid_distance(x, y)/self._lambda)

    def _epanechnikov(self, t):
        if np.abs(t) < 1:
            return 3.0 / 4.0 * (1 - t * t)
        else:
            return 0

    def _gaussian(self, t):
        return 1/np.sqrt(2 * np.pi) * np.power(np.e, -1/2 * t ** 2)


class KNN:
    def __init__(self, k):
        self.k = k

    def fit(self, x, y):
        self.x = x
        self.y = y

    def predict(self, x):
        return np.array(list(map(self._predict, x)))

    def _predict(self, x):
        dist_knn = map(partial(euclid_distance, x), self.x[0:self.k])
        y_knn = self.y[0:self.k]
        sorted_pair = sorted(zip(dist_knn, y_knn))

        dist_knn = [x for x, _ in sorted_pair]
        y_knn = [y for _, y in sorted_pair]

        for x_index, element in enumerate(self.x[self.k-1:]):
            distance = euclid_distance(x, element)
            if distance < dist_knn[self.k-1]:
                j = 0
                for index in range(len(dist_knn)):
                    if distance > dist_knn[index-1] and distance <= dist_knn[index]:
                        j = index
                        break

                y_knn[j+1:self.k] = y_knn[j:self.k-1]
                dist_knn[j+1:self.k] = dist_knn[j:self.k-1]

                dist_knn[j] = distance
                y_knn[j] = self.y[x_index+self.k-1]

        if dist_knn[0] == 0.0:
            return y_knn[0]

        pairs = list(zip(dist_knn, y_knn))

        sum1 = sum(map(lambda pair: (1/pair[0] ** 2)*pair[1], pairs))
        sum2 = sum(map(lambda pair: (1/pair[0] ** 2), pairs))

        return sum1/sum2


def euclid_distance(a, b):
    return np.linalg.norm(a-b)


def calculate_rmse(y_true, y_predict):
    return np.sqrt(((y_predict - y_true) ** 2).mean())


class LinearRegression:
    def __init__(self, learning_rate=0.05, regularization_rate=0.0011, max_iters=10000, max_err=0.001, degree=2):
        self._alpha = learning_rate
        self._lambda = regularization_rate
        self._lambda2 = regularization_rate
        self._max_iters = max_iters
        self._max_err = max_err
        self._degree = degree

    def fit(self, x, y):
        return self.fit_gradient_descent(x, y)

    def fit_gradient_descent(self, x, y):
        self._theta = np.zeros(1 + N + int(N * (N + 1) / 2)).tolist()
        x_t = np.transpose(x)

        for _ in range(1, self._max_iters):
            y_predict = self.predict(x)
            mse = self._calculate_mse(y, y_predict)
            if mse < self._max_err:
                return

            d_t0 = (y_predict - y).mean()
            self._theta[0] -= self._alpha * d_t0

            for i in range(0, N):
                d_ti = ((y_predict - y) * x_t[i]).mean()
                if self._theta[i + 1] < 0:
                    lasso = -self._lambda2
                elif self._theta[i + 1] > 0:
                    lasso = self._lambda2
                else:
                    lasso = 0
                self._theta[i + 1] = self._theta[i + 1] * \
                    (1 + self._alpha * self._lambda) - self._alpha * d_ti
                self._theta[i + 1] -= lasso

            count = N + 1
            for i in range(0, N):
                for j in range(i, N):
                    d_ti = ((y_predict - y) * x_t[i] * x_t[j]).mean()

                    if self._theta[count] < 0:
                        lasso = -self._lambda2
                    elif self._theta[count] > 0:
                        lasso = self._lambda2
                    else:
                        lasso = 0
                    self._theta[count] = self._theta[count] * \
                        (1 + self._alpha * self._lambda) - self._alpha * d_ti
                    self._theta[count] -= lasso
                    count += 1

    def fit_closed_form(self, x, y):
        ones = np.ones(len(x))
        x = np.column_stack((ones, x))
        x_t = np.transpose(x)
        self._theta = np.dot(np.dot(np.linalg.inv(np.dot(x_t, x)), x_t), y)

    def predict(self, x):
        # Initialize prediction for each row (sample).
        predictions = [self._theta[0]] * len(x)
        # Convert rows to columns (per feature).
        x = np.transpose(x)

        for i in range(0, N):
            predictions += self._theta[i + 1] * x[i]

        count = N + 1
        for i in range(0, N):
            for j in range(i, N):
                predictions += self._theta[count] * x[i] * x[j]
                count += 1

        return predictions

    def _calculate_mse(self, y_true, y_predict):
        n = len(y_true)
        return np.sum((y_predict - y_true) ** 2) / (2 * n)


def normalize_data(data):
    if Z_SCORE:
        return normalize_z_score(data)
    else:
        return normalize_min_max(data)


def normalize_z_score(data):
    """Normalizes data around 0."""
    mean = data.mean()
    std = data.std()
    data = (data - mean) / std
    return data, mean, std


def normalize_min_max(data):
    """Normalizes data between 0 and 1."""
    min_val = data.min()
    max_val = data.max()
    data = (data - min_val) / (max_val - min_val)
    return data, min_val, max_val


def normalize(data, min_val, max_val):
    if Z_SCORE:
        return (data - min_val) / max_val
    else:
        return (data - min_val) / (max_val - min_val)


def denormalize(data, min_val, max_val, y_row_index):
    max_val = max_val[y_row_index]
    min_val = min_val[y_row_index]
    if Z_SCORE:
        return data * max_val + min_val
    else:
        return data * (max_val - min_val) + min_val


def encode_data(data):
    if LABEL_ENCODING:
        return label_encoding(data)
    else:
        return one_hot_encoding(data)


def label_encoding(data):
    data[FTR_RANK] = data[FTR_RANK].map(
        {'Prof': 0, 'AssocProf': 1, 'AsstProf': 2})
    data[FTR_DISCIPLINE] = data[FTR_DISCIPLINE].map(
        {'A': 0, 'B': 1})
    data[FTR_SEX] = data[FTR_SEX].map(
        {'Female': 0, 'Male': 1})
    return data


def one_hot_encoding(data):
    data = data.assign(prof=pd.Series(data[FTR_RANK].map(
        {'Prof': 1, 'AssocProf': 0, 'AsstProf': 0})).values)
    data = data.assign(assocprof=pd.Series(data[FTR_RANK].map(
        {'Prof': 0, 'AssocProf': 1, 'AsstProf': 0})).values)
    data = data.assign(asstprof=pd.Series(data[FTR_RANK].map(
        {'Prof': 0, 'AssocProf': 0, 'AsstProf': 1})).values)
    data = data.drop(FTR_RANK, axis=1)

    data = data.assign(a=pd.Series(data[FTR_DISCIPLINE].map(
        {'A': 1, 'B': 0})).values)
    data = data.assign(b=pd.Series(data[FTR_DISCIPLINE].map(
        {'A': 0, 'B': 1})).values)
    data = data.drop(FTR_DISCIPLINE, axis=1)

    data = data.assign(female=pd.Series(data[FTR_SEX].map(
        {'Female': 1, 'Male': 0})).values)
    data = data.assign(male=pd.Series(data[FTR_SEX].map(
        {'Female': 0, 'Male': 1})).values)
    data = data.drop(FTR_SEX, axis=1)

    return data


def cross_validate(model, train, test=None, k=5):
    """Calculates mean error using k-fold cross-validation."""
    train = train.sample(frac=1, random_state=360)  # Shuffles rows.
    k = k if test is None else 1
    chunks = np.array_split(train, k)
    mean_rmse = 0

    for i in range(0, k):
        if VALIDATING:
            # Use i-th chunk for testing and the rest k-1 chunks for training.
            train = pd.concat([chunks[c] for c in range(0, k) if c != i])
            test = chunks[i]

        train, norm_min, norm_max = normalize_data(train)
        test = normalize(test, norm_min, norm_max)

        model.fit(
            train.drop(FTR_SALARY, axis=1).values, train[FTR_SALARY].values)

        y_predict = model.predict(test.drop(FTR_SALARY, axis=1).values)
        y_true = test[FTR_SALARY].values

        # Denormalize data to get real error.
        y_row_index = train.columns.get_loc(FTR_SALARY)
        y_predict = denormalize(y_predict, norm_min, norm_max, y_row_index)
        y_true = denormalize(y_true, norm_min, norm_max, y_row_index)

        rmse = calculate_rmse(y_true, y_predict)
        if VALIDATING:
            print('(iter {0}) rmse={1}'.format(i + 1, rmse))
        mean_rmse += rmse

    mean_rmse /= k
    print(mean_rmse)
    return mean_rmse


def visualize(data, feature1, feature2, feature3=FTR_SALARY):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(data[feature1], data[feature2], data[feature3],
               zdir='z', s=20, c=None, depthshade=True)
    plt.show()


def main():
    model = LinearRegression()
    # model = KNN(10)
    # model = KernelRegression()
    train = pd.read_csv(sys.argv[1])
    train = encode_data(train)

    if VISUALIZE:
        visualize(train, FTR_PHD, FTR_SERVICE)
        return

    if VALIDATING:
        cross_validate(model, train)
    else:
        test = pd.read_csv(sys.argv[2])
        test = encode_data(test)
        cross_validate(model, train, test)


if __name__ == '__main__':
    main()
