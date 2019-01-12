import sys
import pandas as pd
import numpy as np

# Visualize data and exit program.
VISUALIZE = False

if VISUALIZE:
    import matplotlib.pyplot as plt


class SimpleLinearRegression:
    def __init__(self, start_point=[0, 0], learning_rate=0.1, max_iters=10000, max_err=0.01):
        self._start_point = start_point
        self._learning_rate = learning_rate
        self._max_iters = max_iters
        self._max_err = max_err
        self._theta = start_point

    def fit_gradient_descent(self, x, y):
        for _ in range(1, self._max_iters):
            y_predict = self.predict(x)
            mse = self._calculate_mse(y, y_predict)
            if mse < self._max_err:
                return

            d_t0 = (y_predict - y).mean()
            d_t1 = ((y_predict - y) * x).mean()
            self._theta[0] -= self._learning_rate * d_t0
            self._theta[1] -= self._learning_rate * d_t1

    def fit_closed_form(self, x, y):
        ones = np.ones(len(x))
        x = np.column_stack((ones, x))
        x_t = np.transpose(x)
        self._theta = np.dot(np.dot(np.linalg.inv(np.dot(x_t, x)), x_t), y)

    def predict(self, x):
        return x * self._theta[1] + self._theta[0]

    def calculate_rmse(self, y_true, y_predict):
        return np.sqrt(((y_predict - y_true) ** 2).mean())

    def _calculate_mse(self, y_true, y_predict):
        n = len(y_true)
        return np.sum((y_predict - y_true) ** 2) / (2 * n)


def reject_outliers_ratio(data, m=2):
    ratio = data['size'] / data['weight']
    return data[abs(ratio - np.mean(ratio)) < m * np.std(ratio)]


def split_dataset(dataset, ratio=0.8):
    """Random train-test split."""
    train = dataset.sample(frac=ratio, random_state=360)
    test = dataset.drop(train.index)
    return train, test


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
    return (data - min_val) / max_val


def denormalize(data, min_val, max_val):
    return data * max_val + min_val


def main():
    # dataset = pd.read_csv(sys.argv[1])
    # train, test = split_dataset(dataset, 0.7)
    train = pd.read_csv(sys.argv[1])
    test = pd.read_csv(sys.argv[2])
    train = reject_outliers_ratio(train, 2)

    # Normalize data and save params for prediction:
    train, norm_min, norm_max = normalize_z_score(train)
    test = normalize(test, norm_min, norm_max)

    model = SimpleLinearRegression()
    model.fit_closed_form(train['size'].values, train['weight'].values)
    # model.fit_gradient_descent(train['size'].values, train['weight'].values)

    y_predict = model.predict(test['size'].values)
    y_true = test['weight'].values

    # Denormalize data to get real error:
    y_predict = denormalize(y_predict, norm_min[1], norm_max[1])
    y_true = denormalize(y_true, norm_min[1], norm_max[1])
    print(model.calculate_rmse(y_true, y_predict))

    if VISUALIZE:
        x = np.array([-5, 5])
        y = model.predict(x)
        plt.plot(x, y)
        plt.plot(train['size'], train['weight'], 'bo')
        plt.plot(test['size'], test['weight'], 'rx')
        plt.show()


if __name__ == '__main__':
    main()
