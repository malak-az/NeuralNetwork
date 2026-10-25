
import numpy as np


class NN:
    def __init__(self, lr=0.01, epochs=1000, bias=True, threshold=0.1):
        self.lr = lr
        self.epochs = epochs
        self.bias = bias
        self.threshold = threshold
        self.all_mse = []

    def perc(self, X, Y):
        n, features = X.shape
        self.w = np.random.uniform(-0.01, 0.01, features)
        self.b = np.random.uniform(-0.01, 0.01) if self.bias else 0

        for epoch in range(self.epochs):
            errors = 0

            for i in range(n):
                x = X[i]
                y = Y[i]

                pred = 1 if np.dot(x, self.w) + self.b >= 0 else -1
                error = y - pred

                if error != 0:
                    self.w += self.lr * error * x

                    if self.bias:
                        self.b += self.lr * error

                    errors += 1

            if errors == 0:
                break

        return self

    def adaline(self, X, Y):
        n, features = X.shape
        self.w = np.random.uniform(-0.01, 0.01, features)
        self.b = np.random.uniform(-0.01, 0.01) if self.bias else 0
        self.all_mse = []

        for epoch in range(self.epochs):
            for i in range(n):
                x = X[i]
                y = Y[i]

                pred = np.dot(x, self.w) + self.b
                error = y - pred

                self.w += self.lr * error * x

                if self.bias:
                    self.b += self.lr * error

            pred = X @ self.w + self.b
            mse = np.mean((Y - pred) ** 2)
            self.all_mse.append(mse)

            if mse < self.threshold:
                break

        return self

    def predict(self, X):
        output = X @ self.w + self.b
        return np.where(output >= 0, 1, -1)


def confusion_matrix(y_true, y_pred):
    cm = np.zeros((2, 2), dtype=int)

    for i in range(len(y_true)):
        actual = y_true[i]
        pred = y_pred[i]

        row = 0 if actual == 1 else 1
        col = 0 if pred == 1 else 1

        cm[row, col] += 1

    return cm

def accuracy(cm):
    return np.trace(cm) / np.sum(cm)
