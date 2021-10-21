import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


class LeastSquaresClassifier():
    def __init__(self):
        return

    def fit(self, X, y):
        self.X = X
        self.y = y

        betas = np.dot(np.dot(np.linalg.inv(np.dot(np.transpose(X), X)), np.transpose(X)), y)

        predictions = np.dot(betas, np.transpose(X))
        errors = np.array([])
        for i in np.arange(0, 1, 0.001):
            classes = [1 if x > i else 0 for x in predictions]
            mse = np.mean((classes - y) ** 2)
            errors = np.append(errors, mse)
        min_threshold_index = np.argmin(errors)
        min_threshold = 0.001 * min_threshold_index

        self.betas = betas
        self.threshold = min_threshold

    def predict(self, X):
        predictions = np.dot(self.betas, np.transpose(X))
        classes = [1 if x > self.threshold else 0 for x in predictions]

        return classes

    def plot(self):
        fig, ax = plt.subplots()
        sns.scatterplot(x=self.X[:, 0], y=self.X[:, 1], hue=self.y, ax=ax)

        b = (self.threshold - self.betas[0] * self.X[:, 0]) / self.betas[1]
        sns.lineplot(x=self.X[:, 0], y=b, color="black")
        ax.set_ylim(min(self.X[:, 1]) - 1, max(self.X[:, 1]) + 1)
        plt.show()