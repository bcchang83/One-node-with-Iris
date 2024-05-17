import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


def sigmoid_func(x):
    return 1 / (1 + np.exp(-x))


def MSE(d, y):
    return 0.5 * np.sum(y - d) ** 2


lr = 0.001
epoch = 500
trial = 10
iris = datasets.load_iris()
df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
df["target"] = iris.target
df = df.loc[(df.target == 1) | (df.target == 2)]
X = np.array(df[iris.feature_names])
Y = np.array(df["target"])
Y = Y - 1
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.3)
acc_train_trial = []
acc_test_trial = []
for t in range(trial):
    w = np.array([0.2, 0.2, 0.2, 0.2])
    # w = np.random.ranf(4)
    b = 0.2
    # b = np.random.ranf(1)
    loss_train_list = []
    loss_test_list = []
    acc_train_list = []
    acc_test_list = []

    for _ in range(epoch):
        loss_tmp = 0
        ans_train_list = []
        for i in range(len(y_train)):
            y_pred = sigmoid_func(np.matmul(w, x_train[i]) + b)
            loss_tmp += MSE(y_train[i], y_pred)
            ans_train_list.append(y_pred > 0.5)
            w = w - lr * (y_pred - y_train[i]) * y_pred * (1 - y_pred) * x_train[i]
            b = b - lr * (y_pred - y_train[i]) * y_pred * (1 - y_pred)
        loss_train_list.append(loss_tmp)
        acc_train_list.append(accuracy_score(y_train, ans_train_list))

        y_pred_test = sigmoid_func(np.matmul(w, x_test.T) + b)
        loss_test_list.append(MSE(y_test, y_pred_test))
        acc_test_list.append(accuracy_score(y_test, y_pred_test > 0.5))

    plt.figure(1 + 2 * t)
    plt.title("Trial {}, Cost".format(t + 1))
    plt.xlabel("Epoch")
    plt.ylabel("Cost")
    plt.plot(loss_train_list, label="training cost")
    plt.plot(loss_test_list, label="testing cost")
    plt.legend()
    plt.figure(2 + 2 * t)
    plt.title("Trial {}, Accuracy".format(t + 1))
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.plot(acc_train_list, label="training accuracy")
    plt.plot(acc_test_list, label="testing accuracy")
    plt.legend()
    acc_train_trial.append(accuracy_score(y_train, ans_train_list))
    acc_test_trial.append(accuracy_score(y_test, y_pred_test > 0.5))
plt.figure()
plt.title(
    "Accuracy in every trial\nAvg. training accuracy = {:.3}, Avg. testing accuracy = {:.3}".format(
        np.average(acc_train_trial), np.average(acc_test_trial)
    )
)
plt.plot(acc_train_trial, label="training accuracy")
plt.plot(acc_test_trial, label="testing accuracy")
plt.ylim(min(min(acc_train_trial), min(acc_test_trial)) - 0.1, 1.05)
plt.xlabel("Trials")
plt.ylabel("Accuracy")
plt.legend()
