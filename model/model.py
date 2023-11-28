import datetime
import numpy as np
import pandas as pd
from loguru import logger
import matplotlib.pyplot as plt


from sklearn.model_selection import KFold
from sklearn.model_selection import learning_curve
from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import validation_curve
from sklearn.model_selection import train_test_split

from sklearn.metrics import mean_squared_error, roc_auc_score

import warnings
warnings.filterwarnings("ignore")


class SklearnWrapper(object):
    def __init__(self, clf, seed=0, params=None):
        params['random_state'] = seed        
        self.clf = clf(**params)
            
    def train(self, x_train, y_train):        
        self.clf.fit(x_train, y_train)    
    
    def predict(self, x):        
        return self.clf.predict(x)


def get_sklearn_model(model_name, param=None):
    # 朴素贝叶斯
    if model_name == "NB":
        from sklearn.naive_bayes import MultinomialNB

        model = MultinomialNB(alpha=0.01)
    # 逻辑回归
    elif model_name == "LR":
        from sklearn.linear_model import LogisticRegression

        model = LogisticRegression(penalty="l2")
    # KNN
    elif model_name == "KNN":
        from sklearn.neighbors import KNeighborsClassifier

        model = KNeighborsClassifier()
    # 随机森林
    elif model_name == "RF":
        from sklearn.ensemble import RandomForestClassifier

        model = RandomForestClassifier()
    # 决策树
    elif model_name == "DT":
        from sklearn import tree

        model = tree.DecisionTreeClassifier()
    # 向量机
    elif model_name == "SVC":
        from sklearn.svm import SVC

        model = SVC(kernel="rbf")
    # GBDT
    elif model_name == "GBDT":
        from sklearn.ensemble import GradientBoostingClassifier

        model = GradientBoostingClassifier()
    # XGBoost
    elif model_name == "XGB":
        from xgboost import XGBClassifier

        model = XGBClassifier()
    # lightGBM
    elif model_name == "LGB":
        from lightgbm import LGBMClassifier

        model = LGBMClassifier()
    else:
        print("wrong model name!")
        return

    if param is not None:
        model.set_params(**param)
    return model


def get_oof(clf, x_train, y_train, x_test, n_splits=5):
    """
    Performs K-Fold cross-validation on the training data and makes out-of-fold predictions.
    Also, makes predictions on the test data.

    Parameters:
        - clf: The classifier to be trained and used for predictions.
        - x_train: Training data features.
        - y_train: Training data target.
        - x_test: Test data features.
        - n_splits: Number of folds for K-Fold cross-validation. Default is 5.

    Returns:
        - oof_train: Out-of-fold predictions for the training data.
        - oof_test: Average predictions on the test data over all folds.
    """
    kf = KFold(n_splits=n_splits)
    oof_train = np.zeros((x_train.shape[0],))
    oof_test = np.zeros((x_test.shape[0],))
    oof_test_skf = np.empty((n_splits, x_test.shape[0]))

    for i, (train_index, valid_index) in enumerate(kf.split(x_train, y_train)):
        trn_x, val_x = x_train.iloc[train_index], x_train.iloc[valid_index]
        trn_y, val_y = y_train[train_index], y_train[valid_index]
        
        clf.train(trn_x, trn_y)
        oof_train[valid_index] = clf.predict(val_x)
        oof_test_skf[i, :] = clf.predict(x_test)

    oof_test[:] = oof_test_skf.mean(axis=0)
    return oof_train.reshape(-1, 1), oof_test.reshape(-1, 1)


def plot_learning_curve(
    estimator,
    title,
    X,
    y,
    ylim=None,
    cv=None,
    n_jobs=-1,
    scoring='accuracy',
    train_sizes=[0.01, 0.02, 0.05, 0.1, 0.2, 0.3],
):
    fig, ax = plt.subplots()
    ax.set_title(title)
    if ylim is not None:
        ax.set_ylim(*ylim)
    ax.set_xlabel("Training examples")
    ax.set_ylabel("Score")
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=cv, scoring=scoring, n_jobs=n_jobs, train_sizes=train_sizes
    )
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    ax.grid()

    ax.fill_between(
        train_sizes,
        train_scores_mean - train_scores_std,
        train_scores_mean + train_scores_std,
        alpha=0.1,
        color="r",
    )
    ax.fill_between(
        train_sizes,
        test_scores_mean - test_scores_std,
        test_scores_mean + test_scores_std,
        alpha=0.1,
        color="g",
    )
    ax.plot(train_sizes, train_scores_mean, "o-", color="r", label="Training score")
    ax.plot(train_sizes, test_scores_mean, "o-", color="g", label="Cross-validation score")

    ax.legend(loc="best")
    return fig


def plot_validation_curve(estimator, X, y, param_name, param_range, cv=10, scoring='accuracy', n_jobs=1):
    # Create a figure
    fig, ax = plt.subplots()

    train_scores, test_scores = validation_curve(
        estimator, X, y, param_name=param_name, param_range=param_range,
        cv=cv, scoring=scoring, n_jobs=n_jobs)

    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)

    ax.set_title("Validation Curve with SVM")
    ax.set_xlabel("$\gamma$")
    ax.set_ylabel("Score")
    ax.set_ylim(0.0, 1.1)
    ax.semilogx(param_range, train_scores_mean, label="Training score", color="r")
    ax.fill_between(param_range, train_scores_mean - train_scores_std,
                    train_scores_mean + train_scores_std, alpha=0.2, color="r")
    ax.semilogx(param_range, test_scores_mean, label="Cross-validation score",
                color="g")
    ax.fill_between(param_range, test_scores_mean - test_scores_std,
                    test_scores_mean + test_scores_std, alpha=0.2, color="g")
    ax.legend(loc="best")

    return fig


if __name__ == "__main__":
    # Example usage
    from sklearn.datasets import load_digits
    from sklearn.svm import SVC
    digits = load_digits()
    X, y = digits.data, digits.target
    param_range = np.logspace(-6, -1, 5)

    fig = plot_validation_curve(SVC(), X, y, param_name="gamma", param_range=param_range)
    # You can now display or save fig as needed