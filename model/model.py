import datetime
import numpy as np
import pandas as pd
from loguru import logger

import matplotlib.pyplot as plt
from sklearn.model_selection import learning_curve
from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import StratifiedKFold


from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, roc_auc_score

import warnings
warnings.filterwarnings("ignore")


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
    plt.figure()
    plt.title(title)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel("Training examples")
    plt.ylabel("Score")
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=cv, scoring=scoring, n_jobs=n_jobs, train_sizes=train_sizes
    )
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    plt.grid()

    plt.fill_between(
        train_sizes,
        train_scores_mean - train_scores_std,
        train_scores_mean + train_scores_std,
        alpha=0.1,
        color="r",
    )
    plt.fill_between(
        train_sizes,
        test_scores_mean - test_scores_std,
        test_scores_mean + test_scores_std,
        alpha=0.1,
        color="g",
    )
    plt.plot(train_sizes, train_scores_mean, "o-",
             color="r", label="Training score")
    plt.plot(
        train_sizes, test_scores_mean, "o-", color="g", label="Cross-validation score"
    )

    plt.legend(loc="best")
    return plt

