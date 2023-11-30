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

""" dataset """
def get_id_df(df):
    #返回ID列
    return df[id_col_names]

def get_target_df(df):
    #返回Target列
    return df[target_col_name]

def get_predictors_df(df):
    #返回特征列
    predictors = [f for f in df.columns if f not in id_target_cols]
    return df[predictors]

def read_featurefile_train(featurename): 
    #按特征名读取训练集
    df = pd.read_csv(
        featurepath + 'train_' + featurename + '.csv', 
        sep=',' , 
        encoding = "utf-8")
    df.fillna(0,inplace=True)
    return df

def read_featurefile_test(featurename): 
    #按特征名读取测试集
    df=pd.read_csv(
        featurepath + 'test_' + featurename + '.csv', 
        sep=',' , 
        encoding = "utf-8")
    df.fillna(0,inplace=True)
    return df

def read_data(featurename): 
    #按特征名读取数据
    traindf = read_featurefile_train(featurename)
    testdf = read_featurefile_test(featurename)
    return traindf,testdf  

def standardize_df(train_data, test_data=None, 
                   id_cols=[], 
                   target_col='label', 
                   scaler=None, scaler_type='minmax'):
    """
    Standardize the feature columns of train and test datasets using either Min-Max scaling or Standard scaling.

    Args:
    - train_data (pd.DataFrame): The training dataset.
    - test_data (pd.DataFrame): The test dataset.
    - id_cols (list): List of column names to be excluded from scaling (like IDs).
    - target_col (str): The name of the target column in the training data.
    - scaler_type (str): Type of scaler to use ('minmax' or 'standard').

    Returns:
    - tuple: A tuple containing the standardized training and test dataframes.
    """
    if not isinstance(train_data, pd.DataFrame) or (test_data is not None and not isinstance(test_data, pd.DataFrame)):
        raise ValueError("train_data and test_data must be pandas DataFrames")

    if scaler_type not in ['minmax', 'standard']:
        raise ValueError("scaler_type must be either 'minmax' or 'standard'")

    features_columns = [col for col in train_data.columns if col not in id_cols + [target_col]]
    
    if scaler is None:
        if scaler_type == 'minmax':
            from sklearn.preprocessing import MinMaxScaler
            scaler = MinMaxScaler()
        else:
            from sklearn.preprocessing import StandardScaler
            scaler = StandardScaler()

        scaler.fit(train_data[features_columns])

    train_data_scaled = pd.DataFrame(
        scaler.transform(train_data[features_columns]), 
        columns=features_columns,
        index=train_data.index
    )
    if test_data is not None:
        test_data_scaled = pd.DataFrame(
            scaler.transform(test_data[features_columns]), 
            columns=features_columns,
            index=test_data.index
        )

    if target_col in list(train_data):
        train_data_scaled[target_col] = train_data[target_col]
    
    for col in id_cols:
        train_data_scaled[col] = train_data[col]
        if test_data is not None:
            test_data_scaled[col] = test_data[col]

    return train_data_scaled, test_data_scaled if test_data is not None else None, scaler


""" model """
class SklearnWrapper(object):
    def __init__(self, clf, seed=42, params=None):
        params["random_state"] = seed
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
    scoring="accuracy",
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
    ax.plot(
        train_sizes, test_scores_mean, "o-", color="g", label="Cross-validation score"
    )

    ax.legend(loc="best")
    return fig


def plot_validation_curve(
    estimator, X, y, param_name, param_range, cv=10, scoring="accuracy", n_jobs=1
):
    # Create a figure
    fig, ax = plt.subplots()

    train_scores, test_scores = validation_curve(
        estimator,
        X,
        y,
        param_name=param_name,
        param_range=param_range,
        cv=cv,
        scoring=scoring,
        n_jobs=n_jobs,
    )

    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)

    ax.set_title("Validation Curve with SVM")
    ax.set_xlabel("$\gamma$")
    ax.set_ylabel("Score")
    ax.set_ylim(0.0, 1.1)
    ax.semilogx(param_range, train_scores_mean, label="Training score", color="r")
    ax.fill_between(
        param_range,
        train_scores_mean - train_scores_std,
        train_scores_mean + train_scores_std,
        alpha=0.2,
        color="r",
    )
    ax.semilogx(
        param_range, test_scores_mean, label="Cross-validation score", color="g"
    )
    ax.fill_between(
        param_range,
        test_scores_mean - test_scores_std,
        test_scores_mean + test_scores_std,
        alpha=0.2,
        color="g",
    )
    ax.legend(loc="best")

    return fig


if __name__ == "__main__":
    """plot_validation_curve"""
    from sklearn.datasets import load_digits
    from sklearn.svm import SVC

    digits = load_digits()
    X, y = digits.data, digits.target
    param_range = np.logspace(-6, -1, 5)

    fig = plot_validation_curve(
        SVC(), X, y, param_name="gamma", param_range=param_range
    )

    # Example usage
    # train_scaled, test_scaled = standardize_df(train_df, test_df, ['id'], 'label', 'standard')
