import numpy as np
import pandas as pd
import xgboost as xgb
import lightgbm as lgb

from itertools import combinations
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier

from sklearn.feature_selection import RFECV
from sklearn.feature_selection import SelectFromModel, SelectKBest, f_classif
from sklearn.feature_selection import SequentialFeatureSelector

from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold


def null_importance_feature_selection(X, y, feature_names, n_estimators=100, random_state=42, n_jobs=-1):
    """
    Performs null importance feature selection to identify useful features in a dataset.
    This method involves comparing the importance of features with their importance when
    the target variable is shuffled. If the importance of a feature with the original target
    is not significantly higher than its importance with the shuffled target, the feature 
    may be considered as not useful.

    Parameters:
        - X: Features dataset.
        - y: Target variable.
        - feature_names: List of feature names.
        - n_estimators: Number of trees in the forest. Default is 100.
        - random_state: Random state for reproducibility. Default is 42.

    Returns:
        - A DataFrame with feature names, their original importance, shuffled importance,
          and a boolean indicating whether the feature is considered useful or not.
        - A list of feature names that are considered useful.
    """

    def calculate_feature_importance(X, y):
        model = RandomForestClassifier(n_estimators=n_estimators, random_state=random_state, n_jobs=n_jobs)
        model.fit(X, y)
        return model.feature_importances_

    original_importance = calculate_feature_importance(X, y)

    shuffled_y = np.random.permutation(y)
    shuffled_importance = calculate_feature_importance(X, shuffled_y)

    feature_comparison = pd.DataFrame({
        'Feature': feature_names,
        'Original Importance': original_importance,
        'Shuffled Importance': shuffled_importance
    })

    feature_comparison['Useful'] = feature_comparison['Original Importance'] > feature_comparison['Shuffled Importance']
    useful_feature_names = feature_comparison[feature_comparison.Useful].Feature.values.tolist()
    return feature_comparison, useful_feature_names


def feature_selection(X, y, method='model', k=None, threshold=None, model=None):
    """
    特征选择：基于模型或者基于统计方法。

    :param X: 特征数据集（DataFrame 格式）
    :param y: 目标变量
    :param method: 选择的方法，'model' 或 'kbest'
    :param k: 如果使用 'kbest'，选择的特征数量
    :param threshold: 如果使用 'model'，基于模型权重的阈值
    :param model: 如果使用 'model'，基础模型，默认为随机森林
    :return: 选择的特征列表
    """
    if method == 'model':
        if model is None:
            model = RandomForestClassifier()
        selector = SelectFromModel(model, threshold=threshold)
    elif method == 'kbest':
        if k is None:
            k = 'all'
        selector = SelectKBest(f_classif, k=k)
    else:
        raise ValueError("Method must be 'model' or 'kbest'")

    # 拟合特征选择器
    selector.fit(X, y)

    # 获取选择的特征
    selected_features = X.columns[selector.get_support()]

    # 返回选择的特征
    return selected_features


def sequential_feature_selection(X, y, direction='forward', n_features_to_select=None, cv=5, scoring='accuracy'):
    """
    使用前向或后向特征选择。

    :param X: 特征数据集（DataFrame 格式）
    :param y: 目标变量
    :param direction: 特征选择的方向，'forward' 或 'backward'
    :param n_features_to_select: 选择的特征数量，如果为 None，则自动选择
    :param cv: 交叉验证的折数
    :param scoring: 评估指标
    :return: 选择的特征列表
    """
    # 初始化基础模型
    model = RandomForestClassifier()

    # 初始化 Sequential Feature Selector
    sfs = SequentialFeatureSelector(
        model, 
        n_features_to_select=n_features_to_select, 
        direction=direction,
        cv=cv, scoring=scoring
    )

    # 拟合 Sequential Feature Selector
    sfs.fit(X, y)

    # 获取选择的特征
    selected_features = X.columns[sfs.get_support()]

    # 返回选择的特征
    return selected_features


def rfecv_feature_selection(X, y, estimator=None, cv=5, scoring='accuracy', step=1, verbose=0, n_jobs=-1):
    """
    使用 RFECV 进行特征选择。

    :param X: 特征数据集（DataFrame 格式）
    :param y: 目标变量
    :param estimator: 基础模型，默认为随机森林
    :param cv: 交叉验证的折数
    :param scoring: 评估指标
    :param step: 每次迭代中要移除的特征数量
    :param verbose: 控制输出的详细程度
    :return: 选择的特征列表
    """
    if estimator is None:
        estimator = RandomForestClassifier()

    rfecv = RFECV(estimator, step=step, cv=StratifiedKFold(cv), scoring=scoring, verbose=verbose, n_jobs=n_jobs)
    rfecv.fit(X, y)

    print("Optimal number of features : %d" % rfecv.n_features_)
    print("Best features : %s" % list(X.columns[rfecv.support_]))

    return X.columns[rfecv.support_]


def best_feature_subset(X, y, num_features, cv=5, metric='accuracy'):
    """
    使用交叉验证找到最佳特征子集。

    :param X: 特征数据集（DataFrame 格式）
    :param y: 目标变量
    :param num_features: 考虑的特征子集大小
    :param cv: 交叉验证的折数
    :param metric: 评估指标（如 'accuracy', 'roc_auc'）
    :return: 最佳特征子集和对应的性能分数
    """
    best_score = 0
    best_subset = None

    # 遍历所有可能的特征子集
    for features in combinations(X.columns, num_features):
        X_subset = X[list(features)]

        # 使用 LightGBM 进行交叉验证
        model = lgb.LGBMClassifier()
        score = np.mean(cross_val_score(model, X_subset, y, cv=cv, scoring=metric))

        if score > best_score:
            best_score = score
            best_subset = features

    return best_subset, best_score


def calculate_feature_importance_lgb(x, y, importance_type='gain', params=None, num_boost_round=100, 
                                     threshold=None, top_n=None, cumulative_importance=None):
    """
    使用 LightGBM 计算并返回按特征重要性排序后的特征及其重要性，并可根据指定的参数进行特征选择。

    :param x: 特征数据（DataFrame 格式）
    :param y: 目标变量
    :param importance_type: 特征重要性类型 ('split', 'gain')
    :param params: LightGBM 模型参数
    :param num_boost_round: 提升轮数
    :param threshold: 重要性阈值
    :param top_n: 选择重要性排名前 N 的特征
    :param cumulative_importance: 累积重要性百分比阈值
    :return: 根据特定参数筛选后的特征名称及其重要性，以及未被选中的特征及其重要性
    """
    # 设置默认参数（如果未提供）
    if params is None:
        params = {
            'objective': 'binary',
            'metric': 'binary_logloss',
            'verbose': -1
        }

    # 创建 Dataset 对象
    lgb_data = lgb.Dataset(x, label=y)

    # 训练模型
    model = lgb.train(params, lgb_data, num_boost_round=num_boost_round)

    # 获取特征重要性
    importance = model.feature_importance(importance_type=importance_type)
    feature_names = model.feature_name()
    importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': importance})

    # 按重要性排序
    importance_df.sort_values(by='Importance', ascending=False, inplace=True)

    # 复制一份作为未被选中的特征 DataFrame
    unimportance_df = importance_df.copy()

    # 应用特征选择逻辑
    if threshold is not None:
        importance_df = importance_df[importance_df['Importance'] >= threshold]
        unimportance_df = unimportance_df[unimportance_df['Importance'] < threshold]
    elif top_n is not None:
        importance_df = importance_df.head(top_n)
        unimportance_df = unimportance_df.iloc[top_n:]
    elif cumulative_importance is not None:
        importance_df['Cumulative Importance'] = importance_df['Importance'].cumsum()
        importance_df = importance_df[importance_df['Cumulative Importance'] <= cumulative_importance]
        unimportance_df['Cumulative Importance'] = unimportance_df['Importance'].cumsum()
        unimportance_df = unimportance_df[unimportance_df['Cumulative Importance'] > cumulative_importance]

    # 返回筛选后的特征名称和对应的重要性，以及未被选中的特征
    return importance_df, unimportance_df


def calculate_feature_importance_xgb(x, y, importance_type='weight', params=None, num_boost_round=10, 
                                 threshold=None, top_n=None, cumulative_importance=None):
    """
    计算并返回按特征重要性排序后的特征及其重要性，并可根据指定的参数进行特征选择。

    :param x: 特征数据（DataFrame 格式）
    :param y: 目标变量
    :param importance_type: 特征重要性类型 ('weight', 'gain', 'cover')
    :param params: XGBoost 模型参数
    :param num_boost_round: 提升轮数
    :param threshold: 重要性阈值
    :param top_n: 选择重要性排名前 N 的特征
    :param cumulative_importance: 累积重要性百分比阈值
    :return: 根据特定参数筛选后的特征名称及其重要性，以及未被选中的特征及其重要性
    """
    # 设置默认参数（如果未提供）
    if params is None:
        params = {
            'max_depth': 10,
            'subsample': 1,
            'verbose_eval': True,
            'seed': 12,
            'objective': 'binary:logistic'
        }

    # 创建 DMatrix 对象
    xgtrain = xgb.DMatrix(x, label=y)

    # 训练模型
    bst = xgb.train(params, xgtrain, num_boost_round=num_boost_round)

    # 计算特征重要性
    importance = bst.get_score(importance_type=importance_type)

    # 将特征重要性字典转换为 DataFrame
    importance_df = pd.DataFrame(importance.items(), columns=['Feature', 'Importance'])

    # 按重要性排序
    importance_df.sort_values(by='Importance', ascending=False, inplace=True)

    # 复制一份作为未被选中的特征 DataFrame
    unimportance_df = pd.DataFrame()

    # 应用特征选择逻辑
    if threshold is not None:
        importance_df = importance_df[importance_df['Importance'] >= threshold]
        unimportance_df = unimportance_df[unimportance_df['Importance'] < threshold]
    elif top_n is not None:
        importance_df = importance_df.head(top_n)
        unimportance_df = unimportance_df.iloc[top_n:]
    elif cumulative_importance is not None:
        importance_df['Cumulative Importance'] = importance_df['Importance'].cumsum()
        importance_df = importance_df[importance_df['Cumulative Importance'] <= cumulative_importance]
        unimportance_df['Cumulative Importance'] = unimportance_df['Importance'].cumsum()
        unimportance_df = unimportance_df[unimportance_df['Cumulative Importance'] > cumulative_importance]

    # 返回筛选后的特征名称和对应的重要性，以及未被选中的特征
    return importance_df, unimportance_df


if __name__ == "__main__":
    # Load iris dataset
    iris = load_iris()
    X = iris.data
    y = iris.target

    # Perform null importance feature selection
    feature_importance_analysis, useful_feature_names = null_importance_feature_selection(X, y, iris.feature_names)
    print(f"useful_feature_names: {useful_feature_names}")
    print(feature_importance_analysis)


    # 使用基于模型的方法
    selected_features_model = feature_selection(X, y, method='model')
    print("Selected Features (Model-based):", selected_features_model)

    # 使用 SelectKBest
    selected_features_kbest = feature_selection(X, y, method='kbest', k=5)
    print("Selected Features (KBest):", selected_features_kbest)


    # 前向特征选择
    forward_selected_features = sequential_feature_selection(X, y, direction='forward')
    print("Forward Selected Features:", forward_selected_features)

    # 后向特征消除
    backward_eliminated_features = sequential_feature_selection(X, y, direction='backward')
    print("Backward Eliminated Features:", backward_eliminated_features)

    selected_features = rfecv_feature_selection(X, y)
    print("Selected Features:", selected_features)


    # 找到最佳的两个特征组合
    best_subset, best_subset_score = best_feature_subset(X, y, num_features=2)
    print("Best Feature Subset:", best_subset)
    print("Best Subset Score:", best_subset_score)


    important_features, unimportant_features = calculate_feature_importance_lgb(x, y, 'gain', top_n=10)
    print("Unimportant Features and Importances:", unimportant_features)

    important_features, unimportant_features = calculate_feature_importance(x, y, 'gain', top_n=10)
    print("Unimportant Features and Importances:", unimportant_features)
