# %%
from sklearn import metrics
from features import get_id_df, get_target_df, get_predictors_df, read_data
from cfg import *
import numpy as np
import pandas as pd
import datetime
from loguru import logger

from sklearn.linear_model import LogisticRegression

import matplotlib.pyplot as plt
from sklearn.model_selection import learning_curve
from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import StratifiedKFold


from sklearn.model_selection import train_test_split  # 切分数据
from sklearn.metrics import mean_squared_error, roc_auc_score

from sklearn.model_selection import learning_curve
from sklearn.model_selection import ShuffleSplit

import warnings
warnings.filterwarnings("ignore")


# %%

def standize_df(train_data, test_data):
    from sklearn import preprocessing

    features_columns = [
        f for f in test_data.columns if f not in id_target_cols]
    min_max_scaler = preprocessing.MinMaxScaler()
    min_max_scaler = min_max_scaler.fit(train_data[features_columns])

    train_data_scaler = min_max_scaler.transform(train_data[features_columns])
    test_data_scaler = min_max_scaler.transform(test_data[features_columns])

    train_data_scaler = pd.DataFrame(train_data_scaler)
    train_data_scaler.columns = features_columns

    test_data_scaler = pd.DataFrame(test_data_scaler)
    test_data_scaler.columns = features_columns

    train_data_scaler['label'] = train_data['label']
    train_data_scaler[id_col_names] = train_data[id_col_names]
    test_data_scaler[id_col_names] = test_data[id_col_names]

    return train_data_scaler, test_data_scaler


def get_sklearn_model(model_name, param=None):
    # 朴素贝叶斯
    if model_name == 'NB':
        from sklearn.naive_bayes import MultinomialNB
        model = MultinomialNB(alpha=0.01)
    # 逻辑回归
    elif model_name == 'LR':
        from sklearn.linear_model import LogisticRegression
        model = LogisticRegression(penalty='l2')
    # KNN
    elif model_name == 'KNN':
        from sklearn.neighbors import KNeighborsClassifier
        model = KNeighborsClassifier()
    # 随机森林
    elif model_name == 'RF':
        from sklearn.ensemble import RandomForestClassifier
        model = RandomForestClassifier()
    # 决策树
    elif model_name == 'DT':
        from sklearn import tree
        model = tree.DecisionTreeClassifier()
    # 向量机
    elif model_name == 'SVC':
        from sklearn.svm import SVC
        model = SVC(kernel='rbf')
    # GBDT
    elif model_name == 'GBDT':
        from sklearn.ensemble import GradientBoostingClassifier
        model = GradientBoostingClassifier()
    # XGBoost
    elif model_name == 'XGB':
        from xgboost import XGBClassifier
        model = XGBClassifier()
    # lightGBM
    elif model_name == 'LGB':
        from lightgbm import LGBMClassifier
        model = LGBMClassifier()
    else:
        print("wrong model name!")
        return

    if param is not None:
        model.set_params(**param)
    return model


def plot_learning_curve(estimator, title, X, y, ylim=None, cv=None,
                        n_jobs=-1, train_sizes=[0.01, 0.02, 0.05, 0.1, 0.2, 0.3]):
    plt.figure()
    plt.title(title)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel("Training examples")
    plt.ylabel("Score")
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=cv, scoring=myeval, n_jobs=n_jobs, train_sizes=train_sizes)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    plt.grid()

    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
             label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
             label="Cross-validation score")

    plt.legend(loc="best")
    return plt


def plot_curve_single(traindf, classifier, cvnum=5, train_sizes=[0.01, 0.02, 0.05, 0.1, 0.2, 0.3]):
    # 画算法的学习曲线,为加快画图速度，最多选 20% 数据
    X = get_predictors_df(traindf)
    y = get_target_df(traindf)

    estimator = get_sklearn_model(classifier)
    title = "learning curve of "+classifier+", cv:"+str(cvnum)

    plot_learning_curve(estimator, title, X, y,
                        ylim=(0, 1.01),
                        cv=cvnum,
                        train_sizes=train_sizes)


def myauc(test):
    """
    #性能评价函数
    #本赛题目标是预测投放的优惠券是否核销。
    #针对此任务及一些相关背景知识，使用优惠券核销预测的平均AUC（ROC曲线下面积）作为评价标准。 
    #即对每个优惠券coupon_id单独计算核销预测的AUC值，再对所有优惠券的AUC值求平均作为最终的评价标准。
    # coupon平均auc计算
    """
    testgroup = test.groupby(['coupon_id'])
    aucs = []
    for i in testgroup:
        coupon_df = i[1]
        # 测算AUC必须大于1个类别
        if len(coupon_df['label'].unique()) < 2:
            continue
        auc = metrics.roc_auc_score(coupon_df['label'], coupon_df['pred'])
        aucs.append(auc)
    return np.average(aucs)


def test_model(traindf, classifier):
    train = traindf[traindf.date_received < 20160515].copy()
    test = traindf[traindf.date_received >= 20160515].copy()

    train_data = get_predictors_df(train).copy()
    train_target = get_target_df(train).copy()
    test_data = get_predictors_df(test).copy()
    test_target = get_target_df(test).copy()

    clf = get_sklearn_model(classifier)
    clf.fit(train_data, train_target)
    result = clf.predict_proba(test_data)[:, 1]
    test['pred'] = result
    score = metrics.roc_auc_score(test_target, result)
    score_coupon = myauc(test)
    logger.debug(
        f"{classifier}, AUC: {score:.3f}, Coupon AUC: {score_coupon:.3f}")


def test_model_split(traindf, classifier):
    target = get_target_df(traindf).copy()

    train_all, test_all, train_target, test_target = train_test_split(
        traindf, target, test_size=0.2, random_state=0)

    train_data = get_predictors_df(train_all).copy()
    test_data = get_predictors_df(test_all).copy()

    clf = get_sklearn_model(classifier)
    clf.fit(train_data, train_target)
    result = clf.predict_proba(test_data)[:, 1]

    test = test_all.copy()
    test['pred'] = result

    score = metrics.roc_auc_score(test_target, result)
    score_coupon = myauc(test)
    logger.debug(
        f"{classifier}, AUC: {score:.3f}, Coupon AUC: {score_coupon:.3f}")


# 对算法进行分析
def classifier_df_score(train_feat, classifier, cvnum=5, param=None):
    clf = get_sklearn_model(classifier, param)
    train = train_feat.copy()
    target = get_target_df(train_feat).copy()
    kf = StratifiedKFold(n_splits=cvnum)

    scores = []
    score_coupons = []
    for k, (train_index, test_index) in enumerate(kf.split(train, target)):
        train_data, test_data = train.iloc[train_index], train.iloc[test_index]
        train_target, test_target = target[train_index], target[test_index]
        
        clf.fit(get_predictors_df(train_data), train_target)
        
        train_pred = clf.predict_proba(get_predictors_df(train_data))[:, 1]
        test_pred = clf.predict_proba(get_predictors_df(test_data))[:, 1]

        score_test = roc_auc_score(test_target, test_pred)
        test_data['pred'] = test_pred
        score_coupon_test = myauc(test_data)

        scores.append(score_test)
        score_coupons.append(score_coupon_test)

    print(f"{classifier}, 总体AUC: {np.mean(scores):.4f}, Coupon AUC: {np.mean(score_coupons):.4f}")


# %%
# 所有的特征都是上一节生成的
train_f1, test_f1 = read_data('f1')
# 因为要使用KNN等进行测试，所以需要归一化
train_f1, test_f1 = standize_df(train_f1, test_f1)


# %%
test_model(train_f1, 'LR')
plot_curve_single(train_f1, 'LR')

# %%
test_model(train_f1, 'NB')
plot_curve_single(train_f1, 'NB')

# %%
test_model(train_f1, 'DT')
plot_curve_single(train_f1, 'DT')

# %%
test_model(train_f1, 'RF')
plot_curve_single(train_f1, 'RF')

# %%
test_model(train_f1, 'LGB')
plot_curve_single(train_f1, 'LGB')

# %%
test_model(train_f1, 'XGB')
plot_curve_single(train_f1, 'XGB')

# %%
""" # 3 用不同的特征训练，对比分析 """
train_f2, test_f2 = read_data('sf2')
train_f2, test_f2 = standize_df(train_f2, test_f2)

train_f3, test_f3 = read_data('sf3')
train_f3, test_f3 = standize_df(train_f3, test_f3)


# %%
print('特征f1逻辑回归成绩')
test_model(train_f1, 'LR')

print('特征sf2逻辑回归成绩')
test_model(train_f2, 'LR')

print('特征sf3逻辑回归成绩')
test_model(train_f3, 'LR')

# %%
# 数据读取
train_f1, test_f1 = read_data('f1')
train_f2, test_f2 = read_data('sf2')
train_f3, test_f3 = read_data('sf3')

#%%
# 简单交叉验证F1
target = get_target_df(train_f1).copy()
traindf = train_f1.copy()

# 切分数据 训练数据80% 验证数据20%
train_all, test_all, train_target, test_target = train_test_split(
    traindf, target, test_size=0.2, random_state=0)

train_data = get_predictors_df(train_all).copy()
test_data = get_predictors_df(test_all).copy()

clf = LogisticRegression()
clf.fit(train_data, train_target)
train_pred = clf.predict_proba(train_data)[:, 1]
test_pred = clf.predict_proba(test_data)[:, 1]

score_train = roc_auc_score(train_target, train_pred)
score_test = roc_auc_score(test_target, test_pred)

train_all['pred'] = train_pred
test_all['pred'] = test_pred
print("LogisticRegression train 总体AUC:   ", score_train)
print("LogisticRegression test 总体AUC:   ", score_test)
print("LogisticRegression train Coupon AUC:   ", myauc(train_all))
print("LogisticRegression test Coupon AUC:   ", myauc(test_all))

# %%
train = train_f1.copy()
train.head()

print ('特征f1, 不同模型5折训练Score：')
classifier_df_score(train,'NB', 5)
classifier_df_score(train,'LR', 5)
classifier_df_score(train,'RF', 5)
classifier_df_score(train,'LGB', 5)

# %%
plot_curve_single(train_f2, 'DT', 5, [0.1,0.2,0.3,0.5])
# %%
plot_curve_single(train_f3,'LGB',5,[0.1,0.2,0.3,0.5])
# %%
