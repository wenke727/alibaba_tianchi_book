#%%
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from datetime import date
import datetime as dt
from scipy import stats
import warnings
warnings.filterwarnings("ignore")

from cfg import *

#%%
""" eda """
def plot_qq_with_subfigures(dataframe, column):
    """
    Plot a Q-Q plot for the specified column in the given dataframe based on a condition
    using subfigures.

    Parameters:
    dataframe (DataFrame): The dataframe containing the data.
    condition (Series): A boolean Series indicating the condition to filter the dataframe.
    column (str): The column for which to plot the Q-Q plot.
    """
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))

    # Distribution plot
    sns.distplot(dataframe[column], fit=stats.norm, ax=axes[0])

    # Q-Q plot
    stats.probplot(dataframe[column], plot=axes[1])

    return fig

#%%
""" 将特征数值化 """
def get_discount_rate(s):
    #计算折扣率，将满减和折扣统一
    s = str(s)
    if s=='null':
        return -1
    s = s.split(fd_seperator)
    if len(s) == 1:
        return float(s[0])
    else:
        return round((1.0-float(s[1])/float(s[0])),3)

def get_if_fd(s):
    #获取是否满减（full reduction promotion）
    s = str(s)
    s = s.split(fd_seperator)
    if len(s)==1:
        return 0
    else:
        return 1
        
def get_full_value(s):
    #获取满减的条件
    s = str(s)
    s = s.split(fd_seperator)
    if len(s)==1:
        #return 'null'
        return np.nan
    else:
        return int(s[0])
        
def get_reduction_value(s):
    #获取满减的优惠     
    s = str(s)
    s = s.split(fd_seperator)
    if len(s) == 1:
        #return 'null'
        return np.nan
    else:
        return int(s[1])

def get_day_gap(s):
    #获取日期间隔，输入内容为Date_received:Date
    s = s.split(fd_seperator)
    if s[0]=='null':
        return -1
    if s[1]=='null':
        return -1
    else:    
        return (date(int(s[0][0:4]),int(s[0][4:6]),int(s[0][6:8])) - date(int(s[1][0:4]),int(s[1][4:6]),int(s[1][6:8]))).days

def get_label(s):
    #获取Label，输入内容为Date:Date_received
    s = s.split(fd_seperator)
    if s[0]=='null':
        return 0
    if s[1]=='null':
        return -1
    elif (date(int(s[0][0:4]),int(s[0][4:6]),int(s[0][6:8]))-date(int(s[1][0:4]),int(s[1][4:6]),int(s[1][6:8]))).days<=15:
        return 1
    else:
        return 0

def add_discount(df):
    #增加折扣相关特征
    df['if_fd']=df['discount_rate'].apply(get_if_fd)
    df['full_value']=df['discount_rate'].apply(get_full_value)
    df['reduction_value']=df['discount_rate'].apply(get_reduction_value)
    df['discount_rate']=df['discount_rate'].apply(get_discount_rate)
    df.distance=df.distance.replace('null',np.nan)
    return df

def add_day_gap(df):
    #计算日期间隔  
    df['day_gap']=df['date'].astype('str') + ':' +  df['date_received'].astype('str')
    df['day_gap']=df['day_gap'].apply(get_day_gap)
    return df

def add_label(df):
    #获取label
    df['label']=df['date'].astype('str') + ':' +  df['date_received'].astype('str')
    df['label']=df['label'].apply(get_label)
    return df

def is_firstlastone(x):
    if x==0:
        return 1
    elif x>0:
        return 0
    else:
        #return -1
        return np.nan

def get_day_gap_before(s):
    date_received,dates = s.split('-')
    dates = dates.split(':')
    gaps = []
    for d in dates:
        #将时间差转化为天数
        this_gap = (dt.date(int(date_received[0:4]),int(date_received[4:6]),int(date_received[6:8]))-dt.date(int(d[0:4]),int(d[4:6]),int(d[6:8]))).days
        if this_gap>0:
            gaps.append(this_gap)
    if len(gaps)==0:
        #return -1
        return np.nan
    else:
        return min(gaps)
    
def get_day_gap_after(s):
    date_received,dates = s.split('-')
    dates = dates.split(':')
    gaps = []
    for d in dates:
        this_gap = (dt.datetime(int(d[0:4]),int(d[4:6]),int(d[6:8]))-dt.datetime(int(date_received[0:4]),int(date_received[4:6]),int(date_received[6:8]))).days
        if this_gap>0:
            gaps.append(this_gap)
    if len(gaps)==0:
        #return -1
        return np.nan
    else:
        return min(gaps)
    
    
""" 统计特征处理函数 """
def add_agg_feature_names(df, df_group, group_cols, value_col, agg_ops, col_names):
    # df: 添加特征的dataframe
    # df_group: 特征生成的数据集
    # group_cols: group by 的列
    # value_col: 被统计的列
    # agg_ops:处理方式 包括：count,mean,sum,std,max,min,nunique
    # colname: 新特征的名称
    df_group[value_col]=df_group[value_col].astype('float')
    df_agg = pd.DataFrame(df_group.groupby(group_cols)[value_col].agg(agg_ops)).reset_index()
    df_agg.columns = group_cols + col_names
    df = df.merge(df_agg, on=group_cols, how="left")
    return df    

def add_agg_feature(df, df_group, group_cols, value_col, agg_ops, keyword):
    # 统计特征处理函数
    # 名称按照keyword+'_'+value_col+'_'+op 自动增加
    col_names=[]
    for op in agg_ops:
        col_names.append(keyword+'_'+value_col+'_'+op)
    df=add_agg_feature_names(df, df_group, group_cols, value_col, agg_ops, col_names)
    return df

def add_count_new_feature(df, df_group, group_cols, new_feature_name):
    # 增加一个count特征
    df_group[new_feature_name] = 1
    df_group = df_group.groupby(group_cols).agg('sum').reset_index()
    df = df.merge(df_group, on=group_cols, how="left")
    return df


""" 特征群生成 """
def get_merchant_feature(feature):
    # 获取商家相关特征
    merchant = feature[['merchant_id','coupon_id','distance','date_received','date']].copy()
    t = merchant[['merchant_id']].copy()
    #删除重复行数据
    t.drop_duplicates(inplace=True)

    #卖出的商品
    t1 = merchant[merchant.date!='null'][['merchant_id']].copy()
    merchant_feature=add_count_new_feature(t, t1, 'merchant_id', 'total_sales')

    #使用了优惠券消费的商品，正样本
    t2 = merchant[(merchant.date!='null')&(merchant.coupon_id!='null')][['merchant_id']].copy()
    merchant_feature=add_count_new_feature(merchant_feature, t2, 'merchant_id', 'sales_use_coupon')

    #商品的优惠券的总数量
    t3 = merchant[merchant.coupon_id != 'null'][['merchant_id']].copy()
    merchant_feature=add_count_new_feature(merchant_feature, t3, 'merchant_id', 'total_coupon')
    
    #商品销量和距离的关系
    t4 = merchant[(merchant.date != 'null')&(merchant.coupon_id != 'null')&(merchant.distance != 'null')][['merchant_id','distance']].copy()
    t4.distance=t4.distance.astype('int')
    merchant_feature=add_agg_feature(merchant_feature, t4, ['merchant_id'], 'distance', ['min','max','mean','median'], 'merchant')

    #将数据中的NaN用0来替换
    merchant_feature.sales_use_coupon = merchant_feature.sales_use_coupon.replace(np.nan,0)
    #优惠券的使用率
    merchant_feature['merchant_coupon_transfer_rate'] = merchant_feature.sales_use_coupon.astype('float')/merchant_feature.total_coupon
    #即卖出商品中使用优惠券的占比
    merchant_feature['coupon_rate'] = merchant_feature.sales_use_coupon.astype('float') / merchant_feature.total_sales
    #将数据中的NaN用0来替换
    merchant_feature.total_coupon = merchant_feature.total_coupon.replace(np.nan,0)

    return merchant_feature

def get_user_feature(feature):
    # 获取用户相关特征 
    #for dataset3
    user = feature[['user_id','merchant_id','coupon_id','discount_rate','distance','date_received','date']].copy()

    t = user[['user_id']].copy()
    t.drop_duplicates(inplace=True)
     
    #客户一共买的商品
    t1 = user[user.date!='null'][['user_id','merchant_id']].copy()
    t1.drop_duplicates(inplace=True)
    t1 = t1[['user_id']]
    user_feature=add_count_new_feature(t, t1, 'user_id', 'count_merchant')

    #客户使用优惠券线下购买距离商店的距离统计最大，最小，平均，中位值
    t2 = user[(user.date!='null')&(user.coupon_id!='null')&(user.distance != 'null')][['user_id','distance']]
    t2.distance=t2.distance.astype('int')
    user_feature=add_agg_feature(user_feature, t2, ['user_id'], 'distance', ['min','max','mean','median'], 'user')

    #客户使用优惠券购买的次数
    t7 = user[(user.date!='null')&(user.coupon_id!='null')][['user_id']]
    user_feature=add_count_new_feature(user_feature, t7, 'user_id', 'buy_use_coupon')
    
    #客户使购买的总次数
    t8 = user[user.date!='null'][['user_id']]
    user_feature=add_count_new_feature(user_feature, t8, 'user_id', 'buy_total')
    
    #客户收到优惠券的总数
    t9 = user[user.coupon_id!='null'][['user_id']]
    user_feature=add_count_new_feature(user_feature, t9, 'user_id', 'coupon_received')
    
    #客户从收优惠券到消费的时间间隔，统计最大，最小，平均，中位值
    t10 = user[(user.date_received!='null')&(user.date!='null')][['user_id','date_received','date']]
    t10 = add_day_gap(t10)
    t10 = t10[['user_id','day_gap']]
    user_feature=add_agg_feature(user_feature, t10, ['user_id'], 'day_gap', ['min','max','mean','median'], 'user')

    user_feature.count_merchant = user_feature.count_merchant.replace(np.nan,0)
    user_feature.buy_use_coupon = user_feature.buy_use_coupon.replace(np.nan,0)
    user_feature['buy_use_coupon_rate'] = user_feature.buy_use_coupon.astype('float') / user_feature.buy_total.astype('float')
    user_feature['user_coupon_transfer_rate'] = user_feature.buy_use_coupon.astype('float') / user_feature.coupon_received.astype('float')
    user_feature.buy_total = user_feature.buy_total.replace(np.nan,0)
    user_feature.coupon_received = user_feature.coupon_received.replace(np.nan,0)
    return user_feature

def get_user_merchant_feature(feature):
    t = feature[['user_id','merchant_id']].copy()
    t.drop_duplicates(inplace=True)

    #一个客户在一个商家一共买的次数
    t0 = feature[['user_id','merchant_id','date']].copy()
    t0 = t0[t0.date!='null'][['user_id','merchant_id']]    
    user_merchant=add_count_new_feature(t, t0, ['user_id','merchant_id'], 'user_merchant_buy_total')
    
    #一个客户在一个商家一共收到的优惠券
    t1 = feature[['user_id','merchant_id','coupon_id']]
    t1 = t1[t1.coupon_id!='null'][['user_id','merchant_id']]
    user_merchant=add_count_new_feature(user_merchant, t1, ['user_id','merchant_id'], 'user_merchant_received')
    
    #一个客户在一个商家使用优惠券购买的次数
    t2 = feature[['user_id','merchant_id','date','date_received']]
    t2 = t2[(t2.date!='null')&(t2.date_received!='null')][['user_id','merchant_id']]
    user_merchant=add_count_new_feature(user_merchant, t2, ['user_id','merchant_id'], 'user_merchant_buy_use_coupon')
    
    #一个客户在一个商家浏览的次数
    t3 = feature[['user_id','merchant_id']]
    user_merchant=add_count_new_feature(user_merchant, t3, ['user_id','merchant_id'], 'user_merchant_any')
    
    #一个客户在一个商家没有使用优惠券购买的次数
    t4 = feature[['user_id','merchant_id','date','coupon_id']]
    t4 = t4[(t4.date!='null')&(t4.coupon_id=='null')][['user_id','merchant_id']]
    user_merchant=add_count_new_feature(user_merchant, t4, ['user_id','merchant_id'], 'user_merchant_buy_common')
    
    user_merchant.user_merchant_buy_use_coupon = user_merchant.user_merchant_buy_use_coupon.replace(np.nan,0)
    user_merchant.user_merchant_buy_common = user_merchant.user_merchant_buy_common.replace(np.nan,0)
    user_merchant['user_merchant_coupon_transfer_rate'] = user_merchant.user_merchant_buy_use_coupon.astype('float') / user_merchant.user_merchant_received.astype('float')
    user_merchant['user_merchant_coupon_buy_rate'] = user_merchant.user_merchant_buy_use_coupon.astype('float') / user_merchant.user_merchant_buy_total.astype('float')
    user_merchant['user_merchant_rate'] = user_merchant.user_merchant_buy_total.astype('float') / user_merchant.user_merchant_any.astype('float')
    user_merchant['user_merchant_common_buy_rate'] = user_merchant.user_merchant_buy_common.astype('float') / user_merchant.user_merchant_buy_total.astype('float')
    return user_merchant
    
def get_leakage_feature(dataset):
    #提取穿越特征
    t = dataset[['user_id']].copy()
    t['this_month_user_receive_all_coupon_count'] = 1
    t = t.groupby('user_id').agg('sum').reset_index()
    
    t1 = dataset[['user_id','coupon_id']].copy()
    t1['this_month_user_receive_same_coupn_count'] = 1
    t1 = t1.groupby(['user_id','coupon_id']).agg('sum').reset_index()
        
    t2 = dataset[['user_id','coupon_id','date_received']].copy()
    t2.date_received = t2.date_received.astype('str')
    #如果出现相同的用户接收相同的优惠券在接收时间上用‘：’连接上第n次接受优惠券的时间
    t2 = t2.groupby(['user_id','coupon_id'])['date_received'].agg(lambda x:':'.join(x)).reset_index()
    #将接收时间的一组按着':'分开，这样就可以计算接受了优惠券的数量,apply是合并
    t2['receive_number'] = t2.date_received.apply(lambda s:len(s.split(':')))
    t2 = t2[t2.receive_number > 1]
    #最大接受的日期
    t2['max_date_received'] = t2.date_received.apply(lambda s:max([int(d) for d in s.split(':')]))
    #最小的接收日期
    t2['min_date_received'] = t2.date_received.apply(lambda s:min([int(d) for d in s.split(':')]))
    t2 = t2[['user_id','coupon_id','max_date_received','min_date_received']]

    t3 = dataset[['user_id','coupon_id','date_received']]
    #将两表融合只保留左表数据,这样得到的表，相当于保留了最近接收时间和最远接受时间
    t3 = pd.merge(t3,t2,on=['user_id','coupon_id'],how='left')
    #这个优惠券最近接受时间
    t3['this_month_user_receive_same_coupon_lastone']= t3.max_date_received-t3.date_received.astype(int)
    #这个优惠券最远接受时间
    t3['this_month_user_receive_same_coupon_firstone'] = t3.date_received.astype(int)-t3.min_date_received
    
    t3.this_month_user_receive_same_coupon_lastone = t3.this_month_user_receive_same_coupon_lastone.apply(is_firstlastone)
    t3.this_month_user_receive_same_coupon_firstone = t3.this_month_user_receive_same_coupon_lastone.apply(is_firstlastone)
    t3 = t3[['user_id','coupon_id','date_received','this_month_user_receive_same_coupon_lastone','this_month_user_receive_same_coupon_firstone']]
       
    #提取第四个特征,一个用户所接收到的所有优惠券的数量
    t4 = dataset[['user_id','date_received']].copy()
    t4['this_day_receive_all_coupon_count'] = 1
    t4 = t4.groupby(['user_id','date_received']).agg('sum').reset_index()

    #提取第五个特征,一个用户不同时间所接收到不同优惠券的数量
    t5 = dataset[['user_id','coupon_id','date_received']].copy()
    t5['this_day_user_receive_same_coupon_count'] = 1
    t5 = t5.groupby(['user_id','coupon_id','date_received']).agg('sum').reset_index()
    
    #一个用户不同优惠券 的接受时间
    t6 = dataset[['user_id','coupon_id','date_received']].copy()
    t6.date_received = t6.date_received.astype('str')
    t6 = t6.groupby(['user_id','coupon_id'])['date_received'].agg(lambda x:':'.join(x)).reset_index()
    t6.rename(columns={'date_received':'dates'},inplace = True)
    
    t7 = dataset[['user_id','coupon_id','date_received']]
    t7 = pd.merge(t7,t6,on=['user_id','coupon_id'],how='left')
    t7['date_received_date'] = t7.date_received.astype('str')+'-'+t7.dates
    t7['day_gap_before'] = t7.date_received_date.apply(get_day_gap_before)
    t7['day_gap_after'] = t7.date_received_date.apply(get_day_gap_after)
    t7 = t7[['user_id','coupon_id','date_received','day_gap_before','day_gap_after']]
    
    other_feature = pd.merge(t1,t,on='user_id')
    other_feature = pd.merge(other_feature,t3,on=['user_id','coupon_id'])
    other_feature = pd.merge(other_feature,t4,on=['user_id','date_received'])
    other_feature = pd.merge(other_feature,t5,on=['user_id','coupon_id','date_received'])
    other_feature = pd.merge(other_feature,t7,on=['user_id','coupon_id','date_received'])
    return other_feature


""" 不同版本的函数 """
def f1(dataset,if_train):
    #特征1只有最基础的特征
    result=add_discount(dataset) 
    result.drop_duplicates(inplace=True)
    if if_train:
        result=add_label(result)
    return result

def f2(dataset,feature,if_train):
    #特征2增加Merchant,user特征   
    result=add_discount(dataset)
    
    merchant_feature=get_merchant_feature(feature)
    result=result.merge(merchant_feature, on='merchant_id', how="left")
    
    user_feature=get_user_feature(feature)
    result=result.merge(user_feature, on='user_id', how="left")
    
    user_merchant=get_user_merchant_feature(feature)
    result=result.merge(user_merchant, on=['user_id','merchant_id'], how="left")
    
    result.drop_duplicates(inplace=True)
    
    if if_train:
        result=add_label(result)
     
    return result

def f3(dataset,feature,if_train):
    #特征3增加leakage特征
       
    result=add_discount(dataset)
    
    merchant_feature=get_merchant_feature(feature)
    result=result.merge(merchant_feature, on='merchant_id', how="left")
    
    user_feature=get_user_feature(feature)
    result=result.merge(user_feature, on='user_id', how="left")
    
    user_merchant=get_user_merchant_feature(feature)
    result=result.merge(user_merchant, on=['user_id','merchant_id'], how="left")
    
    leakage_feature=get_leakage_feature(dataset)
    result=result.merge(leakage_feature, on=['user_id','coupon_id','date_received'],how='left')
    
    result.drop_duplicates(inplace=True)
    if if_train:
        result=add_label(result)
     
    return result
    
""" 特征集成及输出 """
def normal_feature_generate(feature_function):
    # 生成不滑窗的特征
    # 特征名：训练集：train_版本函数，测试集:test_版本函数
    off_train = pd.read_csv(datapath + 'ccf_offline_stage1_train.csv', header=0, keep_default_na=False)
    off_train.columns=['user_id','merchant_id','coupon_id','discount_rate','distance','date_received','date']

    off_test = pd.read_csv(datapath+'ccf_offline_stage1_test_revised.csv',header=0,keep_default_na=False)
    off_test.columns = ['user_id','merchant_id','coupon_id','discount_rate','distance','date_received']
    
    #取时间大于'20160501'是为了数据量少点，模型算的快一点，如果时间够的话，可以不加这个限制
    off_train = off_train[(off_train.coupon_id!='null')&(off_train.date_received!='null')&(off_train.date_received>='20160501')]
   
    dftrain=feature_function(off_train,True)
    
    dftest=feature_function(off_test,False)
    
    dftrain.drop(['date'], axis=1, inplace=True)
    dftrain.drop(['merchant_id'], axis=1, inplace=True)
    dftest.drop(['merchant_id'], axis=1, inplace=True)
    
    #输出特征
    print('输出特征')
    dftrain.to_csv(featurepath+'train_'+feature_function.__name__+'.csv',index=False,sep=',')
    dftest.to_csv(featurepath+'test_'+feature_function.__name__+'.csv',index=False,sep=',')

def slide_feature_generate(feature_function):
    #生成滑窗特征
    # 特征名：训练集：train_s版本函数，测试集:test_s版本函数, s是slide滑窗的意思
    off_train = pd.read_csv(datapath+'ccf_offline_stage1_train.csv',header=0,keep_default_na=False)
    off_train.columns=['user_id','merchant_id','coupon_id','discount_rate','distance','date_received','date']

    off_test = pd.read_csv(datapath+'ccf_offline_stage1_test_revised.csv',header=0,keep_default_na=False)
    off_test.columns = ['user_id','merchant_id','coupon_id','discount_rate','distance','date_received']
    
    #交叉训练集一：收到券的日期大于4月14日和小于5月14日
    dataset1 = off_train[(off_train.date_received>='201604014')&(off_train.date_received<='20160514')]
    #交叉训练集一特征：线下数据中领券和用券日期大于1月1日和小于4月13日
    feature1 = off_train[(off_train.date>='20160101')&(off_train.date<='20160413')|((off_train.date=='null')&(off_train.date_received>='20160101')&(off_train.date_received<='20160413'))]
    
    #交叉训练集二：收到券的日期大于5月15日和小于6月15日
    dataset2 = off_train[(off_train.date_received>='20160515')&(off_train.date_received<='20160615')]
    #交叉训练集二特征：线下数据中领券和用券日期大于2月1日和小于5月14日
    feature2 = off_train[(off_train.date>='20160201')&(off_train.date<='20160514')|((off_train.date=='null')&(off_train.date_received>='20160201')&(off_train.date_received<='20160514'))]
    
    #测试集
    dataset3 = off_test
    #测试集特征 :线下数据中领券和用券日期大于3月15日和小于6月30日的
    feature3 = off_train[((off_train.date>='20160315')&(off_train.date<='20160630'))|((off_train.date=='null')&(off_train.date_received>='20160315')&(off_train.date_received<='20160630'))]
    
    dftrain1=feature_function(dataset1,feature1,True)
    dftrain2=feature_function(dataset2,feature2,True)
    dftrain=pd.concat([dftrain1,dftrain2],axis=0)
       
    dftest=feature_function(dataset3,feature3,False)
    
    dftrain.drop(['date'], axis=1, inplace=True)
    dftrain.drop(['merchant_id'], axis=1, inplace=True)
    dftest.drop(['merchant_id'], axis=1, inplace=True)
    
    #输出特征
    print('输出特征')
    dftrain.to_csv(featurepath+'train_s'+feature_function.__name__+'.csv',index=False,sep=',')
    dftest.to_csv(featurepath+'test_s'+feature_function.__name__+'.csv',index=False,sep=',')

""" 特征读取 """
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

#%%
if __name__ == "__main__":
    # #重新读取数据，将null原样保持，方便处理
    # off_train = pd.read_csv('../data/ccf_offline_stage1_train.csv',keep_default_na=False)
    # off_train.columns=['user_id','merchant_id','coupon_id','discount_rate','distance','date_received','date']

    # off_test = pd.read_csv('../data/ccf_offline_stage1_test_revised.csv',keep_default_na=False)
    # off_test.columns = ['user_id','merchant_id','coupon_id','discount_rate','distance','date_received']

    # on_train = pd.read_csv('../data/ccf_online_stage1_train.csv',keep_default_na=False)
    # on_train.columns = ['user_id','merchant_id','action','coupon_id','discount_rate','date_received','date']

    # off_train[['user_id','merchant_id','coupon_id']] = off_train[['user_id','merchant_id','coupon_id']].astype(str)
    # off_test[['user_id','merchant_id','coupon_id']] = off_test[['user_id','merchant_id','coupon_id']].astype(str)
    # on_train[['user_id','merchant_id','coupon_id']] = on_train[['user_id','merchant_id','coupon_id']].astype(str)

    # #拷贝数据，免得调试的时候重读文件
    # dftrain = off_train.copy()
    # dftest = off_test.copy()

    # dftrain = add_feature(dftrain)
    # dftrain = add_label(dftrain)
    # dftest = add_feature(dftest)
    

    # plot_qq_with_subfigures(
    #     dftrain[(dftrain.label>=0)&(dftrain.distance>=0)], 
    #     column='distance');

    # plot_qq_with_subfigures(
    #     dftrain[(dftrain.label>=0)&(dftrain.distance>=0)], 
    #     column='discount_rate');

    """ 特征工程 """
    #f1
    normal_feature_generate(f1)
    #sf2
    slide_feature_generate(f2)
    #sf3
    slide_feature_generate(f3)    

# %%
