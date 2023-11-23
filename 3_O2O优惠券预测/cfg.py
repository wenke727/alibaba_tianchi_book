fd_seperator = ":"

datapath = '../data/' 
featurepath = './data/feature/' 
resultpath = './data/result/'
tmppath = './data/tmp/'
scorepath = './data/score/'

target_col_name = 'label'
id_col_names = ['user_id', 'coupon_id', 'date_received']
id_target_cols = ['user_id','coupon_id','date_received','label']
myeval='roc_auc'
cvscore = 0