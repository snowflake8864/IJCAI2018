# -*- coding:utf-8 -*-
import pandas as pd
import numpy as np
import datetime
import scipy as sp
import xgboost as xgb
from sklearn.model_selection import train_test_split


# 评价函数
def logloss(act, pred):
    epsilon = 1e-15
    pred = sp.maximum(epsilon, pred)
    pred = sp.minimum(1-epsilon, pred)
    ll = sum(act*sp.log(pred) + sp.subtract(1, act)*sp.log(sp.subtract(1, pred)))
    ll = ll * -1.0/len(act)
    return ll


# 转换UNIX时间为正常时间
def timestamp2datetime(timestamp):
    dt = datetime.datetime.utcfromtimestamp(timestamp) + datetime.timedelta(hours=8)
    return dt.strftime('%Y-%m-%d %H:%M:%S')

'''
# df训练集的数据量：478138 日期：2018-09-18~2018-09-24
# df_test的预测量：18371 日期：2018-09-25
'''
dir = 'data/oria/'
df = pd.read_table(dir + 'round1_ijcai_18_train_20180301.txt', engine='python', sep=" ")
df_test = pd.read_table(dir + 'round1_ijcai_18_test_a_20180301.txt', engine='python', sep=" ")

# 转换df的时间属性
date_list = df['context_timestamp'].values
date_lste = []
for dates in date_list:
    date_lste.append(timestamp2datetime(dates))
df['riqi'] = date_lste


# 转换df_test的时间属性
date_li = df_test['context_timestamp'].values
date_ls = []
for dates in date_li:
    date_ls.append(timestamp2datetime(dates))
df_test['riqi'] = date_ls

'''
# 划分训练集和测试集 训练集2018-09-18~2018-09-23 测试集2018-09-24

'''
train = df[(df.riqi >= '2018-09-18 00:00:00') & (df.riqi <= '2018-09-23 23:59:59')]
test = df[(df.riqi >= '2018-09-24 00:00:00') & (df.riqi <= '2018-09-24 23:59:59')]

df_train_data = train.drop(['is_trade'], axis=1)

# 先将不能用的数据剔除 item_category_list, item_property_list, predict_category_property, riqi
df_train_data = train.drop(['is_trade', 'item_category_list', 'item_property_list', 'predict_category_property', 'riqi'], axis=1)

df_train_target = train['is_trade']

train_x, test_x, train_y, test_y = train_test_split(df_train_data, df_train_target, test_size=0.2, random_state=502)

params = {'max_depth':8,
          'nthread':25,
          'eta':0.1,
          'eval_metric':'logloss',
          'objective':'binary:logistic',
          'subsample':0.7,
          'colsample_bytree':0.5,
          'silent':1,
          'seed':1123,
          'min_child_weight':10
          #'scale_pos_weight':0.5
          }
num_boost_round = 300

dtrain = xgb.DMatrix(train_x, train_y)
dvalid = xgb.DMatrix(test_x, test_y)

watchlist = [(dtrain, 'train'), (dvalid, 'eval')]
gbm = xgb.train(params, dtrain, num_boost_round, evals=watchlist,
                early_stopping_rounds=200, verbose_eval=True)

res = gbm.predict(xgb.DMatrix(test_x))
print '本地cv:'
print logloss(test_y, res)
#  线下成绩： 0.0885740627643

'''
测试文件的生成
'''
df_test = df_test.drop(['item_category_list', 'item_property_list', 'predict_category_property', 'riqi'], axis=1)
answer = gbm.predict(xgb.DMatrix(df_test))

pd_result = pd.DataFrame({'instance_id': df_test["instance_id"], 'predicted_score': answer})
pd_result.to_csv('result_xgb_v1.txt', index=False, sep=' ', columns={'instance_id', 'predicted_score'})
print '完成训练'
