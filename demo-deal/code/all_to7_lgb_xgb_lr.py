# coding: UTF-8
import pandas as pd
import time
from xgboost import XGBClassifier
import xgboost as xgb
import numpy as np
from datetime import datetime,timedelta
import pickle,os
from dateutil.parser import parse
from sklearn.utils import shuffle
from sklearn.metrics import roc_auc_score
import lightgbm as lgb 
import datetime
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

data_train=pd.read_csv('../data/round2_ijcai_18_train_20180425.txt',delim_whitespace=True)
#data_test_a=pd.read_csv('/test_a_trade_250.txt',delim_whitespace=True)
#data_test_a['is_trade'] = 0
#data_train=pd.concat([data_train,data_test_a]).reset_index(drop=True)
#data_train=data_train.drop_duplicates(['instance_id'])

data_train['item_category_1'],data_train['item_category_2'],data_train['item_category_3']=data_train['item_category_list'].str.split(';',2).str
del data_train['item_category_list']
data_train['predict_category_property_A'],data_train['predict_category_property_B'],data_train['predict_category_property_C']=data_train['predict_category_property'].str.split(';',2).str
del data_train['predict_category_property']
data_train['predict_category_property_A'],data_train['predict_category_property_A_1']=data_train['predict_category_property_A'].str.split(':',1).str
data_train['predict_category_property_A_1'],data_train['predict_category_property_A_2'],data_train['predict_category_property_A_3']=data_train['predict_category_property_A_1'].str.split(',',2).str
del data_train['predict_category_property_A_3']

data_train['predict_category_property_B'],data_train['predict_category_property_B_1']=data_train['predict_category_property_B'].str.split(':',1).str
data_train['predict_category_property_B_1'],data_train['predict_category_property_B_2'],data_train['predict_category_property_B_3']=data_train['predict_category_property_B_1'].str.split(',',2).str
del data_train['predict_category_property_B_3']

data_train['predict_category_property_C'],data_train['predict_category_property_C_1']=data_train['predict_category_property_C'].str.split(':',1).str
data_train['predict_category_property_C_1'],data_train['predict_category_property_C_2'],data_train['predict_category_property_C_3']=data_train['predict_category_property_C_1'].str.split(',',2).str
data_train['predict_category_property_C_1'],data_train['predict_category_property_C_3']=data_train['predict_category_property_C_1'].str.split(';',1).str
data_train['predict_category_property_C_2'],data_train['predict_category_property_C_3']=data_train['predict_category_property_C_2'].str.split(';',1).str
del data_train['predict_category_property_C_3']
#del data_train['predict_category_property_C']

data_train['item_property_list_1'],data_train['item_property_list_2'],data_train['item_property_list_3'],data_train['item_property_list_4']=data_train['item_property_list'].str.split(';',3).str
del data_train['item_property_list_4']
del data_train['item_property_list']
del data_train['item_category_3']
#data_train=data_train.fillna(-1)

##处理类目类特征，将类目类特征分为一列一列
#data_test=pd.read_csv('../data/round2_ijcai_18_test_a_20180425.txt',delim_whitespace=True)
print('load data_test...')
data_test_b=pd.read_csv('../data/round2_ijcai_18_test_b_20180510.txt',delim_whitespace=True)
print('data_test_b:%d' %(len(data_test_b)))
data_test_a=pd.read_csv('../data/round2_ijcai_18_test_a_20180425.txt',delim_whitespace=True)
print('data_test_a:%d' %(len(data_test_a)))
data_test=pd.concat([data_test_b,data_test_a]).reset_index(drop=True)


data_test['item_category_1'],data_test['item_category_2'],data_test['item_category_3']=data_test['item_category_list'].str.split(';',2).str
del data_test['item_category_list']
data_test['predict_category_property_A'],data_test['predict_category_property_B'],data_test['predict_category_property_C']=data_test['predict_category_property'].str.split(';',2).str
del data_test['predict_category_property']

data_test['predict_category_property_A'],data_test['predict_category_property_A_1']=data_test['predict_category_property_A'].str.split(':',1).str
data_test['predict_category_property_A_1'],data_test['predict_category_property_A_2'],data_test['predict_category_property_A_3']=data_test['predict_category_property_A_1'].str.split(',',2).str
del data_test['predict_category_property_A_3']

data_test['predict_category_property_B'],data_test['predict_category_property_B_1']=data_test['predict_category_property_B'].str.split(':',1).str
data_test['predict_category_property_B_1'],data_test['predict_category_property_B_2'],data_test['predict_category_property_B_3']=data_test['predict_category_property_B_1'].str.split(',',2).str
del data_test['predict_category_property_B_3']

data_test['predict_category_property_C'],data_test['predict_category_property_C_1']=data_test['predict_category_property_C'].str.split(':',1).str
data_test['predict_category_property_C_1'],data_test['predict_category_property_C_2'],data_test['predict_category_property_C_3']=data_test['predict_category_property_C_1'].str.split(',',2).str
data_test['predict_category_property_C_1'],data_test['predict_category_property_C_3']=data_test['predict_category_property_C_1'].str.split(';',1).str
data_test['predict_category_property_C_2'],data_test['predict_category_property_C_3']=data_test['predict_category_property_C_2'].str.split(';',1).str
del data_test['predict_category_property_C_3']
#del data_test['predict_category_property_C']

data_test['item_property_list_1'],data_test['item_property_list_2'],data_test['item_property_list_3'],data_test['item_property_list_4']=data_test['item_property_list'].str.split(';',3).str
del data_test['item_property_list_4']
del data_test['item_property_list']

data_train.to_csv('../data/data_train_ori.csv')
data_test.to_csv('../data/data_test_ori.csv')

print('load_data_ori...')
data_train=pd.read_csv('../data/data_train_ori_b.csv')
print(len(data_train))
#data_train['times']=data_train['times'].astype(str)
data_test=pd.read_csv('../data/data_test_ori_b.csv')
print(len(data_test))
#data_test['times']=data_test['times'].astype(str)

def convert_data(data):
    data["times"] = data["context_timestamp"].apply(lambda x: datetime.datetime.fromtimestamp(x))
    data["day"] = data["times"].apply(lambda x: x.day)
    data["hour"] = data["times"].apply(lambda x: x.hour)
    data['min'] = data['times'].apply(lambda x: x.minute)
    data['day']=data['day'].astype('int')
    data['hour']=data['hour'].astype('int')
    data['min']=data['min'].astype('int')
    
    # 小时均值特征
    grouped = data.groupby('user_id')['hour'].mean().reset_index()
    grouped.columns = ['user_id', 'user_mean_hour']
    data = data.merge(grouped, how='left', on='user_id')
    grouped = data.groupby('item_id')['hour'].mean().reset_index()
    grouped.columns = ['item_id', 'item_mean_hour']
    data = data.merge(grouped, how='left', on='item_id')
    grouped = data.groupby('shop_id')['hour'].mean().reset_index()
    grouped.columns = ['shop_id', 'shop_mean_hour']
    data = data.merge(grouped, how='left', on='shop_id')
    grouped = data.groupby('item_city_id')['hour'].mean().reset_index()
    grouped.columns = ['item_city_id', 'city_mean_hour']
    data = data.merge(grouped, how='left', on='item_city_id')
    grouped = data.groupby('item_brand_id')['hour'].mean().reset_index()
    grouped.columns = ['item_brand_id', 'brand_mean_hour']
    data = data.merge(grouped, how='left', on='item_brand_id')
    # 小时var特征
    '''
    grouped = data.groupby('user_id')['hour'].var().reset_index()
    grouped.columns = ['user_id', 'user_var_hour']
    data = data.merge(grouped, how='left', on='user_id')
    grouped = data.groupby('item_id')['hour'].var().reset_index()
    grouped.columns = ['item_id', 'item_var_hour']
    data = data.merge(grouped, how='left', on='item_id')
    grouped = data.groupby('shop_id')['hour'].var().reset_index()
    grouped.columns = ['shop_id', 'shop_var_hour']
    data = data.merge(grouped, how='left', on='shop_id')
    grouped = data.groupby('item_city_id')['hour'].var().reset_index()
    grouped.columns = ['item_city_id', 'city_var_hour']
    data = data.merge(grouped, how='left', on='item_city_id')
    grouped = data.groupby('item_brand_id')['hour'].var().reset_index()
    grouped.columns = ['item_brand_id', 'brand_var_hour']
    data = data.merge(grouped, how='left', on='item_brand_id')
    '''
    #天均值特征
    grouped = data.groupby('user_id')['day'].mean().reset_index()
    grouped.columns = ['user_id', 'user_mean_day']
    data = data.merge(grouped, how='left', on='user_id')
    grouped = data.groupby('item_id')['day'].mean().reset_index()
    grouped.columns = ['item_id', 'item_mean_day']
    data = data.merge(grouped, how='left', on='item_id')
    grouped = data.groupby('shop_id')['day'].mean().reset_index()
    grouped.columns = ['shop_id', 'shop_mean_day']
    data = data.merge(grouped, how='left', on='shop_id')
    grouped = data.groupby('item_city_id')['day'].mean().reset_index()
    grouped.columns = ['item_city_id', 'city_mean_day']
    data = data.merge(grouped, how='left', on='item_city_id')
    grouped = data.groupby('item_brand_id')['day'].mean().reset_index()
    grouped.columns = ['item_brand_id', 'brand_mean_day']
    data = data.merge(grouped, how='left', on='item_brand_id')
    
    #天var特征
    grouped = data.groupby('user_id')['day'].var().reset_index()
    grouped.columns = ['user_id', 'user_var_day']
    data = data.merge(grouped, how='left', on='user_id')
    grouped = data.groupby('item_id')['day'].var().reset_index()
    grouped.columns = ['item_id', 'item_var_day']
    data = data.merge(grouped, how='left', on='item_id')
    grouped = data.groupby('shop_id')['day'].var().reset_index()
    grouped.columns = ['shop_id', 'shop_var_day']
    data = data.merge(grouped, how='left', on='shop_id')
    grouped = data.groupby('item_city_id')['day'].var().reset_index()
    grouped.columns = ['item_city_id', 'city_var_day']
    data = data.merge(grouped, how='left', on='item_city_id')
    grouped = data.groupby('item_brand_id')['day'].var().reset_index()
    grouped.columns = ['item_brand_id', 'brand_var_day']
    data = data.merge(grouped, how='left', on='item_brand_id')

    #天小时均值特征
    grouped = data.groupby(['user_id', 'day'])['hour'].mean().reset_index()
    grouped.columns = ['user_id', 'day', 'user_mean_day_hour']
    data = data.merge(grouped, how='left', on=['user_id', 'day'])
    grouped = data.groupby(['item_id', 'day'])['hour'].mean().reset_index()
    grouped.columns = ['item_id', 'day', 'item_mean_day_hour']
    data = data.merge(grouped, how='left', on=['item_id', 'day'])
    grouped = data.groupby(['shop_id', 'day'])['hour'].mean().reset_index()
    grouped.columns = ['shop_id', 'day', 'shop_mean_day_hour']
    data = data.merge(grouped, how='left', on=['shop_id', 'day'])
    grouped = data.groupby(['item_city_id', 'day'])['hour'].mean().reset_index()
    grouped.columns = ['item_city_id', 'day', 'city_mean_day_hour']
    data = data.merge(grouped, how='left', on=['item_city_id', 'day'])
    grouped = data.groupby(['item_brand_id', 'day'])['hour'].mean().reset_index()
    grouped.columns = ['item_brand_id', 'day', 'brand_mean_day_hour']
    data = data.merge(grouped, how='left', on=['item_brand_id', 'day'])
    
    #天小时var特征
    grouped = data.groupby(['user_id', 'day'])['hour'].var().reset_index()
    grouped.columns = ['user_id', 'day', 'user_var_day_hour']
    data = data.merge(grouped, how='left', on=['user_id', 'day'])
    grouped = data.groupby(['item_id', 'day'])['hour'].var().reset_index()
    grouped.columns = ['item_id', 'day', 'item_var_day_hour']
    data = data.merge(grouped, how='left', on=['item_id', 'day'])
    grouped = data.groupby(['shop_id', 'day'])['hour'].var().reset_index()
    grouped.columns = ['shop_id', 'day', 'shop_var_day_hour']
    data = data.merge(grouped, how='left', on=['shop_id', 'day'])
    grouped = data.groupby(['item_city_id', 'day'])['hour'].var().reset_index()
    grouped.columns = ['item_city_id', 'day', 'city_var_day_hour']
    data = data.merge(grouped, how='left', on=['item_city_id', 'day'])
    grouped = data.groupby(['item_brand_id', 'day'])['hour'].var().reset_index()
    grouped.columns = ['item_brand_id', 'day', 'brand_var_day_hour']
    data = data.merge(grouped, how='left', on=['item_brand_id', 'day'])

    return data


data_train["times"] = data_train["context_timestamp"].apply(lambda x: datetime.datetime.fromtimestamp(x))
data_train["day"] = data_train["times"].apply(lambda x: x.day)

data_test['is_trade'] = 0
print(len(data_test))

train_data = pd.concat([data_train,data_test],).reset_index(drop=True)
del data_train
del data_test

train_data["datetime"] = train_data["context_timestamp"].apply(lambda x: datetime.datetime.fromtimestamp(x))
train_data["day"] = train_data["datetime"].apply(lambda x: x.day)
train_data["hour"] = train_data["datetime"].apply(lambda x: x.hour)

##按时间排序
train_data = train_data.sort_values('context_timestamp')
train_data["item_category_list"] = train_data["item_category_list"].apply(lambda x: x.split(";"))
train_data["item_property_list"] = train_data["item_property_list"].apply(lambda x: x.split(";"))

categories = train_data["predict_category_property"].apply(lambda x: x.split(";"))
train_data["num_query_cat"] = categories.apply(lambda x: len(x))
for i in range(categories.apply(lambda x: len(x)).max()):
    train_data["category_"+str(i)] = categories.apply(lambda x: x[i].split(":")[0] if len(x)>i else "-1")
    train_data["category_"+str(i)+"_props"] = categories.apply(lambda x: x[i].split(":")[1].split(",") if len(x)>i and x[i].split(":")[0] != "-1" else ["-1"])

#start_time='25'    
#end_time='26'

#train_data=train_data[(train_data.day >= int(start_time)) & (train_data.day <int(end_time))]


train_data["num_item_category"] = train_data["item_category_list"].apply(lambda x: len(x)-x.count("-1"))
train_data["num_item_property"] = train_data["item_property_list"].apply(lambda x: len(x)-x.count("-1")) 


#####################################
##统计各类别在此次出现前的count数
def count_cat_prep(df,column,newcolumn):
    count_dict = {}
    df[newcolumn] = 0
    data = df[[column,newcolumn]].values
    for cat_list in data:
        if cat_list[0] not in count_dict:
            count_dict[cat_list[0]] = 0
            cat_list[1] = 0
        else:
            count_dict[cat_list[0]] += 1
            cat_list[1] = count_dict[cat_list[0]]
    df[[column,newcolumn]] = data

train_data['user_item_id'] = train_data['user_id'].astype(str)+"_"+train_data['item_id'].astype(str)
train_data['user_shop_id'] = train_data['user_id'].astype(str)+"_"+train_data['shop_id'].astype(str)
train_data['user_brand_id'] = train_data['user_id'].astype(str)+"_"+train_data['item_brand_id'].astype(str)
train_data['item_category'] = train_data['item_category_list'].astype(str)
train_data['user_category_id'] = train_data['user_id'].astype(str)+"_"+train_data['item_category'].astype(str)
train_data['user_context_id'] = train_data['user_id'].astype(str)+"_"+train_data['context_id'].astype(str)
train_data['user_city_id'] = train_data['user_id'].astype(str)+"_"+train_data['item_city_id'].astype(str)
##统计各类别在总样本中的count数
for column in ['user_id','item_id','item_brand_id','shop_id','user_item_id','user_shop_id','user_brand_id','user_category_id','context_id','item_city_id','user_context_id','user_city_id']:
    count_cat_prep(train_data,column,column+'_click_count_prep')
    
for column in ['user_id','item_id','item_brand_id','shop_id','user_item_id','user_shop_id','user_brand_id','user_category_id','context_id','item_city_id','user_context_id','user_city_id']:
    train_data = train_data.join(train_data[column].value_counts(),on = column ,rsuffix = '_count')
  
print('gen_gaptime ...')  
##前一次或后一次点击与现在的时间差（trick）

def lasttime_delta(column):    
    train_data[column+'_lasttime_delta'] = 0
    data = train_data[['context_timestamp',column,column+'_lasttime_delta']].values
    lasttime_dict = {}
    for df_list in data:
        if df_list[1] not in lasttime_dict:
            df_list[2] = -1
            lasttime_dict[df_list[1]] = df_list[0]
        else:
            df_list[2] = df_list[0] - lasttime_dict[df_list[1]]
            lasttime_dict[df_list[1]] = df_list[0]
    train_data[['context_timestamp',column,column+'_lasttime_delta']] = data

def nexttime_delta(column):    
    train_data[column+'_nexttime_delta'] = 0
    data = train_data[['context_timestamp',column,column+'_nexttime_delta']].values
    nexttime_dict = {}
    for df_list in data:
        if df_list[1] not in nexttime_dict:
            df_list[2] = -1
            nexttime_dict[df_list[1]] = df_list[0]
        else:
            df_list[2] = nexttime_dict[df_list[1]] - df_list[0]
            nexttime_dict[df_list[1]] = df_list[0]
    train_data[['context_timestamp',column,column+'_nexttime_delta']]= data

for column in ['user_id','item_id','item_brand_id','shop_id','user_item_id','user_shop_id','user_brand_id','user_category_id','context_id','item_city_id','user_context_id','user_city_id']:
    lasttime_delta(column)
    
train_data = train_data.sort_values('context_timestamp',ascending=False)

for column in ['user_id','item_id','item_brand_id','shop_id','user_item_id','user_shop_id','user_brand_id','user_category_id','context_id','item_city_id','user_context_id','user_city_id']:
    nexttime_delta(column)
    
train_data = train_data.sort_values('context_timestamp')


a_prep=train_data[['instance_id','user_id_click_count_prep','item_id_click_count_prep','item_brand_id_click_count_prep','shop_id_click_count_prep','user_item_id_click_count_prep','user_shop_id_click_count_prep',
                   'user_brand_id_click_count_prep','user_category_id_click_count_prep','context_id_click_count_prep','item_city_id_click_count_prep','user_context_id_click_count_prep','user_city_id_click_count_prep']]
a_count=train_data[['instance_id','user_id_count','item_id_count','item_brand_id_count','shop_id_count','user_item_id_count','user_shop_id_count','user_brand_id_count','user_category_id_count',
                    'context_id_count','item_city_id_count','user_context_id_count','user_city_id_count']]

a_instance=train_data['instance_id']
a_gap_time=train_data[['user_id_lasttime_delta','item_id_lasttime_delta','item_brand_id_lasttime_delta','shop_id_lasttime_delta','user_item_id_lasttime_delta',
'user_shop_id_lasttime_delta','user_brand_id_lasttime_delta','user_category_id_lasttime_delta','context_id_lasttime_delta','item_city_id_lasttime_delta',
'user_context_id_lasttime_delta','user_city_id_lasttime_delta',
'user_id_nexttime_delta','item_id_nexttime_delta','item_brand_id_nexttime_delta','shop_id_nexttime_delta','user_item_id_nexttime_delta',
'user_shop_id_nexttime_delta','user_brand_id_nexttime_delta','user_category_id_nexttime_delta','context_id_nexttime_delta','item_city_id_nexttime_delta',
'user_context_id_nexttime_delta','user_city_id_nexttime_delta',
]]


a_gap_time=pd.concat([a_instance,a_gap_time],axis=1)
a_prep=a_prep.sort_index()
a_count=a_count.sort_index()
a_gap_time=a_gap_time.sort_index()

print('gen_trade ...')
a_instance=train_data['instance_id']
##统计各类别的trade转化数
def trade_prep_count(column):
    train_data[column+'_trade_prep_count'] = 0
    for day in train_data.day.unique():
        print('day:%d' %(day))
        if day == 31:
            train_data.loc[train_data.day==day,column+'_trade_prep_count'] = -1
        else:
            trade_dict = train_data[train_data.day<day].groupby(column)['is_trade'].sum().to_dict()
            train_data.loc[train_data.day==day,column+'_trade_prep_count'] = train_data.loc[train_data.day==day,column].apply(lambda x: trade_dict[x] if x in trade_dict else 0)

for column in ['user_id','item_id','item_brand_id','shop_id','user_item_id','user_shop_id','user_brand_id','user_category_id','context_id','item_city_id','user_context_id','user_city_id']:
    trade_prep_count(column)
a_trade_prep_count=train_data[['user_id_trade_prep_count','item_id_trade_prep_count','item_brand_id_trade_prep_count','shop_id_trade_prep_count',
'user_item_id_trade_prep_count','user_shop_id_trade_prep_count','user_brand_id_trade_prep_count','user_category_id_trade_prep_count',
'context_id_trade_prep_count','item_city_id_trade_prep_count','user_context_id_trade_prep_count','user_city_id_trade_prep_count']]
a_trade_prep_count=pd.concat([a_instance,a_trade_prep_count],axis=1)
a_trade_prep_count=a_trade_prep_count.sort_index()

del a_prep['instance_id']
del a_count['instance_id']
del a_gap_time['instance_id']
del a_trade_prep_count['instance_id']
a_prep.to_csv('../data/other_feat/count_prep_b.csv',index=False)
a_count.to_csv('../data/other_feat/count_b.csv',index=False)
a_gap_time.to_csv('../data/other_feat/gap_time_b.csv',index=False)
a_trade_prep_count.to_csv('../data/other_feat/trade_prep_b.csv',index=False)

print('load other_feat...')
count_prep=pd.read_csv('../data/other_feat/count_prep_b.csv')
count=pd.read_csv('../data/other_feat/count_b.csv')
gap_time=pd.read_csv('../data/other_feat/gap_time_b.csv')
trade_prep_count=pd.read_csv('../data/other_feat/trade_prep_b.csv')
#count_after=pd.read_csv('../data/other_feat/count_after.csv')
del count_prep['instance_id']
del count['instance_id']
del gap_time['instance_id']
del trade_prep_count['instance_id']
'''

##清洗数据
def clean_data(data_test,data_train):
    '''data_test=data_test.drop_duplicates(['user_id'])
    data_test=data_test[['instance_id','user_id']]
    clean_train_data=pd.merge(data_test,data_train,how='left',on='user_id')
    del clean_train_data['instance_id_x']
    clean_train_data.rename(columns={'instance_id_y':'instance_id'},inplace=True)

    clean_train_data=clean_train_data.dropna(thresh=25)'''
    clean_train_data=data_train
    clean_train_data=clean_train_data.drop_duplicates(['instance_id'])

    return clean_train_data

##产生固定时间内的数据
def time_gap_data(start_time,end_time,data_test,data_train):
    clean_train_data=clean_data(data_test,data_train)
    time_data=clean_train_data[(clean_train_data.times >= start_time) & (clean_train_data.times<=end_time)]
    return time_data

##计算两时间间的小时数
def get_hours(start_time,end_time):
    d=parse(end_time)-parse(start_time)
    hours=int(d.days*24+d.seconds/3600)
    return hours

def convert_data(data):
    data["times"] = data["context_timestamp"].apply(lambda x: datetime.datetime.fromtimestamp(x))
    data["day"] = data["times"].apply(lambda x: x.day)
    data["hour"] = data["times"].apply(lambda x: x.hour)
    data['min'] = data['times'].apply(lambda x: x.minute)
    data['day']=data['day'].astype('int')
    data['hour']=data['hour'].astype('int')
    data['min']=data['min'].astype('int')
    data['tm_hour']=data['hour']+data['min']/60
    data['tm_hour_sin']=data['tm_hour'].map(lambda x:np.sin((x-12)/24*2*np.pi))
    data['tm_hour_cos']=data['tm_hour'].map(lambda x:np.cos((x-12)/24*2*np.pi))
    data_time=data[['user_id','day','hour','min']]
    
    # 小时均值特征,存在leak
    grouped = data.groupby('user_id')['hour'].mean().reset_index()
    grouped.columns = ['user_id', 'user_mean_hour']
    data = data.merge(grouped, how='left', on='user_id')
    grouped = data.groupby('item_id')['hour'].mean().reset_index()
    grouped.columns = ['item_id', 'item_mean_hour']
    data = data.merge(grouped, how='left', on='item_id')
    grouped = data.groupby('shop_id')['hour'].mean().reset_index()
    grouped.columns = ['shop_id', 'shop_mean_hour']
    data = data.merge(grouped, how='left', on='shop_id')
    grouped = data.groupby('item_city_id')['hour'].mean().reset_index()
    grouped.columns = ['item_city_id', 'city_mean_hour']
    data = data.merge(grouped, how='left', on='item_city_id')
    grouped = data.groupby('item_brand_id')['hour'].mean().reset_index()
    grouped.columns = ['item_brand_id', 'brand_mean_hour']
    data = data.merge(grouped, how='left', on='item_brand_id')
    #天均值特征
    grouped = data.groupby('user_id')['day'].mean().reset_index()
    grouped.columns = ['user_id', 'user_mean_day']
    data = data.merge(grouped, how='left', on='user_id')
    grouped = data.groupby('item_id')['day'].mean().reset_index()
    grouped.columns = ['item_id', 'item_mean_day']
    data = data.merge(grouped, how='left', on='item_id')
    grouped = data.groupby('shop_id')['day'].mean().reset_index()
    grouped.columns = ['shop_id', 'shop_mean_day']
    data = data.merge(grouped, how='left', on='shop_id')
    grouped = data.groupby('item_city_id')['day'].mean().reset_index()
    grouped.columns = ['item_city_id', 'city_mean_day']
    data = data.merge(grouped, how='left', on='item_city_id')
    grouped = data.groupby('item_brand_id')['day'].mean().reset_index()
    grouped.columns = ['item_brand_id', 'brand_mean_day']
    data = data.merge(grouped, how='left', on='item_brand_id')
    #天小时均值特征
    grouped = data.groupby(['user_id', 'day'])['hour'].mean().reset_index()
    grouped.columns = ['user_id', 'day', 'user_mean_day_hour']
    data = data.merge(grouped, how='left', on=['user_id', 'day'])
    grouped = data.groupby(['item_id', 'day'])['hour'].mean().reset_index()
    grouped.columns = ['item_id', 'day', 'item_mean_day_hour']
    data = data.merge(grouped, how='left', on=['item_id', 'day'])
    grouped = data.groupby(['shop_id', 'day'])['hour'].mean().reset_index()
    grouped.columns = ['shop_id', 'day', 'shop_mean_day_hour']
    data = data.merge(grouped, how='left', on=['shop_id', 'day'])
    grouped = data.groupby(['item_city_id', 'day'])['hour'].mean().reset_index()
    grouped.columns = ['item_city_id', 'day', 'city_mean_day_hour']
    data = data.merge(grouped, how='left', on=['item_city_id', 'day'])
    grouped = data.groupby(['item_brand_id', 'day'])['hour'].mean().reset_index()
    grouped.columns = ['item_brand_id', 'day', 'brand_mean_day_hour']
    data = data.merge(grouped, how='left', on=['item_brand_id', 'day'])
    
    '''
    data_price_day5=data[['item_id','item_price_level']][data.day==5].drop_duplicates('item_id').rename(columns={'item_price_level':'item_price_level_day5'})
    data_price_day6=data[['item_id','item_price_level']][data.day==6].drop_duplicates('item_id').rename(columns={'item_price_level':'item_price_level_day6'})
    data_price_day7=data[['item_id','item_price_level']][data.day==7].drop_duplicates('item_id').rename(columns={'item_price_level':'item_price_level_day7'})
    
    data_price_all=pd.merge(data_price_day7,data_price_day6,'left',on='item_id')
    data_price_all=pd.merge(data_price_all,data_price_day5,'left',on='item_id')
    data_price_all['price_gap_7_6']=(data_price_all['item_price_level_day6']-data_price_all['item_price_level_day7'])/data_price_all['item_price_level_day6']
    data_price_all['price_gap_7_5']=(data_price_all['item_price_level_day5']-data_price_all['item_price_level_day7'])/data_price_all['item_price_level_day5']
    data=pd.merge(data,data_price_all[['item_id','price_gap_7_6','price_gap_7_5']],'left',on='item_id')
    print(data['price_gap_7_6'].unique())
    print(data['price_gap_7_5'].unique())
    
    data_sales_day5=data[['item_id','item_sales_level']][data.day==5].drop_duplicates('item_id').rename(columns={'item_sales_level':'item_sales_level_day5'})
    data_sales_day6=data[['item_id','item_sales_level']][data.day==6].drop_duplicates('item_id').rename(columns={'item_sales_level':'item_sales_level_day6'})
    data_sales_day7=data[['item_id','item_sales_level']][data.day==7].drop_duplicates('item_id').rename(columns={'item_sales_level':'item_sales_level_day7'})
    
    data_sales_all=pd.merge(data_sales_day7,data_sales_day6,'left',on='item_id')
    data_sales_all=pd.merge(data_sales_all,data_sales_day5,'left',on='item_id')
    data_sales_all['sales_gap_7_6']=(data_sales_all['item_sales_level_day7']-data_sales_all['item_sales_level_day6'])/data_sales_all['item_sales_level_day7']
    data_sales_all['sales_gap_7_5']=(data_sales_all['item_sales_level_day7']-data_sales_all['item_sales_level_day5'])/data_sales_all['item_sales_level_day7']
    data=pd.merge(data,data_sales_all[['item_id','sales_gap_7_6','sales_gap_7_5']],'left',on='item_id')
    print(data['sales_gap_7_6'].unique())
    print(data['sales_gap_7_5'].unique())
    
    data_collected_day5=data[['item_id','item_collected_level']][data.day==5].drop_duplicates('item_id').rename(columns={'item_collected_level':'item_collected_level_day5'})
    data_collected_day6=data[['item_id','item_collected_level']][data.day==6].drop_duplicates('item_id').rename(columns={'item_collected_level':'item_collected_level_day6'})
    data_collected_day7=data[['item_id','item_collected_level']][data.day==7].drop_duplicates('item_id').rename(columns={'item_collected_level':'item_collected_level_day7'})
    
    data_collected_all=pd.merge(data_collected_day7,data_collected_day6,'left',on='item_id')
    data_collected_all=pd.merge(data_collected_all,data_collected_day5,'left',on='item_id')
    data_collected_all['collected_gap_7_6']=(data_collected_all['item_collected_level_day7']-data_collected_all['item_collected_level_day6'])/data_collected_all['item_collected_level_day7']
    data_collected_all['collected_gap_7_5']=(data_collected_all['item_collected_level_day7']-data_collected_all['item_collected_level_day5'])/data_collected_all['item_collected_level_day7']
    data=pd.merge(data,data_collected_all[['item_id','collected_gap_7_6','collected_gap_7_5']],'left',on='item_id')
    print(data['collected_gap_7_6'].unique())
    print(data['collected_gap_7_5'].unique())
    
    data_pv_day5=data[['item_id','item_pv_level']][data.day==5].drop_duplicates('item_id').rename(columns={'item_pv_level':'item_pv_level_day5'})
    data_pv_day6=data[['item_id','item_pv_level']][data.day==6].drop_duplicates('item_id').rename(columns={'item_pv_level':'item_pv_level_day6'})
    data_pv_day7=data[['item_id','item_pv_level']][data.day==7].drop_duplicates('item_id').rename(columns={'item_pv_level':'item_pv_level_day7'})
    
    data_pv_all=pd.merge(data_pv_day7,data_pv_day6,'left',on='item_id')
    data_pv_all=pd.merge(data_pv_all,data_pv_day5,'left',on='item_id')
    data_pv_all['pv_gap_7_6']=(data_pv_all['item_pv_level_day7']-data_pv_all['item_pv_level_day6'])/data_pv_all['item_pv_level_day7']
    data_pv_all['pv_gap_7_5']=(data_pv_all['item_pv_level_day7']-data_pv_all['item_pv_level_day5'])/data_pv_all['item_pv_level_day7']
    data=pd.merge(data,data_pv_all[['item_id','pv_gap_7_6','pv_gap_7_5']],'left',on='item_id')
    print(data['pv_gap_7_6'].unique())
    print(data['pv_gap_7_5'].unique())'''
    
    print('user_hour ...')
    #shop_time=data[['day','predict_category_property_A_1','hour','min']]
    user_query_day = data.groupby(['user_id', 'day']).size().reset_index().rename(columns={0: 'user_query_day'})    
    user_query_day_hour = data.groupby(['user_id', 'day', 'hour']).size().reset_index().rename(columns={0: 'user_query_day_hour'})   
    user_query_day_hour_min = data.groupby(['user_id', 'day', 'hour','min']).size().reset_index().rename(columns={0: 'user_query_day_hour_min'})
    #user_query_day_hour_min_sec = data.groupby(['user_id', 'day', 'hour','min','sec']).size().reset_index().rename(columns={0: 'user_query_day_hour_min_sec'})
    user_day_hourmin_mean= data_time.groupby(['user_id', 'day']).mean().reset_index().rename(columns={'hour': 'mean_hour','min':'mean_min'})
    user_day_hourmin_max= data_time.groupby(['user_id', 'day']).max().reset_index().rename(columns={'hour': 'max_hour','min':'max_min'})
    user_day_hourmin_min= data_time.groupby(['user_id', 'day']).min().reset_index().rename(columns={'hour': 'min_hour','min':'min_min'})

    item_day_click=data.groupby(['user_id', 'day','item_id']).size().reset_index().rename(columns={0: 'item_day_click'})  
    item_day_hour_click=data.groupby(['user_id', 'day','hour','item_id']).size().reset_index().rename(columns={0: 'item_day_hour_click'})    
    
    print('click ...')
    item_city_day_click=data.groupby(['user_id', 'day','item_city_id']).size().reset_index().rename(columns={0: 'item_city_day_click'})  
    item_city_day_hour_click=data.groupby(['user_id', 'day','hour','item_city_id']).size().reset_index().rename(columns={0: 'item_city_day_hour_click'})   
    item_category2_day_click=data.groupby(['user_id', 'day','item_category_2']).size().reset_index().rename(columns={0: 'item_category2_day_click'})  
    item_category2_day_hour_click=data.groupby(['user_id', 'day','hour','item_category_2']).size().reset_index().rename(columns={0: 'item_category2_day_hour_click'}) 
    item_brand_day_click=data.groupby(['user_id', 'day','item_brand_id']).size().reset_index().rename(columns={0: 'item_brand_day_click'})  
    item_brand_day_hour_click=data.groupby(['user_id', 'day','hour','item_brand_id']).size().reset_index().rename(columns={0: 'item_brand_day_hour_click'})   
    item_property_A_day_click=data.groupby(['user_id', 'day','predict_category_property_A']).size().reset_index().rename(columns={0: 'item_property_A_day_click'})  
    item_property_A_day_hour_click=data.groupby(['user_id', 'day','hour','predict_category_property_A']).size().reset_index().rename(columns={0: 'item_property_A_day_hour_click'})
    item_property_A_1_day_click=data.groupby(['user_id', 'day','predict_category_property_A_1']).size().reset_index().rename(columns={0: 'item_property_A_1_day_click'})  
    item_property_A_1_day_hour_click=data.groupby(['user_id', 'day','hour','predict_category_property_A_1']).size().reset_index().rename(columns={0: 'item_property_A_1_day_hour_click'})   
    item_property_B_day_click=data.groupby(['user_id', 'day','predict_category_property_B']).size().reset_index().rename(columns={0: 'item_property_B_day_click'})  
    item_property_B_day_hour_click=data.groupby(['user_id', 'day','hour','predict_category_property_B']).size().reset_index().rename(columns={0: 'item_property_B_day_hour_click'})   
    item_property_B_1_day_click=data.groupby(['user_id', 'day','predict_category_property_B_1']).size().reset_index().rename(columns={0: 'item_property_B_1_day_click'})  
    item_property_B_1_day_hour_click=data.groupby(['user_id', 'day','hour','predict_category_property_B_1']).size().reset_index().rename(columns={0: 'item_property_B_1_day_hour_click'})   
    item_property_C_day_click=data.groupby(['user_id', 'day','predict_category_property_C']).size().reset_index().rename(columns={0: 'item_property_C_day_click'})  
    item_property_C_day_hour_click=data.groupby(['user_id', 'day','hour','predict_category_property_C']).size().reset_index().rename(columns={0: 'item_property_C_day_hour_click'})   
    item_list_1_day_click=data.groupby(['user_id', 'day','item_property_list_1']).size().reset_index().rename(columns={0: 'item_list_1_day_click'})  
    item_list_1_day_hour_click=data.groupby(['user_id', 'day','hour','item_property_list_1']).size().reset_index().rename(columns={0: 'item_list_1_day_hour_click'})  
    item_list_2_day_click=data.groupby(['user_id', 'day','item_property_list_2']).size().reset_index().rename(columns={0: 'item_list_2_day_click'})  
    item_list_2_day_hour_click=data.groupby(['user_id', 'day','hour','item_property_list_2']).size().reset_index().rename(columns={0: 'item_list_2_day_hour_click'}) 
    item_list_3_day_click=data.groupby(['user_id', 'day','item_property_list_3']).size().reset_index().rename(columns={0: 'item_list_3_day_click'})  
    item_list_3_day_hour_click=data.groupby(['user_id', 'day','hour','item_property_list_3']).size().reset_index().rename(columns={0: 'item_list_3_day_hour_click'}) 
    
    print('count')
    '''
    shop_count=data.groupby(['shop_id']).size().reset_index().rename(columns={0: 'shop_count'})
    item_count=data.groupby(['item_id']).size().reset_index().rename(columns={0: 'item_count'})
    user_count=data.groupby(['user_id']).size().reset_index().rename(columns={0: 'user_count'})
    context_count=data.groupby(['context_id']).size().reset_index().rename(columns={0: 'context_count'})
    data = pd.merge(data, shop_count, 'left', on=['shop_id'])
    data = pd.merge(data, item_count, 'left', on=['item_id'])
    data = pd.merge(data, user_count, 'left', on=['user_id'])
    data = pd.merge(data, context_count, 'left', on=['context_id'])'''
    '''
    data_shop_day5=data[['shop_id','shop_count']][data.day==5].drop_duplicates('shop_id').rename(columns={'shop_count':'shop_count_day5'})
    data_shop_day6=data[['shop_id','shop_count']][data.day==6].drop_duplicates('shop_id').rename(columns={'shop_count':'shop_count_day6'})
    data_shop_day7=data[['shop_id','shop_count']][data.day==7].drop_duplicates('shop_id').rename(columns={'shop_count':'shop_count_day7'})
    
    data_shop_all=pd.merge(data_shop_day7,data_shop_day6,'left',on='shop_id')
    data_shop_all=pd.merge(data_shop_all,data_shop_day5,'left',on='shop_id')
    data_shop_all['shop_gap_7_6']=data_shop_all['shop_count_day7']-data_shop_all['shop_count_day6']
    data_shop_all['shop_gap_7_5']=data_shop_all['shop_count_day7']-data_shop_all['shop_count_day5']
    data=pd.merge(data,data_shop_all[['shop_id','shop_gap_7_6','shop_gap_7_5']],'left',on='shop_id')
    
    for c in columns_gap:
        if isinstance(c, (list, tuple)):
            pair = [cc for cc in c]
            c = '_'.join(c)
        else:
            pair = [c]
        
        p1=c
        p2=c+'_click'
        p3=c+'_click_ori'
        p4=c+'_all_sec'
        p5=c+'_sec'
        p6=c+'_min'
        p8=c+'_hour'
        print(p2)

        a1=gap_feat(start_time,end_time,data_test,train_set,p1,p2,p3,p4,p5,p6,p8)
        final_train_set=pd.merge(final_train_set,a1,'left',on='instance_id')'''


    #final_train_set=pd.merge(final_train_set,PH_user_ratio,how='left',on='user_id')
    '''pos_set=final_train_set[final_train_set['is_trade']==1]
    neg_set=final_train_set[final_train_set['is_trade']==0]
    neg_n=int(len(neg_set)/4)
    neg_set=neg_set.sample(neg_n,random_state=0,axis=0)
    #pos_set_1=np.array(pos_set)
    #pos_set_1=pd.DataFrame(np.repeat(pos_set_1,1,axis=0),columns=pos_set.columns)

    final_train_set=pd.concat([neg_set,pos_set],axis=0)
    final_train_set=shuffle(final_train_set)
    #label=final_train_set['is_trade']
    #del final_train_set['is_trade']'''

    label=final_train_set['is_trade']
    del final_train_set['is_trade']
    del final_train_set['instance_id']
    del final_train_set['user_id']

    #del final_train_set['item_id']
    #del final_train_set['shop_id']    
    #del final_train_set['item_city_id']
    #del final_train_set['item_brand_id']
    #del final_train_set['context_page_id']
    
    #del final_train_set['user_gender_id']
    #del final_train_set['user_age_level']
    #del final_train_set['user_occupation_id']
    #del final_train_set['user_star_level']
    return final_train_set,label
    
def make_test_set(train_start_time,train_end_time,test_start_time,test_end_time,data_test,data_train):
   
    test_set=time_gap_data(test_start_time,test_end_time,data_test,data_train)
    user_mess=test_set[['instance_id','user_gender_id','user_age_level','user_occupation_id','user_star_level','user_id','is_trade',
    'day','hour','user_query_day','user_query_day_hour','min','mean_hour','item_day_click','item_day_hour_click','item_brand_count',
    'item_property_A_1_count','item_property_list_1_count','item_property_C_day_click','tm_hour','tm_hour_sin','tm_hour_cos',
    'item_city_day_click','item_city_day_hour_click','item_category2_day_click','item_category2_day_hour_click',
    'item_brand_day_click','item_brand_day_hour_click','item_property_A_day_click','item_property_A_day_hour_click',
    'item_property_A_1_day_click','item_property_A_1_day_hour_click','item_property_B_day_click','item_property_B_day_hour_click',
    'item_property_B_1_day_click','item_property_B_1_day_hour_click','item_property_C_day_hour_click',
    'item_list_1_day_click','item_list_1_day_hour_click','item_list_2_day_click','item_list_2_day_hour_click','item_list_3_day_click','item_list_3_day_hour_click',
    'item_city_count','item_category2_count','item_property_A_count','item_property_B_count','item_property_B_1_count','item_property_C_count',
    'item_property_list_2_count','item_property_list_3_count',
    'user_id_click_count_prep','item_id_click_count_prep','item_brand_id_click_count_prep','shop_id_click_count_prep','user_item_id_click_count_prep','user_shop_id_click_count_prep',
     'user_brand_id_click_count_prep','user_category_id_click_count_prep',
     'user_id_count','item_id_count','item_brand_id_count','shop_id_count','user_item_id_count','user_shop_id_count','user_brand_id_count','user_category_id_count',
     'user_id_lasttime_delta','item_id_lasttime_delta','item_brand_id_lasttime_delta','shop_id_lasttime_delta','user_item_id_lasttime_delta',
'user_shop_id_lasttime_delta','user_brand_id_lasttime_delta','user_category_id_lasttime_delta',
'user_id_nexttime_delta','item_id_nexttime_delta','item_brand_id_nexttime_delta','shop_id_nexttime_delta','user_item_id_nexttime_delta',
'user_shop_id_nexttime_delta','user_brand_id_nexttime_delta','user_category_id_nexttime_delta',
'user_id_trade_prep_count','item_id_trade_prep_count','item_brand_id_trade_prep_count','shop_id_trade_prep_count',

#'user_category_item_id_click_count_prep','user_category_shop_id_click_count_prep',
#'user_category_brand_id_click_count_prep','user_category_city_id_click_count_prep',

'user_mean_hour','item_mean_hour','shop_mean_hour','city_mean_hour','brand_mean_hour',
    'user_mean_day','item_mean_day','shop_mean_day','city_mean_day','brand_mean_day',
    'user_mean_day_hour','item_mean_day_hour','shop_mean_day_hour','city_mean_day_hour',
    'brand_mean_day_hour',
#'price_gap_7_6','price_gap_7_5','sales_gap_7_6','sales_gap_7_5','collected_gap_7_6','collected_gap_7_5','pv_gap_7_6','pv_gap_7_5'
#'user_shop_num_mean','user_shop_positive_mean','user_shop_star_mean','user_shop_service_mean','user_shop_delivery_mean','user_shop_description_mean',
 #   'user_item_price_mean', 'user_item_sales_mean','user_item_collected_mean', 'user_item_pv_mean'
    ]]
#'user_item_id_trade_prep_count','user_shop_id_trade_prep_count','user_brand_id_trade_prep_count','user_category_id_trade_prep_count']]

   # 'user_item_brand_count','user_item_property_A_1_count','user_item_property_A_count','user_item_property_B_count','user_item_property_B_1_count','user_item_property_C_count',
    #'user_item_property_list_1_count','user_item_property_list_2_count','user_item_property_list_3_count','user_item_city_count']]
    item_mess=test_set[['item_id','instance_id','item_price_level','item_sales_level','item_collected_level','item_pv_level',
    'item_city_id','item_brand_id','context_page_id','context_id',]]
    PH_mess=test_set[['instance_id','item_city_id_PH_ctr','item_id_PH_ctr','item_brand_id_PH_ctr','user_id_PH_ctr','shop_id_PH_ctr',]]
    #'user_item_category_2_PH_ctr','user_shop_id_PH_ctr','user_item_brand_id_PH_ctr','user_item_id_PH_ctr','user_city_id_PH_ctr','user_context_id_PH_ctr']]
    shop_mess=test_set[['shop_id','instance_id','shop_star_level','shop_review_num_level','shop_review_positive_rate','shop_score_service','shop_score_delivery','shop_score_description']]
    category_mess=test_set[['instance_id','item_property_list_1','item_property_list_2','item_property_list_3','predict_category_property_A','predict_category_property_B','predict_category_property_C',
    'predict_category_property_A_1','predict_category_property_B_1']]
    category_mess[['item_property_list_1','item_property_list_2','item_property_list_3','predict_category_property_A','predict_category_property_B','predict_category_property_C',
    'predict_category_property_A_1','predict_category_property_B_1']]=category_mess[['item_property_list_1','item_property_list_2','item_property_list_3','predict_category_property_A','predict_category_property_B','predict_category_property_C',
    'predict_category_property_A_1','predict_category_property_B_1']].astype(float)
  
    final_test_set=pd.merge(user_mess,item_mess,how='left',on='instance_id')    
    final_test_set=pd.merge(final_test_set,category_mess,how='left',on='instance_id')
    final_test_set=pd.merge(final_test_set,shop_mess,how='left',on='instance_id')  
    final_test_set=pd.merge(final_test_set,PH_mess,how='left',on='instance_id')  
     
   # final_test_set=pd.merge(final_test_set,train_user_feat,how='left',on='user_id')############
    #final_test_set=pd.merge(final_test_set,train_user_gender_trade_feat,how='left',on='user_gender_id')
    #final_test_set=pd.merge(final_test_set,train_user_age_trade_feat,how='left',on='user_age_level')
    #final_test_set=pd.merge(final_test_set,train_user_occup_trade_feat,how='left',on='user_occupation_id')
    #final_test_set=pd.merge(final_test_set,train_user_star_trade_feat,how='left',on='user_star_level')

    for c in columns_click:
        if isinstance(c, (list, tuple)):
            pair = [cc for cc in c]
            c = '_'.join(c)
        else:
            pair = [c]
        
        p1=c
        p2=c+'_click'
        print(p2)
          
        a1=click_feat(test_start_time,test_end_time,data_test,test_set,p1,p2)
        final_test_set=pd.merge(final_test_set,a1,'left',on='instance_id')
    
    '''for c in columns_gap:
        if isinstance(c, (list, tuple)):
            pair = [cc for cc in c]
            c = '_'.join(c)
        else:
            pair = [c]
        
        p1=c
        p2=c+'_click'
        p3=c+'_click_ori'
        p4=c+'_all_sec'
        p5=c+'_sec'
        p6=c+'_min'
        p8=c+'_hour'
        print(p2)
        a1=gap_feat(test_start_time,test_end_time,data_test,test_set,p1,p2,p3,p4,p5,p6,p8)
        final_test_set=pd.merge(final_test_set,a1,'left',on='instance_id')'''
   # final_test_set=pd.merge(final_test_set,PH_user_ratio,how='left',on='user_id')
    
    label=final_test_set['is_trade']
    del final_test_set['is_trade']
    del final_test_set['instance_id'] 
    del final_test_set['user_id']

    #del final_test_set['item_id']
    #del final_test_set['shop_id']
    #del final_test_set['item_city_id'] 
    #del final_test_set['item_brand_id']
    #del final_test_set['context_page_id']
    #del final_test_set['user_gender_id']
    #del final_test_set['user_age_level']
    #del final_test_set['user_occupation_id']
    #del final_test_set['user_star_level']
   # del final_test_set['context_timestamp']
    return final_test_set,label
    
def make_test_set_2(train_start_time,train_end_time,test_start_time,test_end_time,data_test,data_train):

    test_set=data_test
    #index=test_set[['user_id','item_id','shop_id','is_trade','instance_id']]
    user_mess=test_set[['instance_id','user_gender_id','user_age_level','user_occupation_id','user_star_level','user_id',
    'day','hour','user_query_day','user_query_day_hour','min','mean_hour','item_day_click','item_day_hour_click','item_brand_count',
    'item_property_A_1_count','item_property_list_1_count','item_property_C_day_click','tm_hour','tm_hour_sin','tm_hour_cos',
    'item_city_day_click','item_city_day_hour_click','item_category2_day_click','item_category2_day_hour_click',
    'item_brand_day_click','item_brand_day_hour_click','item_property_A_day_click','item_property_A_day_hour_click',
    'item_property_A_1_day_click','item_property_A_1_day_hour_click','item_property_B_day_click','item_property_B_day_hour_click',
    'item_property_B_1_day_click','item_property_B_1_day_hour_click','item_property_C_day_hour_click',
    'item_list_1_day_click','item_list_1_day_hour_click','item_list_2_day_click','item_list_2_day_hour_click','item_list_3_day_click','item_list_3_day_hour_click',
    'item_city_count','item_category2_count','item_property_A_count','item_property_B_count','item_property_B_1_count','item_property_C_count',
    'item_property_list_2_count','item_property_list_3_count',
    'user_id_click_count_prep','item_id_click_count_prep','item_brand_id_click_count_prep','shop_id_click_count_prep','user_item_id_click_count_prep','user_shop_id_click_count_prep',
     'user_brand_id_click_count_prep','user_category_id_click_count_prep',
     'user_id_count','item_id_count','item_brand_id_count','shop_id_count','user_item_id_count','user_shop_id_count','user_brand_id_count','user_category_id_count',
     'user_id_lasttime_delta','item_id_lasttime_delta','item_brand_id_lasttime_delta','shop_id_lasttime_delta','user_item_id_lasttime_delta',
'user_shop_id_lasttime_delta','user_brand_id_lasttime_delta','user_category_id_lasttime_delta',
'user_id_nexttime_delta','item_id_nexttime_delta','item_brand_id_nexttime_delta','shop_id_nexttime_delta','user_item_id_nexttime_delta',
'user_shop_id_nexttime_delta','user_brand_id_nexttime_delta','user_category_id_nexttime_delta',
'user_id_trade_prep_count','item_id_trade_prep_count','item_brand_id_trade_prep_count','shop_id_trade_prep_count',

#'user_category_item_id_click_count_prep','user_category_shop_id_click_count_prep',
#'user_category_brand_id_click_count_prep','user_category_city_id_click_count_prep',

'user_mean_hour','item_mean_hour','shop_mean_hour','city_mean_hour','brand_mean_hour',
    'user_mean_day','item_mean_day','shop_mean_day','city_mean_day','brand_mean_day',
    'user_mean_day_hour','item_mean_day_hour','shop_mean_day_hour','city_mean_day_hour',
    'brand_mean_day_hour',
#'price_gap_7_6','price_gap_7_5','sales_gap_7_6','sales_gap_7_5','collected_gap_7_6','collected_gap_7_5','pv_gap_7_6','pv_gap_7_5'
#'user_shop_num_mean','user_shop_positive_mean','user_shop_star_mean','user_shop_service_mean','user_shop_delivery_mean','user_shop_description_mean',
 #   'user_item_price_mean', 'user_item_sales_mean','user_item_collected_mean', 'user_item_pv_mean'
 ]]
#'user_item_id_trade_prep_count','user_shop_id_trade_prep_count','user_brand_id_trade_prep_count','user_category_id_trade_prep_count']]
    item_mess=test_set[['item_id','instance_id','item_price_level','item_sales_level','item_collected_level','item_pv_level',
    'item_city_id','item_brand_id','context_page_id','context_id',]]
    PH_mess=test_set[['instance_id','item_city_id_PH_ctr','item_id_PH_ctr','item_brand_id_PH_ctr','user_id_PH_ctr','shop_id_PH_ctr',]]
    #'user_item_category_2_PH_ctr','user_shop_id_PH_ctr','user_item_brand_id_PH_ctr','user_item_id_PH_ctr','user_city_id_PH_ctr','user_context_id_PH_ctr']]
    shop_mess=test_set[['shop_id','instance_id','shop_star_level','shop_review_num_level','shop_review_positive_rate','shop_score_service','shop_score_delivery','shop_score_description']]
    category_mess=test_set[['instance_id','item_property_list_1','item_property_list_2','item_property_list_3','predict_category_property_A','predict_category_property_B','predict_category_property_C',
    'predict_category_property_A_1','predict_category_property_B_1']]
    category_mess[['item_property_list_1','item_property_list_2','item_property_list_3','predict_category_property_A','predict_category_property_B','predict_category_property_C',
    'predict_category_property_A_1','predict_category_property_B_1']]=category_mess[['item_property_list_1','item_property_list_2','item_property_list_3','predict_category_property_A','predict_category_property_B','predict_category_property_C',
    'predict_category_property_A_1','predict_category_property_B_1']].astype(float)

    final_test_set=pd.merge(user_mess,item_mess,how='left',on='instance_id')    
    final_test_set=pd.merge(final_test_set,shop_mess,how='left',on='instance_id')  
    final_test_set=pd.merge(final_test_set,category_mess,how='left',on='instance_id')
    final_test_set=pd.merge(final_test_set,PH_mess,how='left',on='instance_id')  
    #final_test_set=pd.merge(final_test_set,train_user_feat,how='left',on='user_id')#########
    #final_test_set=pd.merge(final_test_set,train_user_gender_trade_feat,how='left',on='user_gender_id')
    #final_test_set=pd.merge(final_test_set,train_user_age_trade_feat,how='left',on='user_age_level')
    #final_test_set=pd.merge(final_test_set,train_user_occup_trade_feat,how='left',on='user_occupation_id')
    #final_test_set=pd.merge(final_test_set,train_user_star_trade_feat,how='left',on='user_star_level')
    
    for c in columns_click:
        if isinstance(c, (list, tuple)):
            pair = [cc for cc in c]
            c = '_'.join(c)
        else:
            pair = [c]
        
        p1=c
        p2=c+'_click'
        print(p2)
          
        a1=click_feat(test_start_time,test_end_time,data_test,test_set,p1,p2)
        final_test_set=pd.merge(final_test_set,a1,'left',on='instance_id')
        
    '''for c in columns_gap:
        if isinstance(c, (list, tuple)):
            pair = [cc for cc in c]
            c = '_'.join(c)
        else:
            pair = [c]
        
        p1=c
        p2=c+'_click'
        p3=c+'_click_ori'
        p4=c+'_all_sec'
        p5=c+'_sec'
        p6=c+'_min'
        p8=c+'_hour'
        print(p2)
          
        a1=gap_feat(test_start_time,test_end_time,data_test,test_set,p1,p2,p3,p4,p5,p6,p8)
        final_test_set=pd.merge(final_test_set,a1,'left',on='instance_id')'''
           

    del final_test_set['user_id']
   # del final_test_set['item_id']
   # del final_test_set['shop_id']
    del final_test_set['instance_id'] 
    return final_test_set
    
def online(data_test,data_train):
    #train_start_time=datetime.datetime.strptime('2018-08-31', "%Y-%m-%d")
    #train_end_time=datetime.datetime.strptime('2018-09-07 11:59:59', "%Y-%m-%d %H:%M:%S")
    train_start_time='2018-09-07 00:00:00'
    train_end_time='2018-09-07 11:59:59'

    #test_start_time=datetime.datetime.strtime('2018-09-07 12:00:00', "%Y-%m-%d %H:%M:%S")
    #test_end_time=datetime.datetime.strptime('2018-09-07 23:59:59', "%Y-%m-%d %H:%M:%S")
    test_start_time='2018-09-07 12:00:00'
    test_end_time='2018-09-07 23:59:59'
    
    print('online_train')
    train_x,train_y=make_train_set(train_start_time,train_end_time,data_test,data_train)
    #train_x=train_x.fillna(-1)
    print('online_train_x:%d' % (len(train_x)))
    print('online_train_y: %d' %(len(train_y)))
    train_x.to_csv('../data/online_train_all_to7_x.csv',index=False)
    train_y.to_csv('../data/online_train_all_to7_y.csv',index=False)

    print('online_test')
    test_x=make_test_set_2(train_start_time,train_end_time,test_start_time,test_end_time,data_test,data_train)
    print(len(test_x))
    test_x.to_csv('../data/online_test_all_to7.csv',index=False)
    
online(data_test,data_train)
def online_submit():


    print('online_train')
    train_x=pd.read_csv('../data/online_train_all_to7_x.csv')
    train_y=pd.read_csv('../data/online_train_all_to7_y.csv', header=None)
    train_x=pd.merge(train_x,ori_train,'left',on='instance_id')
    

    print('online_test')
    test_x=pd.read_csv('../data/online_test_all_to7.csv')
    test_x=pd.merge(test_x,ori_train,'left',on='instance_id')

    
    data_test=test_x['instance_id']
    ###################################################################
    
    train_index=train_x.axes[1]
    X = train_x[train_index]
    y = train_y.values
    X_test = test_x[train_index]
	

    lgb_train = lgb.LGBMClassifier(num_leaves=256, learning_rate=0.01,n_estimators=1830,colsample_bytree = 0.8,
                        subsample = 0.8, min_child_weight=6, n_jobs=20)
    lgb_train.fit(X, y,)
    
    lgb_pre_test_y = lgb_train.predict_proba(test_x[train_index])[:,1]
                                             
    pre_test_y=lgb_pre_test_y

    #pre_test_y=xgb_pre_test_y                                      
    data_test=test_x['instance_id']                                             
    pre_test_y=pd.DataFrame(pre_test_y)
    data_test=data_test.reset_index()
    del data_test['index']
    
    sub_data=pd.concat([data_test,pre_test_y],axis=1)
    sub_data=sub_data[:1209768]
    sub_data.rename(columns={0:'predicted_score'},inplace=True)
    print(sub_data['predicted_score'].mean())
    sub_data.to_csv('../submit/lgb_all_to7.txt',index=False,sep=' ')
    print(len(train_x.columns))
    ##########################################################################lr
    x_train=train_x
    x_valid=test_x
    x_train = np.log10(x_train + 1)
    x_valid = np.log10(x_valid + 1)
    
    where_are_nan = np.isnan(x_train)
    where_are_inf = np.isinf(x_train)
    x_train[where_are_nan] = 0
    x_train[where_are_inf] = 0
    where_are_nan = np.isnan(x_valid)
    where_are_inf = np.isinf(x_valid)
    x_valid[where_are_nan] = 0
    x_valid[where_are_inf] = 0
       
    x_train=x_train.drop(['instance_id','item_id','context_id','shop_id','item_property_list_1','item_property_list_2','item_property_list_3','predict_category_property_A','predict_category_property_B','predict_category_property_C',
    'predict_category_property_A_1','predict_category_property_B_1','item_brand_id','item_city_id','tm_hour','min','item_brand_id_count',
    'item_brand_id_click_count_prep'],axis=1).values
    
    x_valid=x_valid.drop(['instance_id','item_id','context_id','shop_id','item_property_list_1','item_property_list_2','item_property_list_3','predict_category_property_A','predict_category_property_B','predict_category_property_C',
    'predict_category_property_A_1','predict_category_property_B_1','item_brand_id','item_city_id','tm_hour','min','item_brand_id_count',
    'item_brand_id_click_count_prep'],axis=1).values
    
    scale = StandardScaler()
    # scale=MinMaxScaler()
    scale.fit(x_train)
    x_train = scale.transform(x_train)
    x_valid = scale.transform(x_valid)
    
    lr = LogisticRegression(n_jobs=-1, random_state=2018, C=32, max_iter=100)
    lr.fit(x_train,y)
    
    pre_val_lr = lr.predict_proba(x_valid)[:,1]

    #pre_test_y=pre_val_y
    #pre_test_y=pre_test_y
    #pre_test_y=xgb_pre_test_y                                      
    pre_val_lr=pd.DataFrame(pre_val_lr)
    data_test=data_test.reset_index()
    del data_test['index']
    
    sub_data=pd.concat([data_test,pre_val_lr],axis=1)
    sub_data.rename(columns={0:'predicted_score'},inplace=True)
    sub_data.to_csv('../submit/sub_lr_all_to_7.txt',index=False,sep=' ')
    ########################################################################################
    train_index=train_x.axes[1]
    X = train_x[train_index]
    y = train_y.values


    xgb_train=XGBClassifier(max_depth=8,eta=0.05, min_child_weight=6, nthread=20, seed=0,num_boost_round=300)
    xgb_train.fit(X, y,)
    
    pre_val_xgb_y=xgb_train.predict_proba(test_x[train_index])[:,1]

    data_test=test_x['instance_id']
    #pre_val_xgb_y = xgb_train.predict(xgb.DMatrix(test_x[train_index]))
                                             
    pre_test_y=pre_val_xgb_y
                                     
    data_test=test_x['instance_id']                                             
    pre_test_y=pd.DataFrame(pre_test_y)
    data_test=data_test.reset_index()
    del data_test['index']
    
    sub_data=pd.concat([data_test,pre_test_y],axis=1)
    sub_data=sub_data[:1209768]
    sub_data.rename(columns={0:'predicted_score'},inplace=True)
    print(sub_data['predicted_score'].mean())
    sub_data.to_csv('../submit/xgb_all_to_7.txt',index=False,sep=' ')

online_submit()