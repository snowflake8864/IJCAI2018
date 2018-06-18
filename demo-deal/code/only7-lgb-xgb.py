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


import pandas as pd
import numpy as np

import datetime
import math
import gc

##处理类目类特征，将类目类特征分为一列一列
data_train=pd.read_csv('../data/round2_train.txt',delim_whitespace=True)
data_train["times"] = data_train["context_timestamp"].apply(lambda x: datetime.datetime.fromtimestamp(x))
data_train=data_train[(data_train.times>='2018-09-07 00:00:00') & (data_train.times<='2018-09-07 11:59:59')]
print(len(data_train))
#data_train_a=pd.read_csv('/home/cbc/alimama-data/round1_ijcai_18_test_a_20180301.txt',delim_whitespace=True)
#data_train_a['is_trade'] = 0

#data_train=pd.concat([data_train,data_train_a]).reset_index(drop=True)
#data_train=data_train.drop_duplicates(['instance_id'])
length=len(data_train)
data_test_b=pd.read_csv('../data/round2_ijcai_18_test_b_20180510.txt',delim_whitespace=True)
data_test_a=pd.read_csv('../data/round2_ijcai_18_test_a_20180425.txt',delim_whitespace=True)
data_test=pd.concat([data_test_b,data_test_a]).reset_index(drop=True)
data_test['is_trade'] = 0

train_data = pd.concat([data_train,data_test],).reset_index(drop=True)


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

print('gen_count ...')
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

print('gen_trade ...')
##统计各类别的trade转化数
def trade_prep_count(column):
    train_data[column+'_trade_prep_count'] = 0
    for day in train_data.day.unique():
        if day == 6:
            train_data.loc[train_data.day==day,column+'_trade_prep_count'] = -1
        else:
            trade_dict = train_data[train_data.day<day].groupby(column)['is_trade'].sum().to_dict()
            train_data.loc[train_data.day==day,column+'_trade_prep_count'] = train_data.loc[train_data.day==day,column].apply(lambda x: trade_dict[x] if x in trade_dict else 0)

#for column in ['user_id','item_id','item_brand_id','shop_id','user_item_id','user_shop_id','user_brand_id','user_category_id','context_id','item_city_id','user_context_id','user_city_id']:
#    trade_prep_count(column)

'''for i in ['item_price_level', 'item_sales_level',
                 'item_pv_level', 'user_gender_id', 'user_occupation_id',
                'user_age_level', 
                'context_page_id', 'hour', 'shop_review_num_level',
            "shop_star_level",  "user_star_level" , "item_collected_level",
        'item_category']:
    df_train = train_data[train_data.day<25]
    train_data[i+'_PH_ctr'] = 0
    dic_PH = ctr_PH(df_train,train_data,i,10000,0.00001)
    train_data[i+'_PH_ctr'] = train_data[i].apply(lambda x: dic_PH[x])'''
 

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
#a_trade_prep_count=train_data[['user_id_trade_prep_count','item_id_trade_prep_count','item_brand_id_trade_prep_count','shop_id_trade_prep_count',
#'user_item_id_trade_prep_count','user_shop_id_trade_prep_count','user_brand_id_trade_prep_count','user_category_id_trade_prep_count',
#'context_id_trade_prep_count','item_city_id_trade_prep_count','user_context_id_trade_prep_count','user_city_id_trade_prep_count']]

a_gap_time=pd.concat([a_instance,a_gap_time],axis=1)
#a_trade_prep_count=pd.concat([a_instance,a_trade_prep_count],axis=1)

a_prep=a_prep.sort_index()
a_count=a_count.sort_index()
a_gap_time=a_gap_time.sort_index()
#a_trade_prep_count=a_trade_prep_count.sort_index()

a_prep.to_csv('../data/other_feat/count_prep_7_b.csv',index=False)
a_count.to_csv('../data/other_feat/count_7_b.csv',index=False)
a_gap_time.to_csv('../data/other_feat/gap_time_7_b.csv',index=False)


count_prep=pd.read_csv('../data/other_feat/count_prep_7_b.csv')
count=pd.read_csv('../data/other_feat/count_7_b.csv')
gap_time=pd.read_csv('../data/other_feat/gap_time_7_b.csv')
#trade_prep_count=pd.read_csv('../data/other_feat/trade_prep_7.csv')
del count_prep['instance_id']
del count['instance_id']
del gap_time['instance_id']
#del trade_prep_count['instance_id']
##处理类目类特征，将类目类特征分为一列一列
data_train=pd.read_csv('../data/round2_train.txt',delim_whitespace=True)
data_train["times"] = data_train["context_timestamp"].apply(lambda x: datetime.datetime.fromtimestamp(x))
data_train=data_train[(data_train.times>='2018-09-07 00:00:00') & (data_train.times<='2018-09-07 11:59:59')]
print(len(data_train))
#data_test_a=pd.read_csv('/home/cbc/alimama-data/test_a_trade_250.txt',delim_whitespace=True)
#data_test_a['is_trade'] = 0
#data_train=pd.concat([data_train,data_test_a]).reset_index(drop=True)
#data_train=data_train.drop_duplicates(['instance_id'])

#data_train['item_category_list']=data_train['item_category_list'].apply(lambda x:' '.join(x.split(';')))
#item_category_list=CountVectorizer().fit_transform(data_train['item_category_list'])
#data_train=sparse.hstack((item_category_list,data_train))
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
#data_test=pd.read_csv('../data/round2_ijcai_18_test_a_20180425.txt',delim_whitespace=True)
data_test_b=pd.read_csv('../data/round2_ijcai_18_test_b_20180510.txt',delim_whitespace=True)
print('data_test_b:%d' %(len(data_test_b)))
data_test_a=pd.read_csv('../data/round2_ijcai_18_test_a_20180425.txt',delim_whitespace=True)
print('data_test_a:%d' %(len(data_test_a)))
data_test=pd.concat([data_test_b,data_test_a]).reset_index(drop=True)

#data_test['item_category_list']=data_test['item_category_list'].apply(lambda x:' '.join(x.split(';')))
#item_category_list=CountVectorizer().fit_transform(data_test['item_category_list'])
#data_test=sparse.hstack((item_category_list,data_test))

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
#del data_test['item_category_3']
#data_test=data_test.fillna(-1)

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
    data['sec'] = data['times'].apply(lambda x: x.second)
    data['day']=data['day'].astype('int')
    data['hour']=data['hour'].astype('int')
    data['min']=data['min'].astype('int')
    data['tm_hour']=data['hour']+data['min']/60
    data['tm_hour_sin']=data['tm_hour'].map(lambda x:np.sin((x-12)/24*2*np.pi))
    data['tm_hour_cos']=data['tm_hour'].map(lambda x:np.cos((x-12)/24*2*np.pi))
    data_time=data[['user_id','day','hour','min']]
    data=data.reset_index(drop=True)

    #data['shop_review_positive_rate']=data['shop_review_positive_rate'].rank()
    #data['shop_score_service']=data['shop_score_service'].rank()
    #data['shop_score_delivery']=data['shop_score_delivery'].rank()
    #data['shop_score_description']=data['shop_score_description'].rank()
    '''
    data_6=data[data['day']==6]
    data_7=data[data['day']==7]
    data_6=data_6[['instance_id','user_id','item_category_2']].drop_duplicates(['user_id','item_category_2'])
    data_7=data_7[['instance_id','user_id','item_category_2']].drop_duplicates(['user_id','item_category_2'])
    data_user=pd.merge(data_7,data_6,'left',on=['user_id','item_category_2']).dropna(axis=1)
    data_user['user_6_click']=1
    data_user=data_user[['user_id','user_6_click']]
    data=pd.merge(data,data_user,'left',on='user_id')
    '''
    
    print('user_hour ...')
    #shop_time=data[['day','predict_category_property_A_1','hour','min']]
    user_query_day = data.groupby(['user_id', 'day']).size().reset_index().rename(columns={0: 'user_query_day'})    
    user_query_day_hour = data.groupby(['user_id', 'day', 'hour']).size().reset_index().rename(columns={0: 'user_query_day_hour'})   
    user_query_day_hour_min = data.groupby(['user_id', 'day', 'hour','min']).size().reset_index().rename(columns={0: 'user_query_day_hour_min'})
    #user_query_day_hour_min_sec = data.groupby(['user_id', 'day', 'hour','min','sec']).size().reset_index().rename(columns={0: 'user_query_day_hour_min_sec'})
    user_day_hourmin_mean= data_time.groupby(['user_id', 'day']).mean().reset_index().rename(columns={'hour': 'mean_hour','min':'mean_min'})   
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
    item_city_count=data.groupby(['item_city_id']).size().reset_index().rename(columns={0: 'item_city_count'})   
    item_category2_count=data.groupby(['item_category_2']).size().reset_index().rename(columns={0: 'item_category2_count'})   
    item_property_A_count=data.groupby(['predict_category_property_A']).size().reset_index().rename(columns={0: 'item_property_A_count'})   
    item_property_B_count=data.groupby(['predict_category_property_B']).size().reset_index().rename(columns={0: 'item_property_B_count'})   
    item_property_B_1_count=data.groupby(['predict_category_property_B_1']).size().reset_index().rename(columns={0: 'item_property_B_1_count'})   
    item_property_C_count=data.groupby(['predict_category_property_C']).size().reset_index().rename(columns={0: 'item_property_C_count'})   
    item_property_list_2_count=data.groupby(['item_property_list_2']).size().reset_index().rename(columns={0: 'item_property_list_2_count'}) 
    item_property_list_3_count=data.groupby(['item_property_list_3']).size().reset_index().rename(columns={0: 'item_property_list_3_count'}) 
    item_brand_count=data.groupby(['item_brand_id']).size().reset_index().rename(columns={0: 'item_brand_count'})   
    item_property_A_1_count=data.groupby(['predict_category_property_A_1']).size().reset_index().rename(columns={0: 'item_property_A_1_count'})   
    item_property_list_1_count=data.groupby(['item_property_list_1']).size().reset_index().rename(columns={0: 'item_property_list_1_count'}) 
    
    print('user_like')
    item_user_day_hour_click=data.groupby(['item_id', 'day','hour','user_id']).size().reset_index().rename(columns={0: 'item_user_day_hour_click'})   
    shop_user_day_hour_click=data.groupby(['shop_id', 'day','hour','user_id']).size().reset_index().rename(columns={0: 'shop_user_day_hour_click'}) 
    brand_user_day_hour_click=data.groupby(['item_brand_id', 'day','hour','user_id']).size().reset_index().rename(columns={0: 'item_brand_user_day_hour_click'})   
    context_user_day_hour_click=data.groupby(['context_id', 'day','hour','user_id']).size().reset_index().rename(columns={0: 'context_user_day_hour_click'})   
    property_A_user_day_hour_click=data.groupby(['predict_category_property_A', 'day','hour','user_id']).size().reset_index().rename(columns={0: 'property_A_user_day_hour_click'})
    property_A_1_user_day_hour_click=data.groupby(['predict_category_property_A_1', 'day','hour','user_id']).size().reset_index().rename(columns={0: 'property_A_1_user_day_hour_click'})
    property_B_user_day_hour_click=data.groupby(['predict_category_property_B', 'day','hour','user_id']).size().reset_index().rename(columns={0: 'property_B_user_day_hour_click'})
    property_B_1_user_day_hour_click=data.groupby(['predict_category_property_B_1', 'day','hour','user_id']).size().reset_index().rename(columns={0: 'property_B_1_user_day_hour_click'})
    property_C_user_day_hour_click=data.groupby(['predict_category_property_C', 'day','hour','user_id']).size().reset_index().rename(columns={0: 'property_C_user_day_hour_click'})
    list_1_user_day_hour_click=data.groupby(['item_property_list_1', 'day','hour','user_id']).size().reset_index().rename(columns={0: 'list_1_user_day_hour_click'})  
    list_2_user_day_hour_click=data.groupby(['item_property_list_2', 'day','hour','user_id']).size().reset_index().rename(columns={0: 'list_2_user_day_hour_click'})  
    list_3_user_day_hour_click=data.groupby(['item_property_list_3', 'day','hour','user_id']).size().reset_index().rename(columns={0: 'list_3_user_day_hour_click'})  
############      
    print('user_hour_merge ...')
    data = pd.merge(data, user_query_day, 'left', on=['user_id', 'day'])
    data = pd.merge(data, user_query_day_hour, 'left',on=['user_id', 'day', 'hour'])
    data = pd.merge(data, user_query_day_hour_min, 'left',on=['user_id', 'day', 'hour','min'])
    #data = pd.merge(data, user_query_day_hour_min_sec, 'left',on=['user_id', 'day', 'hour','min','sec'])
    data = pd.merge(data, user_day_hourmin_mean, 'left',on=['user_id','day'])
    data = pd.merge(data, item_day_click, 'left',on=['user_id','day','item_id'])
    data = pd.merge(data, item_day_hour_click, 'left',on=['user_id','day','hour','item_id'])
    data = pd.merge(data, item_property_C_day_click, 'left',on=['user_id','day','predict_category_property_C'])       
    data = pd.merge(data, item_brand_count, 'left', on=['item_brand_id'])
    data = pd.merge(data, item_property_A_1_count, 'left', on=['predict_category_property_A_1'])
    data = pd.merge(data, item_property_list_1_count, 'left', on=['item_property_list_1'])
          
    print('click_merge...')
    data = pd.merge(data, item_city_day_click, 'left',on=['user_id','day','item_city_id'])
    data = pd.merge(data, item_city_day_hour_click, 'left',on=['user_id','day','hour','item_city_id'])
    data = pd.merge(data, item_category2_day_click, 'left',on=['user_id','day','item_category_2'])
    data = pd.merge(data, item_category2_day_hour_click, 'left',on=['user_id','day','hour','item_category_2'])
    data = pd.merge(data, item_brand_day_click, 'left',on=['user_id','day','item_brand_id'])
    data = pd.merge(data, item_brand_day_hour_click, 'left',on=['user_id','day','hour','item_brand_id'])
    data = pd.merge(data, item_property_A_day_click, 'left',on=['user_id','day','predict_category_property_A'])
    data = pd.merge(data, item_property_A_day_hour_click, 'left',on=['user_id','day','hour','predict_category_property_A'])
    data = pd.merge(data, item_property_A_1_day_click, 'left',on=['user_id','day','predict_category_property_A_1'])
    data = pd.merge(data, item_property_A_1_day_hour_click, 'left',on=['user_id','day','hour','predict_category_property_A_1'])
    data = pd.merge(data, item_property_B_day_click, 'left',on=['user_id','day','predict_category_property_B'])
    data = pd.merge(data, item_property_B_day_hour_click, 'left',on=['user_id','day','hour','predict_category_property_B'])
    data = pd.merge(data, item_property_B_1_day_click, 'left',on=['user_id','day','predict_category_property_B_1'])
    data = pd.merge(data, item_property_B_1_day_hour_click, 'left',on=['user_id','day','hour','predict_category_property_B_1'])
    data = pd.merge(data, item_property_C_day_hour_click, 'left',on=['user_id','day','hour','predict_category_property_C'])
    data = pd.merge(data, item_list_1_day_click, 'left',on=['user_id','day','item_property_list_1'])
    data = pd.merge(data, item_list_1_day_hour_click, 'left',on=['user_id','day','hour','item_property_list_1'])
    data = pd.merge(data, item_list_2_day_click, 'left',on=['user_id','day','item_property_list_2'])
    data = pd.merge(data, item_list_2_day_hour_click, 'left',on=['user_id','day','hour','item_property_list_2'])
    data = pd.merge(data, item_list_3_day_click, 'left',on=['user_id','day','item_property_list_3'])
    data = pd.merge(data, item_list_3_day_hour_click, 'left',on=['user_id','day','hour','item_property_list_3'])
    data = pd.merge(data, item_property_list_2_count, 'left', on=['item_property_list_2'])
    data = pd.merge(data, item_property_list_3_count, 'left', on=['item_property_list_3'])
           
    print('count_merge ...')
    data = pd.merge(data, item_city_count, 'left', on=['item_city_id'])
    data = pd.merge(data, item_category2_count, 'left', on=['item_category_2'])
    data = pd.merge(data, item_property_A_count, 'left', on=['predict_category_property_A'])
    data = pd.merge(data, item_property_B_count, 'left', on=['predict_category_property_B'])
    data = pd.merge(data, item_property_B_1_count, 'left', on=['predict_category_property_B_1'])
    data = pd.merge(data, item_property_C_count, 'left', on=['predict_category_property_C'])
    
    

    
    return data
    
    
len_train=len(data_train)
all_data=pd.concat([data_train,data_test])
all_data=convert_data(all_data)
data_train=all_data.iloc[:len_train]
data_test=all_data.iloc[len_train:]
###############################################
#all_data=pd.merge(all_data,has_seen,'left',on=['instance_id'])
count_prep_train=count_prep.iloc[:len_train]
count_prep_test=count_prep.iloc[len_train:]
count_train=count.iloc[:len_train]
count_test=count.iloc[len_train:]
gap_time_train=gap_time.iloc[:len_train]
gap_time_test=gap_time.iloc[len_train:]
#trade_prep_count_train=trade_prep_count.iloc[:len_train]
#trade_prep_count_test=trade_prep_count.iloc[len_train:]

data_train=pd.concat([data_train,count_prep_train],axis=1)
data_train=pd.concat([data_train,count_train],axis=1)
data_train=pd.concat([data_train,gap_time_train],axis=1)
#data_train=pd.concat([data_train,trade_prep_count_train],axis=1)

data_test=pd.concat([data_test,count_prep_test],axis=1)
data_test=pd.concat([data_test,count_test],axis=1)
data_test=pd.concat([data_test,gap_time_test],axis=1)
#data_test=pd.concat([data_test,trade_prep_count_test],axis=1)

######################################

#data_train=pd.merge(data_train,data_train_PH,'left',on=['instance_id'])
print(len(data_train))
print(len(data_test))
#data_test=pd.merge(data_test,data_test_PH,'left',on=['instance_id'])
del data_test['is_trade']
del data_test['item_category_3']
'''
print('half...')
data_train.to_csv('../data/data_train_25_half.csv',index=False)
data_test.to_csv('../data/data_test_25_half.csv',index=False)


#data_train_PH=pd.read_csv('../data/data_train_PH.csv')
#data_test_PH=pd.read_csv('../data/data_test_PH.csv')

print('load_data_half...')
data_train=pd.read_csv('../data/data_train_25_half.csv')
data_train=data_train[data_train['day']>=7]
data_train=data_train[data_train['day']!=31]
print(len(data_train))
data_train['times']=data_train['times'].astype(str)
data_test=pd.read_csv('../data/data_test_25_half.csv')
print(len(data_test))
data_test['times']=data_test['times'].astype(str)
print('load_data_half finish...')
'''


 ########################################################################
def click_feat(start_time,end_time,data_test,data_train,p1,p2):
    param1=p1
    param2=p2

    clean_train_data=time_gap_data(start_time,end_time,data_test,data_train)
    user_item_brand_click=clean_train_data[['user_id',param1,'times','instance_id']]
    user_item_brand_click=user_item_brand_click.fillna(-1)##
    
    a=user_item_brand_click.groupby(['user_id',param1]).size()
    a=a.reset_index()
    b=pd.merge(user_item_brand_click,a,how='left',on=['user_id',param1])
    
    data_1=b[b[0]==1]
    del data_1[0]
    data_1[param2]=0
    
    train_tmp=b[b[0]>1]
    
    train_tmp=train_tmp.ix[:,['times',param1,'user_id','instance_id',0]]    
    train_tmp=train_tmp.sort_values(by=['user_id',param1,'times'])
    
    train_tmp_group_first=train_tmp.groupby(by=list(['user_id',param1])).head(1)
    train_tmp_group_first[0]=1
    train_tmp_group_last=train_tmp.groupby(by=list(['user_id',param1])).tail(1)
    train_tmp_group_last[0]=3
       
    train_tmp_first_last=pd.concat([train_tmp_group_first,train_tmp_group_last],axis=0)
    
    final=pd.merge(train_tmp[['instance_id',0]],train_tmp_first_last[['instance_id',0]],how='left',on='instance_id')
    final[param2]=final['0_y'].replace([np.nan],[2])
       # final[param1][final[param2]==-1]=-1
    
    final=final[['instance_id',param2]]
    data_1=data_1[['instance_id',param2]]
    final_tmp=pd.concat([final,data_1],axis=0)
    final_tmp=pd.merge(final_tmp,user_item_brand_click,how='left',on='instance_id')
    final_tmp[[param1,param2]]=final_tmp[[param1,param2]].astype(float)##
    final_tmp[param2][final_tmp[param1]==-1]=-1  ##
  #  del final_tmp['context_timestamp']
   # final_tmp=final_tmp.astype('int64')
    
    #final_tmp.to_csv(param3%(start_time[-2:],end_time[-2:]),index=False)
    return final_tmp[['instance_id',p2]]
 
def gap_feat(start_time,end_time,data_test,data_train,p1,p2,p3,p4,p5,p6,p8):
    param1=p1
    param2=p2
    param3=p3
    param4=p4
    param5=p5
    param6=p6

    param8=p8
    clean_train_data=time_gap_data(start_time,end_time,data_test,data_train)
    user_item_brand_click=clean_train_data[['user_id',param1,'context_timestamp','instance_id']]
    user_item=clean_train_data[['user_id',param1,'instance_id','context_timestamp']]

    a=user_item_brand_click.groupby(['user_id',param1]).size()
    a=a.reset_index()
    b=pd.merge(user_item_brand_click,a,how='left',on=['user_id',param1])
    
    data_1=b[b[0]==1]
    del data_1[0]
    data_1[param2]=0

    train_tmp=b[b[0]>1]

    train_tmp=train_tmp.ix[:,['context_timestamp',param1,'user_id','instance_id',0]]    
    train_tmp=train_tmp.sort_values(by=['user_id',param1,'context_timestamp'])

    train_tmp_group_first=train_tmp.groupby(by=list(['user_id',param1])).head(1)
    train_tmp_group_first[0]=1
    train_tmp_group_last=train_tmp.groupby(by=list(['user_id',param1])).tail(1)
    train_tmp_group_last[0]=3
   
    train_tmp_first_last=pd.concat([train_tmp_group_first,train_tmp_group_last],axis=0)

    final=pd.merge(train_tmp[['instance_id',0]],train_tmp_first_last[['instance_id',0]],how='left',on='instance_id')
    final[param2]=final['0_y'].replace([np.nan],[2])
    
    final_tmp=final[['instance_id',param2]]
    data_1=data_1[['instance_id',param2]]
   # final_tmp=pd.concat([final,data_1],axis=0)
###########
    final_tmp=pd.merge(final_tmp,user_item,how='left',on='instance_id')
    final_tmp=final_tmp.sort_values(by=['user_id',param1,'context_timestamp'])
    item_category2_click_ch=final_tmp.iloc[1:]
    item_category2_click_1=final_tmp.iloc[0:2]
    item_category2_click_ch=pd.concat([item_category2_click_ch,item_category2_click_1],axis=0)
    item_category2_click_ch=item_category2_click_ch.iloc[:-1]
    item_category2_click_ch=item_category2_click_ch.reset_index()
    del item_category2_click_ch['index']

    final_tmp_click_time=final_tmp[[param2,'context_timestamp']]
    final_tmp_click_time .rename(columns={param2:param3,'context_timestamp':'context_timestamp_ori'},inplace=True)  
    final_tmp_click_time=final_tmp_click_time.reset_index()
    del final_tmp_click_time['index']
    concat=pd.concat([item_category2_click_ch,final_tmp_click_time],axis=1)
    concat['click']=concat[param2]-concat[param3]
    concat['time']=concat['context_timestamp']-concat['context_timestamp_ori']
    concat['time'][concat['click']<0]=0
    concat=concat[['instance_id','time']]
    
    concat[param4]=concat['time']
    concat[param5]=concat['time']%60
    concat[param6]=concat['time']//60
    concat[param8]=concat['time']//3600

    final_time=pd.concat([concat,data_1],axis=0)
    del final_time[param2]
    final_time= final_time.fillna(-1)##

    del final_time['time']
    return final_time[['instance_id',p4,p8,p6]]

columns_click = [
                     ['predict_category_property_A'],
                     ['predict_category_property_B'],
                     ['predict_category_property_C'],
                     ['predict_category_property_A_1'],
                     ['predict_category_property_B_1'],
                     ['item_property_list_1'],
                     ['item_property_list_2'],
                     ['item_property_list_3'],
                     ['item_brand_id'],
                     ['item_city_id'],
                     ['item_category_2'],
                     ['item_price_level'],
                     ['item_sales_level'],
                     ['item_collected_level'],
                     ['item_pv_level'],
                     ['context_page_id'],
                     ['shop_review_num_level'],
                     ['shop_review_positive_rate'],
                     ['shop_star_level'],
                     ['shop_score_service'],
                     ['shop_score_delivery'],
                     ['shop_score_description'],
                     ]
columns_gap = [
                     ['predict_category_property_A'],
                     ['predict_category_property_B'],
                     ['predict_category_property_C'],
                     ['predict_category_property_A_1'],
                     ['predict_category_property_B_1'],
                     ['item_property_list_1'],
                     ['item_property_list_2'],
                     ['item_property_list_3'],
                     ['item_category_2'],
                     ['item_price_level'],
                     ['item_sales_level'],
                     ['item_collected_level'],
                     ['item_pv_level'],
                     ['context_page_id'],
                     ['shop_review_num_level'],
                     ['shop_review_positive_rate'],
                     ['shop_star_level'],
                     ['shop_score_service'],
                     ['shop_score_delivery'],
                     ['shop_score_description'],
                     ]
                     

def make_train_set(start_time,end_time,data_test,data_train):
    train_set=time_gap_data(start_time,end_time,data_test,data_train)
    

    user_mess=train_set[['instance_id','user_gender_id','user_age_level','user_occupation_id','user_star_level','user_id','is_trade',
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
#'user_id_trade_prep_count','item_id_trade_prep_count','item_brand_id_trade_prep_count','shop_id_trade_prep_count',
#'item_user_day_hour_click','shop_user_day_hour_click','item_brand_user_day_hour_click','context_user_day_hour_click',
#'property_A_user_day_hour_click','property_A_1_user_day_hour_click','property_B_user_day_hour_click',
#'property_B_1_user_day_hour_click','property_C_user_day_hour_click','list_1_user_day_hour_click','list_2_user_day_hour_click',
#'list_3_user_day_hour_click',
#'user_6_click',
#'hour_item_city_count','hour_item_category2_count','hour_item_property_A_count','hour_item_property_B_count',
#'hour_item_property_B_1_count','hour_item_property_C_count','hour_item_property_list_2_count','hour_item_property_list_3_count',
#'hour_item_brand_count','hour_item_property_A_1_count','hour_item_property_list_1_count',
#'hour_user_count','hour_item_count','hour_shop_count','hour_context_count',
#'user_shop_min15_click'
]]
#'user_item_id_trade_prep_count','user_shop_id_trade_prep_count','user_brand_id_trade_prep_count','user_category_id_trade_prep_count']]

    #'user_item_brand_count','user_item_property_A_1_count','user_item_property_A_count','user_item_property_B_count','user_item_property_B_1_count','user_item_property_C_count',
    #'user_item_property_list_1_count','user_item_property_list_2_count','user_item_property_list_3_count','user_item_city_count']]
    item_mess=train_set[['item_id','instance_id','item_price_level','item_sales_level','item_collected_level','item_pv_level',
    'item_city_id','item_brand_id','context_page_id','context_id']]
    #PH_mess=train_set[['instance_id','item_city_id_PH_ctr','item_id_PH_ctr','item_brand_id_PH_ctr','user_id_PH_ctr','shop_id_PH_ctr']]
    shop_mess=train_set[['shop_id','instance_id','shop_star_level','shop_review_num_level','shop_review_positive_rate','shop_score_service','shop_score_delivery','shop_score_description']]
    category_mess=train_set[['instance_id','item_property_list_1','item_property_list_2','item_property_list_3','predict_category_property_A','predict_category_property_B','predict_category_property_C',
    'predict_category_property_A_1','predict_category_property_B_1']]
    category_mess[['item_property_list_1','item_property_list_2','item_property_list_3','predict_category_property_A','predict_category_property_B','predict_category_property_C',
    'predict_category_property_A_1','predict_category_property_B_1']]=category_mess[['item_property_list_1','item_property_list_2','item_property_list_3','predict_category_property_A','predict_category_property_B','predict_category_property_C',
    'predict_category_property_A_1','predict_category_property_B_1']].astype(float)
    #index=train_set[['user_id','item_id','shop_id','is_trade','instance_id']]
 
    final_train_set=pd.merge(user_mess,item_mess,how='left',on='instance_id')    
    final_train_set=pd.merge(final_train_set,category_mess,how='left',on='instance_id') 
    final_train_set=pd.merge(final_train_set,shop_mess,how='left',on='instance_id')  
    #final_train_set=pd.merge(final_train_set,PH_mess,how='left',on='instance_id')  
    #final_train_set=pd.merge(final_train_set,train_user_feat,how='left',on='user_id')##################
    #final_train_set=pd.merge(final_train_set,train_user_gender_trade_feat,how='left',on='user_gender_id')
    #final_train_set=pd.merge(final_train_set,train_user_age_trade_feat,how='left',on='user_age_level')
    #final_train_set=pd.merge(final_train_set,train_user_occup_trade_feat,how='left',on='user_occupation_id')
    #final_train_set=pd.merge(final_train_set,train_user_star_trade_feat,how='left',on='user_star_level')
 
    for c in columns_click:
        if isinstance(c, (list, tuple)):
            pair = [cc for cc in c]
            c = '_'.join(c)
        else:
            pair = [c]
        
        p1=c
        p2=c+'_click'
        print(p2)
          
        a1=click_feat(start_time,end_time,data_test,data_train,p1,p2)
        final_train_set=pd.merge(final_train_set,a1,'left',on='instance_id')

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

        a1=gap_feat(start_time,end_time,data_test,data_train,p1,p2,p3,p4,p5,p6,p8)
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
    #del final_train_set['instance_id']
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
#'user_id_trade_prep_count','item_id_trade_prep_count','item_brand_id_trade_prep_count','shop_id_trade_prep_count',
#'item_user_day_hour_click','shop_user_day_hour_click','item_brand_user_day_hour_click','context_user_day_hour_click',
#'property_A_user_day_hour_click','property_A_1_user_day_hour_click','property_B_user_day_hour_click',
#'property_B_1_user_day_hour_click','property_C_user_day_hour_click','list_1_user_day_hour_click','list_2_user_day_hour_click',
#'list_3_user_day_hour_click',
#'user_6_click',
#'hour_item_city_count','hour_item_category2_count','hour_item_property_A_count','hour_item_property_B_count',
#'hour_item_property_B_1_count','hour_item_property_C_count','hour_item_property_list_2_count','hour_item_property_list_3_count',
#'hour_item_brand_count','hour_item_property_A_1_count','hour_item_property_list_1_count',
#'hour_user_count','hour_item_count','hour_shop_count','hour_context_count',
#'user_shop_min15_click'
]]
#'user_item_id_trade_prep_count','user_shop_id_trade_prep_count','user_brand_id_trade_prep_count','user_category_id_trade_prep_count']]

   # 'user_item_brand_count','user_item_property_A_1_count','user_item_property_A_count','user_item_property_B_count','user_item_property_B_1_count','user_item_property_C_count',
    #'user_item_property_list_1_count','user_item_property_list_2_count','user_item_property_list_3_count','user_item_city_count']]
    item_mess=test_set[['item_id','instance_id','item_price_level','item_sales_level','item_collected_level','item_pv_level',
    'item_city_id','item_brand_id','context_page_id','context_id']]
    #PH_mess=test_set[['instance_id','item_city_id_PH_ctr','item_id_PH_ctr','item_brand_id_PH_ctr','user_id_PH_ctr','shop_id_PH_ctr']]
    shop_mess=test_set[['shop_id','instance_id','shop_star_level','shop_review_num_level','shop_review_positive_rate','shop_score_service','shop_score_delivery','shop_score_description']]
    category_mess=test_set[['instance_id','item_property_list_1','item_property_list_2','item_property_list_3','predict_category_property_A','predict_category_property_B','predict_category_property_C',
    'predict_category_property_A_1','predict_category_property_B_1']]
    category_mess[['item_property_list_1','item_property_list_2','item_property_list_3','predict_category_property_A','predict_category_property_B','predict_category_property_C',
    'predict_category_property_A_1','predict_category_property_B_1']]=category_mess[['item_property_list_1','item_property_list_2','item_property_list_3','predict_category_property_A','predict_category_property_B','predict_category_property_C',
    'predict_category_property_A_1','predict_category_property_B_1']].astype(float)
  
    final_test_set=pd.merge(user_mess,item_mess,how='left',on='instance_id')    
    final_test_set=pd.merge(final_test_set,category_mess,how='left',on='instance_id')
    final_test_set=pd.merge(final_test_set,shop_mess,how='left',on='instance_id')  
    #final_test_set=pd.merge(final_test_set,PH_mess,how='left',on='instance_id')  
     
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
          
        a1=click_feat(test_start_time,test_end_time,data_test,data_train,p1,p2)
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
          
        a1=gap_feat(test_start_time,test_end_time,data_test,data_train,p1,p2,p3,p4,p5,p6,p8)
        final_test_set=pd.merge(final_test_set,a1,'left',on='instance_id')'''
   # final_test_set=pd.merge(final_test_set,PH_user_ratio,how='left',on='user_id')
    
    label=final_test_set['is_trade']
    del final_test_set['is_trade']
    #del final_test_set['instance_id'] 
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
#'user_id_trade_prep_count','item_id_trade_prep_count','item_brand_id_trade_prep_count','shop_id_trade_prep_count',
#'item_user_day_hour_click','shop_user_day_hour_click','item_brand_user_day_hour_click','context_user_day_hour_click',
#'property_A_user_day_hour_click','property_A_1_user_day_hour_click','property_B_user_day_hour_click',
#'property_B_1_user_day_hour_click','property_C_user_day_hour_click','list_1_user_day_hour_click','list_2_user_day_hour_click',
#'list_3_user_day_hour_click',
#'user_6_click',
#'hour_item_city_count','hour_item_category2_count','hour_item_property_A_count','hour_item_property_B_count',
#'hour_item_property_B_1_count','hour_item_property_C_count','hour_item_property_list_2_count','hour_item_property_list_3_count',
#'hour_item_brand_count','hour_item_property_A_1_count','hour_item_property_list_1_count',
#'hour_user_count','hour_item_count','hour_shop_count','hour_context_count',
#'user_shop_min15_click'
]]
#'user_item_id_trade_prep_count','user_shop_id_trade_prep_count','user_brand_id_trade_prep_count','user_category_id_trade_prep_count']]
    item_mess=test_set[['item_id','instance_id','item_price_level','item_sales_level','item_collected_level','item_pv_level',
    'item_city_id','item_brand_id','context_page_id','context_id']]
    #PH_mess=test_set[['instance_id','item_city_id_PH_ctr','item_id_PH_ctr','item_brand_id_PH_ctr','user_id_PH_ctr','shop_id_PH_ctr']]
    shop_mess=test_set[['shop_id','instance_id','shop_star_level','shop_review_num_level','shop_review_positive_rate','shop_score_service','shop_score_delivery','shop_score_description']]
    category_mess=test_set[['instance_id','item_property_list_1','item_property_list_2','item_property_list_3','predict_category_property_A','predict_category_property_B','predict_category_property_C',
    'predict_category_property_A_1','predict_category_property_B_1']]
    category_mess[['item_property_list_1','item_property_list_2','item_property_list_3','predict_category_property_A','predict_category_property_B','predict_category_property_C',
    'predict_category_property_A_1','predict_category_property_B_1']]=category_mess[['item_property_list_1','item_property_list_2','item_property_list_3','predict_category_property_A','predict_category_property_B','predict_category_property_C',
    'predict_category_property_A_1','predict_category_property_B_1']].astype(float)

    final_test_set=pd.merge(user_mess,item_mess,how='left',on='instance_id')    
    final_test_set=pd.merge(final_test_set,shop_mess,how='left',on='instance_id')  
    final_test_set=pd.merge(final_test_set,category_mess,how='left',on='instance_id')
    #final_test_set=pd.merge(final_test_set,PH_mess,how='left',on='instance_id')  
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
          
        a1=click_feat(test_start_time,test_end_time,data_test,data_train,p1,p2)
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
        a1=gap_feat(test_start_time,test_end_time,data_test,data_train,p1,p2,p3,p4,p5,p6,p8)
        final_test_set=pd.merge(final_test_set,a1,'left',on='instance_id')'''
        
    del final_test_set['user_id']
   # del final_test_set['item_id']
   # del final_test_set['shop_id']
    #del final_test_set['instance_id'] 
    return final_test_set
    
def online_submit(data_test,data_train):
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
    # train_x=train_x.fillna(-1)
    print('online_train_x:%d' % (len(train_x)))
    print('online_train_y: %d' %(len(train_y)))
    train_x.to_csv('../data/online_train_x_7_b.csv',index=False)
    train_y.to_csv('../data/online_train_y_7_b.csv',index=False)
    
    print('online_test')
    test_x=make_test_set_2(train_start_time,train_end_time,test_start_time,test_end_time,data_test,data_train)
    print(len(test_x))
    test_x.to_csv('../data/online_test_7_b.csv',index=False)
    
    train_index=train_x.axes[1]
    X = train_x[train_index]
    y = train_y.values

    lgb_train = lgb.LGBMClassifier(num_leaves=64, learning_rate=0.01,n_estimators=1775,colsample_bytree =0.8,
                        subsample = 0.8,max_depth=7, min_child_weight=6, n_jobs=20)
    lgb_train.fit(X, y,)
    
    data_test=data_test['instance_id']
    lgb_pre_test_y = lgb_train.predict_proba(test_x[train_index])[:,1]

    data_test=test_x['instance_id']                                             
    pre_test_y=pd.DataFrame(lgb_pre_test_y)
    data_test=data_test.reset_index()
    del data_test['index']
    
    sub_data=pd.concat([data_test,pre_test_y],axis=1)
    print(len(sub_data))
    sub_data=sub_data[:1209768]
    sub_data.rename(columns={0:'predicted_score'},inplace=True)
    print('trade_mean:%d' %(sub_data['predicted_score'].mean()))
    sub_data.to_csv('../submit/lgb_only7.txt',index=False,sep=' ')
    
    #########################################xgb
    train_index=train_x.axes[1]
    params = {'max_depth': 6,
              'nthread': 25,
              'eta': 0.01,
              'eval_metric': 'logloss',
              'objective': 'binary:logistic',
              'subsample': 0.85,
              'colsample_bytree': 0.85,
              'silent': 1,
              'seed': 0,
              'min_child_weight': 6
              #'scale_pos_weight':0.5
              }

    
    num_boost_round = 2607
    
    X = train_x[train_index]
    y = train_y.values
    X_test = test_x[train_index]

    dtrain = xgb.DMatrix(X, y)
    xgb_train = xgb.train(params, dtrain, num_boost_round, verbose_eval=True)
    
    xgb_pre_test_y =xgb_train.predict(xgb.DMatrix(X_test))
    
    data_test=test_x['instance_id']                                             
    pre_test_y=pd.DataFrame(lgb_pre_test_y)
    data_test=data_test.reset_index()
    del data_test['index']
    
    sub_data=pd.concat([data_test,pre_test_y],axis=1)
    print(len(sub_data))
    sub_data=sub_data[:1209768]
    sub_data.rename(columns={0:'predicted_score'},inplace=True)
    print('trade_mean:%d' %(sub_data['predicted_score'].mean()))
    sub_data.to_csv('../submit/xgb_only7.txt',index=False,sep=' ')
    
    
online_submit(data_test,data_train)