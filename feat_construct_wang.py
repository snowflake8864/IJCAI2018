#coding:utf-8
import pandas as pd
import numpy as np
import time
import datetime
import gc
from sklearn.metrics import log_loss
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
from scipy import sparse
import lightgbm as lgb

import warnings

def ignore_warn(*args ,**kwargs):
    pass
warnings.warn = ignore_warn

def pre_process(data):

    print("Preprocessing...")
    
    data['time'] = pd.to_datetime(data.context_timestamp, unit='s')
    data['time'] = data['time'].apply(lambda x: x + datetime.timedelta(hours=8))
    data['week'] = data['time'].dt.weekday
    data['day'] = data['time'].apply(lambda x: int(str(x)[8:10]))
    data['hour'] = data['time'].apply(lambda x: int(str(x)[11:13]))   

    #将列表特征进行分解
    for i in range(3):
        data['category_%d'%(i)] = data['item_category_list'].apply(
            lambda x:x.split(";")[i] if len(x.split(";")) > i else " ")
    
    for i in range(3):
        data['property_%d'%(i)] = data['item_property_list'].apply(
            lambda x:x.split(";")[i] if len(x.split(";")) > i else " ")

    ###############0401, 时间划分################
    data.loc[(data['hour'] >= 11) & (data['hour'] <= 13 ), 'preiod'] = 2
    data.loc[(data['hour'] >= 0) & (data['hour'] <= 10 ), 'preiod'] = 1
    data.loc[(data['hour'] == 14), 'preiod'] = 1
    data.loc[(data['hour'] >= 15) & (data['hour'] <= 22), 'preiod'] = 0
    ############################################
 
    del data['predict_category_property']
    #del data['item_category_list']
    del data['item_property_list']

    return data

def do_tran(data, leng, flag):
    
    if flag == True:
        '''
        进行one-hot,labelEncoder,特征缩减的转换
        '''
        print("features transform...")
    
        onehot_trans = ['user_gender_id', 'user_age_level', 'user_occupation_id','category_1','category_2']
        label_trans = ('item_city_id', 'context_page_id', 'shop_id', 'item_brand_id','property_0','property_1','property_2')

        data['item_city_id'] = data['item_city_id'].apply(str)
        data['context_page_id'] = data['context_page_id'].apply(str)
        data['shop_id'] = data['shop_id'].apply(str)
        data['item_brand_id'] = data['item_brand_id'].apply(str)
    

        data = pd.get_dummies(data, columns=onehot_trans)

        for c in label_trans:
            lbl = LabelEncoder() 
            lbl.fit(list(data[c].values)) 
            data[c] = lbl.transform(list(data[c].values))
    
    elif flag == False:
        data['property_0'] = data['property_0'].apply(int)
        data['property_1'] = data['property_1'].apply(int)
        data['property_2'] = data['property_2'].apply(int)
        data['category_1'] = data['category_1'].apply(int)
        onehot_trans = ['category_2']
        data = pd.get_dummies(data, columns=onehot_trans)
 
    del data['category_0']
    
    return data

def do_Trick(data):
    #重复次数处理
    temp=data.groupby(['user_id'])['time'].count().reset_index()
    temp['user_large_2']=1*(temp['time']>2)
    temp = temp.drop(['time'], axis=1)
    data = pd.merge(data, temp, how='left', on='user_id')
    del temp
    temp=data.groupby(['item_id'])['time'].count().reset_index()
    temp['item_large_2']=1*(temp['time']>2)
    temp = temp.drop(['time'], axis=1)
    data = pd.merge(data, temp, how='left', on='item_id')
    del temp
    
    #标记出现位置
    subset = ['item_id', 'shop_id', 'user_id', 'item_city_id', 'item_brand_id']
    data['maybe'] = 0
    pos = data.duplicated(subset=subset, keep=False)
    data.loc[pos, 'maybe'] = 1
    pos = (~data.duplicated(subset=subset, keep='first')) & data.duplicated(subset=subset, keep=False)
    data.loc[pos, 'maybe'] = 2
    pos = (~data.duplicated(subset=subset, keep='last')) & data.duplicated(subset=subset, keep=False)
    data.loc[pos, 'maybe'] = 3

    features_trans = ['maybe']
    data = pd.get_dummies(data, columns=features_trans)
    data['maybe_0'] = data['maybe_0'].astype(np.int8)
    data['maybe_1'] = data['maybe_1'].astype(np.int8)
    data['maybe_2'] = data['maybe_2'].astype(np.int8)
    data['maybe_3'] = data['maybe_3'].astype(np.int8)
    
    #时间差Trick,首次,最后一次与当前时刻的时间差
    temp = data.loc[:,['context_timestamp', 'item_id', 'shop_id', 'user_id', 'item_city_id', 'item_brand_id']].drop_duplicates(subset=subset, keep='first')
    temp.rename(columns={'context_timestamp': 'diffTime_first'}, inplace=True)
    data = pd.merge(data, temp, how='left', on=subset)
    data['diffTime_first'] = data['context_timestamp'] - data['diffTime_first']
    del temp,pos
    gc.collect()
    temp = data.loc[:,['context_timestamp', 'item_id', 'shop_id', 'user_id', 'item_city_id', 'item_brand_id']].drop_duplicates(subset=subset, keep='last')
    temp.rename(columns={'context_timestamp': 'diffTime_last'}, inplace=True)
    data = pd.merge(data, temp, how='left', on=subset)
    data['diffTime_last'] = data['diffTime_last'] - data['context_timestamp']
    del temp
    gc.collect()
    data.loc[~data.duplicated(subset=subset, keep=False), ['diffTime_first', 'diffTime_last']] = -1 #置0会变差
    
    ####################0401新添加特征
    data['last_click'] = data['context_timestamp']
    pos = data.duplicated(subset=subset, keep=False)
    data.loc[pos, 'last_click'] = data.loc[pos, 'last_click'].diff(periods=1)
    pos = ~data.duplicated(subset=subset, keep='first')
    data.loc[pos, 'last_click'] = -1
    data['next_click'] = data['context_timestamp']
    pos = data.duplicated(subset=subset, keep=False)
    data.loc[pos, 'next_click'] = -1 * data.loc[pos, 'next_click'].diff(periods=-1)
    pos = ~data.duplicated(subset=subset, keep='last')
    data.loc[pos, 'next_click'] = -1
    del pos
    data['maybe_4']=data['maybe_1']+data['maybe_2']
    data['maybe_5']=data['maybe_1']+data['maybe_3']
    data['diffTime_span']=data['diffTime_last']+data['diffTime_first']

    return data

def do_Trick1(data):
    '''
    user_id点击时间差
    '''
    subset = ['user_id']
    data['umaybe'] = 0
    pos = data.duplicated(subset=subset, keep=False)
    data.loc[pos, 'umaybe'] = 1
    pos = (~data.duplicated(subset=subset, keep='first')) & data.duplicated(subset=subset, keep=False)
    data.loc[pos, 'umaybe'] = 2
    pos = (~data.duplicated(subset=subset, keep='last')) & data.duplicated(subset=subset, keep=False)
    data.loc[pos, 'umaybe'] = 3
    del pos
    gc.collect()
    features_trans = ['umaybe']
    data = pd.get_dummies(data, columns=features_trans)
    data['umaybe_0'] = data['umaybe_0'].astype(np.int8)
    data['umaybe_1'] = data['umaybe_1'].astype(np.int8)
    data['umaybe_2'] = data['umaybe_2'].astype(np.int8)
    data['umaybe_3'] = data['umaybe_3'].astype(np.int8)
    
    subset = ['user_id']
    temp = data[['context_timestamp', 'user_id']]
    temp = temp.drop_duplicates(subset=subset, keep='first')
    temp.rename(columns={'context_timestamp': 'udiffTime_first'}, inplace=True)
    data = pd.merge(data, temp, how='left', on=subset)
    data['udiffTime_first'] = data['context_timestamp'] - data['udiffTime_first']
    del temp
    gc.collect()
    temp = data[['context_timestamp', 'user_id']]
    temp = temp.drop_duplicates(subset=subset, keep='last')
    temp.rename(columns={'context_timestamp': 'udiffTime_last'}, inplace=True)
    data = pd.merge(data, temp, how='left', on=subset)
    data['udiffTime_last'] = data['udiffTime_last'] - data['context_timestamp']
    del temp
    gc.collect()
    data.loc[~data.duplicated(subset=subset, keep=False), ['udiffTime_first', 'udiffTime_last']] = -1

    return data

def do_Trick2(data):
    '''
    item_id出现时间差
    '''
    subset = ['item_id']
    data['imaybe'] = 0
    pos = data.duplicated(subset=subset, keep=False)
    data.loc[pos, 'imaybe'] = 1
    pos = (~data.duplicated(subset=subset, keep='first')) & data.duplicated(subset=subset, keep=False)
    data.loc[pos, 'imaybe'] = 2
    pos = (~data.duplicated(subset=subset, keep='last')) & data.duplicated(subset=subset, keep=False)
    data.loc[pos, 'imaybe'] = 3
    del pos
    gc.collect()
    features_trans = ['imaybe']
    data = pd.get_dummies(data, columns=features_trans)
    data['imaybe_0'] = data['imaybe_0'].astype(np.int8)
    data['imaybe_1'] = data['imaybe_1'].astype(np.int8)
    data['imaybe_2'] = data['imaybe_2'].astype(np.int8)
    data['imaybe_3'] = data['imaybe_3'].astype(np.int8)

    temp = data[['context_timestamp', 'item_id']]
    temp = temp.drop_duplicates(subset=subset, keep='first')
    temp.rename(columns={'context_timestamp': 'idiffTime_first'}, inplace=True)
    data = pd.merge(data, temp, how='left', on=subset)
    data['idiffTime_first'] = data['context_timestamp'] - data['idiffTime_first']
    del temp
    gc.collect()
    temp = data[['context_timestamp', 'item_id']]
    temp = temp.drop_duplicates(subset=subset, keep='last')
    temp.rename(columns={'context_timestamp': 'idiffTime_last'}, inplace=True)
    data = pd.merge(data, temp, how='left', on=subset)
    data['idiffTime_last'] = data['idiffTime_last'] - data['context_timestamp']
    del temp
    gc.collect()
    data.loc[~data.duplicated(subset=subset, keep=False), ['idiffTime_first', 'idiffTime_last']] = -1

    return data

def do_Trick3(data):

    #user_id，第一次点击，中间点击，最后一次点击
    temp=data.ix[:,['context_timestamp','user_id']]
    temp=temp.sort_values(by=['user_id','context_timestamp'])
    temp['click_user_lab']=0

    preix=-1
    prerow={'context_timestamp': -1,
    'user_id':  -1,
    'click_user_lab':  -1}
    
    #取行遍历
    for ix, row in temp.iterrows():
        if(row['user_id']==prerow['user_id']):
            row['click_user_lab']=2
            if prerow['click_user_lab'] != 2:
                prerow['click_user_lab'] = 1
        elif prerow['click_user_lab'] ==2:
            prerow['click_user_lab'] = 3
        preix=ix
        prerow=row
    temp = temp.sort_index()

    data['click_user_lab'] = temp['click_user_lab']
    del temp
    gc.collect()

    #item_id，第一次点击，中间点击，最后一次点击
    temp=data.ix[:,['context_timestamp','item_id']]
    temp=temp.sort_values(by=['item_id','context_timestamp'])
    temp['click_item_lab']=0

    preix=-1
    prerow={'context_timestamp': -1,
    'item_id':  -1,
    'click_item_lab':  -1}

    for ix, row in temp.iterrows():
        if(row['item_id']==prerow['item_id']):
            row['click_item_lab']=2
            if prerow['click_item_lab'] != 2:
                prerow['click_item_lab'] = 1
        elif prerow['click_item_lab'] ==2:
            prerow['click_item_lab'] = 3
        preix=ix
        prerow=row
    temp = temp.sort_index()

    data['click_item_lab'] = temp['click_item_lab']
    del temp
    gc.collect()

    #加入连续点击同一个user_id,item_id特征，第一次点击，中间点击，最后一次点击
    temp=data.ix[:,['context_timestamp', 'item_id', 'user_id']]
    temp=temp.sort_values(by=['user_id', 'item_id', 'context_timestamp'])
    temp['click_user_item_lab']=0

    preix=-1
    prerow={'context_timestamp': -1,
    'item_id ': -1,
    'user_id':  -1,
    'click_user_item_lab':  -1}

    for ix, row in temp.iterrows():
        if(row['user_id']==prerow['user_id'] and row['item_id']==prerow['item_id']):
            row['click_user_item_lab']=2
            if prerow['click_user_item_lab'] != 2:
                prerow['click_user_item_lab'] = 1
        elif prerow['click_user_item_lab'] ==2:
            prerow['click_user_item_lab'] = 3
        preix=ix
        prerow=row
    temp=temp.sort_index()

    data['click_user_item_lab'] = temp['click_user_item_lab']
    del temp
    gc.collect()

    #加入连续点击同一个user_id, item_brand_id特征，第一次点击，中间点击
    temp=data.ix[:,['context_timestamp','item_brand_id','user_id']]
    temp=temp.sort_values(by=['user_id','item_brand_id','context_timestamp'])
    temp['click_user_brand_lab']=0

    preix=-1
    prerow={'context_timestamp': -1,
    'item_brand_id': -1,
    'user_id':  -1,
    'click_user_brand_lab':  -1}

    for ix, row in temp.iterrows():
        if(row['user_id']==prerow['user_id'] and row['item_brand_id']==prerow['item_brand_id']):
            row['click_user_brand_lab']=2
            if prerow['click_user_brand_lab'] != 2:
                prerow['click_user_brand_lab'] = 1
        elif prerow['click_user_brand_lab'] ==2:
            prerow['click_user_brand_lab'] = 3
        preix=ix
        prerow=row
    temp = temp.sort_index()

    data['click_user_brand_lab'] = temp['click_user_brand_lab']
    del temp
    gc.collect()

    #加入连续点击同一个user_id, shop_id特征，第一次点击，中间点击
    temp=data.ix[:,['context_timestamp','shop_id','user_id']]
    temp=temp.sort_values(by=['user_id','shop_id','context_timestamp'])
    temp['click_user_shop_lab']=0

    preix=-1
    prerow={'context_timestamp': -1,
    'shop_id': -1,
    'user_id':  -1,
    'click_user_shop_lab':  -1}

    for ix, row in temp.iterrows():
        if(row['user_id']==prerow['user_id'] and row['shop_id']==prerow['shop_id']):
            row['click_user_shop_lab']=2
            if prerow['click_user_shop_lab'] != 2:
                prerow['click_user_shop_lab'] = 1
        elif prerow['click_user_shop_lab'] == 2:
            prerow['click_user_shop_lab'] = 3
        preix=ix
        prerow=row
    temp = temp.sort_index()

    data['click_user_shop_lab'] = temp['click_user_shop_lab']
    del temp
    gc.collect()

    #加入连续点击同一个user_id, item_city_id特征，第一次点击，中间点击
    temp=data.ix[:,['context_timestamp','item_city_id','user_id']]
    temp=temp.sort_values(by=['user_id','item_city_id','context_timestamp'])
    temp['click_user_city_lab']=0

    preix=-1
    prerow={'context_timestamp': -1,
    'item_city_id': -1,
    'user_id':  -1,
    'click_user_city_lab':  -1}

    for ix, row in temp.iterrows():
        if(row['user_id']==prerow['user_id'] and row['item_city_id']==prerow['item_city_id']):
            row['click_user_city_lab']=2
            if prerow['click_user_city_lab'] != 2:
                prerow['click_user_city_lab'] = 1
        elif prerow['click_user_city_lab'] ==2:
            prerow['click_user_city_lab'] = 3
        preix=ix
        prerow=row
    temp = temp.sort_index()

    data['click_user_city_lab'] = temp['click_user_city_lab']
    del temp
    gc.collect()
    return data


def do_Trick4(data):
    #同一天内时间差
    subset = ['user_id', 'day']
    temp = data[['context_timestamp', 'user_id', 'day']]
    temp = temp.drop_duplicates(subset=subset, keep='first')
    temp.rename(columns={'context_timestamp': 'u_day_diffTime_first'}, inplace=True)
    data = pd.merge(data, temp, how='left', on=subset)
    data['u_day_diffTime_first'] = data['context_timestamp'] - data['u_day_diffTime_first']
    del temp
    gc.collect()
    temp = data[['context_timestamp', 'user_id', 'day']]
    temp = temp.drop_duplicates(subset=subset, keep='last')
    temp.rename(columns={'context_timestamp': 'u_day_diffTime_last'}, inplace=True)
    data = pd.merge(data, temp, how='left', on=subset)
    data['u_day_diffTime_last'] = data['u_day_diffTime_last'] - data['context_timestamp']
    del temp
    gc.collect()
    data.loc[~data.duplicated(subset=subset, keep=False), ['u_day_diffTime_first', 'u_day_diffTime_last']] = -1

    subset = ['item_id', 'day']
    temp = data[['context_timestamp', 'item_id', 'day']]
    temp = temp.drop_duplicates(subset=subset, keep='first')
    temp.rename(columns={'context_timestamp': 'i_day_diffTime_first'}, inplace=True)
    data = pd.merge(data, temp, how='left', on=subset)
    data['i_day_diffTime_first'] = data['context_timestamp'] - data['i_day_diffTime_first']
    del temp
    gc.collect()
    temp = data[['context_timestamp','item_id', 'day']]
    temp = temp.drop_duplicates(subset=subset, keep='last')
    temp.rename(columns={'context_timestamp': 'i_day_diffTime_last'}, inplace=True)
    data = pd.merge(data, temp, how='left', on=subset)
    data['i_day_diffTime_last'] = data['i_day_diffTime_last'] - data['context_timestamp']
    del temp
    gc.collect()
    data.loc[~data.duplicated(subset=subset, keep=False), ['i_day_diffTime_first', 'i_day_diffTime_last']] = -1
    return data


#######考虑多特征组合#######
def doSize(data):
    ################效果明显,重点观察###################
    df_shop_item = data[['shop_id', 'item_price_level', 'item_sales_level', 'item_collected_level', 'item_pv_level']]
    df_shop_item.columns = ['shop_id', 'mean_item_price_level', 'mean_item_sales_level', 'mean_item_collected_level', 'mean_item_pv_level']
    shop_item = df_shop_item.groupby(['shop_id']).mean().reset_index()
    data = pd.merge(data, shop_item, 'left', on='shop_id')
   
    #'item_category_list'日点击次数
    df = data[['user_id', 'item_category_list', 'day']]
    df = df.groupby(['user_id', 'item_category_list', 'day']).size().reset_index().rename(columns={0:'user_catery_trade'})
    data = pd.merge(data, df, 'left', on=['user_id', 'item_category_list', 'day'])
    df = data[['item_id', 'item_category_list', 'day']]
    df = df.groupby(['item_id', 'item_category_list', 'day']).size().reset_index().rename(columns={0:'item_catery_trade'})
    data = pd.merge(data, df, 'left', on=['item_id', 'item_category_list', 'day'])
    df = data[['shop_id', 'item_category_list', 'day']]
    df = df.groupby(['shop_id', 'item_category_list', 'day']).size().reset_index().rename(columns={0:'shop_catery_trade'})
    data = pd.merge(data, df, 'left', on=['shop_id', 'item_category_list', 'day'])
    df = data[['item_brand_id', 'item_category_list', 'day']]
    df = df.groupby(['item_brand_id', 'item_category_list', 'day']).size().reset_index().rename(columns={0:'brand_catery_trade'})
    data = pd.merge(data, df, 'left', on=['item_brand_id', 'item_category_list', 'day'])
    df = data[['item_city_id', 'item_category_list', 'day']]
    df = df.groupby(['item_city_id', 'item_category_list', 'day']).size().reset_index().rename(columns={0:'city_catery_trade'})
    data = pd.merge(data, df, 'left', on=['item_city_id', 'item_category_list', 'day'])
    df = data[['user_gender_id', 'item_category_list', 'day']]
    df = df.groupby(['user_gender_id', 'item_category_list', 'day']).size().reset_index().rename(columns={0:'gender_catery_trade'})
    data = pd.merge(data, df, 'left', on=['user_gender_id', 'item_category_list', 'day'])
    df = data[['user_age_level', 'item_category_list', 'day']]
    df = df.groupby(['user_age_level', 'item_category_list', 'day']).size().reset_index().rename(columns={0:'age_catery_trade'})
    data = pd.merge(data, df, 'left', on=['user_age_level', 'item_category_list', 'day'])
    df = data[['user_occupation_id', 'item_category_list', 'day']]
    df = df.groupby(['user_occupation_id', 'item_category_list', 'day']).size().reset_index().rename(columns={0:'occupation_catery_trade'})
    data = pd.merge(data, df, 'left', on=['user_occupation_id', 'item_category_list', 'day'])
    del df
    del data['item_category_list']
    ###################################

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

    #单个特征, 日,小时点击次数
    user_query_day = data.groupby(['user_id', 'day']).size(
         ).reset_index().rename(columns={0: 'user_query_day'})
    data = pd.merge(data, user_query_day, 'left', on=['user_id', 'day'])
    user_query_day_hour = data.groupby(['user_id', 'day', 'hour']).size().reset_index().rename(
        columns={0: 'user_query_day_hour'})
    data = pd.merge(data, user_query_day_hour, 'left',
                    on=['user_id', 'day', 'hour'])

    shop_query_day = data.groupby(['shop_id', 'day']).size(
         ).reset_index().rename(columns={0: 'shop_query_day'})
    data = pd.merge(data, shop_query_day, 'left', on=['shop_id', 'day'])
    shop_query_day_hour = data.groupby(['shop_id', 'day', 'hour']).size().reset_index().rename(
        columns={0: 'shop_query_day_hour'})
    data = pd.merge(data, shop_query_day_hour, 'left',
                    on=['shop_id', 'day', 'hour'])

    item_query_day = data.groupby(['item_id', 'day']).size(
         ).reset_index().rename(columns={0: 'item_query_day'})
    data = pd.merge(data, item_query_day, 'left', on=['item_id', 'day'])
    item_query_day_hour = data.groupby(['item_id', 'day', 'hour']).size().reset_index().rename(
        columns={0: 'item_query_day_hour'})
    data = pd.merge(data, item_query_day_hour, 'left',
                    on=['item_id', 'day', 'hour'])

    brand_query_day = data.groupby(['item_brand_id', 'day']).size(
         ).reset_index().rename(columns={0: 'brand_query_day'})
    data = pd.merge(data, brand_query_day, 'left', on=['item_brand_id', 'day'])
    brand_query_day_hour = data.groupby(['item_brand_id', 'day', 'hour']).size().reset_index().rename(
        columns={0: 'brand_query_day_hour'})
    data = pd.merge(data, brand_query_day_hour, 'left',
                    on=['item_brand_id', 'day', 'hour'])

    city_query_day = data.groupby(['item_city_id', 'day']).size(
         ).reset_index().rename(columns={0: 'city_query_day'})
    data = pd.merge(data, city_query_day, 'left', on=['item_city_id', 'day'])
    city_query_day_hour = data.groupby(['item_city_id', 'day', 'hour']).size().reset_index().rename(
        columns={0: 'city_query_day_hour'})
    data = pd.merge(data, city_query_day_hour, 'left',
                    on=['item_city_id', 'day', 'hour'])

    
    #####两特征组合,总,日出现个数   XX特征和item_brand_id组合
    # occupation_brand = data.groupby(['user_occupation_id', 'item_brand_id']).size().reset_index().rename(
    #     columns={0: 'occupation_brand'})
    # data = pd.merge(data, occupation_brand, 'left',
    #                 on=['user_occupation_id', 'item_brand_id'])
    occupation_brand_day = data.groupby(['user_occupation_id', 'item_brand_id', 'day']).size().reset_index().rename(
        columns={0: 'occupation_brand_day'})
    data = pd.merge(data, occupation_brand_day, 'left',
                    on=['user_occupation_id', 'item_brand_id', 'day'])
    
    # age_brand = data.groupby(['user_age_level', 'item_brand_id']).size().reset_index().rename(
    #     columns={0: 'age_brand'})
    # data = pd.merge(data, age_brand, 'left',
    #                 on=['user_age_level', 'item_brand_id'])
    age_brand_day = data.groupby(['user_age_level', 'item_brand_id', 'day']).size().reset_index().rename(
        columns={0: 'age_brand_day'})
    data = pd.merge(data, age_brand_day, 'left',
                    on=['user_age_level', 'item_brand_id', 'day'])
    
    # gender_brand = data.groupby(['user_gender_id', 'item_brand_id']).size().reset_index().rename(
    #     columns={0: 'gender_brand'})
    # data = pd.merge(data, gender_brand, 'left',
    #                 on=['user_gender_id', 'item_brand_id'])
    gender_brand_day = data.groupby(['user_gender_id', 'item_brand_id', 'day']).size().reset_index().rename(
        columns={0: 'gender_brand_day'})
    data = pd.merge(data, gender_brand_day, 'left',
                    on=['user_gender_id', 'item_brand_id', 'day'])

    # user_brand = data.groupby(['user_id', 'item_brand_id']).size().reset_index().rename(
    #     columns={0: 'user_brand'})
    # data = pd.merge(data, user_brand, 'left',
    #                 on=['user_id', 'item_brand_id'])
    user_brand_day = data.groupby(['user_id', 'item_brand_id', 'day']).size().reset_index().rename(
        columns={0: 'user_brand_day'})
    data = pd.merge(data, user_brand_day, 'left',
                    on=['user_id', 'item_brand_id', 'day'])

    # page_brand = data.groupby(['context_page_id', 'item_brand_id']).size().reset_index().rename(
    #     columns={0: 'page_brand'})
    # data = pd.merge(data, page_brand, 'left',
    #                 on=['context_page_id', 'item_brand_id'])
    page_brand_day = data.groupby(['context_page_id', 'item_brand_id', 'day']).size().reset_index().rename(
        columns={0: 'page_brand_day'})
    data = pd.merge(data, page_brand_day, 'left',
                    on=['context_page_id', 'item_brand_id', 'day'])
    
    #####两特征组合,总,日出现个数   XX特征和item_id组合
    # user_item = data.groupby(['user_id', 'item_id']).size().reset_index().rename(
    #     columns={0: 'user_item'})
    # data = pd.merge(data, user_item, 'left',
    #                 on=['user_id', 'item_id'])
    user_item_day = data.groupby(['user_id', 'item_id', 'day']).size().reset_index().rename(
        columns={0: 'user_item_day'})
    data = pd.merge(data, user_item_day, 'left',
                    on=['user_id', 'item_id', 'day'])

    # city_item = data.groupby(['item_city_id', 'item_id']).size().reset_index().rename(
    #     columns={0: 'city_item'})
    # data = pd.merge(data, city_item, 'left',
    #                 on=['item_city_id', 'item_id'])
    city_item_day = data.groupby(['item_city_id', 'item_id', 'day']).size().reset_index().rename(
        columns={0: 'city_item_day'})
    data = pd.merge(data, city_item_day, 'left',
                    on=['item_city_id', 'item_id', 'day'])

    # page_item = data.groupby(['context_page_id', 'item_id']).size().reset_index().rename(
    #     columns={0: 'page_item'})
    # data = pd.merge(data, page_item, 'left',
    #                 on=['context_page_id', 'item_id'])
    page_item_day = data.groupby(['context_page_id', 'day', 'item_id']).size().reset_index().rename(
        columns={0: 'page_item_day'})
    data = pd.merge(data, page_item_day, 'left',
                    on=['context_page_id', 'day', 'item_id'])
    
    #####两特征组合,总,日出现个数   XX特征和shop_id组合
    # user_shop = data.groupby(['user_id', 'shop_id']).size().reset_index().rename(
    #     columns={0: 'user_shop'})
    # data = pd.merge(data, user_shop, 'left',
    #                 on=['user_id', 'shop_id'])
    user_shop_day = data.groupby(['user_id', 'shop_id', 'day']).size().reset_index().rename(
        columns={0: 'user_shop_day'})
    data = pd.merge(data, user_shop_day, 'left',
                    on=['user_id', 'shop_id', 'day'])
    
    return data

def doActive(data):
    # 活跃item_id数特征
    # add = pd.DataFrame(data.groupby(["item_brand_id"]).item_id.nunique()).reset_index()
    # add.columns = ["item_brand_id", "item_brand_active_item"]
    # data = data.merge(add, on=["item_brand_id"], how="left") 
    # add = pd.DataFrame(data.groupby(["item_city_id"]).item_id.nunique()).reset_index()
    # add.columns = ["item_city_id", "item_city_active_item"]
    # data = data.merge(add, on=["item_city_id"], how="left") 
    # add = pd.DataFrame(data.groupby(["user_id"]).item_id.nunique()).reset_index()
    # add.columns = ["user_id", "user_active_item"]
    # data = data.merge(add, on=["user_id"], how="left")
    # add = pd.DataFrame(data.groupby(["user_age_level"]).item_id.nunique()).reset_index()
    # add.columns = ["user_age_level", "age_active_item"]
    # data = data.merge(add, on=["user_age_level"], how="left")
    # add = pd.DataFrame(data.groupby(["shop_id"]).item_id.nunique()).reset_index()
    # add.columns = ["shop_id", "shop_active_item"]
    # data = data.merge(add, on=["shop_id"], how="left")
    # 日活跃item_id数特征
    add = pd.DataFrame(data.groupby(["item_brand_id","day"]).item_id.nunique()).reset_index()
    add.columns = ["item_brand_id", "day", "item_brand_day_active_item"]
    data = data.merge(add, on=["item_brand_id", "day"], how="left") 
    add = pd.DataFrame(data.groupby(["item_city_id", "day"]).item_id.nunique()).reset_index()
    add.columns = ["item_city_id", "day", "item_city_day_active_item"]
    data = data.merge(add, on=["item_city_id", "day"], how="left") 
    add = pd.DataFrame(data.groupby(["user_id", "day"]).item_id.nunique()).reset_index()
    add.columns = ["user_id", "day", "user_day_active_item"]
    data = data.merge(add, on=["user_id", "day"], how="left")
    add = pd.DataFrame(data.groupby(["user_age_level", "day"]).item_id.nunique()).reset_index()
    add.columns = ["user_age_level", "day", "age_day_active_item"]
    data = data.merge(add, on=["user_age_level", "day"], how="left")
    add = pd.DataFrame(data.groupby(["shop_id", "day"]).item_id.nunique()).reset_index()
    add.columns = ["shop_id", "day", "shop_day_active_item"]
    data = data.merge(add, on=["shop_id", "day"], how="left")

    # 活跃city数特征
    # add = pd.DataFrame(data.groupby(["item_id"]).item_city_id.nunique()).reset_index()
    # add.columns = ["item_id", "item_active_city"]
    # data = data.merge(add, on=["item_id"], how="left")
    # add = pd.DataFrame(data.groupby(["shop_id"]).item_city_id.nunique()).reset_index()
    # add.columns = ["shop_id", "shop_active_city"]
    # data = data.merge(add, on=["shop_id"], how="left")
    # add = pd.DataFrame(data.groupby(["user_id"]).item_city_id.nunique()).reset_index()
    # add.columns = ["user_id", "user_active_city"]
    # data = data.merge(add, on=["user_id"], how="left")
    # add = pd.DataFrame(data.groupby(["item_brand_id"]).item_city_id.nunique()).reset_index()
    # add.columns = ["item_brand_id", "brand_active_city"]
    # data = data.merge(add, on=["item_brand_id"], how="left")
    # 日活跃city数特征
    add = pd.DataFrame(data.groupby(["item_id", "day"]).item_city_id.nunique()).reset_index()
    add.columns = ["item_id", "day","item_day_active_city"]
    data = data.merge(add, on=["item_id", "day"], how="left")
    add = pd.DataFrame(data.groupby(["shop_id", "day"]).item_city_id.nunique()).reset_index()
    add.columns = ["shop_id", "day", "shop_day_active_city"]
    data = data.merge(add, on=["shop_id", "day"], how="left")
    add = pd.DataFrame(data.groupby(["user_id", "day"]).item_city_id.nunique()).reset_index()
    add.columns = ["user_id", "day", "user_day_active_city"]
    data = data.merge(add, on=["user_id", "day"], how="left")
    add = pd.DataFrame(data.groupby(["item_brand_id", "day"]).item_city_id.nunique()).reset_index()
    add.columns = ["item_brand_id", "day", "brand_day_active_city"]
    data = data.merge(add, on=["item_brand_id", "day"], how="left")

    #活跃user数特征
    # add = pd.DataFrame(data.groupby(["item_id"]).user_id.nunique()).reset_index()
    # add.columns = ["item_id", "item_active_user"]
    # data = data.merge(add, on=["item_id"], how="left")
    # add = pd.DataFrame(data.groupby(["item_brand_id"]).user_id.nunique()).reset_index()
    # add.columns = ["item_brand_id", "brand_active_user"]
    # data = data.merge(add, on=["item_brand_id"], how="left")
    # add = pd.DataFrame(data.groupby(["item_city_id"]).user_id.nunique()).reset_index()
    # add.columns = ["item_city_id", "city_active_user"]
    # data = data.merge(add, on=["item_city_id"], how="left")
    # add = pd.DataFrame(data.groupby(["shop_id"]).user_id.nunique()).reset_index()
    # add.columns = ["shop_id", "shop_active_user"]
    # data = data.merge(add, on=["shop_id"], how="left")
    #日活跃user数特征
    add = pd.DataFrame(data.groupby(["item_id", "day"]).user_id.nunique()).reset_index()
    add.columns = ["item_id", "day", "item_day_active_user"]
    data = data.merge(add, on=["item_id", "day"], how="left")
    add = pd.DataFrame(data.groupby(["item_brand_id", "day"]).user_id.nunique()).reset_index()
    add.columns = ["item_brand_id", "day", "brand_day_active_user"]
    data = data.merge(add, on=["item_brand_id", "day"], how="left")
    add = pd.DataFrame(data.groupby(["item_city_id", "day"]).user_id.nunique()).reset_index()
    add.columns = ["item_city_id", "day", "city_day_active_user"]
    data = data.merge(add, on=["item_city_id", "day"], how="left")
    add = pd.DataFrame(data.groupby(["shop_id", "day"]).user_id.nunique()).reset_index()
    add.columns = ["shop_id", "day", "shop_day_active_user"]
    data = data.merge(add, on=["shop_id", "day"], how="left")

    # 活跃shop数特征
    # add = pd.DataFrame(data.groupby(["item_id"]).shop_id.nunique()).reset_index()
    # add.columns = ["item_id", "item_active_shop"]
    # data = data.merge(add, on=["item_id"], how="left")
    # add = pd.DataFrame(data.groupby(["item_city_id"]).shop_id.nunique()).reset_index()
    # add.columns = ["item_city_id", "city_active_shop"]
    # data = data.merge(add, on=["item_city_id"], how="left")
    # add = pd.DataFrame(data.groupby(["user_id"]).shop_id.nunique()).reset_index()
    # add.columns = ["user_id", "user_active_shop"]
    # data = data.merge(add, on=["user_id"], how="left")
    # add = pd.DataFrame(data.groupby(["item_brand_id"]).shop_id.nunique()).reset_index()
    # add.columns = ["item_brand_id", "brand_active_shop"]
    # data = data.merge(add, on=["item_brand_id"], how="left")
    # 日活跃shop数特征
    add = pd.DataFrame(data.groupby(["item_id", "day"]).shop_id.nunique()).reset_index()
    add.columns = ["item_id", "day", "item_day_active_shop"]
    data = data.merge(add, on=["item_id", "day"], how="left")
    add = pd.DataFrame(data.groupby(["item_city_id", "day"]).shop_id.nunique()).reset_index()
    add.columns = ["item_city_id", "day", "city_day_active_shop"]
    data = data.merge(add, on=["item_city_id", "day"], how="left")
    add = pd.DataFrame(data.groupby(["user_id", "day"]).shop_id.nunique()).reset_index()
    add.columns = ["user_id", "day", "user_day_active_shop"]
    data = data.merge(add, on=["user_id", "day"], how="left")
    add = pd.DataFrame(data.groupby(["item_brand_id", "day"]).shop_id.nunique()).reset_index()
    add.columns = ["item_brand_id", "day", "brand_day_active_shop"]
    data = data.merge(add, on=["item_brand_id", "day"], how="left")

    # 活跃brand数特征
    # add = pd.DataFrame(data.groupby(["item_id"]).item_brand_id.nunique()).reset_index()
    # add.columns = ["item_id", "item_active_brand"]
    # data = data.merge(add, on=["item_id"], how="left")
    # add = pd.DataFrame(data.groupby(["item_city_id"]).item_brand_id.nunique()).reset_index()
    # add.columns = ["item_city_id", "city_active_brand"]
    # data = data.merge(add, on=["item_city_id"], how="left")
    # add = pd.DataFrame(data.groupby(["user_id"]).item_brand_id.nunique()).reset_index()
    # add.columns = ["user_id", "user_active_brand"]
    # data = data.merge(add, on=["user_id"], how="left")
    # add = pd.DataFrame(data.groupby(["shop_id"]).item_brand_id.nunique()).reset_index()
    # add.columns = ["shop_id", "shop_active_brand"]
    # data = data.merge(add, on=["shop_id"], how="left")
    # 日活跃brand数特征
    add = pd.DataFrame(data.groupby(["item_id", "day"]).item_brand_id.nunique()).reset_index()
    add.columns = ["item_id", "day", "item_day_active_brand"]
    data = data.merge(add, on=["item_id", "day"], how="left")
    add = pd.DataFrame(data.groupby(["item_city_id", "day"]).item_brand_id.nunique()).reset_index()
    add.columns = ["item_city_id", "day", "city_day_active_brand"]
    data = data.merge(add, on=["item_city_id", "day"], how="left")
    add = pd.DataFrame(data.groupby(["user_id", "day"]).item_brand_id.nunique()).reset_index()
    add.columns = ["user_id", "day", "user_day_active_brand"]
    data = data.merge(add, on=["user_id", "day"], how="left")
    add = pd.DataFrame(data.groupby(["shop_id", "day"]).item_brand_id.nunique()).reset_index()
    add.columns = ["shop_id", "day", "shop_day_active_brand"]
    data = data.merge(add, on=["shop_id", "day"], how="left")

    #活跃user, item, brand, city, shop小时数
    add = pd.DataFrame(data.groupby(["user_id", "day"]).hour.nunique()).reset_index()
    add.columns = ["user_id", "day", "user_active_hour"]
    data = data.merge(add, on=["user_id", "day"], how="left")
    add = pd.DataFrame(data.groupby(["item_id", "day"]).hour.nunique()).reset_index()
    add.columns = ["item_id", "day", "item_active_hour"]
    data = data.merge(add, on=["item_id", "day"], how="left")
    add = pd.DataFrame(data.groupby(["shop_id", "day"]).hour.nunique()).reset_index()
    add.columns = ["shop_id", "day", "shop_active_hour"]
    data = data.merge(add, on=["shop_id", "day"], how="left")
    add = pd.DataFrame(data.groupby(["item_brand_id", "day"]).hour.nunique()).reset_index()
    add.columns = ["item_brand_id", "day", "brand_active_hour"]
    data = data.merge(add, on=["item_brand_id", "day"], how="left")
    add = pd.DataFrame(data.groupby(["item_city_id", "day"]).hour.nunique()).reset_index()
    add.columns = ["item_city_id", "day", "city_active_hour"]
    data = data.merge(add, on=["item_city_id", "day"], how="left")

    #活跃user, item, brand, city, shop年龄数
    add = pd.DataFrame(data.groupby(["user_id", "day"]).user_age_level.nunique()).reset_index()
    add.columns = ["user_id", "day", "user_active_age"]
    data = data.merge(add, on=["user_id", "day"], how="left")
    add = pd.DataFrame(data.groupby(["item_id", "day"]).user_age_level.nunique()).reset_index()
    add.columns = ["item_id", "day", "item_active_age"]
    data = data.merge(add, on=["item_id", "day"], how="left")
    add = pd.DataFrame(data.groupby(["shop_id", "day"]).user_age_level.nunique()).reset_index()
    add.columns = ["shop_id", "day", "shop_active_age"]
    data = data.merge(add, on=["shop_id", "day"], how="left")
    add = pd.DataFrame(data.groupby(["item_brand_id", "day"]).user_age_level.nunique()).reset_index()
    add.columns = ["item_brand_id", "day", "brand_active_age"]
    data = data.merge(add, on=["item_brand_id", "day"], how="left")
    add = pd.DataFrame(data.groupby(["item_city_id", "day"]).user_age_level.nunique()).reset_index()
    add.columns = ["item_city_id", "day", "city_active_age"]
    data = data.merge(add, on=["item_city_id", "day"], how="left")

    #活跃user, item, brand, city, shop职业
    add = pd.DataFrame(data.groupby(["user_id", "day"]).user_occupation_id.nunique()).reset_index()
    add.columns = ["user_id", "day", "user_active_occupation"]
    data = data.merge(add, on=["user_id", "day"], how="left")
    add = pd.DataFrame(data.groupby(["item_id", "day"]).user_occupation_id.nunique()).reset_index()
    add.columns = ["item_id", "day", "item_active_occupation"]
    data = data.merge(add, on=["item_id", "day"], how="left")
    add = pd.DataFrame(data.groupby(["shop_id", "day"]).user_occupation_id.nunique()).reset_index()
    add.columns = ["shop_id", "day", "shop_active_occupation"]
    data = data.merge(add, on=["shop_id", "day"], how="left")
    add = pd.DataFrame(data.groupby(["item_brand_id", "day"]).user_occupation_id.nunique()).reset_index()
    add.columns = ["item_brand_id", "day", "brand_active_occupation"]
    data = data.merge(add, on=["item_brand_id", "day"], how="left")
    add = pd.DataFrame(data.groupby(["item_city_id", "day"]).user_occupation_id.nunique()).reset_index()
    add.columns = ["item_city_id", "day", "city_active_occupation"]
    data = data.merge(add, on=["item_city_id", "day"], how="left")

    #活跃user, item, brand, city, shop职业
    add = pd.DataFrame(data.groupby(["user_id", "day"]).user_gender_id.nunique()).reset_index()
    add.columns = ["user_id", "day", "user_active_gender"]
    data = data.merge(add, on=["user_id", "day"], how="left")
    add = pd.DataFrame(data.groupby(["item_id", "day"]).user_gender_id.nunique()).reset_index()
    add.columns = ["item_id", "day", "item_active_gender"]
    data = data.merge(add, on=["item_id", "day"], how="left")
    add = pd.DataFrame(data.groupby(["shop_id", "day"]).user_gender_id.nunique()).reset_index()
    add.columns = ["shop_id", "day", "shop_active_gender"]
    data = data.merge(add, on=["shop_id", "day"], how="left")
    add = pd.DataFrame(data.groupby(["item_brand_id", "day"]).user_gender_id.nunique()).reset_index()
    add.columns = ["item_brand_id", "day", "brand_active_gender"]
    data = data.merge(add, on=["item_brand_id", "day"], how="left")
    add = pd.DataFrame(data.groupby(["item_city_id", "day"]).user_gender_id.nunique()).reset_index()
    add.columns = ["item_city_id", "day", "city_active_gender"]
    data = data.merge(add, on=["item_city_id", "day"], how="left")
    
    #有关context_page_id的活跃数
    add = pd.DataFrame(data.groupby(["user_id", "day"]).context_page_id.nunique()).reset_index()
    add.columns = ["user_id", "day", "user_day_active_page"]
    data = data.merge(add, on=["user_id", "day"], how="left")
    add = pd.DataFrame(data.groupby(["item_id", "day"]).context_page_id.nunique()).reset_index()
    add.columns = ["item_id", "day", "item_day_active_page"]
    data = data.merge(add, on=["item_id", "day"], how="left")
    add = pd.DataFrame(data.groupby(["item_brand_id", "day"]).context_page_id.nunique()).reset_index()
    add.columns = ["item_brand_id", "day", "brand_day_active_page"]
    data = data.merge(add, on=["item_brand_id", "day"], how="left")
    add = pd.DataFrame(data.groupby(["user_age_level", "day"]).context_page_id.nunique()).reset_index()
    add.columns = ["user_age_level", "day", "age_day_active_page"]
    data = data.merge(add, on=["user_age_level", "day"], how="left")
    add = pd.DataFrame(data.groupby(["user_occupation_id", "day"]).context_page_id.nunique()).reset_index()
    add.columns = ["user_occupation_id", "day", "occupation_day_active_page"]
    data = data.merge(add, on=["user_occupation_id", "day"], how="left")

    add = pd.DataFrame(data.groupby(["user_id", "user_gender_id", "user_occupation_id", "user_age_level", "day"]).context_page_id.nunique()).reset_index()
    add.columns = ["user_id", "user_gender_id", "user_occupation_id", "user_age_level", "day", "all_user_day_active_page"]
    data = data.merge(add, on=["user_id", "user_gender_id", "user_occupation_id", "user_age_level", "day"], how="left")

    return data

def main():
    path = './data/'
    train = pd.read_table(path+"round1_ijcai_18_train_20180301.txt",sep=" ")
    #train = train.drop_duplicates(subset='instance_id')
    test = pd.read_table(path+'round1_ijcai_18_test_a_20180301.txt',sep=" ")
    data = pd.concat([train, test])
    
    data = pre_process(data)
    print(data.shape)
    
    ###########挖掘新的特征###########

    data = do_Trick(data)  # 重复特征, 多特征group时间差
    print('do_Trick')
    print(data.shape)
    data = do_Trick1(data)  # user_id点击时间差
    print('do_Trick1')
    print(data.shape)
    data = do_Trick2(data)  # item_id
    print('do_Trick2')
    print(data.shape)
    data = do_Trick3(data)  # 标记首次点击,中间点击,最后点击
    print('do_Trick3')
    print(data.shape)
    data = do_Trick4(data)  # 同一天内时间差
    print('do_Trick4')
    print(data.shape)
    data = doSize(data)  # 单个特征, 日,小时点击次数, 效果不好
    print('doSize')
    print(data.shape)
    data = doActive(data)  # 活跃特征
    print('doActive')
    print(data.shape)
    gc.collect()

    ############挖掘新的特征###########

    data = do_tran(data, train.shape[0], False)#特征转换, True进行特征转换, False不进行特征转换
    data.to_csv(path+'creat_feat_wang.csv', index=False)
    

if __name__ == '__main__':
    main()

'''
去除部分leak
[
 'user_gender_id_user_age_level_shop_id_rate',
 'item_id_item_brand_id_context_page_id_rate',
 'item_id_item_brand_id_item_city_id_rate',
 'user_occupation_id_item_brand_id_rate',
 'user_age_level_item_id_rate',
 'item_brand_id_shop_id_rate',
 'item_id_item_city_id_rate',
 'user_age_level_rate',
 'item_id_rate',
 'shop_id_rate',
 
 'diffTime_first',
 'diffTime_last',
 'udiffTime_first',
 'udiffTime_last',
 'idiffTime_first',
 'idiffTime_last',
 'u_day_diffTime_first',
 'u_day_diffTime_last',
 'i_day_diffTime_first',
 'i_day_diffTime_last',
 'maybe_0',
 'umaybe_2',
 'umaybe_3',
 'click_user_lab',
 'click_user_shop_lab',
 'click_user_item_lab',
 'click_user_brand_lab',
 'click_user_city_lab',
 
 'preiod',
 'sale_collect',
 'user_large_2',
 'property_0',
 'day',
 'hour',
 'category_1',
 'shop_query_day',
 
 'context_id',
 'shop_id',
 'context_page_id',
 'user_gender_id',
 'user_star_level',
 'user_age_level',
 'item_pv_level',
 'item_sales_level',
 'item_price_level',
 'shop_review_num_level',
 'shop_review_positive_rate',

 'item_city_day_active_item',
 'user_day_active_shop',
 'shop_day_active_user',
 'user_day_active_item',
 'user_day_active_city',
 'brand_day_active_shop',
 'item_day_active_page',
 'brand_day_active_page',
 'user_day_active_brand',
 'city_day_active_brand',

 'mean_item_price_level', 
 'mean_item_sales_level', 
 'mean_item_collected_level', 
 'mean_item_pv_level',
 'user_catery_trade']

0.08053特征
['user_gender_id_user_age_level_shop_id_rate',
 'preiod',
 'user_gender_id',
 'user_occupation_id_item_brand_id_rate',
 'user_star_level',
 'user_large_2',
 'user_day_active_shop',
 'shop_day_active_user',
 'property_0',
 'user_age_level',
 'udiffTime_last',
 'item_pv_level',
 'item_sales_level',
 'item_price_level',
 'item_id_item_brand_id_context_page_id_rate',
 'user_age_level_item_id_rate',
 'item_city_day_active_item',
 'maybe_0',
 'click_user_lab',
 'idiffTime_first',
 'user_day_active_item',
 'shop_id',
 'day',
 'diffTime_first',
 'user_age_level_rate',
 'user_day_active_city',
 'udiffTime_first',
 'brand_active_shop',
 'item_id_rate',
 'shop_id_rate',
 'idiffTime_last',
 'user_item',
 'click_user_shop_lab',
 'item_id_item_city_id_rate',
 'click_user_item_lab',
 'umaybe_3',
 'hour',
 'category_1',
 'brand_day_active_shop',
 'shop_review_num_level',
 'context_id',
 'sale_collect',
 'shop_review_positive_rate',
 'item_day_active_page',
 'brand_day_active_page',
 'context_page_id',
 'item_id_item_brand_id_item_city_id_rate',
 'item_brand_id_shop_id_rate',
 'user_day_active_brand',
 'umaybe_2',
 'diffTime_last',
 'click_user_brand_lab',
 'click_user_city_lab',
 'city_day_active_brand',
 'shop_query_day',
 'u_day_diffTime_first',
 'u_day_diffTime_last',
 'i_day_diffTime_first',
 'i_day_diffTime_last',
 'mean_item_price_level', 
 'mean_item_sales_level', 
 'mean_item_collected_level', 
 'mean_item_pv_level',
 'user_catery_trade']
 
 ['user_gender_id_user_age_level_shop_id_rate',
 'preiod',
 'item_sales_level_user_occ_prob',
 'occupation_brand',
 'user_gender_id',
 'user_occupation_id_user_age_prob',
 'user_occupation_id_item_brand_id_rate',
 'user_star_level',
 'user_large_2',
 'shop_id_user_prob',
 'property_0',
 'user_age_level',
 'udiffTime_last',
 'item_pv_level',
 'item_sales_level',
 'item_pv_level_user_gender_prob',
 'shop_review_num_level_user_gender_prob',
 'item_price_level',
 'item_id_item_brand_id_context_page_id_rate',
 'user_age_level_item_id_rate',
 'item_price_level_shop_cnt',
 'item_collected_level_price_prob',
 'item_id_shop_rev_cnt',
 'maybe_0',
 'click_user_lab',
 'idiffTime_first',
 'item_price_level_city_cnt',
 'shop_id',
 'user_shop',
 'day',
 'diffTime_first',
 'user_age_level_rate',
 'item_brand_id_shop_rev_prob',
 'udiffTime_first',
 'item_id_rate',
 'shop_id_rate',
 'idiffTime_last',
 'user_item',
 'click_user_shop_lab',
 'item_id_item_city_id_rate',
 'click_user_item_lab',
 'umaybe_3',
 'hour',
 'category_1',
 'shop_review_num_level',
 'item_pv_level_user_cnt',
 'context_id',
 'item_price_level_city_prob',
 'sale_collect',
 'shop_review_positive_rate',
 'item_brand_id_user_prob',
 'item_brand_id_shop_prob',
 'user_age_level_user_gender_prob',
 'context_page_id',
 'shop_star_level_user_age_prob',
 'item_id_item_brand_id_item_city_id_rate',
 'item_brand_id_shop_id_rate',
 'umaybe_2',
 'diffTime_last',
 'item_sales_level_user_age_prob',
 'click_user_brand_lab',
 'click_user_city_lab',
 'shop_query_day',
 'u_day_diffTime_first',
 'u_day_diffTime_last',
 'i_day_diffTime_first',
 'i_day_diffTime_last',
 'item_pv_level_salse_prob',
 'shop_active_brand',
 'shop_active_user',
 'user_active_item',
 'user_active_brand',
 'user_active_shop',
 'item_active_page',
 'item_active_user',
 'item_active_age',
 'brand_active_shop',
 'brand_active_page',
 'item_day_active_page',
 'brand_day_active_page',
 'user_day_active_shop',
 'shop_day_active_user',
 'user_day_active_item',
 'user_day_active_city',
 'brand_day_active_shop',
 'user_day_active_brand',
 'city_day_active_brand',
 'item_city_day_active_item',
 'mean_item_price_level', 'mean_item_sales_level', 'mean_item_collected_level', 'mean_item_pv_level','user_catery_trade'] 这个是最优特征

'''