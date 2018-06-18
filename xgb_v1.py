# -*- coding:utf-8 -*-
import pandas as pd
import numpy as np
import datetime
import scipy as sp
import xgboost as xgb
import operator
from sklearn.preprocessing import LabelEncoder
import gc


# 评价函数
def logloss(act, pred):
    epsilon = 1e-15
    pred = sp.maximum(epsilon, pred)
    pred = sp.minimum(1-epsilon, pred)
    ll = sum(act*sp.log(pred) + sp.subtract(1, act)*sp.log(sp.subtract(1, pred)))
    ll = ll * -1.0/len(act)
    return ll


# 转换标签编码
def LabEncode(data, col):
    le = LabelEncoder()
    ipl = le.fit_transform(data[col])
    data[col] = ipl
    return data


# 转换UNIX时间为正常时间
def timestamp2datetime(timestamp):
    dt = datetime.datetime.utcfromtimestamp(timestamp) + datetime.timedelta(hours=8)
    return dt.strftime('%Y-%m-%d %H:%M:%S')


# 用户在当前小时内和当天的点击量统计特征，以及当前所在的小时
def convert_data(data):
    data['time'] = data.context_timestamp.apply(timestamp2datetime)
    data['day'] = data.time.apply(lambda x: int(x[8:10]))
    data['hour'] = data.time.apply(lambda x: int(x[11:13]))
    data['hour_sin'] = data['hour'].apply(lambda x: np.pi*np.sin(int(x)))
    data['hour_cos'] = data['hour'].apply(lambda x: np.pi*np.cos(int(x)))
    return data


# 从不同时间粒度（天，小时）
# 提取用户点击行为的统计特征
def convert_item_user(data):
    user_item_day = data.groupby(['user_id', 'item_id', 'day']).size(
    ).reset_index().rename(columns={0: 'user_item_day'})
    data = pd.merge(data, user_item_day, 'left', on=['user_id', 'item_id', 'day'])
    user_item_day_hour = data.groupby(['user_id', 'item_id', 'day', 'hour']).size().reset_index().rename(
        columns={0: 'user_item_day_hour'})
    data = pd.merge(data, user_item_day_hour, 'left',
                    on=['user_id', 'item_id', 'day', 'hour'])
    user_query_day = data.groupby(['user_id', 'day']).size().reset_index().rename(columns={0: 'user_query_day'})
    data = pd.merge(data, user_query_day, 'left', on=['user_id', 'day'])
    user_query_day_hour = data.groupby(['user_id', 'day', 'hour']).size().reset_index().rename(columns={0: 'user_query_day_hour'})
    data = pd.merge(data, user_query_day_hour, 'left', on=['user_id', 'day', 'hour'])
    return data


# 列别特征去中间的列别
def category_feat(data):
    data['item_category_list'] = data.item_category_list.apply(lambda s: (int([d for d in s.split(';')][1])))
    return data


# 商品按照类别、品牌、属性和城市做一个价格、销量、收藏、展示的均值
def item_mean_ratio(df, colname):
    grouped = df.groupby([colname])
    meancols = ['item_price_level', 'item_sales_level', 'item_collected_level', 'item_pv_level']
    df_g = grouped[meancols].mean().reset_index()
    colnames = [i for i in df_g.columns]
    for i in range(len(colnames)):
        if colnames[i] != colname:
            colnames[i] += '_mean_by_'+colname.split('_')[1]
    df_g.columns = colnames
    df=pd.merge(df, df_g, how='left', on=colname)
    colnames = colnames[1:]
    for i in range(len(colnames)):
        df[colnames[i]+'_ratio'] = df[meancols[i]]/df[colnames[i]]
    return df


# 和下一次点击的时间间隔
def nexttime_delta(column):
    data[column+'_nexttime_delta'] = 0
    train_data = data[['context_timestamp', column, column+'_nexttime_delta']].values
    nexttime_dict = {}
    for df_list in train_data:
        if df_list[1] not in nexttime_dict:
            df_list[2] = -1
            nexttime_dict[df_list[1]] = df_list[0]
        else:
            df_list[2] = nexttime_dict[df_list[1]] - df_list[0]
            nexttime_dict[df_list[1]] = df_list[0]
    data[['context_timestamp', column, column+'_nexttime_delta']] = train_data
    return data


# 添加属性特征
def properity_feat(data):
    for i in range(3):
        data['property_%d'%(i)] = data['item_property_list'].apply(
            lambda x: int(x.split(";")[i]) if len(x.split(";")) > i else -1)
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
    temp['click_user_item_lab'] = 0

    preix=-1
    prerow={'context_timestamp': -1,
            'item_id ': -1,
            'user_id': -1,
            'click_user_item_lab': -1}

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

    data['click_user_item_lab']=temp['click_user_item_lab']
    del temp
    gc.collect()

    #加入连续点击同一个user_id, item_brand_id特征，第一次点击，中间点击
    temp=data.ix[:,['context_timestamp',  'item_brand_id', 'user_id']]
    temp=temp.sort_values(by=['user_id','item_brand_id', 'context_timestamp'])
    temp['click_user_brand_lab'] = 0

    preix=-1
    prerow={'context_timestamp': -1,
            'item_brand_id': -1,
            'user_id': -1,
            'click_user_brand_lab': -1}

    for ix, row in temp.iterrows():
        if(row['user_id'] == prerow['user_id'] and row['item_brand_id'] == prerow['item_brand_id']):
            row['click_user_brand_lab'] = 2
            if prerow['click_user_brand_lab'] != 2:
                prerow['click_user_brand_lab'] = 1
        elif prerow['click_user_brand_lab'] == 2:
            prerow['click_user_brand_lab'] = 3
        preix = ix
        prerow = row
    temp = temp.sort_index()

    data['click_user_brand_lab'] = temp['click_user_brand_lab']
    del temp
    gc.collect()

    #加入连续点击同一个user_id, shop_id特征，第一次点击，中间点击
    temp = data.ix[:, ['context_timestamp', 'shop_id', 'user_id']]
    temp = temp.sort_values(by=['user_id', 'shop_id', 'context_timestamp'])
    temp['click_user_shop_lab'] = 0

    preix=-1
    prerow={'context_timestamp': -1,
            'shop_id': -1,
            'user_id': -1,
            'click_user_shop_lab': -1}

    for ix, row in temp.iterrows():
        if(row['user_id']==prerow['user_id'] and row['shop_id']==prerow['shop_id']):
            row['click_user_shop_lab']=2
            if prerow['click_user_shop_lab'] != 2:
                prerow['click_user_shop_lab'] = 1
        elif prerow['click_user_shop_lab'] ==2:
            prerow['click_user_shop_lab'] = 3
        preix=ix
        prerow=row
    temp = temp.sort_index()

    data['click_user_shop_lab'] = temp['click_user_shop_lab']
    del temp
    gc.collect()

    #加入连续点击同一个user_id, item_city_id特征，第一次点击，中间点击
    temp=data.ix[:, ['context_timestamp', 'item_city_id', 'user_id']]
    temp=temp.sort_values(by=['user_id', 'item_city_id', 'context_timestamp'])
    temp['click_user_city_lab'] = 0

    preix=-1
    prerow={'context_timestamp': -1,
            'item_city_id': -1,
            'user_id': -1,
            'click_user_city_lab': -1}

    for ix, row in temp.iterrows():
        if(row['user_id']==prerow['user_id'] and row['item_city_id']==prerow['item_city_id']):
            row['click_user_city_lab']=2
            if prerow['click_user_city_lab'] != 2:
                prerow['click_user_city_lab'] = 1
        elif prerow['click_user_city_lab'] == 2:
            prerow['click_user_city_lab'] = 3
        preix = ix
        prerow = row
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
    temp = data[['context_timestamp', 'item_id', 'day']]
    temp = temp.drop_duplicates(subset=subset, keep='last')
    temp.rename(columns={'context_timestamp': 'i_day_diffTime_last'}, inplace=True)
    data = pd.merge(data, temp, how='left', on=subset)
    data['i_day_diffTime_last'] = data['i_day_diffTime_last'] - data['context_timestamp']
    del temp
    gc.collect()
    data.loc[~data.duplicated(subset=subset, keep=False), ['i_day_diffTime_first', 'i_day_diffTime_last']] = -1
    return data


def feat():
    del_feat = ['user_id', 'context_id', 'instance_id', 'time', 'item_property_list', 'predict_category_property', 'context_timestamp', 'is_trade']

    features = []
    for feature in data.columns.values.tolist():
        if feature not in del_feat:
            features.append(feature)
    return features


# 本地交叉验证
def xgbCV(trian, test):
    features = feat()
    X = train[features]
    y = train['is_trade'].values
    X_test = test[features]
    y_test = test['is_trade'].values

    params = {'max_depth': 7,
              'nthread': 25,
              'eta': 0.05,
              'eval_metric': 'logloss',
              'objective': 'binary:logistic',
              'subsample': 0.7,
              'colsample_bytree': 0.5,
              'silent': 1,
              'seed': 1,
              'min_child_weight': 10
              #'scale_pos_weight':0.5
              }
    num_boost_round = 500

    dtrain = xgb.DMatrix(X, y)
    dvalid = xgb.DMatrix(X_test, y_test)

    watchlist = [(dtrain, 'train'), (dvalid, 'eval')]
    gbm = xgb.train(params, dtrain, num_boost_round, evals=watchlist,
                    early_stopping_rounds=200, verbose_eval=True)

    res = gbm.predict(xgb.DMatrix(X_test))
    print '本地cv:'
    print logloss(y_test, res)
    # 输出特征的重要性

    importance = gbm.get_fscore()
    importance = sorted(importance.items(), key=operator.itemgetter(1))
    df = pd.DataFrame(importance, columns=['feature', 'fscore'])
    print df
    return gbm.best_iteration


def submit(train, test, best_iter):
    features = feat()
    X = train[features]
    y = train['is_trade'].values
    X_test = test[features]

    params = {'max_depth': 7,
              'nthread': 25,
              'eta': 0.05,
              'eval_metric': 'logloss',
              'objective': 'binary:logistic',
              'subsample': 0.7,
              'colsample_bytree': 0.5,
              'silent': 1,
              'seed': 1,
              'min_child_weight': 10,
              'num_round': best_iter+1
              }

    dtrain = xgb.DMatrix(X, y)
    watchlist = [(dtrain, 'train')]
    gbm = xgb.train(params, dtrain, num_boost_round=best_iter, evals=watchlist,
                    early_stopping_rounds=200, verbose_eval=True)
    # gbm = xgb.train(params, dtrain, verbose_eval=True)
    '''
    测试文件的生成
    '''
    answer = gbm.predict(xgb.DMatrix(X_test))
    pd_result = pd.DataFrame({'instance_id': df_test["instance_id"], 'predicted_score': answer})
    pd_result.to_csv('result/result_xgb_v2.txt', index=False, sep=" ", float_format='%.6f')
    print '完成训练'


if __name__ == '__main__':
    '''
    # 划分训练集和测试集 训练集2018-09-18~2018-09-23 测试集2018-09-24
    '''
    dir = 'data/oria/'
    df = pd.read_table(dir + 'round1_ijcai_18_train_20180301.txt', engine='python', sep=" ")
    df_test = pd.read_table(dir + 'round1_ijcai_18_test_b_20180418.txt', engine='python', sep=" ")

    # 删除重复的 instance_id
    # df = df.drop_duplicates(['instance_id'])
    data = pd.concat([df, df_test])

    # 类别的特征 采用第二个特征
    data = category_feat(data)
    # 用户、商品 每天 每小时 点击特征
    data = convert_data(data)
    data = convert_item_user(data)
    # 商品的均值和率
    data = item_mean_ratio(data, 'item_category_list')
    data = item_mean_ratio(data, 'item_brand_id')
    data = item_mean_ratio(data, 'item_city_id')
    # 和下一次点击的时间间隔
    for column in ['user_id', 'item_id', 'item_brand_id', 'shop_id']:
        data = nexttime_delta(column)
    # 添加属性特征
    data = properity_feat(data)
    # 添加trick4特征
    data = do_Trick4(data)
    # 添加trick3特征
    data = do_Trick3(data)
    # 线下测试
    train = data.loc[data.day < 24]  # 18,19,20,21,22,23,24
    test = data.loc[data.day == 24]  # 暂时先使用第24天作为验证集
    best_iter = xgbCV(train, test)
    # 线上提交
    train = data[data.is_trade.notnull()]
    test = data[data.is_trade.isnull()]
    submit(train, test, best_iter)

    # 线下：0.0806596554941
    # 线上：0.08127
    # 添加trick4特征
    # 线下：0.0802056933493
    # 线上：0.08106
    # 添加trick3特征
    # 线下：0.0798451259784
    # 线上：0.08096
