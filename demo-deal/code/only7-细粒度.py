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


data_train=pd.read_csv('../data/round2_train.txt',delim_whitespace=True)
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
#data_test=pd.read_csv('../data/round2_ijcai_18_test_a_20180425.txt',delim_whitespace=True)
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
###########################################
##处理类目类特征，将类目类特征分为一列一列

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
train_data = train_data.sort_values('context_timestamp',ascending=True)
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
    print(len(df))

    
train_data['user_category_id'] = train_data['user_id'].astype(str)+"_"+train_data['item_category_list'].astype(str)
train_data['user_category_item_id'] = train_data['user_id'].astype(str)+"_"+train_data['item_category_list'].astype(str)+"_"+train_data['item_id'].astype(str)
train_data['user_category_shop_id'] =train_data['user_id'].astype(str)+"_" +train_data['item_category_list'].astype(str)+"_"+train_data['shop_id'].astype(str)
train_data['user_category_brand_id'] =train_data['user_id'].astype(str)+"_"+train_data['item_category_list'].astype(str)+"_"+train_data['item_brand_id'].astype(str)
train_data['user_category_city_id'] = train_data['user_id'].astype(str)+"_"+train_data['item_category_list'].astype(str)+"_"+train_data['item_city_id'].astype(str)

train_data['user_property_id'] = train_data['user_id'].astype(str)+"_"+train_data['item_property_list'].astype(str)
train_data['user_property_item_id'] = train_data['user_id'].astype(str)+"_"+train_data['item_property_list'].astype(str)+"_"+train_data['item_id'].astype(str)
train_data['user_property_shop_id'] =train_data['user_id'].astype(str)+"_" +train_data['item_property_list'].astype(str)+"_"+train_data['shop_id'].astype(str)
train_data['user_property_brand_id'] =train_data['user_id'].astype(str)+"_"+train_data['item_property_list'].astype(str)+"_"+train_data['item_brand_id'].astype(str)
train_data['user_property_city_id'] = train_data['user_id'].astype(str)+"_"+train_data['item_property_list'].astype(str)+"_"+train_data['item_city_id'].astype(str)
##统计各类别在总样本中的count数
columns=[
'user_category_id','user_category_item_id','user_category_shop_id',
'user_category_brand_id','user_category_city_id',
'user_property_id','user_property_item_id',
'user_property_shop_id','user_property_brand_id','user_property_city_id'
]
for column in columns:
    print(column)
    count_cat_prep(train_data,column,column+'_click_count_before')
    
query_before=train_data[[
'user_category_id_click_count_before','user_category_item_id_click_count_before',
'user_category_shop_id_click_count_before','user_category_brand_id_click_count_before','user_category_city_id_click_count_before',
'user_property_id_click_count_before','user_property_item_id_click_count_before',
'user_property_shop_id_click_count_before','user_property_brand_id_click_count_before','user_property_city_id_click_count_before',
]]

train_data = train_data.sort_values('context_timestamp',ascending=False)
for column in columns:
    print(column)
    count_cat_prep(train_data,column,column+'_click_count_after')
    
query_after=train_data[[
'user_category_id_click_count_after','user_category_item_id_click_count_after',
'user_category_shop_id_click_count_after','user_category_brand_id_click_count_after','user_category_city_id_click_count_after',
'user_property_id_click_count_after','user_property_item_id_click_count_after',
'user_property_shop_id_click_count_after','user_property_brand_id_click_count_after','user_property_city_id_click_count_after',
]]

query_before.to_csv('../data/other_feat/query_before.csv',index=False)
query_after.to_csv('../data/other_feat/query_after.csv',index=False)
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
####################################################################################
def convert_all_to7_data(data):
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



###############################################################

# -*- coding: UTF-8 -*-
import time
from datetime import datetime
from datetime import timedelta
import pandas as pd
import pickle
import os
import math
import numpy as np
import gc
import lightgbm as lgb
from sklearn.metrics import log_loss

from get_feature import *

train_path = '../data/round2_train.txt'

def user_buy_feat(start_date):
	data = load_data(train_path)
	data = data[data.day < start_date]
	user = pd.DataFrame(data[['user_id']].drop_duplicates())
	user_collect_mean = data[['user_id','item_collected_level']].groupby('user_id').mean().reset_index().rename(columns={'item_collected_level':'user_item_collect_mean'})
	user_price_mean = data[['user_id','item_price_level']].groupby('user_id').mean().reset_index().rename(columns={'item_price_level':'user_item_price_mean'})
	user_sales_mean = data[['user_id','item_sales_level']].groupby('user_id').mean().reset_index().rename(columns={'item_sales_level':'user_item_sales_mean'})
	user_pv_mean = data[['user_id','item_pv_level']].groupby('user_id').mean().reset_index().rename(columns={'item_pv_level':'user_item_pv_mean'})

	user = pd.merge(user,user_collect_mean,how='left',on='user_id',)
	user = pd.merge(user,user_price_mean,how='left',on='user_id',)
	user = pd.merge(user,user_sales_mean,how='left',on='user_id',)
	user = pd.merge(user,user_pv_mean,how='left',on='user_id',)
	
    return user

def shop_buy_feat(start_date):
    
	data = load_data(train_path)
	data = data[data.day < start_date]
	shop = pd.DataFrame(data[['shop_id']].drop_duplicates())
	shop_user_gender0_ratio = data[['shop_id','user_gender_id']].groupby(['shop_id'])['user_gender_id']\
		.agg([lambda x: np.mean(x==0)]).reset_index().rename(columns={'<lambda>': 'shop_user_gender0_ratio'})
	shop_user_gender1_ratio = data[['shop_id','user_gender_id']].groupby(['shop_id'])['user_gender_id']\
		.agg([lambda x: np.mean(x==1)]).reset_index().rename(columns={'<lambda>': 'shop_user_gender1_ratio'})
	shop_user_age_mean = data[['shop_id','user_age_level']].groupby('shop_id').mean().reset_index().rename(columns={'user_age_level':'shop_user_age_mean'})  
	shop_user_star_mean = data[['shop_id','user_star_level']].groupby('shop_id').mean().reset_index().rename(columns={'user_star_level':'shop_user_star_mean'})
	shop_item_price_mean = data[['shop_id','item_price_level']].groupby('shop_id').mean().reset_index().rename(columns={'item_price_level':'shop_item_price_mean'})
	tmp = data[['shop_id','item_id']].drop_duplicates()
	shop_item_num = tmp.groupby('shop_id').count().reset_index().rename(columns={'item_id':'shop_item_num'})  

	shop = pd.merge(shop,shop_user_gender0_ratio,on='shop_id',how='left')
	shop = pd.merge(shop,shop_user_gender1_ratio,on='shop_id',how='left')
	shop = pd.merge(shop,shop_user_age_mean,on='shop_id',how='left')
	shop = pd.merge(shop,shop_user_star_mean,on='shop_id',how='left') 
	shop = pd.merge(shop,shop_item_price_mean,on='shop_id',how='left') 
	shop = pd.merge(shop,shop_item_num,on='shop_id',how='left')

    return shop

def item_buy_feat(start_date):    
	data = load_data(train_path)
	data = data[data.day < start_date]
	item = pd.DataFrame(data[['item_id']].drop_duplicates())
	item_user_gender0_ratio = data[['item_id','user_gender_id']].groupby(['item_id'])['user_gender_id']\
		.agg([lambda x: np.mean(x==0)]).reset_index().rename(columns={'<lambda>': 'item_user_gender0_ratio'})
	item_user_gender1_ratio = data[['item_id','user_gender_id']].groupby(['item_id'])['user_gender_id']\
		.agg([lambda x: np.mean(x==1)]).reset_index().rename(columns={'<lambda>': 'item_user_gender1_ratio'})
	item_user_age_mean = data[['item_id','user_age_level']].groupby('item_id').mean().reset_index().rename(columns={'user_age_level':'item_user_age_mean'})  
	item_user_star_mean = data[['item_id','user_star_level']].groupby('item_id').mean().reset_index().rename(columns={'user_star_level':'item_user_star_mean'})

	item = pd.merge(item,item_user_gender0_ratio,on='item_id',how='left')
	item = pd.merge(item,item_user_gender1_ratio,on='item_id',how='left')
	item = pd.merge(item,item_user_age_mean,on='item_id',how='left')
	item = pd.merge(item,item_user_star_mean,on='item_id',how='left') 

	item.to_csv(dump_path, index=False)
    return item

def item_brand_buy_feat(start_date):
    
	data = load_data(train_path)
	data = data[data.day < start_date]
	item_brand = pd.DataFrame(data[['item_brand_id']].drop_duplicates())
	item_brand_user_gender0_ratio = data[['item_brand_id','user_gender_id']].groupby(['item_brand_id'])['user_gender_id']\
		.agg([lambda x: np.mean(x==0)]).reset_index().rename(columns={'<lambda>': 'item_brand_user_gender0_ratio'})
	item_brand_user_gender1_ratio = data[['item_brand_id','user_gender_id']].groupby(['item_brand_id'])['user_gender_id']\
		.agg([lambda x: np.mean(x==1)]).reset_index().rename(columns={'<lambda>': 'item_brand_user_gender1_ratio'})
	item_brand_user_age_mean = data[['item_brand_id','user_age_level']].groupby('item_brand_id').mean().reset_index().rename(columns={'user_age_level':'item_brand_user_age_mean'})  
	item_brand_user_star_mean = data[['item_brand_id','user_star_level']].groupby('item_brand_id').mean().reset_index().rename(columns={'user_star_level':'item_brand_user_star_mean'})

	item_brand = pd.merge(item_brand,item_brand_user_gender0_ratio,on='item_brand_id',how='left')
	item_brand = pd.merge(item_brand,item_brand_user_gender1_ratio,on='item_brand_id',how='left')
	item_brand = pd.merge(item_brand,item_brand_user_age_mean,on='item_brand_id',how='left')
	item_brand = pd.merge(item_brand,item_brand_user_star_mean,on='item_brand_id',how='left') 

    return item_brand

def user_item_shop_feat(lab,dat,start_date):
    
	data = dat.copy()
	del dat

	data['hour2'] = data['hour'].map(lambda x: 1 if x<=12 else 2)
	user = pd.DataFrame(data[['user_id']].drop_duplicates())
	user_item_collect_mean = data[['user_id','item_collected_level']].groupby('user_id').mean().reset_index().rename(columns={'item_collected_level':'user_item_collect_mean'})
	user_item_price_mean = data[['user_id','item_price_level']].groupby('user_id').mean().reset_index().rename(columns={'item_price_level':'user_item_price_mean'})
	user_item_sales_mean = data[['user_id','item_sales_level']].groupby('user_id').mean().reset_index().rename(columns={'item_sales_level':'user_item_sales_mean'})
	user_item_pv_mean = data[['user_id','item_pv_level']].groupby('user_id').mean().reset_index().rename(columns={'item_pv_level':'user_item_pv_mean'})

	user = pd.merge(user,user_item_collect_mean,on='user_id',how='left')
	user = pd.merge(user,user_item_price_mean,on='user_id',how='left')
	user = pd.merge(user,user_item_sales_mean,on='user_id',how='left')
	user = pd.merge(user,user_item_pv_mean,on='user_id',how='left')
   
	shop = pd.DataFrame(data[['shop_id']].drop_duplicates())
	shop_user_gender0_ratio = data[['shop_id','user_gender_id']].groupby(['shop_id'])['user_gender_id']\
		.agg([lambda x: np.mean(x==0)]).reset_index().rename(columns={'<lambda>': 'shop_user_gender0_ratio'})
	shop_user_gender1_ratio = data[['shop_id','user_gender_id']].groupby(['shop_id'])['user_gender_id']\
		.agg([lambda x: np.mean(x==1)]).reset_index().rename(columns={'<lambda>': 'shop_user_gender1_ratio'})
	shop_user_age_mean = data[['shop_id','user_age_level']].groupby('shop_id').mean().reset_index().rename(columns={'user_age_level':'shop_user_age_mean'})  
	shop_user_star_mean = data[['shop_id','user_star_level']].groupby('shop_id').mean().reset_index().rename(columns={'user_star_level':'shop_user_star_mean'})
	shop_item_price_mean = data[['shop_id','item_price_level']].groupby('shop_id').mean().reset_index().rename(columns={'item_price_level':'shop_item_price_mean'})
	tmp = data[['shop_id','item_id']].drop_duplicates()
	shop_item_num = tmp.groupby('shop_id').count().reset_index().rename(columns={'item_id':'shop_item_num'})  

	shop = pd.merge(shop,shop_user_gender0_ratio,on='shop_id',how='left')
	shop = pd.merge(shop,shop_user_gender1_ratio,on='shop_id',how='left')
	shop = pd.merge(shop,shop_user_age_mean,on='shop_id',how='left')
	shop = pd.merge(shop,shop_user_star_mean,on='shop_id',how='left') 
	shop = pd.merge(shop,shop_item_price_mean,on='shop_id',how='left') 
	shop = pd.merge(shop,shop_item_num,on='shop_id',how='left')

	item = pd.DataFrame(data[['item_id']].drop_duplicates())
	item_user_gender0_ratio = data[['item_id','user_gender_id']].groupby(['item_id'])['user_gender_id']\
		.agg([lambda x: np.mean(x==0)]).reset_index().rename(columns={'<lambda>': 'item_user_gender0_ratio'})
	item_user_gender1_ratio = data[['item_id','user_gender_id']].groupby(['item_id'])['user_gender_id']\
		.agg([lambda x: np.mean(x==1)]).reset_index().rename(columns={'<lambda>': 'item_user_gender1_ratio'})
	item_user_age_mean = data[['item_id','user_age_level']].groupby('item_id').mean().reset_index().rename(columns={'user_age_level':'item_user_age_mean'})  
	item_user_star_mean = data[['item_id','user_star_level']].groupby('item_id').mean().reset_index().rename(columns={'user_star_level':'item_user_star_mean'})

	item = pd.merge(item,item_user_gender0_ratio,on='item_id',how='left')
	item = pd.merge(item,item_user_gender1_ratio,on='item_id',how='left')
	item = pd.merge(item,item_user_age_mean,on='item_id',how='left')
	item = pd.merge(item,item_user_star_mean,on='item_id',how='left') 

	item_brand = pd.DataFrame(data[['item_brand_id']].drop_duplicates())
	item_brand_user_gender0_ratio = data[['item_brand_id','user_gender_id']].groupby(['item_brand_id'])['user_gender_id']\
		.agg([lambda x: np.mean(x==0)]).reset_index().rename(columns={'<lambda>': 'item_brand_user_gender0_ratio'})
	item_brand_user_gender1_ratio = data[['item_brand_id','user_gender_id']].groupby(['item_brand_id'])['user_gender_id']\
		.agg([lambda x: np.mean(x==1)]).reset_index().rename(columns={'<lambda>': 'item_brand_user_gender1_ratio'})
	item_brand_user_age_mean = data[['item_brand_id','user_age_level']].groupby('item_brand_id').mean().reset_index().rename(columns={'user_age_level':'item_brand_user_age_mean'})  
	item_brand_user_star_mean = data[['item_brand_id','user_star_level']].groupby('item_brand_id').mean().reset_index().rename(columns={'user_star_level':'item_brand_user_star_mean'})

	item_brand = pd.merge(item_brand,item_brand_user_gender0_ratio,on='item_brand_id',how='left')
	item_brand = pd.merge(item_brand,item_brand_user_gender1_ratio,on='item_brand_id',how='left')
	item_brand = pd.merge(item_brand,item_brand_user_age_mean,on='item_brand_id',how='left')
	item_brand = pd.merge(item_brand,item_brand_user_star_mean,on='item_brand_id',how='left') 


	data = data[['instance_id','hour2','user_id','shop_id','item_id','item_brand_id']]
	data = pd.merge(data,user,on='user_id',how='left')
	data = pd.merge(data,shop,on='shop_id',how='left')
	data = pd.merge(data,item,on='item_id',how='left')
	data = pd.merge(data,item_brand,on='item_brand_id',how='left')
	data = data[['instance_id','hour2','user_item_collect_mean','user_item_price_mean','user_item_sales_mean','user_item_pv_mean',
			'shop_user_gender0_ratio','shop_user_gender1_ratio','shop_user_age_mean','shop_user_star_mean','shop_item_price_mean','shop_item_num',
			'item_user_gender0_ratio','item_user_gender1_ratio','item_user_age_mean','item_user_star_mean',
			'item_brand_user_gender0_ratio','item_brand_user_gender1_ratio','item_brand_user_age_mean','item_brand_user_star_mean',
			]]
	data.columns = ['instance_id','hour2','user_day_item_collect_mean','user_day_item_price_mean','user_day_item_sales_mean','user_day_item_pv_mean',
			'shop_day_user_gende0_ratio','shop_day_user_gender1_ratio','shop_day_user_age_mean','shop_day_user_star_mean','shop_day_item_price_mean','shop_day_item_num',
			'item_day_user_gende0_ratio','item_day_user_gender1_ratio','item_day_user_age_mean','item_day_user_star_mean',
			'item_brand_day_user_gende0_ratio','item_brand_day_user_gender1_ratio','item_brand_day_user_age_mean','item_brand_day_user_star_mean',
			]
        
    return data        

def count_feat(lab,dat,start_date):
	data = dat.copy()
	del dat

	user = pd.DataFrame(data[['user_id']].drop_duplicates())
	shop = pd.DataFrame(data[['shop_id']].drop_duplicates())
	item = pd.DataFrame(data[['item_id']].drop_duplicates())
	user_shop = pd.DataFrame(data[['user_id','shop_id']].drop_duplicates())
	user_item = pd.DataFrame(data[['user_id','item_id']].drop_duplicates())
	shop_item = pd.DataFrame(data[['shop_id','item_id']].drop_duplicates())

	user_day_num = data[['user_id','item_id']].groupby('user_id').count().reset_index().rename(columns={'item_id':'user_day_num'})
	user_item_day_num = data[['user_id','item_id','shop_id']].groupby(['user_id','item_id']).count().reset_index().rename(columns={'shop_id':'user_item_day_num'})
	user_shop_day_num = data[['user_id','item_id','shop_id']].groupby(['user_id','shop_id']).count().reset_index().rename(columns={'item_id':'user_shop_day_num'})        
  
	shop_day_num = data[['shop_id','item_id']].groupby('shop_id').count().reset_index().rename(columns={'item_id':'shop_day_num'})
	shop_item_day_num = data[['user_id','item_id','shop_id']].groupby(['shop_id','item_id']).count().reset_index().rename(columns={'user_id':'shop_item_day_num'})
	shop_item_day_type = data[['shop_id','item_id']].drop_duplicates().groupby(['shop_id']).count().reset_index().rename(columns={'item_id':'shop_item_day_type'})

	item_day_num = data[['shop_id','item_id']].groupby('item_id').count().reset_index().rename(columns={'shop_id':'item_day_num'})

	shop_day_hour_num = data[['shop_id','hour','item_id']].groupby(['shop_id','hour']).count().reset_index().rename(columns={'item_id':'shop_day_hour_num'})
	item_day_hour_num = data[['shop_id','hour','item_id']].groupby(['item_id','hour']).count().reset_index().rename(columns={'shop_id':'item_day_hour_num'})
	shop_item_day_hour_num = data[['user_id','item_id','shop_id','hour']].groupby(['shop_id','item_id','hour']).count().reset_index().rename(columns={'user_id':'shop_item_day_hour_num'})

	user = pd.merge(user,user_day_num,on='user_id',how='left')
	shop = pd.merge(shop,shop_day_num,on='shop_id',how='left')
	shop = pd.merge(shop,shop_item_day_type,on='shop_id',how='left')
	item = pd.merge(item,item_day_num,on='item_id',how='left')
	user_shop = pd.merge(user_shop,user_shop_day_num,on=['user_id','shop_id'],how='left')
	user_item = pd.merge(user_item,user_item_day_num,on=['user_id','item_id'],how='left')
	shop_item = pd.merge(shop_item,shop_item_day_num,on=['shop_id','item_id'],how='left')

	data = data[['instance_id','user_id','shop_id','item_id','hour']]
	data = pd.merge(data,user,on='user_id',how='left')
	data = pd.merge(data,shop,on='shop_id',how='left')
	data = pd.merge(data,item,on='item_id',how='left')
	data = pd.merge(data,user_shop,on=['user_id','shop_id'],how='left')
	data = pd.merge(data,user_item,on=['user_id','item_id'],how='left')
	data = pd.merge(data,shop_item,on=['shop_id','item_id'],how='left')

	data = pd.merge(data,shop_day_hour_num,on=['shop_id','hour'],how='left')
	data = pd.merge(data,item_day_hour_num,on=['item_id','hour'],how='left')
	data = pd.merge(data,shop_item_day_hour_num,on=['shop_id','item_id','hour'],how='left')

	data['user_shop_day_rt'] = data['user_shop_day_num'] / data['user_day_num']
	data['user_item_day_rt'] = data['user_item_day_num'] / data['user_day_num']
	data['shop_user_day_rt'] = data['user_shop_day_num'] / data['shop_day_num']
	data['item_user_day_rt'] = data['user_item_day_num'] / data['item_day_num']
	data['shop_item_day_rt'] = data['shop_item_day_num'] / data['shop_day_num']

	data = data[['instance_id','shop_day_num','shop_item_day_type','shop_item_day_num','item_day_num','user_shop_day_rt','user_item_day_rt',
	'shop_user_day_rt','item_user_day_rt','shop_item_day_rt','shop_day_hour_num','item_day_hour_num','shop_item_day_hour_num']]
      
    return data 



def cate_feat(feat,start_date):
    
	data = load_data(train_path)
	data = data[data.day < start_date]
	for i in range(0, 3):
			data['item_category_list_' + str(i)] = data['item_category_list'].map(
				lambda x: str(str(x).split(';')[i]) if len(str(x).split(';')) > i else '')
	if int(start_date)==0:
		gap = 1
	else:
		gap = int(start_date)

	single_click = data.groupby([feat]).size().reset_index().rename(columns={0: feat+'_click_num'})
	single_click_use = data[data.is_trade == 1]
	single_click_use = single_click_use.groupby([feat]).size().reset_index().rename(columns={0: feat+'_click_use'})
	data = pd.merge(single_click, single_click_use, how='left', on=feat)
	data.fillna(0, inplace=True)
	data[feat+'_click_use'] = data[feat+'_click_use'] / gap
	data[feat+'_click_num'] = data[feat+'_click_num'] / gap
	data[feat+'_click_ratio'] = data[feat+'_click_use'] / data[feat+'_click_num']


    return data


def click_day_feat(dat,feat,start_date):    
	data = dat.copy()
	del dat
	data = data[data.hour < 12]
	single_click = data.groupby([feat]).size().reset_index().rename(columns={0: feat+'_day_click_num'})
	single_click_use = data[data.is_trade == 1]
	single_click_use = single_click_use.groupby([feat]).size().reset_index().rename(columns={0: feat+'_day_click_use'})
	data = pd.merge(single_click, single_click_use, how='left', on=feat)
	data.fillna(0, inplace=True)
	data[feat+'_day_click_ratio'] = data[feat+'_day_click_use'] / data[feat+'_day_click_num']


    return data





def item_ch_feat(dat,start_date):
  
	data = load_data(train_path)
	data = data[data.day < start_date]
	tst1 = data[['item_id','item_sales_level']].groupby(['item_id'])['item_sales_level'].mean().reset_index().rename(columns={'item_sales_level': 'item_sales_level_mean'})
	tst2 = data[['item_id','item_price_level']].groupby(['item_id'])['item_price_level'].mean().reset_index().rename(columns={'item_price_level': 'item_price_level_mean'})
	
	df = dat.copy()
	del dat             
	tst0 = pd.merge(df[['item_id','item_sales_level']],tst1,on='item_id',how='left')
	tst0['item_sales_level_ch'] = tst0['item_sales_level_mean'] / tst0['item_sales_level']

	tst3 = pd.merge(df[['item_id','item_price_level']],tst2,on='item_id',how='left')
	tst3['item_price_level_ch'] = tst3['item_price_level_mean'] / tst3['item_price_level']

	df['item_sales_level_ch'] = tst0['item_sales_level_ch']
	df['item_price_level_ch'] = tst3['item_price_level_ch']  
	df['item_sales_level_ch'].fillna(1.0,inplace=True)
	df['item_price_level_ch'].fillna(1.0,inplace=True)

    return data


# def get_()



def gap_time_feat(lab,dat,start_date):
	data = dat.copy()
	data1 = data.copy()
	del dat

	def bef1_count(x,y):
		count=0
		all=str(y).split('|')
		for i in all:
		   if (int(x)>int(i))&((int(x)-int(i))<3600):
			   count=count+1
		return count 
	data1['now_date']=data1['now_date'].astype('str')
	tst1 = data1[['user_id','now_date']].groupby('user_id')['now_date'].agg(lambda x:'|'.join(x)).reset_index().rename(columns={'now_date':'user_click_time'})
	tst1 = pd.merge(data[['user_id','now_date']],tst1,on='user_id',how='left')
	tst1['user_bef1_count'] = tst1.apply(lambda x:bef1_count(x['now_date'],x['user_click_time']),axis=1)
	
	tst2 = data1[['user_id','shop_id','now_date']].groupby(['user_id','shop_id'])['now_date'].agg(lambda x:'|'.join(x)).reset_index().rename(columns={'now_date':'user_shop_click_time'})
	tst2 = pd.merge(data[['user_id','shop_id','now_date']],tst2,on=['user_id','shop_id'],how='left')
	tst2['user_shop_bef1_count'] = tst2.apply(lambda x:bef1_count(x['now_date'],x['user_shop_click_time']),axis=1)

	tst3 = data1[['user_id','item_id','now_date']].groupby(['user_id','item_id'])['now_date'].agg(lambda x:'|'.join(x)).reset_index().rename(columns={'now_date':'user_item_click_time'})
	tst3 = pd.merge(data[['user_id','item_id','now_date']],tst3,on=['user_id','item_id'],how='left')
	tst3['user_item_bef1_count'] = tst3.apply(lambda x:bef1_count(x['now_date'],x['user_item_click_time']),axis=1)

	def bef6_count(x,y):
		count=0
		all=str(y).split('|')
		for i in all:
		   if (int(x)>int(i))&((int(x)-int(i))<21600):
			   count=count+1
		return count 
	data1['now_date']=data1['now_date'].astype('str')
	tst4 = data1[['user_id','now_date']].groupby('user_id')['now_date'].agg(lambda x:'|'.join(x)).reset_index().rename(columns={'now_date':'user_click_time'})
	tst4 = pd.merge(data[['user_id','now_date']],tst4,on='user_id',how='left')
	tst4['user_bef6_count'] = tst4.apply(lambda x:bef6_count(x['now_date'],x['user_click_time']),axis=1)
	
	tst5 = data1[['user_id','shop_id','now_date']].groupby(['user_id','shop_id'])['now_date'].agg(lambda x:'|'.join(x)).reset_index().rename(columns={'now_date':'user_shop_click_time'})
	tst5 = pd.merge(data[['user_id','shop_id','now_date']],tst5,on=['user_id','shop_id'],how='left')
	tst5['user_shop_bef6_count'] = tst5.apply(lambda x:bef6_count(x['now_date'],x['user_shop_click_time']),axis=1)

	tst6 = data1[['user_id','item_id','now_date']].groupby(['user_id','item_id'])['now_date'].agg(lambda x:'|'.join(x)).reset_index().rename(columns={'now_date':'user_item_click_time'})
	tst6 = pd.merge(data[['user_id','item_id','now_date']],tst6,on=['user_id','item_id'],how='left')
	tst6['user_item_bef6_count'] = tst6.apply(lambda x:bef6_count(x['now_date'],x['user_item_click_time']),axis=1)

	def aft1_count(x,y):
		count=0
		all=str(y).split('|')
		for i in all:
		   if (int(x)<int(i))&((int(i)-int(x))<3600):
			   count=count+1
		return count 
	data1['now_date']=data1['now_date'].astype('str')
	tst7 = data1[['user_id','now_date']].groupby('user_id')['now_date'].agg(lambda x:'|'.join(x)).reset_index().rename(columns={'now_date':'user_click_time'})
	tst7 = pd.merge(data[['user_id','now_date']],tst7,on='user_id',how='left')
	tst7['user_aft1_count'] = tst7.apply(lambda x:aft1_count(x['now_date'],x['user_click_time']),axis=1)
	
	tst8 = data1[['user_id','shop_id','now_date']].groupby(['user_id','shop_id'])['now_date'].agg(lambda x:'|'.join(x)).reset_index().rename(columns={'now_date':'user_shop_click_time'})
	tst8 = pd.merge(data[['user_id','shop_id','now_date']],tst8,on=['user_id','shop_id'],how='left')
	tst8['user_shop_aft1_count'] = tst8.apply(lambda x:aft1_count(x['now_date'],x['user_shop_click_time']),axis=1)

	tst9 = data1[['user_id','item_id','now_date']].groupby(['user_id','item_id'])['now_date'].agg(lambda x:'|'.join(x)).reset_index().rename(columns={'now_date':'user_item_click_time'})
	tst9 = pd.merge(data[['user_id','item_id','now_date']],tst9,on=['user_id','item_id'],how='left')
	tst9['user_item_aft1_count'] = tst9.apply(lambda x:aft1_count(x['now_date'],x['user_item_click_time']),axis=1)

	def aft6_count(x,y):
		count=0
		all=str(y).split('|')
		for i in all:
		   if (int(x)<int(i))&((int(i)-int(x))<21600):
			   count=count+1
		return count 
	data1['now_date']=data1['now_date'].astype('str')
	tst10 = data1[['user_id','now_date']].groupby('user_id')['now_date'].agg(lambda x:'|'.join(x)).reset_index().rename(columns={'now_date':'user_click_time'})
	tst10 = pd.merge(data[['user_id','now_date']],tst10,on='user_id',how='left')
	tst10['user_aft6_count'] = tst10.apply(lambda x:aft6_count(x['now_date'],x['user_click_time']),axis=1)
	
	tst11 = data1[['user_id','shop_id','now_date']].groupby(['user_id','shop_id'])['now_date'].agg(lambda x:'|'.join(x)).reset_index().rename(columns={'now_date':'user_shop_click_time'})
	tst11 = pd.merge(data[['user_id','shop_id','now_date']],tst11,on=['user_id','shop_id'],how='left')
	tst11['user_shop_aft6_count'] = tst11.apply(lambda x:aft6_count(x['now_date'],x['user_shop_click_time']),axis=1)

	tst12 = data1[['user_id','item_id','now_date']].groupby(['user_id','item_id'])['now_date'].agg(lambda x:'|'.join(x)).reset_index().rename(columns={'now_date':'user_item_click_time'})
	tst12 = pd.merge(data[['user_id','item_id','now_date']],tst12,on=['user_id','item_id'],how='left')
	tst12['user_item_aft6_count'] = tst12.apply(lambda x:aft6_count(x['now_date'],x['user_item_click_time']),axis=1)

	def bef6_count(x,y):
		count=0
		all=str(y).split('|')
		for i in all:
		   if (int(x)>int(i))&((int(x)-int(i))<21600):
			   count=count+1
		return count 
	data1['now_date']=data1['now_date'].astype('str')
	tst13 = data1[['item_id','now_date']].groupby('item_id')['now_date'].agg(lambda x:'|'.join(x)).reset_index().rename(columns={'now_date':'item_click_time'})
	tst13 = pd.merge(data[['item_id','now_date']],tst13,on='item_id',how='left')
	tst13['item_befor6_count'] = tst13.apply(lambda x:bef6_count(x['now_date'],x['item_click_time']),axis=1)

	def aft6_count(x,y):
		count=0
		all=str(y).split('|')
		for i in all:
		   if (int(x)<int(i))&((int(i)-int(x))<21600):
			   count=count+1
		return count 
	data1['now_date']=data1['now_date'].astype('str')
	tst14 = data1[['item_id','now_date']].groupby('item_id')['now_date'].agg(lambda x:'|'.join(x)).reset_index().rename(columns={'now_date':'item_click_time'})
	tst14 = pd.merge(data[['item_id','now_date']],tst14,on='item_id',how='left')
	tst14['item_after6_count'] = tst14.apply(lambda x:aft6_count(x['now_date'],x['item_click_time']),axis=1)

	data = pd.concat([data[['instance_id']],tst1['user_bef1_count']],axis=1)
	data['user_shop_bef1_count'] = tst2['user_shop_bef1_count']
	data['user_item_bef1_count'] = tst3['user_item_bef1_count']
	data['user_bef6_count'] = tst4['user_bef6_count']
	data['user_shop_bef6_count'] = tst5['user_shop_bef6_count']
	data['user_item_bef6_count'] = tst6['user_item_bef6_count']
	data['user_aft1_count'] = tst7['user_aft1_count']
	data['user_shop_aft1_count'] = tst8['user_shop_aft1_count']
	data['user_item_aft1_count'] = tst9['user_item_aft1_count']
	data['user_aft6_count'] = tst10['user_aft6_count']
	data['user_shop_aft6_count'] = tst11['user_shop_aft6_count']
	data['user_item_aft6_count'] = tst12['user_item_aft6_count']
	data['item_befor6_count'] = tst13['item_befor6_count']
	data['item_after6_count'] = tst14['item_after6_count']      

	data = data[['instance_id','user_bef1_count','user_shop_bef1_count','user_item_bef1_count','user_bef6_count',
	'user_shop_bef6_count','user_item_bef6_count','user_aft1_count','user_shop_aft1_count','user_item_aft1_count',
	'user_aft6_count','user_shop_aft6_count','user_item_aft6_count','item_befor6_count','item_after6_count']]
      
    return data 

def shift_feat(lab,dat,start_date):
    
	data = dat.copy()
	del dat
	data = data[['instance_id','user_id','shop_id','item_id','item_brand_id','context_timestamp']]
	
	data['context_timestamp'] = data['context_timestamp'].astype(int)
	data = data.sort_values('context_timestamp').reset_index()
	data['user_day_af2click'] = data.groupby(['user_id']).context_timestamp.shift(-2) - data.context_timestamp
	data['user_day_pre2click'] = data.context_timestamp - data.groupby(['user_id']).context_timestamp.shift(2)
	data['user_day_shop_af2click'] = data.groupby(['user_id','shop_id']).context_timestamp.shift(-2) - data.context_timestamp
	data['user_day_shop_pre2click'] = data.context_timestamp - data.groupby(['user_id','shop_id']).context_timestamp.shift(2)
	data['user_day_item_af2click'] = data.groupby(['user_id','item_id']).context_timestamp.shift(-2) - data.context_timestamp
	data['user_day_item_pre2click'] = data.context_timestamp - data.groupby(['user_id','item_id']).context_timestamp.shift(2)

	data['item_day_afclick'] = data.groupby(['item_id']).context_timestamp.shift(-1) - data.context_timestamp
	data['item_day_preClick'] = data.context_timestamp - data.groupby(['item_id']).context_timestamp.shift(1)

	data['shop_day_afclick'] = data.groupby(['shop_id']).context_timestamp.shift(-1) - data.context_timestamp
	data['shop_day_preclick'] = data.context_timestamp - data.groupby(['shop_id']).context_timestamp.shift(1)

	data['item_brand_day_afclick'] = data.groupby(['item_brand_id']).context_timestamp.shift(-1) - data.context_timestamp
	data['item_brand_day_preclick'] = data.context_timestamp - data.groupby(['item_brand_id']).context_timestamp.shift(1)

	data = data[['instance_id', 'user_day_af2click', 'user_day_pre2click', 
	  'user_day_shop_af2click', 'user_day_shop_pre2click', 'user_day_item_af2click', 'user_day_item_pre2click',
	  'item_day_afclick','item_day_preClick','shop_day_afclick','shop_day_preclick','item_brand_day_afclick',
	  'item_brand_day_preclick'
	  ]]
     

    return data

def train_feat(start_date): 
    user_buy_feat = user_buy_feat(start_date)
    shop_buy_feat = shop_buy_feat(start_date)
    item_buy_feat = item_buy_feat(start_date)
    item_brand_buy_feat = item_brand_buy_feat(start_date)
    item_brand_click_feature = single_click_feature('item_brand_id',start_date)
    item_sales_level_click_feature = single_click_feature('item_sales_level',start_date)
    item_category_list_1_click_feature = cate_feat('item_category_list_1',start_date)
    user_hour_feature = group_rt_feature('user_id','hour',start_date)
    shop_hour_feature = group_rt_feature('shop_id','hour',start_date)
    item_hour_feature = group_rt_feature('item_id','hour',start_date)
    shop_age_feature = group_rt_feature('shop_id','user_age_level',start_date)
    shop_star_feature = group_rt_feature('shop_id','user_star_level',start_date)
    item_age_feature = group_rt_feature('item_id','user_age_level',start_date)
    item_star_feature = group_rt_feature('item_id','user_star_level',start_date)
    leak2_feauture = user_item_shop_feat('train',train_data,start_date)
    leak3_feauture = count_feat('train',train_data,start_date)
    user_day_click_feature = click_day_feat(train_data,'user_id',start_date)
    shop_day_click_feature = click_day_feat(train_data,'shop_id',start_date)
    item_day_click_feature = click_day_feat(train_data,'item_id',start_date)
    item_brand_day_click_feature = click_day_feat(train_data,'item_brand_id',start_date)
    item_city_day_click_feature = click_day_feat(train_data,'item_city_id',start_date)
    item_category_list_1_day_click_feature = click_day_feat(train_data,'item_category_list_1',start_date)
    item_ch_feat = item_ch_feat(train_data,start_date)
    leak5_feauture = gap_time_feat('train',train_data,start_date)
    leak6_feauture = shift_feat('train',train_data,start_date)

    train_data = pd.merge(train_data,user_buy_feat,on='user_id',how='left')
    train_data = pd.merge(train_data,shop_buy_feat,on='shop_id',how='left')
    train_data = pd.merge(train_data,item_buy_feat,on='item_id',how='left')
    train_data = pd.merge(train_data,item_brand_buy_feat,on='item_brand_id',how='left')
    train_data = pd.merge(train_data,item_brand_click_feature,on='item_brand_id',how='left')
    train_data = pd.merge(train_data,item_sales_level_click_feature,on='item_sales_level',how='left')    
    train_data = pd.merge(train_data,item_category_list_1_click_feature,on='item_category_list_1',how='left')
    train_data = pd.merge(train_data,user_hour_feature,on=['user_id','hour'],how='left')
    train_data = pd.merge(train_data,shop_hour_feature,on=['shop_id','hour'],how='left')
    train_data = pd.merge(train_data,item_hour_feature,on=['item_id','hour'],how='left')
    train_data = pd.merge(train_data,shop_age_feature,on=['shop_id','user_age_level'],how='left')
    train_data = pd.merge(train_data,shop_star_feature,on=['shop_id','user_star_level'],how='left')
    train_data = pd.merge(train_data,item_age_feature,on=['item_id','user_age_level'],how='left')
    train_data = pd.merge(train_data,item_star_feature,on=['item_id','user_star_level'],how='left')
    train_data = pd.merge(train_data,leak2_feauture,on='instance_id',how='left')
    train_data = pd.merge(train_data,leak3_feauture,on='instance_id',how='left')
    train_data = pd.merge(train_data,user_day_click_feature,on='user_id',how='left')
    train_data = pd.merge(train_data,shop_day_click_feature,on='shop_id',how='left')
    train_data = pd.merge(train_data,item_day_click_feature,on='item_id',how='left')
    train_data = pd.merge(train_data,item_brand_day_click_feature,on='item_brand_id',how='left')
    train_data = pd.merge(train_data,item_city_day_click_feature,on='item_city_id',how='left')
    train_data = pd.merge(train_data,item_category_list_1_day_click_feature,on='item_category_list_1',how='left') 

    train_data = pd.merge(train_data,item_ch_feat,on='instance_id',how='left')
    train_data = pd.merge(train_data,leak5_feauture,on='instance_id',how='left')
    train_data = pd.merge(train_data,leak6_feauture,on='instance_id',how='left')

    train_data['user_day_tofir_gap'] = train_data['now_date'] - train_data['user_day_first']

    user_day_mean_time = train_data[['user_id','now_date']].groupby('user_id').mean().reset_index().rename(columns={'now_date':'user_day_mean_time'})
    shop_day_mean_time = train_data[['shop_id','now_date']].groupby('shop_id').mean().reset_index().rename(columns={'now_date':'shop_day_mean_time'})
    item_day_mean_time = train_data[['item_id','now_date']].groupby('item_id').mean().reset_index().rename(columns={'now_date':'item_day_mean_time'})
    train_data = pd.merge(train_data,user_day_mean_time,on='user_id',how='left')
    train_data = pd.merge(train_data,shop_day_mean_time,on='shop_id',how='left')
    train_data = pd.merge(train_data,item_day_mean_time,on='item_id',how='left')

    train_data.to_csv("%s_%s_add.csv" % (start_date, start_date+1), index=None)

def test_feat(start_date):
 
    test_data = pd.read_csv('test1_%s_%s.csv'%(start_date,start_date+1))
    user_buy_feat = user_buy_feat(start_date)
    shop_buy_feat = shop_buy_feat(start_date)
    item_buy_feat = item_buy_feat(start_date)
    item_brand_buy_feat = item_brand_buy_feat(start_date)
    item_brand_click_feature = single_click_feature('item_brand_id',start_date)
    item_sales_level_click_feature = single_click_feature('item_sales_level',start_date)
    item_category_list_1_click_feature = cate_feat('item_category_list_1',start_date)
    user_hour_feature = group_rt_feature('user_id','hour',start_date)
    shop_hour_feature = group_rt_feature('shop_id','hour',start_date)
    item_hour_feature = group_rt_feature('item_id','hour',start_date)
    shop_age_feature = group_rt_feature('shop_id','user_age_level',start_date)
    shop_star_feature = group_rt_feature('shop_id','user_star_level',start_date)
    item_age_feature = group_rt_feature('item_id','user_age_level',start_date)
    item_star_feature = group_rt_feature('item_id','user_star_level',start_date)
    leak2_feauture = user_item_shop_feat('test1',test_data,start_date)
    leak3_feauture = count_feat('test1',test_data,start_date)
    user_day_click_feature = click_day_feat(test_data,'user_id',start_date)
    shop_day_click_feature = click_day_feat(test_data,'shop_id',start_date)
    item_day_click_feature = click_day_feat(test_data,'item_id',start_date)
    item_brand_day_click_feature = click_day_feat(test_data,'item_brand_id',start_date)
    item_city_day_click_feature = click_day_feat(test_data,'item_city_id',start_date)
    item_category_list_1_day_click_feature = click_day_feat(test_data,'item_category_list_1',start_date)
    item_ch_feat = item_ch_feat(test_data,start_date)
    leak5_feauture = gap_time_feat('test1',test_data,start_date)
    leak6_feauture = shift_feat('test1',test_data,start_date)

    test_data = pd.merge(test_data,user_buy_feat,on='user_id',how='left')
    test_data = pd.merge(test_data,shop_buy_feat,on='shop_id',how='left')
    test_data = pd.merge(test_data,item_buy_feat,on='item_id',how='left')
    test_data = pd.merge(test_data,item_brand_buy_feat,on='item_brand_id',how='left')
    test_data = pd.merge(test_data,item_brand_click_feature,on='item_brand_id',how='left')
    test_data = pd.merge(test_data,item_sales_level_click_feature,on='item_sales_level',how='left')
    test_data = pd.merge(test_data,item_category_list_1_click_feature,on='item_category_list_1',how='left')
    test_data = pd.merge(test_data,user_hour_feature,on=['user_id','hour'],how='left')
    test_data = pd.merge(test_data,shop_hour_feature,on=['shop_id','hour'],how='left')
    test_data = pd.merge(test_data,item_hour_feature,on=['item_id','hour'],how='left')
    test_data = pd.merge(test_data,shop_age_feature,on=['shop_id','user_age_level'],how='left')
    test_data = pd.merge(test_data,shop_star_feature,on=['shop_id','user_star_level'],how='left')
    test_data = pd.merge(test_data,item_age_feature,on=['item_id','user_age_level'],how='left')
    test_data = pd.merge(test_data,item_star_feature,on=['item_id','user_star_level'],how='left')
    test_data = pd.merge(test_data,leak2_feauture,on='instance_id',how='left')
    test_data = pd.merge(test_data,leak3_feauture,on='instance_id',how='left')

    test_data = pd.merge(test_data,user_day_click_feature,on='user_id',how='left')
    test_data = pd.merge(test_data,shop_day_click_feature,on='shop_id',how='left')
    test_data = pd.merge(test_data,item_day_click_feature,on='item_id',how='left')
    test_data = pd.merge(test_data,item_brand_day_click_feature,on='item_brand_id',how='left')
    test_data = pd.merge(test_data,item_city_day_click_feature,on='item_city_id',how='left')
    test_data = pd.merge(test_data,item_category_list_1_day_click_feature,on='item_category_list_1',how='left')  

    test_data = pd.merge(test_data,item_ch_feat,on='instance_id',how='left')
    test_data = pd.merge(test_data,leak5_feauture,on='instance_id',how='left')
    test_data = pd.merge(test_data,leak6_feauture,on='instance_id',how='left')

    test_data['user_day_tofir_gap'] = test_data['now_date'] - test_data['user_day_first']

    user_day_mean_time = test_data[['user_id','now_date']].groupby('user_id').mean().reset_index().rename(columns={'now_date':'user_day_mean_time'})
    shop_day_mean_time = test_data[['shop_id','now_date']].groupby('shop_id').mean().reset_index().rename(columns={'now_date':'shop_day_mean_time'})
    item_day_mean_time = test_data[['item_id','now_date']].groupby('item_id').mean().reset_index().rename(columns={'now_date':'item_day_mean_time'})
    test_data = pd.merge(test_data,user_day_mean_time,on='user_id',how='left')
    test_data = pd.merge(test_data,shop_day_mean_time,on='shop_id',how='left')
    test_data = pd.merge(test_data,item_day_mean_time,on='item_id',how='left')

    test_data.to_csv("test1_%s_%s_add.csv" % (start_date, start_date+1), index=None)


test_feat(7)
train_feat(7)
###############################################################################
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

##清洗数据
def clean_data(data_test,data_train):
 
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

def convert_only7_data(data):
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
count_prep_train=a_prep.iloc[:len_train]
count_prep_test=a_prep.iloc[len_train:]
count_train=a_count.iloc[:len_train]
count_test=a_count.iloc[len_train:]
gap_time_train=a_gap_time.iloc[:len_train]
gap_time_test=a_gap_time.iloc[len_train:]
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

 

    label=final_train_set['is_trade']
    del final_train_set['is_trade']
    #del final_train_set['instance_id']
    del final_train_set['user_id']

 
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
       

   # final_test_set=pd.merge(final_test_set,PH_user_ratio,how='left',on='user_id')
    
    label=final_test_set['is_trade']
    del final_test_set['is_trade']
    #del final_test_set['instance_id'] 
    del final_test_set['user_id']


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
           

        
    del final_test_set['user_id']
   # del final_test_set['item_id']
   # del final_test_set['shop_id']
    #del final_test_set['instance_id'] 
    return final_test_set
    
def offline_test(data_test,data_train):
    #train_start_time=datetime.datetime.strptime('2018-08-31','%Y-%m-%d')
    #train_end_time=datetime.datetime.strptime('2018-09-06 11:59:59','%Y-%m-%d %H:%M:%S')
    train_start_time='2018-09-07 00:00:00'
    train_end_time='2018-09-07 9:59:59'

    #train_val_start_time=datetime.datetime.strptime('2018-09-06 12:00:00','%Y-%m-%d %H:%M:%S')
    #train_val_end_time=datetime.datetime.strptime('2018-09-06 23:59:59','%Y-%m-%d %H:%M:%S')
    train_val_start_time='2018-09-07 10:00:00'
    train_val_end_time='2018-09-07 11:59:59'

    print('off_train')
    train_x,train_y=make_train_set(train_start_time,train_end_time,data_test,data_train)
    #train_x=train_x.fillna(-1)
    print('off_train_x:%d' % (len(train_x)))
    print('off_train_y: %d' %(len(train_y)))
    train_x.to_csv('../data/train_x_7_b.csv',index=False)
    train_y.to_csv('../data/train_y_7_b.csv',index=False)

    print('off_val')
    train_val_x,train_val_y=make_test_set(train_start_time,train_end_time,train_val_start_time,train_val_end_time,data_test,data_train)
    train_val_x=train_val_x.fillna(-1)
    print('off_val_x:%d' % (len(train_val_x)))
    print('off_val_y: %d' %(len(train_val_y)))
    train_val_x.to_csv('../data/val_x_7_b.csv',index=False)
    train_val_y.to_csv('../data/val_y_7_b.csv',index=False)
    
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
    #train_x=train_x.fillna(-1)
    print('online_train_x:%d' % (len(train_x)))
    print('online_train_y: %d' %(len(train_y)))
    train_x.to_csv('../data/online_train_x_7_b.csv',index=False)
    train_y.to_csv('../data/online_train_y_7_b.csv',index=False)
    
    print('online_test')
    test_x=make_test_set_2(train_start_time,train_end_time,test_start_time,test_end_time,data_test,data_train)
    print(len(test_x))
    test_x.to_csv('../data/online_test_7_b.csv',index=False)
    
offline_test(data_test,data_train)
online_submit(data_test,data_train)
###################################################3
print('load_data_ori...')
data_train=pd.read_csv('../data/data_train_ori_b.csv')
print(len(data_train))
#data_train['times']=data_train['times'].astype(str)
data_test=pd.read_csv('../data/data_test_ori_b.csv')
print(len(data_test))
#data_test['times']=data_test['times'].astype(str)
print('load_data_ori finish...')

def convert_before1_data(data):
    data["times"] = data["context_timestamp"].apply(lambda x: datetime.datetime.fromtimestamp(x))
    data["day"] = data["times"].apply(lambda x: x.day)
    data["hour"] = data["times"].apply(lambda x: x.hour)
    data['min'] = data['times'].apply(lambda x: x.minute)
    data['day']=data['day'].astype('int')
    data['hour']=data['hour'].astype('int')
    data['min']=data['min'].astype('int')
    return data
len_train=len(data_train)
all_data=pd.concat([data_train,data_test]).reset_index(drop=True)
all_data=convert_before1_data(all_data)



columns_cate = [
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
                     ['shop_star_level'],
                     ['item_id'],
                     ['shop_id'],
                     ]
                     
columns_level = [
                     ['item_price_level'],
                     ['item_collected_level'],
                     ['item_sales_level'],
                     ['item_pv_level'],
                     ['shop_review_num_level'],
                     ['shop_review_positive_rate'],
                     ['shop_star_level'],
                     ['shop_score_service'],
                     ['shop_score_delivery'],
                     ['shop_score_description'],

]

columns_user= [
                     ['user_gender_id'],
                     ['user_age_level'],
                     ['user_occupation_id'],
                     ['user_star_level'],
                     

]


def feat_before_1day():
    data=all_data[(all_data['times']>='2018-08-31 00:00:00') & (all_data['times']<='2018-09-07 23:59:59')]
    
    for c in columns_level:
        if isinstance(c, (list, tuple)):
            pair = [cc for cc in c]
            c = '_'.join(c)
        else:
            pair = [c]
        p1=c
        p2=c+'_before_1day_mean'
        p3=c+'_before_1day_var'
        p4=c+'_before_1day_std'
        print(p2)
        data_feat=data[['user_id',p1]]
        mean_feat=data_feat[['user_id',p1]].groupby('user_id').mean().reset_index().rename(columns={p1: p2}) 
        data=pd.merge(data,mean_feat,'left',on=['user_id'])
        
        data_feat_1=data[['user_id',p1]]
        var_feat=data_feat_1[['user_id',p1]].groupby('user_id').var().reset_index().rename(columns={p1: p3}) 
        data=pd.merge(data,var_feat,'left',on=['user_id'])
        
        data_feat_2=data[['user_id',p1]]
        std_feat=data_feat_2[['user_id',p1]].groupby('user_id').std().reset_index().rename(columns={p1: p4}) 
        data=pd.merge(data,std_feat,'left',on=['user_id'])
        
    #data.to_csv('../my_feat/only7_2.txt',index=False)
    
    
    for c in columns_user:
        if isinstance(c, (list, tuple)):
            pair = [cc for cc in c]
            c = '_'.join(c)
        else:
            pair = [c]
        p1=c
        p2='shop_'+c+'_before_1day_mean'
        p3='shop_'+c+'_before_1day_var'
        p4='shop_'+c+'_before_1day_std'
        data_shop_1=data[['shop_id',p1]]
        mean_feat=data_shop_1[['shop_id',p1]].groupby('shop_id').mean().reset_index().rename(columns={p1: p2}) 
        data=pd.merge(data,mean_feat,'left',on=['shop_id'])
        
        data_shop_2=data[['shop_id',p1]]
        var_feat=data_shop_2[['shop_id',p1]].groupby('shop_id').var().reset_index().rename(columns={p1: p3}) 
        data=pd.merge(data,var_feat,'left',on=['shop_id'])
        
        data_shop_3=data[['shop_id',p1]]
        std_feat=data_shop_3[['shop_id',p1]].groupby('shop_id').std().reset_index().rename(columns={p1: p4}) 
        data=pd.merge(data,std_feat,'left',on=['shop_id'])
        
        p2='item'+c+'_before_1day_mean'
        p3='item'+c+'_before_1day_var'
        p4='item'+c+'_before_1day_std'
        data_item_1=data[['item_id',p1]]
        mean_feat=data_item_1[['item_id',p1]].groupby('item_id').mean().reset_index().rename(columns={p1: p2}) 
        data=pd.merge(data,mean_feat,'left',on=['item_id'])
        
        data_item_2=data[['item_id',p1]]
        var_feat=data_item_2[['item_id',p1]].groupby('item_id').var().reset_index().rename(columns={p1: p3}) 
        data=pd.merge(data,var_feat,'left',on=['item_id'])
        
        data_item_3=data[['item_id',p1]]
        std_feat=data_item_3[['item_id',p1]].groupby('item_id').std().reset_index().rename(columns={p1: p4}) 
        data=pd.merge(data,std_feat,'left',on=['item_id'])
        
        
        p2='brand'+c+'_before_1day_mean'
        p3='brand'+c+'_before_1day_var'
        p4='brand'+c+'_before_1day_std'
        data_brand_1=data[['item_brand_id',p1]]
        mean_feat=data_brand_1[['item_brand_id',p1]].groupby('item_brand_id').mean().reset_index().rename(columns={p1: p2}) 
        data=pd.merge(data,mean_feat,'left',on=['item_brand_id'])
        
        data_brand_2=data[['item_brand_id',p1]]
        var_feat=data_brand_2[['item_brand_id',p1]].groupby('item_brand_id').var().reset_index().rename(columns={p1: p3}) 
        data=pd.merge(data,var_feat,'left',on=['item_brand_id'])
        
        data_brand_3=data[['item_brand_id',p1]]
        std_feat=data_brand_3[['item_brand_id',p1]].groupby('item_brand_id').std().reset_index().rename(columns={p1: p4}) 
        data=pd.merge(data,std_feat,'left',on=['item_brand_id'])
        
        
        p2='city'+c+'_before_1day_mean'
        p3='city'+c+'_before_1day_var'
        p4='city'+c+'_before_1day_std'
        data_city_1=data[['item_city_id',p1]]
        mean_feat=data_city_1[['item_city_id',p1]].groupby('item_city_id').mean().reset_index().rename(columns={p1: p2}) 
        data=pd.merge(data,mean_feat,'left',on=['item_city_id'])
        
        data_city_2=data[['item_city_id',p1]]
        var_feat=data_city_2[['item_city_id',p1]].groupby('item_city_id').var().reset_index().rename(columns={p1: p3}) 
        data=pd.merge(data,var_feat,'left',on=['item_city_id'])
        
        data_city_3=data[['item_city_id',p1]]
        std_feat=data_city_3[['item_city_id',p1]].groupby('item_city_id').std().reset_index().rename(columns={p1: p4}) 
    #data=pd.merge(data,std_feat,'left',on=['item_city_id'])
        
        print(p2)
        #data.to_csv('../my_feat/only7_3.txt',index=False)'''
    
    for c in columns_cate:
        if isinstance(c, (list, tuple)):
            pair = [cc for cc in c]
            c = '_'.join(c)
        else:
            pair = [c]
        p1=c
        p2=c+'_before_1day_count'
        print(p2)
        count_feat=data.groupby(['user_id',p1]).size().reset_index().rename(columns={0: p2}) 
        data=pd.merge(data,count_feat,'left',on=['user_id',p1])
    data.to_csv('../my_feat/only7_1.txt',index=False)
    
    
        
    
    return data

feat=feat_before_1day()
feat=feat[(feat['times']>='2018-09-07 00:00:00') & (feat['times']<='2018-09-07 23:59:59')]
print(len(feat))
feat=feat.drop(['context_id', 'context_page_id', 'context_timestamp',
       'is_trade', u'item_brand_id', 'item_category_1', 'item_category_2',
        'item_category_3', 'item_city_id', 'item_collected_level',
       'item_id', 'item_price_level', 'item_property_list_1',
       'item_property_list_2', 'item_property_list_3', 'item_pv_level',
        'item_sales_level', 'predict_category_property_A',
        'predict_category_property_A_1', 'predict_category_property_A_2',
        'predict_category_property_B', 'predict_category_property_B_1',
        'predict_category_property_B_2', 'predict_category_property_C',
        'predict_category_property_C_1', 'predict_category_property_C_2',
        'shop_id', 'shop_review_num_level', 'shop_review_positive_rate',
       'shop_score_delivery', 'shop_score_description',
        'shop_score_service', 'shop_star_level', 'user_age_level',
        'user_gender_id', 'user_id', 'user_occupation_id',
        'user_star_level',  'day', 'hour', 'min',
 ],axis=1)
 
feat.to_csv('../my_feat/before_1day_all.txt',index=False)
#################################################

print('load_data_ori...')
data_train=pd.read_csv('../data/data_train_ori_b.csv')
print(len(data_train))
#data_train['times']=data_train['times'].astype(str)
data_test=pd.read_csv('../data/data_test_ori_b.csv')
print(len(data_test))
#data_test['times']=data_test['times'].astype(str)
print('load_data_ori finish...')

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
    return data


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
                     #--单个--                     
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
                     
len_train=len(data_train)
all_data=pd.concat([data_train,data_test])
all_data=convert_data(all_data)


start_time='2018-08-31'
end_time='2018-09-07 23:59:59'
for c in columns_click:
        if isinstance(c, (list, tuple)):
            pair = [cc for cc in c]
            c = '_'.join(c)
        else:
            pair = [c]
        
        p1=c
        p2=c+'_click'
        print(p2)
        a1=click_feat(start_time,end_time,data_test,all_data,p1,p2)
        all_data=pd.merge(all_data,a1,'left',on='instance_id')
    
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
    a1=gap_feat(start_time,end_time,data_test,all_data,p1,p2,p3,p4,p5,p6,p8)
    all_data=pd.merge(all_data,a1,'left',on='instance_id')

#from functools import reduce
#columns_click=reduce(lambda x,y:x+y, columns_click)
#columns=[]
#for i in columns_click:
#    i=i+'_click'
#    columns.append(columns)

all_data=all_data[['instance_id',
'predict_category_property_A_click','predict_category_property_B_click',
'predict_category_property_C_click','predict_category_property_A_1_click','predict_category_property_B_1_click',
'item_property_list_1_click','item_property_list_2_click','item_property_list_2_click','item_brand_id_click',
'item_city_id_click','item_category_2_click','item_price_level_click','item_sales_level_click','item_collected_level_click',
'item_pv_level_click','context_page_id_click','shop_review_num_level_click','shop_review_positive_rate_click',
'shop_star_level_click','shop_star_level_click','shop_score_service_click','shop_score_description_click',

'predict_category_property_A_all_sec','predict_category_property_B_all_sec','predict_category_property_C_all_sec',
'predict_category_property_A_1_all_sec','predict_category_property_B_1_all_sec','item_property_list_1_all_sec',
'item_property_list_2_all_sec','item_property_list_2_all_sec',
'item_category_2_all_sec','item_price_level_all_sec','item_sales_level_all_sec','item_collected_level_all_sec',
'item_pv_level_all_sec','context_page_id_all_sec','shop_review_num_level_all_sec','shop_review_positive_rate_all_sec',
'shop_star_level_all_sec','shop_star_level_all_sec','shop_score_service_all_sec','shop_score_description_all_sec',


'predict_category_property_A_hour','predict_category_property_B_hour','predict_category_property_C_hour',
'predict_category_property_A_1_hour','predict_category_property_B_1_hour','item_property_list_1_hour',
'item_property_list_2_hour','item_property_list_2_hour',
'item_category_2_hour','item_price_level_hour','item_sales_level_hour','item_collected_level_hour',
'item_pv_level_hour','context_page_id_hour','shop_review_num_level_hour','shop_review_positive_rate_hour',
'shop_star_level_hour','shop_star_level_hour','shop_score_service_hour','shop_score_description_hour']]

data_train=all_data.iloc[:len_train]
print('data_train:%d' %(len(data_train)))
data_test=all_data.iloc[len_train:]
print('data_test:%d' %(len(data_test)))

data_train.to_csv('../data/data_train_all_gaptime_click.csv',index=False)
data_test.to_csv('../data/data_test_all_gaptime_click.csv',index=False)

def convert_data(data):
    data["times"] = data["context_timestamp"].apply(lambda x: datetime.datetime.fromtimestamp(x))
    data["day"] = data["times"].apply(lambda x: x.day)
    data["hour"] = data["times"].apply(lambda x: x.hour)
    data['min'] = data['times'].apply(lambda x: x.minute)
    data['day']=data['day'].astype('int')
    data['hour']=data['hour'].astype('int')
    data['min']=data['min'].astype('int')

    return data
    
len_train=len(data_train)
all_data=pd.concat([data_train,data_test])
all_data=convert_data(all_data)
data_train=all_data.iloc[:len_train]
data_test=all_data.iloc[len_train:]

data_train_gaptime_click=pd.read_csv('../data/data_train_all_gaptime_click.csv')
data_test_gaptime_click=pd.read_csv('../data/data_test_all_gaptime_click.csv')
data_train=pd.merge(data_train,data_train_gaptime_click,'left',on=['instance_id'])
data_test=pd.merge(data_test,data_test_gaptime_click,'left',on=['instance_id'])

def online_submit():
    ori_train=data_train[(data_train['times']>='2018-08-31') & (data_train['times']<='2018-09-07 11:59:59')]
    ori_train=ori_train[['instance_id','times']]
    #'item_user_day_click','item_user_day_hour_click','shop_user_day_click','shop_user_day_hour_click','item_brand_user_day_click',
    #'item_brand_user_day_hour_click','context_user_day_click','context_user_day_hour_click']]

    print('ori_train:%d' % (len(ori_train)))
    
    
    print('online_train')
    train_x=pd.read_csv('../data/online_train_x_all_b.csv')
    train_y=pd.read_csv('../data/online_train_y_all_b.csv', header=None)
    train_x=pd.merge(train_x,ori_train,'left',on='instance_id')
    
    train_x_y=pd.concat([train_x,train_y],axis=1).reset_index(drop=True).rename(columns={0:'label'})
    train_all=train_x_y[train_x_y['times']<='2018-09-06 23:59:59']#.sample(frac=0.6)
    train_predict_7=train_x_y[train_x_y['times']>='2018-09-07 00:00:00']
    
    train_x=train_all.drop(['times','label'],axis=1)
    train_y=train_all['label']
    
    print('online_train_x:%d' % (len(train_x)))
    print('online_train_y: %d' %(len(train_y)))

    print('online_test')
    test_x=pd.read_csv('../data/online_test_all_b.csv')
    train_predict_7=train_predict_7.drop(['times','label'],axis=1)
    test_x=pd.concat([train_predict_7,test_x]).reset_index(drop=True)
    print(len(test_x))
    
    data_test=test_x['instance_id']
    ###################################################################
    
    train_index=train_x.axes[1]
    X = train_x[train_index]
    y = train_y.values
    X_test = test_x[train_index]
	

    lgb_train = lgb.LGBMClassifier(num_leaves=128, learning_rate=0.01,n_estimators=2000,colsample_bytree =0.8,
                        subsample = 0.8,max_depth=9, min_child_weight=6, n_jobs=20)
    lgb_train.fit(X, y,)
    
    lgb_pre_test_y = lgb_train.predict_proba(test_x[train_index])[:,1]
                                             
    pre_test_y=lgb_pre_test_y

    #pre_test_y=xgb_pre_test_y                                      
    #data_test=test_x['instance_id']                                             
    pre_test_y=pd.DataFrame(pre_test_y)
    data_test=data_test.reset_index()
    del data_test['index']
    
    sub_data=pd.concat([data_test,pre_test_y],axis=1)
    #sub_data=sub_data[:1209768]
    sub_data.rename(columns={0:'predicted_score'},inplace=True)
    print(sub_data['predicted_score'].mean())
    sub_data.to_csv('../data/predict_7.txt',index=False,sep=' ')
    
online_submit()
#############################################################
import time
import pandas as pd
import numpy as np
from datetime import datetime
from collections import Counter
train = pd.read_csv('/data/hufeng/alimama/stage2/round2_train.txt',sep=' ') #10432036
data_test_b=pd.read_csv('../data/round2_ijcai_18_test_b_20180510.txt',delim_whitespace=True)
data_test_a=pd.read_csv('../data/round2_ijcai_18_test_a_20180425.txt',delim_whitespace=True)
data_test=pd.concat([data_test_b,data_test_a]).reset_index(drop=True)
train = pd.concat([data_train,data_test],).reset_index(drop=True)
def timestamp_datetime(value):
    format = '%Y-%m-%d %H:%M:%S'
    value = time.localtime(value)
    dt = time.strftime(format, value)
    return dt

train['times'] = train.context_timestamp.apply(timestamp_datetime)
train['day'] = train.times.apply(lambda x: int(x[8:10]))
train.day[train.day==31] = 0
def compute_item_category_2(x):
    item_category_list = x.split(';')
    if len(item_category_list)==2:
        return item_category_list[-1]+'_no_category_2'
    if len(item_category_list)==3:
        return item_category_list[-1]

train['item_category_len'] = train.item_category_list.apply(lambda x :len(x.split(';')))
train['item_category_0'] = train.item_category_list.apply(lambda x :x.split(';')[0])
train['item_category_1'] = train.item_category_list.apply(lambda x :x.split(';')[1]) 
train['item_category_2'] = train.item_category_list.apply(compute_item_category_2)

train.item_sales_level.replace(-1,float(train.item_sales_level.median()),inplace=True)
train.user_age_level.replace(-1,float(train.user_age_level.median()),inplace=True)
train.user_occupation_id.replace(-1,float(train.user_occupation_id.mode()),inplace=True)
train.user_star_level.replace(-1,float(train.user_star_level.median()),inplace=True)
train.shop_review_positive_rate.replace(-1,float(train.shop_review_positive_rate.median()),inplace=True)
train.shop_score_service.replace(-1,float(train.shop_score_service.median()),inplace=True)
train.shop_score_delivery.replace(-1,float(train.shop_score_delivery.median()),inplace=True)
train.shop_score_description.replace(-1,float(train.shop_score_description.median()),inplace=True)

feat03 = train[(train['day']>=0)&(train['day']<=3)] 
feat45 = train[(train['day']>=4)&(train['day']<=5)] 
feat6 = train[(train['day']>=6)&(train['day']<=6)] 
feat03.sort_values(by=['times'],inplace=True)
feat45.sort_values(by=['times'],inplace=True)
feat6.sort_values(by=['times'],inplace=True)
label = train[train['day']==7] 
label.sort_values(by=['times'],inplace=True)

label['item_property_len'] = label.item_property_list.apply(lambda x : len(x.split(';')))

label['predict_category_len'] =  label['predict_category_property'].apply(lambda x :len(x.split(';')))

def predict_category_feat(x):
    category_property_list = x.split(';')
    category_list = []
    for category_property in category_property_list:
        category_list.append(category_property.split(':')[0])
    return category_list

label['predict_category'] = label['predict_category_property'].apply(predict_category_feat)
label['item_category_0'] = label.item_category_list.apply(lambda x :x.split(';')[0])
def item_category01_feat(x):
    item_category_0 = x.split('_')[0]
    predict_category_list = eval(x.split('_')[1])
    if item_category_0 in predict_category_list:
        return 1
    else:
        return 0

label['item_category_0_predict'] = label['item_category_0'] + '_' + label['predict_category'].astype('str')
label['item_category_0_in'] = label['item_category_0_predict'].apply(item_category01_feat)

label['item_category_1_predict'] = label['item_category_1'] + '_' + label['predict_category'].astype('str')
label['item_category_1_in'] = label['item_category_1_predict'].apply(item_category01_feat)

def item_category2(x):
    item_category_2 = x.split('_')[0]
    predict_category_list = eval(x.split('_')[1])
    item_category_len = float(x.split('_')[2])
    if item_category_len==2:
        return 0
    else:
        if item_category_2 in predict_category_list:
            return 1
        else:
            return 0

label['item_category_2_predict_category'] = label['item_category_2'] + '_' + label['predict_category'].astype('str') + '_' + label['item_category_len'].astype('str')
label['item_category_2_in'] = label['item_category_2_predict_category'].apply(item_category2)

def compute_item_category2_not_in(x):
    item_category_2 = x.split('_')[0]
    predict_category_list = eval(x.split('_')[1])
    item_category_len = float(x.split('_')[2])
    if (item_category_len==3)&(item_category_2 not in predict_category_list):
        return 1
    else:
        return 0

label['item_category_2_no'] = label['item_category_2_predict_category'].apply(compute_item_category2_not_in)

label['item_category_01'] = label['item_category_0_in'] + label['item_category_1_in']

label['item_category_02'] = label['item_category_0_in'] + label['item_category_2_in']

label['item_category_12'] = label['item_category_1_in'] + label['item_category_2_in']

label['item_category_012'] = label['item_category_0_in'] + label['item_category_1_in'] + label['item_category_2_in']

label['item_category_012_rate'] = label.item_category_012.astype('float')/label.item_category_len.astype('float')

label['item_category_012_proportion'] = label.item_category_012.astype('float')/label.predict_category_len.astype('float')

def compute_item_category_position(x):
    item_category_0 = x.split('_')[0]
    predict_category_list = eval(x.split('_')[1])
    if item_category_0 not in predict_category_list:
        return 20
    else:
        return predict_category_list.index(item_category_0)+1

label['item_category_0_position'] = label['item_category_0_predict'].apply(compute_item_category_position)


label['item_category_0_position_div'] = label.item_category_0_position.astype('float')/label.predict_category_len.astype('float')

label['item_category_1_position'] = label['item_category_1_predict'].apply(compute_item_category_position)

label['item_category_1_position_div'] = label.item_category_1_position.astype('float')/label.predict_category_len.astype('float')

label['item_category_10_position_diff'] = label.item_category_1_position.astype('float') - label.item_category_0_position.astype('float')

label['item_category_10_position_diff_div'] = label.item_category_10_position_diff.astype('float')/label.predict_category_len.astype('float')

def compute_category_property_num(x):
    item_category = x.split('_')[0]
    predict_category_property = x.split('_')[1]
    predict_category_property_list = predict_category_property.split(';')
    for category_property in predict_category_property_list:
        if category_property.split(':')[0]==item_category:
            property_all = category_property.split(':')[1]
            if property_all=='-1':
                return 0
            else:
                return len(property_all.split(','))
    return 0


label['item_category_0_predict_property'] = label['item_category_0'] + '_' + label['predict_category_property'].astype('str')
label['category_0_property_num'] = label['item_category_0_predict_property'].apply(compute_category_property_num)

def category_property_cross_num(x):
    item_category = x.split('_')[0]
    item_property_all = x.split('_')[1]
    item_property_list = item_property_all.split(';')
    predict_category_property = x.split('_')[2]
    predict_category_property_list = predict_category_property.split(';')
    for category_property in predict_category_property_list:
        if category_property.split(':')[0]==item_category:
            predict_property_all = category_property.split(':')[1]
            predict_property_list = predict_property_all.split(',')
            return len(set(item_property_list)&set(predict_property_list))
    return 0

label['item_category_0_item_property_list_predict'] = label['item_category_0'] + '_' + label['item_property_list'] + '_' + label['predict_category_property'].astype('str')
label['category_0_property_cross_count'] = label['item_category_0_item_property_list_predict'].apply(category_property_cross_num)

label['category_0_property_cross'] = label.category_0_property_cross_count.astype('float')/label.item_property_len.astype('float')
label.category_0_property_cross.fillna(0,inplace=True)

label['category_0_property_rate'] = label.category_0_property_cross_count.astype('float')/label.category_0_property_num.astype('float')
label.category_0_property_rate.fillna(0,inplace=True)
label.category_0_property_rate.replace(label.category_0_property_rate.max(),1.0,inplace=True)

label['item_category_1_predict_property'] = label['item_category_1'] + '_' + label['predict_category_property'].astype('str')
label['category_1_property_num'] = label['item_category_1_predict_property'].apply(compute_category_property_num)

label['item_category_1_item_property_list_predict'] = label['item_category_1'] + '_' + label['item_property_list'] + '_' + label['predict_category_property'].astype('str')
label['category_1_property_cross_num'] = label['item_category_1_item_property_list_predict'].apply(category_property_cross_num)

label['category_1_property'] = label.category_1_property_cross_num.astype('float')/label.item_property_len.astype('float')
label.category_1_property.fillna(0,inplace=True)

label['category_1_property_rate'] = label.category_1_property_cross_num.astype('float')/label.category_1_property_num.astype('float')
label.category_1_property_rate.fillna(0,inplace=True)


label=label[['insttance_id','times','item_category_0_in','item_category_1_in','item_category_2_in','item_category_2_no','item_category_01',
'item_category_02','item_category_12','item_category_012','item_category_012_rate',
'item_category_012_proportion','item_category_0_position','item_category_0_position_div',
'item_category_1_position','item_category_1_position_div','item_category_10_position_diff',
'item_category_10_position_diff_div','category_0_property_num','category_0_property_cross_count',
'category_0_property_cross','category_0_property_rate','category_1_property_num',
'category_1_property_cross_num','category_1_property','category_1_property_rate']]

label.to_csv('../data/label.txt',index=False)
######################################################################

print('load_data_ori...')
data_train=pd.read_csv('../data/data_train_ori_b.csv')
print(len(data_train))
#data_train['times']=data_train['times'].astype(str)
data_test=pd.read_csv('../data/data_test_ori_b.csv')
print(len(data_test))
#data_test['times']=data_test['times'].astype(str)
print('load_data_ori finish...')
query_after=pd.read_csv('../data/other_feat/user_query_count_after.csv')
query1_after=pd.read_csv('../data/other_feat/user_query1_count_after.csv')
query_before=pd.read_csv('../data/other_feat/user_query_count_before.csv')
query1_before=pd.read_csv('../data/other_feat/user_query1_count_before.csv')



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
    '''
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
    '''
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
    '''
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
    '''

    
    return data


count_prep_tianyin=pd.read_csv('../data/other_feat/query_before.csv')
category_count_after=pd.read_csv('../data/other_feat/query_before.csv')


len_train=len(data_train)
all_data=pd.concat([data_train,data_test]).reset_index(drop=True)
all_data=convert_data(all_data)
all_data=pd.concat([all_data,query_after],axis=1).reset_index(drop=True)
all_data=pd.concat([all_data,query_before],axis=1).reset_index(drop=True)
print(len(all_data))
all_data=pd.concat([all_data,query1_after],axis=1).reset_index(drop=True)
all_data=pd.concat([all_data,query1_before],axis=1).reset_index(drop=True)
print(len(all_data))
data_train=all_data.iloc[:len_train]
print(len(data_train))
data_test=all_data.iloc[len_train:]
print(len(data_test))
del data_test['is_trade']
del data_test['item_category_3']
del data_train['is_trade']
'''
###############################################
#all_data=pd.merge(all_data,has_seen,'left',on=['instance_id'])
count_prep_train=count_prep.iloc[:len_train]
count_prep_test=count_prep.iloc[len_train:]
count_train=count.iloc[:len_train]
count_test=count.iloc[len_train:]
gap_time_train=gap_time.iloc[:len_train]
gap_time_test=gap_time.iloc[len_train:]
trade_prep_count_train=trade_prep_count.iloc[:len_train]
trade_prep_count_test=trade_prep_count.iloc[len_train:]

data_train=pd.concat([data_train,count_prep_train],axis=1)
data_train=pd.concat([data_train,count_train],axis=1)
data_train=pd.concat([data_train,gap_time_train],axis=1)
data_train=pd.concat([data_train,trade_prep_count_train],axis=1)

data_test=pd.concat([data_test,count_prep_test],axis=1)
data_test=pd.concat([data_test,count_test],axis=1)
data_test=pd.concat([data_test,gap_time_test],axis=1)
data_test=pd.concat([data_test,trade_prep_count_test],axis=1)

######################################

#data_train=pd.merge(data_train,data_train_PH,'left',on=['instance_id'])
#data_test=pd.merge(data_test,data_test_PH,'left',on=['instance_id'])
del data_test['is_trade']
del data_test['item_category_3']
'''

#####################################################################
def online_submit():
    ori_train=data_train[(data_train['times']>='2018-09-07 00:00:00') & (data_train['times']<='2018-09-07 11:59:59')]
    ori_train=ori_train[['instance_id',
    #'item_city_id_PH_ctr','item_id_PH_ctr','item_brand_id_PH_ctr',
    #'user_id_PH_ctr','shop_id_PH_ctr',
    #'user_category_item_id_click_count_prep','user_category_shop_id_click_count_prep',
    #'user_category_brand_id_click_count_prep','user_category_city_id_click_count_prep',
    
    'user_mean_hour','item_mean_hour','shop_mean_hour','city_mean_hour','brand_mean_hour',
    'user_mean_day','item_mean_day','shop_mean_day','city_mean_day','brand_mean_day',
    'user_mean_day_hour','item_mean_day_hour','shop_mean_day_hour','city_mean_day_hour',
    'brand_mean_day_hour',
    
    'user_var_hour','item_var_hour','shop_var_hour','city_var_hour','brand_var_hour',
    'user_var_day','item_var_day','shop_var_day','city_var_day','brand_var_day',
    'user_var_day_hour','item_var_day_hour','shop_var_day_hour','city_var_day_hour','brand_var_day_hour',
    
    'price_gap_7_6','price_gap_7_5','sales_gap_7_6','sales_gap_7_5','collected_gap_7_6','collected_gap_7_5','pv_gap_7_6','pv_gap_7_5',
    ]]
    #'item_user_day_click','item_user_day_hour_click','shop_user_day_click','shop_user_day_hour_click','item_brand_user_day_click',
    #'item_brand_user_day_hour_click','context_user_day_click','context_user_day_hour_click']]

    
    predict_7=pd.read_csv('../data/predict_7.txt',sep=' ')
    del predict_7['instance_id']
    part=pd.read_csv('../data/df_label.txt')
    part_train=part[(part['times']>='2018-09-07 00:00:00') & (part['times']<='2018-09-07 11:59:59')]
    del part_train['times']
    part_test=part[(part['times']>='2018-09-07 12:00:00') & (part['times']<='2018-09-07 11:59:59')]
    del part_test['times']
    
    print('online_train')
    train_x=pd.read_csv('../data/online_train_all_to7_x.csv')
    train_y=pd.read_csv('../data/online_train_all_to7_y.csv', header=None)
    train_x=pd.merge(train_x,ori_train,'left',on='instance_id')
    
    train_add=pd.read_csv('7_8_add.csv')
    train_add=train_add.drop(['is_trade','item_category_list','item_property_list','predict_category_property', 'time'],axis=1)
    train_x=pd.merge(train_x,train_add,'left',on='instance_id')
    
    train_only7=pd.read_csv('../data/online_train_x_7_b.csv')
    train_only7=train_only7[['instance_id','item_city_count','item_category2_count','item_property_A_count','item_property_B_count','item_property_B_1_count','item_property_C_count',
    'item_property_list_2_count','item_property_list_3_count',
    'user_id_click_count_prep','item_id_click_count_prep','item_brand_id_click_count_prep','shop_id_click_count_prep','user_item_id_click_count_prep','user_shop_id_click_count_prep',
     'user_brand_id_click_count_prep','user_category_id_click_count_prep',
     'user_id_count','item_id_count','item_brand_id_count','shop_id_count','user_item_id_count','user_shop_id_count','user_brand_id_count','user_category_id_count',
     'user_id_lasttime_delta','item_id_lasttime_delta','item_brand_id_lasttime_delta','shop_id_lasttime_delta','user_item_id_lasttime_delta',
'user_shop_id_lasttime_delta','user_brand_id_lasttime_delta','user_category_id_lasttime_delta',
'user_id_nexttime_delta','item_id_nexttime_delta','item_brand_id_nexttime_delta','shop_id_nexttime_delta','user_item_id_nexttime_delta',
'user_shop_id_nexttime_delta','user_brand_id_nexttime_delta','user_category_id_nexttime_delta',]]
    train_x=pd.merge(train_x,train_only7,'left',on='instance_id')
    
    train_before_1=pd.read_csv('../my_feat/before_1day_all.txt')
    a_time=train_before_1[['instance_id','times']]
    train_before_1_train=train_before_1[(train_before_1['times']>='2018-09-07 00:00:00') & (train_before_1['times']<='2018-09-07 11:59:59')]
    del train_before_1_train['times']
    train_x=pd.merge(train_x,train_before_1_train,'left',on='instance_id')
    train_x=pd.merge(train_x,part_train,'left',on='instance_id')
    train_x=pd.concat([train_x,predict_7[:1077175]],axis=1)

    print(len(train_x))

    
    
    print('online_test')
    test_x=pd.read_csv('../data/online_test_all_to7.csv')
    test_x=pd.merge(test_x,ori_train,'left',on='instance_id')
    
    test_add=pd.read_csv('test1_7_8_add.csv')
    test_add=test_add.drop(['is_trade','item_category_list','item_property_list', 'predict_category_property', 'time'],axis=1)
    test_x=pd.merge(test_x,test_add,'left',on='instance_id')
    
    test_only7=pd.read_csv('../data/online_test_7_b.csv')
    test_only7=test_only7[['instance_id','item_city_count','item_category2_count','item_property_A_count','item_property_B_count','item_property_B_1_count','item_property_C_count',
    'item_property_list_2_count','item_property_list_3_count',
    'user_id_click_count_prep','item_id_click_count_prep','item_brand_id_click_count_prep','shop_id_click_count_prep','user_item_id_click_count_prep','user_shop_id_click_count_prep',
     'user_brand_id_click_count_prep','user_category_id_click_count_prep',
     'user_id_count','item_id_count','item_brand_id_count','shop_id_count','user_item_id_count','user_shop_id_count','user_brand_id_count','user_category_id_count',
     'user_id_lasttime_delta','item_id_lasttime_delta','item_brand_id_lasttime_delta','shop_id_lasttime_delta','user_item_id_lasttime_delta',
'user_shop_id_lasttime_delta','user_brand_id_lasttime_delta','user_category_id_lasttime_delta',
'user_id_nexttime_delta','item_id_nexttime_delta','item_brand_id_nexttime_delta','shop_id_nexttime_delta','user_item_id_nexttime_delta',
'user_shop_id_nexttime_delta','user_brand_id_nexttime_delta','user_category_id_nexttime_delta',]]
    test_x=pd.merge(test_x,test_only7,'left',on='instance_id')
    
    train_before_1_test=train_before_1[(train_before_1['times']>='2018-09-07 12:00:00') & (train_before_1['times']<='2018-09-07 23:59:59')]
    del train_before_1_test['times']
    test_x=pd.merge(test_x,train_before_1_test,'left',on='instance_id')
    test_x=pd.merge(test_x,part_val,'left',on='instance_id')
    predict_7_val=predict_7[1077175:].reset_index(drop=True)
    test_x=pd.concat([test_x,predict_7_val],axis=1)

online_submit()
