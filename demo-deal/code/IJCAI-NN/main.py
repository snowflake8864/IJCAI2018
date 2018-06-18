# -*- coding:utf-8 -*-
from model import NFM
from utils import data_preprocess
from model import DeepFM
from model import AFM
from model import DCN
from model import DIN
from model import FNN
from model import NFM
from model import PNN
import torch
import config
import pandas as pd
import gc
from DataReader import FeatureDictionary, DataParser

#loda data
#dfTrain = pd.read_table(config.TRAIN_FILE, sep=" ")
#dfTest = pd.read_table(config.TEST_FILE, sep=" ")
#dfTrain = pd.read_csv(config.TRAIN_FILE)
#dfTest = pd.read_csv(config.TEST_FILE)

#pick features
features = config.BASIC_COLS + config.STRONG_COLS

# data = pd.read_csv(config.DATA_FILE)

online = True

if online==False :

    print('Valid Model Offline...')
    dfTrain = pd.read_csv(config.TRAIN_FILE)
    dfTest = pd.read_csv(config.TEST_FILE)
    dfTrain = dfTrain[features]
    dfTest = dfTest[features]
    print('CSV OK...')

    # Fillna  By Snake
    dfTrain.fillna(dfTrain.median(),inplace=True)
    dfTrain.fillna(-1,inplace=True)  # Prevent NAN 
    dfTest.fillna(dfTest.median(),inplace=True)
    dfTest.fillna(-1,inplace=True)

    print('Fillna OK...')
    gc.collect()

else:
    print('Online Model ...')
    dfTrain = pd.read_csv(config.ONLINE_TRAIN_FILE)
    dfTest = pd.read_csv(config.ONLINE_TEST_FILE)
    dfTrain = dfTrain[features]
    dfTest = dfTest[features]
    print('CSV OK...')

    # Fillna  By Snake
    dfTrain.fillna(dfTrain.median(),inplace=True)
    dfTrain.fillna(-1,inplace=True)  # Prevent NAN 
    dfTest.fillna(dfTest.median(),inplace=True)
    dfTest.fillna(-1,inplace=True)

    print('Fillna OK...')
    del data

num_epoch = 20

#field cols
#cols = [c for c in features if (not c in config.IGNORE_COLS)]

'''
X_train = dfTrain[cols].values
y_train = dfTrain["is_trade"].values
X_test = dfTest[cols].values
ids_test = dfTest["instance_id"].values
cat_features_indices = [i for i,c in enumerate(cols) if c in config.CATEGORICAL_COLS]
'''

#convert data to (index: value: lable)
fd = FeatureDictionary(dfTrain=dfTrain, dfTest=dfTest,
                       numeric_cols=config.NUMERIC_COLS,
                       ignore_cols=config.IGNORE_COLS)
data_parser = DataParser(feat_dict=fd)
Xi_train, Xv_train, y_train = data_parser.parse(df=dfTrain, has_label=True)
Xi_test, Xv_test, y_test = data_parser.parse(df=dfTest, has_label=True)

#feature_size = fd.feat_dim
feature_sizes = fd.feature_sizes
field_size = len(Xi_train[0]) 


num_epoch = 30

#没有装cuda，不要调cuda
#Xi_train和result_dict['index']不同,一个是统计全局，一个是统计某一个
def run_deepFM():
    deepfm = DeepFM.DeepFM(field_size,feature_sizes,verbose=True,use_cuda=True, weight_decay=0.00005,embedding_size=12,use_fm=False,batch_size=128,use_ffm=True,use_deep=True, n_epochs=num_epoch)
    if online == False:
        deepfm.fit(Xi_train, Xv_train, y_train, Xi_test, Xv_test, y_test, ealry_stopping=True,refit=True)
    else:
        deepfm.fit(Xi_train, Xv_train, y_train, ealry_stopping=False,refit=False)
        pb = deepfm.predict_proba(Xi_test, Xv_test)
        test['predicted_score'] = pb
        test[['instance_id', 'predicted_score']].to_csv('../../submit/deepFM.txt', sep=" ", index=False)

def run_AFM():
    afm = AFM.AFM(field_size, feature_sizes, batch_size=32 * 8, is_shallow_dropout=False, verbose=True, use_cuda=True,
                          weight_decay=0.00002, use_fm=True, use_ffm=False, n_epochs= num_epoch)
    if online = False:
        afm.fit(Xi_train, Xv_train, y_train, Xi_test, Xv_test, y_test, ealry_stopping=True,refit=True)
    else: 
        afm.fit(Xi_train, Xv_train, y_train, ealry_stopping=True,refit=True)
        pb = afm.predict_proba(Xi_test, Xv_test)
        test['predicted_score'] = pb
        test[['instance_id', 'predicted_score']].to_csv('../../submit/AFM.txt', sep=" ", index=False)


def run_DCN():
    dcn = DCN.DCN(field_size, feature_sizes, batch_size=32 * 8, verbose=True, use_cuda=True,
                          weight_decay=0.00002, use_inner_product=True, n_epochs=num_epoch)
    if online = False:
        dcn.fit(Xi_train, Xv_train, y_train, Xi_test, Xv_test, y_test, ealry_stopping=True,refit=True)
    else:
        dcn.fit(Xi_train, Xv_train, y_train, ealry_stopping=True,refit=True)
        pb = dcn.predict_proba(Xi_test, Xv_test)
        test['predicted_score'] = pb
        test[['instance_id', 'predicted_score']].to_csv('../../submit/DCN.txt', sep=" ", index=False)

def run_DIN():
    din = DIN.DIN(field_size, feature_sizes, batch_size=32 * 8, is_shallow_dropout=False, verbose=True, use_cuda=True,
          weight_decay=0.0000002, use_fm=True, use_ffm=False, use_high_interaction=True,interation_type=False,n_epochs=num_epoch)
    if online = False:
        din.fit(Xi_train, Xv_train, y_train, Xi_test, Xv_test, y_test, ealry_stopping=True,refit=True)
    else:
        din.fit(Xi_train, Xv_train, y_train, ealry_stopping=True,refit=True)
        pb = din.predict_proba(Xi_test, Xv_test)
        test['predicted_score'] = pb
        test[['instance_id', 'predicted_score']].to_csv('../../submit/DIN.txt', sep=" ", index=False)

def run_FNN():
    fnn = FNN.FNN(field_size, feature_sizes, batch_size=32 * 16, verbose=True, use_cuda=True,
                          h_depth = 3,pre_weight_decay= 0.0001 ,weight_decay=0.00001,learning_rate=0.0001, use_fm=True, use_ffm=False, n_epochs=num_epoch)
    if online = False:
        fnn.fit(Xi_train, Xv_train, y_train, Xi_test, Xv_test, y_test, ealry_stopping=True,refit=True)
    else:
        fnn.fit(Xi_train, Xv_train, y_train, ealry_stopping=True,refit=True)
        pb = fnn.predict_proba(Xi_test, Xv_test)
        test['predicted_score'] = pb
        test[['instance_id', 'predicted_score']].to_csv('../../submit/FNN.txt', sep=" ", index=False)

if __name__ == '__main__' :
    run_deepFM()
    run_AFM()
    run_DCN()
    run_DIN()
    run_FNN()
    print('Finished ! ...')
