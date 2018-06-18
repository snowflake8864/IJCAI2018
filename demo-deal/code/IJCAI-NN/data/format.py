import pandas as pd

train_data = pd.read_csv('train_all_to7_x.csv')
train_label = pd.read_csv('train_all_to7_y.csv',header=None)
train_label.columns = ['is_trade']
train = pd.concat([train_data,train_label],axis=1)
train.to_csv('train.csv',index=False)

val_data = pd.read_csv('val_all_to7_x.csv')
val_label = pd.read_csv('val_all_to7_y.csv',header=None)
val_label.columns = ['is_trade']
valid = pd.concat([val_data,val_label],axis=1)
valid.to_csv('valid.csv',index=False)
