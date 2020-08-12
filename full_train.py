import pandas as pd
df=pd.read_csv('train_selected.csv')
dfy=pd.read_csv('train_target.csv',header=None)
df['class']=dfy
df.to_csv('train_full.csv',index=False)
df=pd.read_csv('test_selected.csv')
dfy=pd.read_csv('test_target.csv',header=None)
df.to_csv('test_full.csv',index=False)