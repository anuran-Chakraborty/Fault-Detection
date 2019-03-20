import pandas as pd 

df1 = pd.read_csv('train_clean.csv')
df2 = pd.read_csv('test_clean.csv')
df = pd.concat([df1,df2])
df = df.sample(frac=1)


size = df.shape[0]

df1 = df.iloc[ : int(0.7 * size) , : ]
df2 = df.iloc[int(0.7 * size) : , : ]

df1.to_csv('train_clean_shuffled.csv',index = None)
df2.to_csv('test_clean_shuffled.csv', index =None)
