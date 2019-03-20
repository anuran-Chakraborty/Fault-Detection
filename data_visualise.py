import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 

df = pd.read_excel('Acoustic_fault.xls')
df2 = pd.read_excel('Acoustic_fault.xls',2)

c1 = df['c01'].values
# c6 = df['c06'].values
# c11 = df['c11'].values

c2 = df2['c01'].values
# c7 = df2['c0'].values

plt.plot(c1[0:100:1],'^')
# plt.plot(c6[500:1000:1],'<')
#plt.plot(c11[500:1000:1],'>')

plt.plot(c2[0:100:1],'s')
# plt.plot(c7[500:1000:1],'p')

plt.show()

