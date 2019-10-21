import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 

vals = np.array(pd.read_csv("weights.csv",header=None))

for i in range(15):

	curr = vals[i]
	plt.subplot()
	plt.plot(curr)
	
plt.show()	
