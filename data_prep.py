import pandas as pd
import numpy as np
import glob

set1=['c'+str(i).zfill(2) for i in range(1,6)]
set2=['c'+str(i).zfill(2) for i in range(6,11)]
set3=['c'+str(i).zfill(2) for i in range(11,16)]
set4=['c'+str(i).zfill(2) for i in range(16,21)]
set5=['c'+str(i).zfill(2) for i in range(21,26)]

# print(set1)
# print(set2)
# print(set3)
# print(set4)
# print(set5)

def split_files(filename,c,folder=''):

	df_combined=pd.read_csv(filename)
	name_of_file=filename[filename.rfind('/')+1:]
	# print(df_combined)
	print(name_of_file)

	df_train_1=df_combined[set1]
	# print((df_train_1.isna().sum()/df_train_1.shape[0])*100)

	# Now unroll
	unrolled=df_train_1.values.reshape(1,-1)
	df_train_1=pd.DataFrame(unrolled)	

	df_train_2=df_combined[set2]
	# print((df_train_2.isna().sum()/df_train_2.shape[0])*100)

	unrolled=df_train_2.values.reshape(1,-1)
	df_train_2=pd.DataFrame(unrolled)

	df_train_3=df_combined[set3]
	# print((df_train_3.isna().sum()/df_train_3.shape[0])*100)

	unrolled=df_train_3.values.reshape(1,-1)
	df_train_3=pd.DataFrame(unrolled)

	# print(df_train_1)
	# print(df_train_2)
	# print(df_train_3)

	df_test_1=df_combined[set4]
	print('***************************************')
	print((df_test_1.isna().sum()/df_test_1.shape[0])*100)
	df_test_1.fillna(0, inplace=True)
	print((df_test_1.isna().sum()/df_test_1.shape[0])*100)

	# Now unroll
	unrolled=df_test_1.values.reshape(1,-1)
	df_test_1=pd.DataFrame(unrolled)

	df_test_2=df_combined[set5]
	print((df_test_2.isna().sum()/df_test_2.shape[0])*100)
	df_test_2.fillna(0, inplace=True)
	print((df_test_2.isna().sum()/df_test_2.shape[0])*100)
	print('***************************************')
	# Now unroll
	unrolled=df_test_2.values.reshape(1,-1)
	df_test_2=pd.DataFrame(unrolled)

	# Append the rows
	df_train=pd.concat([df_train_1, df_train_2, df_train_3], ignore_index=True)
	df_train['class']=c

	# print(df_train)

	df_test=pd.concat([df_test_1, df_test_2], ignore_index=True)
	df_test['class']=c
	
	# Write to csv
	df_train.to_csv(folder+'csv_train/'+name_of_file, header=False, index=False)
	df_test.to_csv(folder+'csv_test/'+name_of_file, header=False, index=False)


filelist=glob.glob('csv_files/*.csv')
filelist.sort(key=len)
print(filelist)

class_dict={}
i=0
for files in filelist:
	fname=files[files.rfind('/')+1:files.rfind('.')] # Get the name of the file
	print(fname)

	# If single partial discharge
	if(fname.find('_')==-1):
		class_dict[fname]=i # Create a dictionary of classes
		split_files(files,i)
		i=i+1
	# Mixed partial discharge
	else:
		class1=fname[:fname.rfind('_')]
		class2=fname[fname.rfind('_')+1:]
		print(str(class1)+' '+str(class2))
		# Combine the names of two classes
		n_class=str(class_dict[class1])+'_'+str(class_dict[class2])
		split_files(files,i,folder='multi_')
		i=i+1