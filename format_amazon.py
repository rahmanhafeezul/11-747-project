import pandas
import pdb
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.utils import shuffle

file_1 = "amazon/amazon_reviews_us_Digital_Software_v1_00.tsv"
file_2 = "amazon/amazon_reviews_us_Mobile_Apps_v1_00.tsv"
file_3 = "amazon/amazon_reviews_us_Software_v1_00.tsv"

df_1 = pandas.read_csv(file_1, sep='\t', header=0,error_bad_lines=False)
df_2 = pandas.read_csv(file_2, sep='\t', header=0,error_bad_lines=False)
df_3 = pandas.read_csv(file_3, sep='\t', header=0,error_bad_lines=False)
# pdb.set_trace()


X = df_1['review_body'].tolist()
X.extend(df_2['review_body'].tolist())
X.extend(df_3['review_body'].tolist())

y = df_1['star_rating'].tolist()
y.extend(df_2['star_rating'].tolist())
y.extend(df_3['star_rating'].tolist())

X, y = shuffle(X, y)
# indices = np.arange(len(X)).reshape((-1,1))
# np.random.shuffle(indices)
# indices=indices.ravel().reshape((-1,1))
# X = X[indices]
# y = y[indices]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=1)

X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=1)


count=0
# pdb.set_trace()
with open('amazon_train.tsv', 'w') as f:
	to_write = ""
	for i in range(len(X_train)):
		count+=1
		if(count%100==0):

			to_write+=str(X_train[i]).replace("\n", " ")+"\t"+str(int(y_train[i])-1)+"\n"
	f.write(to_write)


count=0
with open('amazon_dev.tsv', 'w') as f:
	to_write = ""
	for i in range(len(X_val)):
		count+=1
		if(count%100==0):
			to_write+=str(X_val[i]).replace("\n", " ")+"\t"+str(int(y_val[i])-1)+"\n"
	f.write(to_write)

count=0
with open('amazon_test.tsv', 'w') as f:
	to_write =""
	for i in range(len(X_test)):
		count+=1
		if(count%100==0):
			to_write+=str(X_test[i]).replace("\n", " ")+"\n"
	f.write(to_write)