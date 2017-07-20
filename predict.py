from sklearn import tree
import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
import math
from sklearn.feature_selection import VarianceThreshold
import pickle
df=pd.read_csv('test.csv')
train, test = train_test_split(df, test_size = 0.2)

clf=tree.DecisionTreeRegressor()
clf= clf.fit(train[['I','II','III','IV']],train[['V']])
test_value= []
np_test=np.asarray(test)
filename='regression_model.sav'
pickle.dump(clf, open(filename, 'wb'))
for x in range(len(test)):
	test_value.append(clf.predict([np_test[x,0:4]]))
np_value=np.asarray(test_value)
testPercen=0
np_1= np.asarray(test)
for x in range(len(test_value)):
	if math.ceil(np_value[x])== math.ceil(np_1[x,4]):
		testPercen +=1
accuracy= (testPercen*100)/ len(test_value)
print ("Accuracy of the model is ", accuracy)
#to visualize the model, creates pdf
'''import pydotplus 
dot_data = tree.export_graphviz(clf, out_file=None) 
graph = pydotplus.graph_from_dot_data(dot_data) 
graph.write_pdf("test.pdf")	'''