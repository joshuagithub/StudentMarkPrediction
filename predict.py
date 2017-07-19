from sklearn import tree
import pandas as pd
from sklearn.model_selection import train_test_split

df=pd.read_csv('test.csv')
train, test = train_test_split(df, test_size = 0.2)
clf=tree.DecisionTreeRegressor()
clf= clf.fit(train[['I','II','III','IV']],train[['V']])
inp= [float(x) for x in input().split()]
a=clf.predict([inp])
print("Predicted %=", a)
test_value= []
for x in test:
	test_value.append(clf.predict([x]))
testPercen=0
for x in range(len(test_value)):
	if math.ceil(test_value[x])== math.ceil(test[x][4]):
		testPercen +=1
accuracy= (testPercen*100)/ len(test_value)


import pydotplus 
dot_data = tree.export_graphviz(clf, out_file=None) 
graph = pydotplus.graph_from_dot_data(dot_data) 
graph.write_pdf("test.pdf")	