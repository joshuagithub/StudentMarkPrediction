import pickle
from sklearn import tree
load_model= pickle.load(open('regression_model.sav', 'rb'))
inp= [float(x) for x in input().split()]
a=load_model.predict([inp])
print("Predicted %=", a)