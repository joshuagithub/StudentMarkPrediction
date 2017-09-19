import pickle
from sklearn import tree

def runmodel(inp):
	load_model= pickle.load(open('regression_model.sav', 'rb'))
	a=load_model.predict([inp])
	return a
	print("Predicted %=", a)
