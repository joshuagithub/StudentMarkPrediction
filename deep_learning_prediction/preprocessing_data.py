import numpy as np

def hot_data(filename):
	labels=[]
	data=[]

	for line in open(filename):

		row = line.split(',')
		labels.append(int(row[16]))
		
		data.append([float(x) for x in row[0:16]])

	np_data = np.matrix(data).astype(np.float32)
	np_labels = np.array(labels).astype(dtype=np.uint8)

	#one-hot matrix
	hot_labels= (np.arange(2) == np_labels[:, None]).astype(np.float32)

	return np_data, hot_labels
	
