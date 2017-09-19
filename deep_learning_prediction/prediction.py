import tensorflow as tf 
import numpy as np
from preprocessing_data import hot_data

filename = "student_data.csv"
test_name= "test.csv"

attributes = tf.placeholder('float', [None,16])
class_label = tf.placeholder('float', [1,2])


layer_1_nodes= 50
layer_2_nodes= 50
layer_3_nodes= 50
layer_4_nodes= 50

epoch= 5

def deep_graph_model(columns):

	#initializing random weights and bias to hidden_layers and size
	hidden_layer_1 = {'weights': tf.Variable(tf.random_normal([16, layer_1_nodes])), 'bias': tf.Variable(tf.random_normal([layer_1_nodes]))}
	hidden_layer_2 = {'weights': tf.Variable(tf.random_normal([layer_1_nodes, layer_2_nodes])), 'bias': tf.Variable(tf.random_normal([layer_2_nodes]))}
	hidden_layer_3 = {'weights': tf.Variable(tf.random_normal([layer_2_nodes, layer_3_nodes])), 'bias': tf.Variable(tf.random_normal([layer_3_nodes]))}
	hidden_layer_4 = {'weights': tf.Variable(tf.random_normal([layer_3_nodes, layer_4_nodes])), 'bias': tf.Variable(tf.random_normal([layer_4_nodes]))}

	output_layer=    {'weights': tf.Variable(tf.random_normal([layer_4_nodes, 2])), 'bias': tf.Variable(tf.random_normal([2]))}

	#calculations ((Input * Weights) + Bias)   ----> activation()

	calc_1 = tf.add(tf.matmul(columns, hidden_layer_1['weights']), hidden_layer_1['bias'])
	calc_1 = tf.nn.relu(calc_1)
	calc_2 = tf.add(tf.matmul(calc_1, hidden_layer_2['weights']), hidden_layer_2['bias'])
	calc_2 = tf.nn.relu(calc_2)
	calc_3 = tf.add(tf.matmul(calc_2, hidden_layer_3['weights']), hidden_layer_3['bias'])
	calc_3 = tf.nn.relu(calc_3)
	calc_4 = tf.add(tf.matmul(calc_3, hidden_layer_4['weights']), hidden_layer_4['bias'])
	calc_4 = tf.nn.softmax(calc_4)
	output = tf.add(tf.matmul(calc_4, output_layer['weights']), output_layer['bias'])

	return output

def student_start():

	prediction= deep_graph_model(attributes)
	cost = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=prediction, labels=class_label))
	optimization = tf.train.AdamOptimizer().minimize(cost)

	with tf.Session() as s:
		s.run(tf.global_variables_initializer())
		for ep in range(epoch):

			loss = 0
			hot_array, hot_label = hot_data(filename)
	
			for i in range(len(hot_array)):
				feed_attributes= hot_array[i:i+1]
				feed_labels= hot_label[i:i+1]
				x, y = s.run([optimization,cost], feed_dict= {attributes: feed_attributes, class_label: feed_labels })
				loss +=y

			print('Epoch:',ep," Loss:",loss)
			
		correct = tf.equal(tf.argmax(prediction,1), tf.argmax(class_label,1))
		accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
		test_data, test_label= hot_data(test_name)
		print ("Accuracy:", accuracy.eval({attributes: test_data, class_label: test_label})*100, "%") 


if __name__ == "__main__":
	student_start() 









	