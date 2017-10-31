import tensorflow as tf 
import numpy as np
from preprocessing_data import hot_data
from shuffle_data import shuffle_split
import os


filename = "student_data.csv"
train_name = "train.csv"
test_name= "test.csv"

attributes = tf.placeholder('float', [None,16])
class_label = tf.placeholder('float')

#to shuffle the data and split 66-34% train and test
shuffle_split(80,filename)

# mean_moving_normal = tf.random_normal(shape=[1000], mean=(5*class_label), stddev=1)
# tf.summary.histogram("normal/moving_mean", mean_moving_normal)

# # # Make a normal distribution with shrinking variance
# variance_shrinking_normal = tf.random_normal(shape=[1000], mean=0, stddev=1-(class_label))
# # # Record that distribution too
# tf.summary.histogram("normal/shrinking_variance", variance_shrinking_normal)

# # # Let's combine both of those distributions into one dataset
# normal_combined = tf.concat([mean_moving_normal, variance_shrinking_normal], 0)
# # # We add another histogram summary to record the combined distribution
# tf.summary.histogram("normal/bimodal", normal_combined)



path= "tmp/histogram_example/" 
writer= tf.summary.FileWriter(path)
summaries = tf.summary.merge_all() 

layer_1_nodes= 100
layer_2_nodes= 100
#layer_3_nodes= 150
#layer_4_nodes= 150
#layer_5_nodes= 150

a_accuracy= []

epoch=4

def deep_graph_model(columns):

	#initializing random weights and bias to hidden_layers and size
	hidden_layer_1 = {'weights': tf.Variable(tf.random_normal([16, layer_1_nodes])), 'bias': tf.Variable(tf.random_normal([layer_1_nodes]))}
	hidden_layer_2 = {'weights': tf.Variable(tf.random_normal([layer_1_nodes, layer_2_nodes])), 'bias': tf.Variable(tf.random_normal([layer_2_nodes]))}
	#hidden_layer_3 = {'weights': tf.Variable(tf.random_normal([layer_2_nodes, layer_3_nodes])), 'bias': tf.Variable(tf.random_normal([layer_3_nodes]))}
	#hidden_layer_4 = {'weights': tf.Variable(tf.random_normal([layer_3_nodes, layer_4_nodes])), 'bias': tf.Variable(tf.random_normal([layer_4_nodes]))}
	#hidden_layer_5 = {'weights': tf.Variable(tf.random_normal([layer_4_nodes, layer_5_nodes])), 'bias': tf.Variable(tf.random_normal([layer_5_nodes]))}
	
	output_layer=    {'weights': tf.Variable(tf.random_normal([layer_2_nodes, 2])), 'bias': tf.Variable(tf.random_normal([2]))}


	calc_1 = tf.add(tf.matmul(columns, hidden_layer_1['weights']), hidden_layer_1['bias'])
	calc_1 = tf.nn.relu(calc_1)
	calc_2 = tf.add(tf.matmul(calc_1, hidden_layer_2['weights']), hidden_layer_2['bias'])
	calc_2 = tf.nn.relu(calc_2)
	#calc_3 = tf.nn.relu(calc_3)
	#calc_4 = tf.add(tf.matmul(calc_3, hidden_layer_4['weights']), hidden_layer_4['bias'])
	#calc_4 = tf.nn.softmax(calc_4)
	#calc_5 = tf.add(tf.matmul(calc_4, hidden_layer_5['weights']), hidden_layer_5['bias'])
	#calc_5 = tf.nn.softmax(calc_5)
	output = tf.add(tf.matmul(calc_2, output_layer['weights']), output_layer['bias'])

	return output


def student_start():
	
	


	prediction= deep_graph_model(attributes)

	with tf.name_scope("cost"):
		cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=class_label))
		optimization = tf.train.AdamOptimizer().minimize(cost)
		tf.summary.scalar("cost", cost)

	with tf.name_scope("accuracy"):
		correct = tf.equal(tf.argmax(prediction,1), tf.argmax(class_label,1))
		accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
			


	test_data, test_label= hot_data(test_name)
	with tf.Session() as s:

		s.run(tf.global_variables_initializer())
		saver= tf.train.Saver()
		
		
		
		
		if (os.path.isfile("model/predict.ckpt")):
			saver.restore(s, "model/predict.ckpt")
		for ep in range(epoch):
			loss = 0
			hot_array, hot_label = hot_data(train_name)
			for i in range(len(hot_array)):
				feed_attributes= hot_array[i:i+1]
				feed_labels= hot_label[i:i+1]
				x, y = s.run([optimization,cost], feed_dict= {attributes: feed_attributes, class_label: feed_labels})
				tf.summary.scalar("cost", cost)
				loss +=y
				k_val= i/len(hot_array)
				summ = s.run(summaries, feed_dict={class_label: k_val})
				
				writer.add_summary(summ, global_step=i)


			print('Epoch:',ep+1," Loss:",loss)

		
		save_path= saver.save(s, "model/predict.ckpt")	
		print("Model saved in file: %s" % save_path)	
		correct = tf.equal(tf.argmax(prediction,1), tf.argmax(class_label,1))
		accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
		print ("Accuracy:", accuracy.eval(feed_dict={attributes: test_data, class_label: test_label})*100, "%")
		return (accuracy.eval(feed_dict={attributes: test_data, class_label: test_label})*100)


if __name__ == "__main__":
	a=student_start()
	# while(a<85):
	# 	shuffle_split(80,filename)
	# 	a=student_start() 

# tensorboard --logdir=/tmp/histogram_example/ 
# tensorboard --logdir=/graph/distribution