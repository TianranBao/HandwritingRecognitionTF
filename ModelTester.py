import tensorflow as tf
import tensorflow_datasets as tfds
import os

current_dir = os.path.dirname(os.path.realpath(__file__)) #get the current directory of the file 
save_path = os.path.join(current_dir, "mlp_model_export") #add file name 'mlp_model_export' to the end of directory

mlp_loaded = tf.saved_model.load(save_path) #load saved model from save path

def accuracy_score(y_pred, y): #calculate accuracy of predicted and true labels
	is_equal = tf.equal(y_pred, y) #compares how close/equal predicted and true labels are 
	return tf.reduce_mean(tf.cast(is_equal, tf.float32)) #return the average of how close each prediction is to the true label 

#use entire MNIST dataset as a test set. create a set of image data and another for label data 
x_test, y_test = tfds.load("mnist", split=["test"], batch_size=-1, as_supervised=True)[0]
test_classes = mlp_loaded(x_test) #calculate the predicted labels based on the imported model
test_acc = accuracy_score(test_classes, y_test) #calculate the accuracy of the predicted labels and actual label 
print(f"Test Accuracy: {test_acc:.3f}")
