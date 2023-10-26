#libraries
import pandas as pd 
import matplotlib #matlab plotting
from matplotlib import pyplot as plt 
import seaborn as sns #graph visualization  
import tempfile #allow python to make temporary files 
import os #OS interface commands 
import tensorflow as tf 
import tensorflow_datasets as tfds #will be getting MNIST data from here

matplotlib.rcParams['figure.figsize'] = [9,6] #set matplot figures to 9 inces by 6 inches
tf.random.set_seed(88) #set seed so results are reproducible 

#create training (60,000 images), validation (10,000 images), and test sets of the MNIST dataset, split into batches of 128 images with labels for supervised learning 
trainData, valData, testData = tfds.load("mnist", split=['train[10000:]', 'train[0:10000]', 'test'], batch_size=128, as_supervised=True)

#create two MNIST datasets with 1500 images each - one for image data and one for label information, each with a batch size spanning the whole dataset 
imageData, labelData = tfds.load("mnist", split=['train[:1500]'], batch_size=-1, as_supervised=True)[0]

#remove unnecessary channel dimension (axis 3)
imageData = tf.squeeze(imageData, axis=3) 

def visualizeMNIST(imgData, num): #make graphics of the first (num) pictures in the visualization set 
	for i in range(num):
		plt.subplot(3,3,1+i) #add to a 3x3 subplot with graph index 1+i 
		plt.axis('off') #no axis
		plt.imshow(imgData[i], cmap='gray') #draw graphs onto plot
		plt.title(f"Label: {labelData[i]}") #label each graph
		plt.subplots_adjust(hspace=.5) #space each graph
	plt.show() #display graph

def histogram(data): #make a histogram of the amount of inputted dataset
	sns.countplot(x=data.numpy())
	plt.xlabel("Number")
	plt.title("Data Distribution")
	plt.show()

def preprocess(imgData, lbls): #reshape MNIST data so it can be processed by the neural net 
	imgData = tf.reshape(imgData, shape=[-1, 784]) #reshape to flatten the data so that the neural net can take a 1x784 input 
	imgData = imgData/255 # make the color values fit between [0, 1] instead of the color values of [0,255]
	return imgData, lbls

trainData, valData = trainData.map(preprocess), valData.map(preprocess) #apply preprocessing functions to the training and validation iamge and label sets

def xavier_init(shape): #creates a weight matrix of size shape x shape initialized by random values from the xavier initialization
	inputDim, outputDim = shape 
	xavierLim = tf.sqrt(6.)/tf.sqrt(tf.cast(inputDim + outputDim, tf.float32)) #define the Xavier initialization with input and output dimensions
	weightVals = tf.random.uniform(shape=(inputDim, outputDim), minval=-xavierLim, maxval=xavierLim, seed=88) #create the random weight matrix using xavier initialization
	return weightVals 

class DenseLayer(tf.Module): 
	def __init__(self, out_dim, weight_init=xavier_init, activation=tf.identity): #redefines DenseLayer's init to have a default weight_init of xavierInit and an identity activation function which just returns the same tensor that is inputted
		self.out_dim = out_dim 
		self.weight_init = weight_init #defaults to xavierInit
		self.activation = activation #defaults to identity activation function
		self.built = False

	def __call__(self, inputData): #when DenseLayer is called as a method
		if not self.built: #if the DenseLayer is not previously built
			self.in_dim = inputData.shape[1] #base input dimension based on inputted data
			self.w = tf.Variable(self.weight_init(shape=(self.in_dim, self.out_dim))) #create the weight matrix in the layer
			self.b = tf.Variable(tf.zeros(shape=(self.out_dim))) #create a bias matrix filled with zeros 
			self.built = True #layer building is complete
		z = tf.add(tf.matmul(inputData, self.w), self.b) #multiply the input data by the weight matrix and then add the bias matrix
		return self.activation(z) #return the modified matrix put through the activation function 

class MLP(tf.Module): #create multilayer perceptron model that will excecute layers in order
	def __init__(self, layers):
		self.layers = layers

	@tf.function #declare __call__ as a tensorflow function to improve performance
	def __call__(self, inputData, preds=False): #preds=False disables predictions being printed after every layer
		for layer in self.layers: #for every layer...
			inputData = layer(inputData) #pass the inputted data through the layer
		return inputData #return final result 

#For this multilayer perceptron, we are using the following archeitechture:
#Forward Pass: ReLU(784x700) x ReLU(700x500) x Softmax(500x10)
#784 (number of pixels), 700 (hidden layer 1), 500 (hidden layer 2), 10 (possible outcomes)

hiddenL1Size = 700
hiddenL2Size = 500
outputSize = 10

mlp_model = MLP([ #initialize a multilayer perceptron with two hidden layers and one output layer
	DenseLayer(out_dim=hiddenL1Size, activation=tf.nn.relu), #first hidden layer with output size 700 and a ReLU activation function
	DenseLayer(out_dim=hiddenL2Size, activation=tf.nn.relu), #second hidden layer with output size 500 and a ReLU activation function
	DenseLayer(out_dim=outputSize)]) # output layer with output size 10

def cross_entropy_loss(y_pred, y): #calculate cross entropy loss 
	#pass simplified true labels and prediction through a softmax function followed by a cross entropy calculation
	sparse_ce = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=y_pred) 
 	return tf.reduce_mean(sparse_ce) #return the mean of the cross entropy values

def accuracy(y_pred, y): #compute accuracy of the model's predictions
	#take the predictions, turn them into a probability distribution using softmax, then use argmax to find the highest probability prediction
	class_preds = tf.argmax(tf.nn.softmax(y_pred), axis=1) 
	is_equal = tf.equal(y, class_preds) #checks if label and prediction are equal
	return tf.reduce_mean(tf.cast(is_equal, tf.float32)) #return accuracy of the prediction from 0.0 to 1.0 by converting tensorflow's boolean to a float value

class Adam: #utilize TensorFlow's Adaptive Moment Estimation optimizer for faster convergence

	def __init__(self, learning_rate =1e-3, beta_1=0.9, beta_2=0.999, ep=1e-7):
		self.beta_1 = beta_1 #weights past gradients' contributions to the current gradient
		self.beta_2 = beta_2 #weights past squared gradients' contributions to the current gradient
		self.learning_rate = learning_rate #determines step size
		self.ep = ep #small value added to denominator to prevent division by zero
		self.t = 1. #keep track of how many updates are done 
		self.v_dvar, self.s_dvar = [], [] #stores first and second moment estimates of the gradient
		self.built = False #is the optimizer initialized? 

