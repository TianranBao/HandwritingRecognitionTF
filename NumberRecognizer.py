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

	def apply_gradients(self, grads, vars): #takes gradients and existing weights to further optimize the learning rate of the network  
		if not self.built: #build optimizer on first run
			for var in vars: 
				v = tf.Variable(tf.zeros(shape=var.shape)) #create zero tensor and assign it to the first moment variable
				s = tf.Variable(tf.zeros(shape=var.shape)) #create zero tensor and assign it to the second moment variable
				self.v_dvar.append(v) #add first moment variable to the list of first moment variable
				self.s_dvar.append(s) #add second moment variable to the list of second moment variable
			self.built = True #build complete

		#apply gradients to model variables
		for i, (d_var, var) in enumerate(zip(grads, vars)): #for each pair of gradient and variable tensor inputs
			#update first moment estimate by multiplying first moment estimate by first moment exponential decay rate + (current first moment gradient)*(1- first moment decay rate)
			self.v_dvar[i].assign(self.beta_1*self.v_dvar[i] + (1-self.beta_1)*d_var) 
			#update second moment estimate by multiplying second moment estimate by second moment exponential decay rate + (1- second moment decay rate)*(current second moment gradient)^2
			self.s_dvar[i].assign(self.beta_2*self.s_dvar[i] + (1-self.beta_2)*tf.square(d_var))
			#scale first moment estimate by dividing first moment estimate by 1 - (first moment exponential decay rate)^(timestep)
			v_dvar_bc = self.v_dvar[i]/(1-(self.beta_1**self.t))
			#scale second moment estimate by dividing second moment estimate by 1 - (second moment exponential decay rate)^(timestep)
			s_dvar_bc = self.s_dvar[i]/(1-(self.beta_2**self.t)) 
			#control size step by multiplying learning rate by bias corrected (first moment estimate)/(sqrt(bias corrected second moment estimate) + epsilon)
			var.assign_sub(self.learning_rate*(v_dvar_bc/(tf.sqrt(s_dvar_bc) + self.ep)))
		self.t += 1 #increase timstep
		return 

def train_step(x_batch, y_batch, loss, acc, model, optimizer): #define training loop 
	with tf.GradientTape() as tape: #use tensorflow gradient tape to "watch" operations 
		y_pred = model(x_batch) #calculate predicted labels
		batch_loss = loss(y_pred, y_batch) #compute loss between predicted and actual labels
	batch_acc = acc(y_pred, y_batch) #calculate the accuracy of the predicted and true labels
	grads = tape.gradient(batch_loss, model.variables) #compute gradient of the batch loss with respect to the model's variables'
	optimizer.apply_gradients(grads, model.variables) #update model parameters based on model variables and loss gradient
	return batch_loss, batch_acc #return loss and accuracy for this batch

def val_step(x_batch, y_batch, loss, acc, model): #calculate performance of model
	y_pred = model(x_batch) #calculate predicted labels
	batch_loss = loss(y_pred, y_batch) #calculate the loss of this batch
	batch_acc = acc(y_pred, y_batch) #calculate the accuracy of this model
	return batch_loss, batch_acc #return loss and accuracy for this batch

def train_model(mlp, train_data, val_data, loss, acc, optimizer, ep): #inputs: neural network model, training data, validation data, loss function, accuracy function, and # of epochs to train for
	train_losses, train_accs = [], [] #store training losses and accuracies
	val_losses, val_accs = [], [] #store validation losses and accuracies 
	for epoch in range(epochs): #train for x epochs
		batch_losses_train, batch_accs_train = [], [] #track loss and accuracy for this batch of training data
		batch_losses_val, batch_accs_val = [], [] #track loss and accuracy for this batch of validation data 

		for x_batch, y_batch in train_data: #for input data and labels in inputted training data
			batch_loss, batch_acc = train_step(x_batch, y_batch, loss, acc, mlp, optimizer) #excecute the training step with given inputs -> updates model variables using calculated gradients
			batch_losses_train.append(batch_loss) #add this batch's losses to batch loss list
			batch_accs_train.append(batch_acc) #add this batch's accuracy to batch accuracy list

		for x_batch, y_batch in val_data: #for input data and labels in inputted validation data
			batch_loss, batch_acc = val_step(x_batch, y_batch, loss, acc, mlp)
			batch_losses_val.append(batch_loss)
			batch_accs_val.append(batch_acc)

		train_loss, train_acc = tf.reduce_mean(batch_losses_train), tf.reduce_mean(batch_accs_train) #take the average of the training loss and accuracy
		val_loss, val_acc = tf.reduce_mean(batch_losses_val), tf.reduce_mean(batch_accs_val) #take the average of the validation loss and accuracy
		train_losses.append(train_loss) #record average training loss for this epoch
		train_accs.append(train_acc) #record average training accuracy for this epoch
		val_losses.append(val_loss) #record average validation loss for this epoch
		val_accs.append(val_acc) #record average validation accuracy for this epoch
		print(f"Epoch: {epoch}")
		print(f"Training loss: {train_loss:.3f}, Training accuracy: {train_acc:.3f}")
		print(f"Validation loss: {val_loss:.3f}, Validation accuracy: {val_acc:.3f}")
	return train_losses, train_accs, val_losses, val_accs #return training losses, accuracies and validation losses, accuracies 

