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