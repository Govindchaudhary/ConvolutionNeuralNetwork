from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense


#initializing the CNN (2 methods defining the sequence of layers or defining a graph)
# we will go with the defining the sequence of layers
classifier = Sequential()

#step-1 convolution layer
#here 32 is the no. of feature detector and 3,3 is the size
#input_shape is the what to expect in input (here 64,64 sized color image 3 indicates colored)
#In computational networks, the activation function of a node defines the output of that node given an input or set of inputs.
#relu is the activation func to be used ie. f(x)=max(x,0)

classifier.add(Convolution2D(32,3,3, input_shape=(64,64,3),activation='relu'))

#step-2 Pooling
#basically reducing the no. of node that will get into  flatteing step ie. to reduce the size of feature maps
#avoiding the expensive computation
#avoids the chances of overfitting and also maintained the special features of the image
#main purpose is the spatial invariance ie. our model can recognize the same image at diff. angle  and position
#here pool_size is basically the box of size 2 x 2 so we are moving this box to the whole convoluted matrix with the stride of 2
#and we take the max of the four number and put it into our pooled matrix


classifier.add(MaxPooling2D(pool_size=(2,2)))

#adding a second convolution layer
#no need of input_shape bcos it gets input from pooled step1
#this is a simple way to inc the accuracy without inc the cost of computation

classifier.add(Convolution2D(32,3,3,activation='relu'))
classifier.add(MaxPooling2D(pool_size=(2,2)))

#step-3 Flattening
#basically converting the feature map matrix to single column matrix of features
#since each feature map contains the information of a special feature of the image 
#hence each node of the flatted matrix represents special feature of the image  

classifier.add(Flatten())

#step-4 Full connected layer
#here the choice of the no. of nodes is completely experiment based but it should not be too small to make it worse and not too big 
#to make it highly computation expensive

classifier.add(Dense(output_dim=128,activation='relu'))
'''
• Sigmoid functions are often used in artificial neural networks to
introduce nonlinearity in the model.
• A neural network element computes a linear combination of its input
signals, and applies a sigmoid function to the result. A reason for its
popularity in neural networks is because the sigmoid function satisfies
a property between the derivative and itself such that it is
computationally easy to perform.
 <-------------------- dervative(sig(t)) = sig(t)(1-sig(t)-------------------------------------------------------->
• Derivatives of the sigmoid function are usually employed in learning
algorithms.

'''

classifier.add(Dense(output_dim=1,activation='sigmoid'))

#compiling the CNN
#we will use loss func same as logistic regression which is lograthmic loss func denoted by crossentropy
classifier.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])

#part-2 fitting the CNN to the images
from keras.preprocessing.image import ImageDataGenerator #we will use it generate the image augmenation
'''
Image augmentation is a technique that is used to artificially expand the data-set. 
This is helpful when we are given a data-set with very few data samples. 
In case of Deep Learning, this situation is bad as the model tends to over-fit when we train it on limited number of data samples.
Image augmentation parameters that are generally used to increase the data sample count are  
zoom,shear, rotation, preprocessing_function and so on.
Usage of these parameters results in generation of images having these attributes during training of Deep Learning model. 
Image samples generated using image augmentation, in general results in increase of existing data sample set by nearly 3x to 4x times.
basically it creates many batches of images and in each batch it will apply some random transformations(like rotating , shifting,cropping etc) 
on some random images and thus we get many more diverse images
'''
#Generate batches of tensor image data with real-time data augmentation
train_datagen = ImageDataGenerator(
        rescale=1./255, #rescale: rescaling factor. Defaults to None. If None or 0, no rescaling is applied, otherwise we multiply the data by the value provided (before applying any other transformation).
        shear_range=0.2,
        zoom_range=0.2,  #Range for random zoom(a float qty)
        horizontal_flip=True) #horizontally flips the image

test_datagen = ImageDataGenerator(rescale=1./255)#rescale all the pixel values between 0 and 1

#actually creating the test set and training set
#flow_from_directory Takes the path to a directory, and generates batches of augmented/normalized data.
#first argument is: path to the target directory. It should contain one subdirectory per class. here dataset/training_set contains two sub-directory one for cat and 1 for dog
training_set = train_datagen.flow_from_directory(
        'dataset/training_set',
        target_size=(64, 64),   #The dimensions to which all images found will be resized.
        batch_size=32,          #size of the batches of data (default: 32) in which some random samples of our data will be included
                                #and that contains the no. of images that will go through our CNN after which the weights will be updated
        class_mode='binary')    #parametr indicating that if your dependent variable is binary or more than two

test_set = test_datagen.flow_from_directory(
        'dataset/test_set',
        target_size=(64, 64),
        batch_size=32,
        class_mode='binary')

#fitting our CNN to the training set while also testing its performance on test set
classifier.fit_generator(
        training_set,
        steps_per_epoch=8000,  #no. of sample images in training set
        epochs=25,
        validation_data=test_set,
        validation_steps=2000)  #no. of images in test set
