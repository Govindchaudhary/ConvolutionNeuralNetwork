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
#basically converting the feature map matrix to single column matri of features

classifier.add(Flatten())

#step-4 Full connected layer

classifier.add(Dense(output_dim=128,activation='relu'))
classifier.add(Dense(output_dim=1,activation='sigmoid'))

#compiling the CNN
classifier.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])

#part-2 fitting the CNN to the images
from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1./255)

training_set = train_datagen.flow_from_directory(
        'dataset/training_set',
        target_size=(64, 64),
        batch_size=32,
        class_mode='binary')

test_set = test_datagen.flow_from_directory(
        'dataset/test_set',
        target_size=(64, 64),
        batch_size=32,
        class_mode='binary')

classifier.fit_generator(
        training_set,
        steps_per_epoch=8000,  #no. of sample images in training set
        epochs=25,
        validation_data=test_set,
        validation_steps=2000)  #no. of images in test set
