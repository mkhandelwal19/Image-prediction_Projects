
# coding: utf-8

# # Importing the Keras libraries and packages

##dropout, layers, epochs


# Importing the Keras libraries and packages
# to Initialize the neural network
from keras.models import Sequential
# Conv2D for 2 dimensional array for images
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense


# # Initialising the CNN


classifier = Sequential()


# # Step 1 - Convolution


# 32 is the number of feature detectors. Typically you start with 32
# input_shape = (64, 64, 3) - force your image to one single format - 3 is for color images
# activation functions removes any negative pixels
classifier.add(Conv2D(32, (3, 3), input_shape = (64, 64, 3), activation = 'relu'))


# # Step 2 - Pooling


# Pooling reduces the size
classifier.add(MaxPooling2D(pool_size = (2, 2)))


# # Adding a second convolutional layer


classifier.add(Conv2D(32, (3, 3), activation = 'relu'))
classifier.add(MaxPooling2D(pool_size = (2, 2)))

#adding 3rd layer
classifier.add(Conv2D(32,(3, 3), activation='relu'))
classifier.add(MaxPooling2D(pool_size = (2,2)))


# # Step 3 - Flattening

classifier.add(Flatten())


# # Step 4 - Full connection


# Add a Hidden Layer . A Number around 100 is a good choice..choose a number 2 to the power  
classifier.add(Dense(units = 128, activation = 'relu'))

# Add an output layer
classifier.add(Dense(units = 1, activation = 'sigmoid'))


# # Compiling the CNN


classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])


# # Part 2 - Fitting the CNN to the images

# Image Augmentation - Preprocessing the images to prevent overfitting (few data to train model) 
# It is a technique to enrich the dataset without adding more images


# https://keras.io/preprocessing/image/#imagedatagenerator-class
from keras.preprocessing.image import ImageDataGenerator

# rescale is feature scaling
# in searing pixels are moved in a fix direction
train_datagen = ImageDataGenerator(rescale = 1./255,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True)



test_datagen = ImageDataGenerator(rescale = 1./255)



training_set = train_datagen.flow_from_directory('dataset/training_set',
                                                 target_size = (64, 64),
                                                 batch_size = 32,
                                                 class_mode = 'binary')


test_set = test_datagen.flow_from_directory('dataset/test_set',
                                            target_size = (64, 64),
                                            batch_size = 32,
                                            class_mode = 'binary')


classifier.fit_generator(training_set,
                         steps_per_epoch = 480,
                         epochs = 2,
                         validation_data = test_set,
                         validation_steps = 160)


# # Part 3 - Making new predictions

# Predict the first image



import numpy as np
from keras.preprocessing import image
test_image = image.load_img('dataset/single_prediction/cat_or_dog_1.jpg', target_size = (64, 64))
# Add a new dimension to make a 3D array
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis = 0)
# Expects 4 dimensions. 4th dimension is batch
result = classifier.predict(test_image)
# know whether 0 is cat or dog
training_set.class_indices
if result[0][0] == 1:
    prediction1 = 'dog'
else:
    prediction1 = 'cat'


prediction1


# Predict the 2nd image


import numpy as np
from keras.preprocessing import image
test_image = image.load_img('dataset/single_prediction/cat_or_dog_2.jpg', target_size = (64, 64))
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis = 0)
result = classifier.predict(test_image)
training_set.class_indices
if result[0][0] == 1:
    prediction2 = 'dog'
else:
    prediction2 = 'cat'

prediction2
