
# coding: utf-8

# # Import the libraries


import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Conv2D, MaxPooling2D, Flatten
import matplotlib.pyplot as plt


# # Load MNIST datasets
# 


(mnist_train_images, mnist_train_labels), (mnist_test_images, mnist_test_labels) = mnist.load_data()


# ## Check the shape of any sample imag
# 


mnist_train_images[1120].shape


# ## Plot a sample image


plt.imshow(mnist_train_images[1120],cmap='gist_gray')
plt.imshow(mnist_train_images[1135],cmap='gist_gray')
plt.imshow(mnist_train_images[500],cmap='gist_gray')
plt.imshow(mnist_train_images[504],cmap='gist_gray')


## Visualize the flattened array



plt.imshow(mnist_train_images[1120].reshape(784,1),cmap='gist_gray',aspect=0.02)


# We need to shape the data differently then before. Since we're treating the data as 2D images of 28x28 pixels instead of a flattened stream of 784 pixels, we need to shape it accordingly. Depending on the data format Keras is set up for, this may be 1x28x28 or 28x28x1 (the "1" indicates a single color channel, as this is just grayscale. If we were dealing with color images, it would be 3 instead of 1 since we'd have red, green, and blue color channels)
# 


from keras import backend as K

if K.image_data_format() == 'channels_first':
    train_images = mnist_train_images.reshape(mnist_train_images.shape[0], 1, 28, 28)
    test_images = mnist_test_images.reshape(mnist_test_images.shape[0], 1, 28, 28)
    input_shape = (1, 28, 28)
else:
    train_images = mnist_train_images.reshape(mnist_train_images.shape[0], 28, 28, 1)
    test_images = mnist_test_images.reshape(mnist_test_images.shape[0], 28, 28, 1)
    input_shape = (28, 28, 1)

train_images = train_images.astype('float32')
test_images = test_images.astype('float32')


# Change each pixel into some number between 0 and 1


train_images /= 255
test_images /= 255


# Change to one hot format


train_labels = keras.utils.to_categorical(mnist_train_labels, 10)
test_labels = keras.utils.to_categorical(mnist_test_labels, 10)


# A method to write the label and print the image


def display_sample(num):
    #Print the one-hot array of this sample's label 
    print(train_labels[num])  
    #Print the label converted back to a number
    label = train_labels[num].argmax(axis=0)
    #Reshape the 768 values to a 28x28 image
    image = train_images[num].reshape([28,28])
    plt.title('Sample: %d  Label: %d' % (num, label))
    plt.imshow(image, cmap=plt.get_cmap('gray_r'))
    plt.show()



display_sample(1120)
display_sample(507)
display_sample(2402)


# Build a Convlution layer. Dropout prevents overfitting


model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3),
                 activation='relu',
                 input_shape=input_shape))
# 64 3x3 kernels
model.add(Conv2D(64, (3, 3), activation='relu'))
# Reduce by taking the max of each 2x2 block
model.add(MaxPooling2D(pool_size=(2, 2)))
# Dropout to avoid overfitting
model.add(Dropout(0.25))
# Flatten the results to one dimension for passing into our final layer
model.add(Flatten())
# A hidden layer to learn with
model.add(Dense(128, activation='relu'))
# Another dropout
model.add(Dropout(0.5))
# Final categorization from 0-9 with softmax
model.add(Dense(10, activation='softmax'))


# Check model summary


model.summary()


# Define the loss function and optimizer


model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])


# Train the mode


history = model.fit(train_images, train_labels,
                    batch_size=32,
                    epochs=2,
                    verbose=2,
                    validation_data=(test_images, test_labels))



score = model.evaluate(test_images, test_labels, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])


# Print test data label and image


for x in range(10):
    test_image = test_images[x].reshape(1,28,28,1)
    predicted_cat = model.predict(test_image,).argmax()
    predicted_cat
    print("predicted value" ,predicted_cat)
    print("image label" , test_labels[x].argmax())
    plt.imshow(test_image.reshape(28,28),cmap='gist_gray')
    plt.show()


# print label mismatch


for x in range(1000):
    test_image = test_images[x].reshape(1,28,28,1)
    predicted_cat = model.predict(test_image,).argmax()
    label = test_labels[x].argmax()
    if (predicted_cat != label):
        plt.title('Prediction: %d Label %d' % (predicted_cat,label))
        plt.imshow(test_image.reshape(28,28),cmap='gist_gray')
        plt.show()

