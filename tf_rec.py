from __future__ import print_function
import keras
from keras.datasets import mnist
from keras.layers import Dense, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.models import Sequential
import matplotlib.pylab as plt
import cv2
import numpy
import sys
import os
from PIL import Image
import PIL
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

batch_size = 128
num_classes = 2
epochs = 10

img_x, img_y = 45, 80 #16:9

def get_training_data(train_folder, test_folder):
    train_files = numpy.array([])   
    labels = numpy.array([])
    test_files = numpy.array([])   
    test_labels = numpy.array([])
    for dirname, dirnames, filenames in os.walk(train_folder+"1/"): #get positive samples
        for fname in filenames:
            try:
                img = Image.open(train_folder+"1/"+fname)
                img = img.resize((80,45))
                img.save(train_folder+"1/"+fname)
                img = cv2.imread(train_folder+"1/"+fname, 0)
                train_files = numpy.append(train_files, img)
                labels = numpy.append(labels, 1)
            except IOError:
                print("error occured") 
    for dirname, dirnames, filenames in os.walk(train_folder+"0/"): #get negative samples
        for fname in filenames:
            try:
                img = Image.open(train_folder+"0/"+fname)
                img = img.resize((80,45))
                img.save(train_folder+"0/"+fname)
                img = cv2.imread(train_folder+"0/"+fname, 0)
                train_files = numpy.append(train_files, img)
                labels = numpy.append(labels, 0)
            except IOError:
                print("error occured")
    for dirname, dirnames, filenames in os.walk(test_folder+"1/"): 
        for fname in filenames:
            try:
                img = Image.open(test_folder+"1/"+fname)
                img = img.resize((80,45))
                img.save(test_folder+"1/"+fname)
                img = cv2.imread(test_folder+"1/"+fname, 0)
                test_files = numpy.append(test_files, img)
                test_labels = numpy.append(test_labels, 1) 
            except IOError:
                print("error occured") 
    for dirname, dirnames, filenames in os.walk(test_folder+"0/"): 
        for fname in filenames:
            try:       
                img = Image.open(test_folder+"0/"+fname)
                img = img.resize((80,45))
                img.save(test_folder+"0/"+fname)
                img = cv2.imread(test_folder+"0/"+fname, 0)
                test_files = numpy.append(test_files, img)
                test_labels = numpy.append(test_labels, 0)   
            except IOError:
                print("error occured")
    train_files.shape = (-1, 45, 80)
    test_files.shape = (-1, 45, 80)
    return (train_files, labels), (test_files, test_labels)

(x_train, y_train), (x_test, y_test) = get_training_data(sys.argv[1], sys.argv[2])
    
x_train = x_train.reshape(x_train.shape[0], img_x, img_y, 1)
x_test = x_test.reshape(x_test.shape[0], img_x, img_y, 1)
input_shape = (img_x, img_y, 1)

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255
print('x_train shape:', x_train.shape)

y_train = keras.utils.to_categorical(y_train, num_classes)
print(y_train)
exit()
y_test = keras.utils.to_categorical(y_test, num_classes)

model = Sequential()
model.add(Conv2D(32, kernel_size=(5, 5), strides=(1, 1),
                 activation='relu',
                 input_shape=input_shape))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
model.add(Conv2D(64, (5, 5), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(1000, activation='relu'))
model.add(Dense(num_classes, activation='softmax'))

model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adam(),
              metrics=['accuracy'])

class AccuracyHistory(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.acc = []

    def on_epoch_end(self, batch, logs={}):
        self.acc.append(logs.get('acc'))

history = AccuracyHistory()

model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          verbose=1,
          validation_data=(x_test, y_test),
          callbacks=[history])
         
score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])
model.save("./model.h5")
plt.plot(range(1, 11), history.acc)
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.show()