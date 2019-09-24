#准确率：95.5%
#没有加正则化项
#不知道怎么用keras读取data_set,还是用的tensorflow
import keras
import numpy as np
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense,Activation,Dropout
from tensorflow.examples.tutorials.mnist import input_data


model = Sequential([
    Dense(500,input_shape= (784,)),
    Activation('relu'),
    Dropout(0.2),

    Dense(500),
    Activation('relu'),
    Dropout(0.2),

    Dense(10),
    Activation('softmax')
])

sgd = keras.optimizers.SGD(lr= 0.8,decay= 0.99,momentum=0.9,nesterov= True)
model.compile(optimizer= 'sgd',
                loss = 'categorical_crossentropy',
                metrics=['accuracy'],
                )

img_rows = 28
img_cols = 28
mnist = input_data.read_data_sets("E:\\text\\py\\DeepLearning\\MNIST_data",one_hot = True)
train_images = mnist.train.images
train_labels = mnist.train.labels
test_images = mnist.test.images
test_labels = mnist.test.labels

model.fit(train_images,train_labels,epochs =20,batch_size= 100,verbose=1, validation_split=0.05)

loss,accuracy = model.evaluate(test_images,test_labels)
print('loss',loss)
print('accuracy',accuracy)
