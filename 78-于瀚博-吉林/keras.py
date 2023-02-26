from tensorflow.keras.datasets import mnist
(train_images,train_labels),(test_images,test_labels)=mnist.load_data()
print('train_images.shape=',train_images.shape)
print('train_labels=',train_labels)
print('test_images.shape=',test_images.shape)
print('test_labels=',test_labels)
digit=test_images[0]
import matplotlib.pyplot as plt
plt.imshow(digit,cmap=plt.cm.binary)
plt.show()
from tensorflow.keras import models
from tensorflow.keras import layers
network=models.Sequential()
network.add(layers.Dense(512,activation='relu',input_shape=(28*28,)))
network.add(layers.Dense(10,activation='softmax'))
network.compile(optimizer='rmsprop',loss='categorical_crossentropy',metrics=['accuracy'])
train_images=train_images.reshape((60000,28*28))
train_images=train_images.astype('float32')/255
test_images=test_images.reshape((10000,28*28))
test_images=test_images.astype('float32')/255
from tensorflow.keras.utils import to_categorical
print('before change:',test_labels[0])
train_labels=to_categorical(train_labels)
test_labels=to_categorical(test_labels)
print('after change:',test_labels[0])
network.fit(train_images,train_labels,epochs=5,batch_size=128)
test_loss,test_acc=network.evaluate(test_images,test_labels,verbose=1)
print(test_loss)
print('test_acc',test_acc)

import numpy as np
(train_images,train_labels),(test_images,test_labels)=mnist.load_data()
img=test_images[2]
plt.imshow(img,cmap=plt.cm.binary)
plt.show()
img=img.reshape((784))
img=np.array(img)
img=np.expand_dims(img,axis=0)
img=img/255.0
pre=network.predict(img)
print(np.argmax(pre))