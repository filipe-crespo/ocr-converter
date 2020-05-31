import cv2
import numpy as np
from matplotlib import pyplot as plt

from keras.preprocessing.image import ImageDataGenerator
from keras.datasets import mnist
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers.convolutional import Conv2D, MaxPooling2D
data = mnist.load_data()

# Divide os dados em dois grupos, train e test
(x_train, y_train), (x_test, y_test) = data

print('Train shape', x_train.shape)
print('Test shape', x_test.shape)

x_train = x_train.reshape(x_train.shape[0], 28, 28, 1).astype('float32')
print(x_train.shape)

x_test = x_test.reshape(x_test.shape[0], 28, 28, 1).astype('float32')
print(x_test.shape)

print(x_train[-1])
x_train = x_train / 255
x_test = x_test / 255
print(x_train[-1])


# Tranformou o vetor em uma matriz que categoriza nossos dados
print(y_train)
y_train = np_utils.to_categorical(y_train)
print(y_train)


print(y_test)
y_test = np_utils.to_categorical(y_test)
print(y_test)

# Numero de opçoes disponivel
num_classes = y_test.shape[1]
print(num_classes)

# inicia o  modelo 
model = Sequential()
# 30 feature maps
# 5X5 kernel
# 28X28 forma px
# 1 PB
# Funcao nao linear
model.add(Conv2D(30,(5, 5),input_shape=(28, 28, 1), activation='relu'))

model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(30,(3, 3),input_shape=(28, 28, 1),activation='relu'))

model.add(MaxPooling2D(pool_size=(2, 2)))

# tecnica de regularicao, desligando neuronios
# probabilidade de 20% estar desligado
model.add(Dropout(0.2))

# converter em um vetfrom keras import backend as Kor de tamanho unico
model.add(Flatten())

model.add(Dense(128, activation='relu'))

model.add(Dense(64, activation='relu'))

model.add(Dense(32, activation='relu'))

model.add(Dense(num_classes, activation='softmax', name='predict'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

model.summary()

# epochs quantas vezes o dado de test sera exibido para o modelo
# batch_size envia de 200 em 200
model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=10, batch_size=2000)

scores = model.evaluate(x_test, y_test, verbose=0)
print("acc: %.2f%%" % (scores[1]*100))

img_pred = cv2.imread("./test/eight.jpg", 0)

plt.imshow(img_pred, cmap='gray')
img_pred = cv2.resize(img_pred, (28,28))

# if  img_pred.shape != (28, 28):
#   img2 = cv2.resize(img_pred, (28,28))
#   imp_pred = img2.reshape(28, 28, -1)
# else:
#   img_pred = img_pred.reshape(28, 28, -1)

img_pred = img_pred.reshape(1, 28, 28, 1)
print(img_pred.shape)

pred = model.predict_classes(img_pred)
pred_proba = model.predict_proba(img_pred)

pred_proba = "%.2f%%" % (pred_proba[0][pred]*100)

print(pred[0], "com a probabilidade de ", pred_proba)

pred = model.predict_classes(img_pred)
pred_proba = model.predict_proba(img_pred)

pred_proba = "%.2f%%" % (pred_proba[0][pred]*100)

print(pred[0], "com a probabilidade de ", pred_proba)