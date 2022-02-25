# -*- coding: utf-8 -*-
"""
Created on Tue Aug 11 20:13:54 2020

@author: thiag
"""

import tensorflow as tf
import numpy as np
import cv2
from matplotlib import pyplot as plt


import pickle
#d = serial.Serial("COM4",9600)

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']


#FUNÇÕES
Y_test = pickle.load(open("testY.pickle", "rb"))


model = tf.keras.models.load_model("MNIST-CNN.model")
print("Testes de predição:")
print()
#roupa = str("C:/Users/thiag/Desktop/Projetos 2020 - Coisas da IA/Quarteto_imbatível/camisa teste.png")

captura = cv2.VideoCapture(0)
cont = 0

while True:
    ret, frame = captura.read()
    cont += 1
    if cont >= 25:
        break
    
new_image = cv2.imwrite("new_img.jpg", frame)
captura.release()
cv2.destroyAllWindows()
     
image = cv2.imread("new_img.jpg")
image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

#cv2.imshow("Image", image)

blurred = cv2.GaussianBlur(image, (5, 5), 0)


print("Blurring...Done.")


(T,threshInv) = cv2.threshold(blurred, 45, 255, cv2.THRESH_BINARY_INV)
resize = cv2.resize(threshInv,(28,28))
plt.imshow(resize,cmap = 'gray')
plt.show()
reshaped = resize.reshape(1,28,28,-1)

print("Testes predição:")
predicao = model.predict([reshaped])

print("Fig #1:")
    # colocar enderço da imagem dentro do "prepare()"
ctg = np.argmax(predicao.flatten())   
print(class_names[ctg])
print("Posição da predição encontrada:", ctg)

print("Done")
