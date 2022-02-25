# -*- coding: utf-8 -*-
"""
Created on Thu May 28 14:59:55 2020

@author: Thiago Kasper de Souza
"""


def serial_send(serial, ctg, color):
    # FUNÇÕES PARA RODAR EM CONJUNTO COM ARDUINO
    sum = (str(ctg) + color)
    print("Código concatenado: ", sum)
    # codifica_dados(int(sum))
    # serial.write(str(codifica_dados(int(sum))).encode())
    serial.close()
    return True


# FUNÇÃO QUE REDIMENSIONA A IMAGEM
def prepare(filepath):  # PREPARAR IMAGEM
    image_size = 40
    image = cv2.imread(filepath)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    resized_image = cv2.resize(image, (image_size, image_size))
    # plt.imshow(resized_image,cmap = 'gray')
    # plt.show()
    return resized_image.reshape(1, image_size, image_size, -1)


def AI_One():
    import cv2
    import tensorflow as tf
    from docs.color.color_analysis import color_distinct
    # import serial ** ARDUINO SUPPORT
    from matplotlib import pyplot as plt
    import numpy as np
    import imageio
    # serial = serial.Serial("COM13",9600) **ARDUINO SUPPORT
    import pyttsx3  # TTS = Text-To-Speech

    engine = pyttsx3.init()
    engine.setProperty('voice', 'pt')

    class_names = ["Bota", "Camisa", "Camiseta", "Tenis"]

    model = tf.keras.models.load_model("../models/AUG_K_TUNED-CNN2.model")
    # model = tf.keras.models.load_model("../models/AUG_K_TUNED-CNN3.model")
    print("Testes de predição:")
    print()

    # TESTES COM IMAGENS PRÉ-DEFINIDAS
    roupa = str("C:/Users/W10/Documents/camiseta (1).jpg")
    # roupa = str("C:/Users/thiag/Desktop/roupas/camisetas/8.jpg")
    # roupa = str("C:/Users/thiag/Desktop/roupas/tenis/2.png")
    # roupa = str("C:/Users/thiag/Desktop/roupas/botas/1.png")

    # CAPTURA DE IMAGEM COM WEBCAM (NO RASP VAI SER DIFERENTE)
    '''
    captura = cv2.VideoCapture(0)
    cont = 0

    while True:
        ret, frame = captura.read()
        cont += 1
        if cont >= 25:
            break

    cv2.imwrite("new_img.jpg", frame)
    captura.release()
    cv2.destroyAllWindows()

    roupa = "new_img.jpg"
'''
    # PREDIÇÃO DA IMAGEM COM CORREÇÃO DO TAMANHO
    prediction = model.predict([prepare(roupa)])
    print("Fig #1:")
    # colocar enderço da imagem dentro do "prepare()"
    ctg = np.argmax(prediction.flatten())
    print(class_names[ctg])

    # CONDIÇÃO PARA FALAR O ARTIGO CERTO: SE CTG
    # VALER ATÉ 3, ARTIGO É FEMININO, SE FOR MAIOR, É MASCULINO

    if ctg < 3 and ctg >= 0:
        engine.say("isso é uma" + class_names[ctg], )
    elif ctg == 3:
        engine.say("isso é um tênis")
    print("Posição da predição encontrada:", ctg)

    # CORREÇÃO DAS CORES DA IMAGEM, PRA APARECER NO PYPLOT
    # + BOUNDING BOX ESTÁTICA CENTRALIZADA (NÃO LOCALIZA NOS CANTOS DA IMG)

    img = imageio.imread(roupa)
    img = cv2.resize(img, (200, 200))
    obj_found = cv2.rectangle(img, (45, 30), (170, 190), (0, 255, 0), 2)
    img_predicted = cv2.putText(obj_found, "Label: {}".format(class_names[ctg]), (12, 25), cv2.FONT_HERSHEY_SIMPLEX,
                                0.58, (0, 254, 0), 2)
    plt.imshow(img_predicted)
    plt.show()
    print()

    # DISTINÇÃO DA COR E FALA DELA
    color = distingue_cor(roupa)
    engine.say("e sua cor é:" + color)
    engine.runAndWait()
    engine.stop()

    # serial_send()

    # DELETA TODAS AS VARIAVEIS PRA POUPAR ESPAÇO E OTIMIZAR O PROGRAMA
    # del class_names
    # del roupa
    del prediction
    del ctg, img, img_predicted
    del color, engine
    # del serial
    # del sum
    # del captura, ret,frame,cont


if __name__ == '__main__':
    AI_One()
