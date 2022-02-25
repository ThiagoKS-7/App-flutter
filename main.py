# -*- coding: utf-8 -*-
"""
Created on Sun Mar 14 12:53:39 2021

@author: Thiago Kasper de Souza
"""

def main():
    # VOZ E OUVIDOS DA IA
    import speech_recognition as sr
    from app.classification import AiOne
    from app.yolo3_img import AiTwo
    import pyttsx3

    engine= pyttsx3.init()
    engine.setProperty('voice','pt')

    model = 'AUG_K_TUNED-CNN2.model'
    class_names = 'coco.names'
    weights = 'yolov3.weights'
    cfg = 'yolov3.cfg'
    text = ''

    Ai_1 = AiOne('Ai_1', model)
    Ai_2 = AiTwo('Ai_2', weights, class_names, cfg)

    r = sr.Recognizer()
    mic = sr.Microphone()

    with mic as fonte:
        r.adjust_for_ambient_noise(fonte)
        engine.say("Saudações, aguardo seu comando")
        engine.runAndWait()
        audio= r.listen(fonte)
        print("um momento...")
        try:
            text= r.recognize_google(audio, language="pt-BR")
            print("Você disse {}".format(text))
        except:
            print("não entendi .-.")

        #FRASES QUE ELA ENTENDE:
    if text =="que roupa é essa":
        Ai_1.predict()
    elif text == "O que é isso" or text == "que é isso" or text == "o que é isso":
        Ai_2.predict()
    else:
        engine.say("Desculpe, não entendi")
        engine.runAndWait()
    engine.stop()

if __name__ == "__main__":
    main()