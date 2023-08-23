# -*- coding: utf-8 -*-
"""
Created on Sat Feb 26 15:03:08 2022

@author: Jesus-Mtz
"""

import cv2
import numpy as np
from datetime import datetime

# crea un objeto clasificador que usa el archivo haarcascade_fullbody.xml para detectar personas en las imágenes.
clasificador = cv2.CascadeClassifier("haarcascade_fullbody.xml")
# crea un objeto cap que abre el vídeo personas.avi para procesarlo.
cap = cv2.VideoCapture("personas.avi")

cuenta = 0
# while que se ejecuta mientras el vídeo esté abierto.
while(cap.isOpened()):
    # lee un fotograma del vídeo y lo guarda en la variable frame.
    # guarda un valor booleano en la variable ret que indica si la lectura fue exitosa
    ret, frame = cap.read()

    #comprobamos si ret es verdadero
    if ret == True:
        # comprueba que es el primer fotograma del vídeo.
        if cuenta == 0:
            # selecciona una región de interés (ROI) en el fotograma. Esta región será la zona marcada donde se quiere detectar personas.
            roi = cv2.selectROI(frame)
            x1 = roi[0]
            y1 = roi[1]
            w1 = roi[2]
            h1 = roi[3]

        if cuenta >= 1:
            # convierte el fotograma a escala de grises y lo guarda en la variable gray.
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            # hace una copia del fotograma y la guarda en la variable img2. Esta copia se usará para recortar las imágenes de las personas detectadas.
            img2=frame.copy()
            '''el método detectMultiScale del objeto clasificador para encontrar las posibles personas en la imagen en escala de grises.
             El método devuelve una lista de tuplas con las coordenadas y el tamaño de cada persona, que se guarda en la variable personas. 
             El método también recibe algunos parámetros que ajustan la precisión y el rendimiento de la detección, como el factor de escala,
              el número mínimo de vecinos, el tamaño mínimo y máximo de las personas.'''
            personas = clasificador.detectMultiScale(gray, scaleFactor=1.2,
                                                     minNeighbors=1,
                                                     minSize=(35,35),
                                                     maxSize=(150,150))
            # manejo posibles errores durante la detección y el procesamiento de las personas.
            try:
                # recorre cada tupla en la lista personas. Cada tupla contiene las coordenadas y el tamaño de una persona, que se guardan en las variables x, y, w, h.
                for (x,y,w,h) in personas:
                    # calcular el centro de la persona usando las coordenadas y el tamaño, y lo guarda en las variables cx, cy.
                    cx = int((x+(w/2)))
                    cy = int((y+(h/2)))
                    # Se crea un array de NumPy con las coordenadas de los vértices de la región de interés, que se guardan en la variable box.
                    box = np.array([[x1,y1], [x1+w1,y1],[x1+w1,y1+h1],
                                    [x1,y1+h1]])
                    # El array se convierte a enteros usando el método np.int0.
                    box = np.int0(box)
                    #  Se crea una lista con el array box, que se guarda en la variable contornos.
                    #  Esta lista se usará para dibujar el contorno de la región de interés en el fotograma.
                    contornos = [box]
                    # Se usa la función pointPolygonTest para comprobar si el centro de la persona está dentro o fuera de la región de interés.
                    # La función devuelve un valor numérico que indica la posición relativa del punto y el polígono, que se guarda en la variable res.
                    # Si el valor es 1, el punto está dentro del polígono. Si es 0, está en el borde. Si es -1, está fuera.
                    res = cv2.pointPolygonTest(box, (cx,cy), False)
                    # Se usa el método drawContours para dibujar el contorno de la región de interés en el fotograma.
                    # El método recibe como argumentos la imagen donde dibujar, la lista de contornos,
                    # el índice del contorno (-1 significa todos), el color (azul) y el grosor (5 píxeles).
                    cv2.drawContours(frame, contornos, -1,(255,0,0),5)

                    # Comprueba si res es 1, es decir, si el centro de la persona está dentro de la región de interés.
                    if res == 1:
                        # usa el método rectangle para dibujar un rectángulo alrededor de la persona en el fotograma.
                        # El método recibe como argumentos la imagen donde dibujar, las coordenadas del vértice superior izquierdo y el inferior derecho del rectángulo,
                        # el color (rojo) y el grosor (2 píxeles).
                        cv2.rectangle(frame,(x,y), (x+w,y+h), (0,0,255),2)
                        # Recorta la imagen de la persona usando las coordenadas y el tamaño, y la guarda en la variable croppedImage.
                        croppedImage = img2[y:y+h, x:x+w]
                        #  Obtiene la fecha y hora actual usando la función datetime.now, y la guarda en la variable date.
                        date = datetime.now()
                        # Se crea una cadena de texto con la ruta y el nombre del archivo donde se guardará la imagen recortada,
                        # usando la variable date para generar un nombre único. La cadena se guarda en la variable nombreArchivo.
                        nombreArchivo = ("./capturas/"+str(date.hour)
                                         +str(date.minute)+str(date.second)
                                         +".jpg")
                        # Se usa el método imwrite para guardar la imagen recortada en el archivo especificado por la variable nombreArchivo.
                        cv2.imwrite(nombreArchivo, croppedImage)

                # Se crean la ventanas para ver el video y otro para los recortes
                cv2.namedWindow("Frame", cv2.WINDOW_NORMAL)
                cv2.namedWindow("Recorte", cv2.WINDOW_NORMAL)
                cv2.imshow("Frame", frame)
                cv2.imshow("Recorte", croppedImage)

                # Se usa el método waitKey para esperar 25 milisegundos a que el usuario presione una tecla. Si la tecla es “q”,
                # se rompe el bucle while y se termina el programa.
                if cv2.waitKey(25) & 0xFF == ord('q'):
                    break
            except:
                pass
        # incrementa la variable cuenta en uno, para indicar que se ha procesado un fotograma más
        cuenta += 1
    # no se pudo leer un fotograma o que se llegó al final del vídeo.
    else:
        break

cap.release()
cv2.destroyAllWindows()