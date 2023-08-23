import cv2

cam = cv2.VideoCapture(0, cv2.CAP_DSHOW)
# Lee la red almacenada en un archivo coffe
net = cv2.dnn.readNetFromCaffe("MobileNetSSD_deploy.prototxt", "MobileNetSSD_deploy.caffemodel")

clases = {
    0:"Fondo", 1:"Avion", 2:"Bicicleta", 3:"Ave", 4:"Bote", 5:"Botella", 6:"Autobus", 7:"Auto", 8:"Gato",
    9:"Silla", 10:"Vaca", 11:"Masa", 12:"Perro", 13:"Caballo", 14:"Motocicleta", 15:"Persona", 16:"Planta en una maceta",
    17:"Oveja", 18:"Sofa", 19:"Trean", 20:"Pantalla"
}
cv2.namedWindow("deteccion", cv2.WINDOW_NORMAL)
conteos = {clase: 0 for clase in clases}

while True:
    res, frame = cam.read()
    # Redimenciona la imagen en 300x300 que acepta el modelo
    fRedimensionado = cv2.resize(frame, (300,300))
    # defino la funcion blodFromImage, recibe la imagen redimencionada, un factor de escala, ademas recibe el tamaño espacial
    # de la imagen de salida y el valor medio de los canales de color. Ademas un valor booleano donde si es falso decimos que los
    # canales estan ordenados de un modo bge. Retorna imagen tipo binariLargeObject
    blob = cv2.dnn.blobFromImage(fRedimensionado, 0.007843, (300,300), (127.5,127.5,127.5), False)
    # Pasamos el blob para que empiece a hacer detecciones
    net.setInput(blob)
    # Guaedo las detecciones
    detecciones = net.forward()
    # Obtener las dimensiones de la imagen redimensionada
    h, w = fRedimensionado.shape[:2]

    # Iteramos por cada prediccion
    for i in range(detecciones.shape[2]):
        # Extraemos los porcentajes de cada deteccion
        confidence = detecciones[0,0,i,2]
        idClase = int(detecciones[0,0,i,1])
        if confidence > 0.75 and idClase != 20:
            conteos[idClase] += 1
            xSup = int(detecciones[0,0,i,3] * h)
            ySup = int(detecciones[0,0,i,4] * w)
            xInf = int(detecciones[0,0,i,5] * h)
            yInf = int(detecciones[0,0,i,6] * w)

            hScaleFactor = frame.shape[0]/300.0
            wScaleFactor = frame.shape[1]/300.0
            xInf = int(wScaleFactor * xInf)
            yInf = int(hScaleFactor * yInf)
            xSup = int(wScaleFactor * xSup)
            ySup = int(hScaleFactor * ySup)

            cv2.rectangle(frame, (xSup, ySup), (xInf, yInf), (0,255,0))
            etiqueta = clases[idClase]+ ":" + str(confidence)
            cv2.putText(frame, etiqueta, (xSup, ySup), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,255),2)

            # Mostrar el número de personas detectadas
            texto = "Personas: {}".format(conteos[15])
            cv2.putText(frame, texto, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

    cv2.imshow("deteccion", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cam.release()
cv2.destroyAllWindows()