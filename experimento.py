import cv2

cam = cv2.VideoCapture(0, cv2.CAP_DSHOW)
net = cv2.dnn.readNetFromCaffe("MobileNetSSD_deploy.prototxt", "MobileNetSSD_deploy.caffemodel")

clases = {
    0: "Fondo", 1: "Avion", 2: "Bicicleta", 3: "Ave", 4: "Bote", 5: "Botella", 6: "Autobus", 7: "Auto", 8: "Gato",
    9: "Silla", 10: "Vaca", 11: "Masa", 12: "Perro", 13: "Caballo", 14: "Motocicleta", 15: "Persona",
    16: "Planta en una maceta",
    17: "Oveja", 18: "Sofa", 19: "Trean", 20: "Pantalla"
}

cv2.namedWindow("deteccion", cv2.WINDOW_NORMAL)
conteos = {clase: 0 for clase in clases}

# Lista para almacenar los trackers
trackers = []

while True:
    res, frame = cam.read()
    fRedimensionado = cv2.resize(frame, (300, 300))
    blob = cv2.dnn.blobFromImage(fRedimensionado, 0.007843, (300, 300), (127.5, 127.5, 127.5), False)
    net.setInput(blob)
    detecciones = net.forward()
    h, w = fRedimensionado.shape[:2]

    for i in range(detecciones.shape[2]):
        confidence = detecciones[0, 0, i, 2]
        idClase = int(detecciones[0, 0, i, 1])

        if confidence > 0.75 and idClase == 15:  # Solo personas
            xSup = int(detecciones[0, 0, i, 3] * w)
            ySup = int(detecciones[0, 0, i, 4] * h)
            xInf = int(detecciones[0, 0, i, 5] * w)
            yInf = int(detecciones[0, 0, i, 6] * h)

            hScaleFactor = frame.shape[0] / 300.0
            wScaleFactor = frame.shape[1] / 300.0
            xInf = int(wScaleFactor * xInf)
            yInf = int(hScaleFactor * yInf)
            xSup = int(wScaleFactor * xSup)
            ySup = int(hScaleFactor * ySup)

            # Inicializar el tracker con el rectángulo delimitador de la persona
            tracker = cv2.TrackerKCF_create()
            bbox = (xSup, ySup, xInf - xSup, yInf - ySup)
            tracker.init(frame, bbox)

            # Agregar el tracker a la lista
            trackers.append(tracker)

            # Contar a la persona
            conteos[idClase] += 1

    # Actualizar los trackers en cada frame
    for tracker in trackers:
        success, bbox = tracker.update(frame)
        if success:
            x, y, w, h = [int(v) for v in bbox]
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # Mostrar conteos y detecciones en la ventana
    for key, value in conteos.items():
        cv2.putText(frame, f"{clases[key]}: {value}", (10, 30 + 20 * key), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

    cv2.imshow("deteccion", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cam.release()
cv2.destroyAllWindows()
