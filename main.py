import cv2
import numpy as np

# Carregue o modelo YOLO v3
net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")

# Carregue as classes
with open("coco.names", "r") as f:
    classes = f.read().strip().split('\n')

# Carregue a imagem que você deseja processar
image = cv2.imread("room.jpg")

# Redimensione a imagem para o tamanho esperado pelo modelo
blob = cv2.dnn.blobFromImage(image, 1/255.0, (416, 416), swapRB=True, crop=False)

# Defina as camadas de saída
layer_names = net.getUnconnectedOutLayersNames()

# Passe a imagem para a rede
net.setInput(blob)
outputs = net.forward(layer_names)

# Processando as saídas
boxes = []
confidences = []
class_ids = []

for output in outputs:
    for detection in output:
        scores = detection[5:]
        class_id = np.argmax(scores)
        confidence = scores[class_id]

        if confidence > 0.5:
            center_x = int(detection[0] * image.shape[1])
            center_y = int(detection[1] * image.shape[0])
            w = int(detection[2] * image.shape[1])
            h = int(detection[3] * image.shape[0])

            x = int(center_x - w / 2)
            y = int(center_y - h / 2)

            boxes.append([x, y, w, h])
            confidences.append(float(confidence))
            class_ids.append(class_id)

# Aplicando a supressão não máxima
indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

# Desenhe as caixas delimitadoras nas imagens
for i in range(len(boxes)):
    if i in indices:
        x, y, w, h = boxes[i]
        label = str(classes[class_ids[i]])
        confidence = confidences[i]
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(image, f"{label} {confidence:.2f}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

# Mostrar a imagem com as caixas delimitadoras
cv2.imwrite("output_room.jpg", image)
