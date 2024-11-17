import cv2
import numpy as np


def FindObjects(outputs, img):
    '''Function to Detect objects'''
    height, width, channels = img.shape
    class_IDs = []
    confidences = []
    boxes = []
    for output in outputs:
        for detection in output:
            scores = detection[5:]
            class_ID = np.argmax(scores)
            confidence = scores[class_ID]

            if confidence > 0.5:
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)

                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_IDs.append(class_ID)

    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.3)
    font = cv2.FONT_HERSHEY_SIMPLEX

    for i in range(len(boxes)):
        if i in indexes:
            x, y, w, h = boxes[i]
            label = f'{classNames[class_IDs[i]].capitalize()} {int(confidences[i]*100)} %'
            color = (0, 0, 0)
            cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
            cv2.putText(img, label, (x, y - 15), font, 0.5, color, 1)


# Loading YOLO
net = cv2.dnn.readNetFromDarknet('yolov3-416.cfg', 'yolov3-416.weights')
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

# Loading coco.names in list
classNames = []

with open('coco.names', 'r') as f:
    classNames = [line.strip() for line in f.readlines()]

# Capturing Video
cap = cv2.VideoCapture(0)

while True:
    success, img = cap.read()

    blob = cv2.dnn.blobFromImage(
        img, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)

    layer_names = net.getLayerNames()
    output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]
    outputs = net.forward(output_layers)

    FindObjects(outputs, img)

    cv2.imshow("image", img)
    img = cv2.resize(img, (416, 416))

    if cv2.waitKey(1) & 0xFF == ord('q'):
        cv2.destroyAllWindows()
        break
