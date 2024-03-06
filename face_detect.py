import json
import cv2
from ultralytics import YOLO
import numpy as np


with open('ref.json', 'r') as json_file:
    reference_features_dict = json.load(json_file)


model = YOLO('yolov8n-face.pt')


scalefactor = 1 / 255.0
size = (96, 96)
mean = (0, 0, 0)
swapRB = True
net1 = cv2.dnn.readNetFromTorch('openface.nn4.small2.v1.t7')


inpWidth = 64       
inpHeight = 64 
scale = 1.0     
avg = [127,127,127]
rgb = False  
classes = [ "neutral", "happiness", "surprise", "sadness", "anger", "disgust", "fear", "contempt" ]
net2 = cv2.dnn.readNet("emotion-ferplus-8.onnx")

def run():
    # global model, scalefactor, size, mean, swapRB, net1, net2, reference_features_dict, classes, 
    video_capture = cv2.VideoCapture(0)

    while True:
        ret, frame = video_capture.read()
        if ret:
            results = model(frame, stream=True)
            for result in results:
                boxes = result.boxes
                for box in boxes:
                    x1, y1, x2, y2 = box.xyxy[0].int().tolist()
                    face = frame[y1:y2, x1:x2]
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

                    blob = cv2.dnn.blobFromImage(face, scalefactor=scalefactor, size=size, mean=mean, swapRB=swapRB)
                    net1.setInput(blob)
                    out = net1.forward()
                    
                    new_face_features = out.flatten()
                    min_distance = float("inf")
                    closest_index = None

                    for person, ref_feature in reference_features_dict.items():
                        distance = np.linalg.norm(ref_feature - new_face_features)
                        if distance < min_distance:
                            min_distance = distance
                            closest_index = person
                    cv2.putText(frame, closest_index, (x2, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)

                    
                    gframe = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    blob = cv2.dnn.blobFromImage(gframe, scale, (inpWidth, inpHeight), avg, rgb, crop=True)
                    
                    net2.setInput(blob)
                    out = net2.forward()
                    out = out.flatten()

                    maxconf = 999
                    for i in range(8):
                        conf = out[i] * 100
                        if conf > maxconf: conf = maxconf
                        if conf < -maxconf: conf = -maxconf
                        cv2.putText(frame, classes[i] + ':', (x2, y1+10+20*(i+1)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (36, 255, 12), 2)
                        rlabel = '%+6.1f' % conf
                        cv2.putText(frame, rlabel, (x2+100, y1+10+20*(i+1)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (36, 255, 12), 2)
                        # cv2.line(msgbox, (mid, 11*i+6), (mid + int(conf*leng/maxconf), 11*i+6), white, 4)




            cv2.imshow("Frame", frame)             

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

if __name__ == "__main__":
    run()