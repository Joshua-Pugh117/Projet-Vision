import json
import math
import cv2
from ultralytics import YOLO
import numpy as np
import pandas as pd
from PIL import Image


with open('ref.json', 'r') as json_file:
    reference_features_dict = json.load(json_file)


model = YOLO('yolov8n-face.pt')

# Variables Face detection
scalefactor = 1 / 255.0
size = (96, 96)
mean = (0, 0, 0)
swapRB = True
net1 = cv2.dnn.readNetFromTorch('openface.nn4.small2.v1.t7')

# Variables Emotion detection
inpWidth = 64       
inpHeight = 64 
scale = 1.0     
avg = [127,127,127]
rgb = False  
classes = [ "neutral", "happiness", "surprise", "sadness", "anger", "disgust", "fear", "contempt" ]
net2 = cv2.dnn.readNet("emotion-ferplus-8.onnx")

# Variables Face alignment
eye_detector = cv2.CascadeClassifier("haarcascade_eye_tree_eyeglasses.xml")
nose_detector = cv2.CascadeClassifier("haarcascade_mcs_nose.xml")

def trignometry_for_distance(a, b):
	return math.sqrt(((b[0] - a[0]) * (b[0] - a[0])) +\
					((b[1] - a[1]) * (b[1] - a[1])))

def Face_Alignment(img_path):
    img_raw = img_path.copy()
    img = img_path.copy()
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    eyes = eye_detector.detectMultiScale(gray_img)

    # for multiple people in an image find the largest 
    # pair of eyes
    new_img = None
    if len(eyes) >= 2:
        eye = eyes[:, 2]
        container1 = []
        for i in range(0, len(eye)):
            container = (eye[i], i)
            container1.append(container)
        df = pd.DataFrame(container1, columns=[
                        "length", "idx"]).sort_values(by=['length'])
        eyes = eyes[df.idx.values[0:2]]

        # deciding to choose left and right eye
        eye_1 = eyes[0]
        eye_2 = eyes[1]
        if eye_1[0] > eye_2[0]:
            left_eye = eye_2
            right_eye = eye_1
        else:
            left_eye = eye_1
            right_eye = eye_2

        # center of eyes
        # center of right eye
        right_eye_center = (
            int(right_eye[0] + (right_eye[2]/2)), 
        int(right_eye[1] + (right_eye[3]/2)))
        right_eye_x = right_eye_center[0]
        right_eye_y = right_eye_center[1]
        cv2.circle(img, right_eye_center, 2, (255, 0, 0), 3)

        # center of left eye
        left_eye_center = (
            int(left_eye[0] + (left_eye[2] / 2)), 
        int(left_eye[1] + (left_eye[3] / 2)))
        left_eye_x = left_eye_center[0]
        left_eye_y = left_eye_center[1]
        cv2.circle(img, left_eye_center, 2, (255, 0, 0), 3)

        # finding rotation direction
        if left_eye_y > right_eye_y:
            print("Rotate image to clock direction")
            point_3rd = (right_eye_x, left_eye_y)
            direction = -1 # rotate image direction to clock
        else:
            print("Rotate to inverse clock direction")
            point_3rd = (left_eye_x, right_eye_y)
            direction = 1 # rotate inverse direction of clock

        cv2.circle(img, point_3rd, 2, (255, 0, 0), 2)
        a = trignometry_for_distance(left_eye_center, 
                                    point_3rd)
        b = trignometry_for_distance(right_eye_center, 
                                    point_3rd)
        c = trignometry_for_distance(right_eye_center, 
                                    left_eye_center)
        cos_a = (b*b + c*c - a*a)/(2*b*c)
        angle = (np.arccos(cos_a) * 180) / math.pi

        if direction == -1:
            angle = 90 - angle
        # else:
        #     angle = -(90-angle)
            # angle = angle

        # rotate image
        new_img = Image.fromarray(img_raw)
        new_img = np.array(new_img.rotate(direction * angle))

    if new_img is None:
        return img_raw
    return new_img

def run():
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

                    # recaler la tete
                    try:
                        face = Face_Alignment(face)
                    except:
                        pass

                    cv2.imshow("Face", face)

#-------------------------------------Detect prenom personne--------------------------------------------------------
                    blob = cv2.dnn.blobFromImage(face, scalefactor=scalefactor, size=size, mean=mean, swapRB=swapRB)
                    net1.setInput(blob)
                    out = net1.forward()
                    
                    new_face_features = out.flatten()
                    min_distance = float("inf")
                    closest_index = None
                    
                    for person in reference_features_dict.keys():
                        for ref_feature in reference_features_dict[person]:
                            distance = np.linalg.norm(ref_feature - new_face_features)
                            if distance < min_distance:
                                min_distance = distance
                                closest_index = person
                            
                    cv2.putText(frame, closest_index, (x2, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)

#-------------------------------------Emotions--------------------------------------------------------                    
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
                        cv2.putText(frame, classes[i] + ':', (x2, y1+10+20*(i+1)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
                        rlabel = '%+6.1f' % conf
                        cv2.putText(frame, rlabel, (x2+100, y1+10+20*(i+1)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
                        
            cv2.imshow("Frame", frame)             

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

if __name__ == "__main__":
    run()