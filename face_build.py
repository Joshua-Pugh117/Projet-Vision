import json
import cv2
from ultralytics import YOLO

model = YOLO('yolov8n-face.pt')
scalefactor = 1 / 255.0
size = (96, 96)
mean = (0, 0, 0)
swapRB = True
with open('ref.json', 'r') as json_file:
    reference_features_dict = json.load(json_file)


def run():
    global model, scalefactor, size, mean, swapRB
    video_capture = cv2.VideoCapture(0)
    cmt = 0

    refs = []
    while True:
        ret, frame = video_capture.read()
        if ret:
            results = model(frame, stream=True)
            for result in results:
                boxes = result.boxes
                if len(boxes) != 0:
                    x1, y1, x2, y2 = boxes[0].xyxy[0].int().tolist()
                    face = frame[y1:y2, x1:x2]
                    # face = cv2.resize(face, (200, 200))
                else:
                    face = frame

                net = cv2.dnn.readNetFromTorch('openface.nn4.small2.v1.t7')
                blob = cv2.dnn.blobFromImage(face, scalefactor=scalefactor, size=size, mean=mean, swapRB=swapRB)
                net.setInput(blob)
                out = net.forward()

                if cmt%5 == 0:
                        refs.append(out.flatten().tolist())

        cv2.imshow("Face", face)
        cmt += 1
        print(len(refs))

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    reference_features_dict["thimaute"] = refs
    with open('ref.json', 'w') as json_file:
        json.dump(reference_features_dict, json_file, indent=2)


if __name__ == "__main__":
    run()