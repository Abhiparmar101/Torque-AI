from flask import Flask, Response
import cv2
import threading
from ultralytics import YOLO

app = Flask(__name__)
outputFrame = None
lock = threading.Lock()

# Load YOLOv8 model
model = YOLO("yolov8n.pt")

# List of class names (ensure this matches the model's labels)
classNames = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat",
              "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
              "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
              "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
              "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
              "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
              "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant", "bed",
              "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone",
              "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
              "teddy bear", "hair drier", "toothbrush"
              ]

def detect_objects():
    global outputFrame, lock

    # Connect to the RTMP stream
    cap = cv2.VideoCapture("rtmp://media1.ambicam.com:1938/dvr30/2d70f9e8-af4b-46bd-a711-f87e449d5624")
    cap.set(3, 640)
    cap.set(4, 480)

    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        # Object detection
        results = model(frame, stream=True)

        # Process results and draw bounding boxes and labels
        for r in results:
            boxes = r.boxes
            for box in boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 255), 3)

                # Get class index
                cls_idx = int(box.cls[0])
                label = classNames[cls_idx] if cls_idx < len(classNames) else 'Unknown'

                # Draw label
                cv2.putText(frame, f"{label} {box.conf[0]:.2f}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

        with lock:
            outputFrame = frame.copy()

def generate():
    global outputFrame, lock
    while True:
        with lock:
            if outputFrame is None:
                continue
            (flag, encodedImage) = cv2.imencode(".jpg", outputFrame)
            if not flag:
                continue

        yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + bytearray(encodedImage) + b'\r\n')

@app.route('/video_feed')
def video_feed():
    return Response(generate(), mimetype="multipart/x-mixed-replace; boundary=frame")

if __name__ == '__main__':
    t = threading.Thread(target=detect_objects)
    t.daemon = True
    t.start()
    app.run(host='0.0.0.0', port=5000, debug=True, threaded=True, use_reloader=False)
