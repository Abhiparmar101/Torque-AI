import cv2
import pandas as pd
import numpy as np
import os
import datetime
import requests
from ultralytics import YOLO

import cvzone

model = YOLO('yolov8s.pt')

cap = cv2.VideoCapture('rtmp://ptz.vmukti.com:80/live-record/SSAM-160711-BACBC')
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))

# Define the ROI coordinates
roi_x = 330  # X-coordinate of the top-left corner of the ROI
roi_y = 150  # Y-coordinate of the top-left corner of the ROI
roi_width = 200   # Width of the ROI
roi_height = 200  # Height of the ROI

my_file = open("coco.txt", "r")
data = my_file.read()
class_list = data.split("\n")

count = 0

# Clear the CSV file if it exists
if os.path.exists('people_count.csv'):
    os.remove('people_count.csv')

while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame = cv2.resize(frame, (640, 480))

    count += 1
    if count % 3 != 0:
        continue

    results = model.predict(frame)
    a = results[0].boxes.data
    a = a.cpu()  # Convert CUDA tensor to CPU tensor
    px = pd.DataFrame(a).astype("float")
    people_count = 0  # Reset people count for each frame
    for index, row in px.iterrows():
        x1 = int(a[index][0])
        y1 = int(a[index][1])
        x2 = int(a[index][2])
        y2 = int(a[index][3])
        d = int(a[index][5])
        c = class_list[d]
        if 'person' in c:
            # Calculate the center point of the detected human
            center_x = int((x1 + x2) / 2)
            center_y = int((y1 + y2) / 2)

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 1)
            cv2.putText(frame, str(c), (x1, y1), cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 255, 255), 1)

            # Check if the center point falls within the ROI
            if roi_x <= center_x <= roi_x + roi_width and roi_y <= center_y <= roi_y + roi_height:
                people_count += 1  # Increment people count for each person detected within the ROI

    # Draw ROI rectangle
    cv2.rectangle(frame, (roi_x, roi_y), (roi_x + roi_width, roi_y + roi_height), (0, 255, 0), 2)

    cv2.putText(frame, f"People Count: {people_count}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
    # resized_frame = cv2.resize(frame, (800, 600))

    cv2.imshow("Frame1", frame)

    if people_count >= 2:
        # Capture and save the image
        camera_id = "PQRS-230349-ABCDE"
        image_name = datetime.datetime.now().strftime("%Y-%m-%d-%H:%M:%S") + camera_id + ".jpg"
        image_path = "/home/torqueai/blobdrive/" + image_name
        cv2.imwrite(image_path, frame)
        print(f"Image captured: {image_name}")

        # Prepare data for CSV
        send_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        img_url = "https://inferenceimage.blob.core.windows.net/inferenceimages/" + image_name
        an_id = 2
        img_count = people_count

        # Save data to CSV
        data = {'cameradid': camera_id, 'sendtime': send_time, 'imgurl': img_url,
                'an_id': an_id, 'ImgCount': img_count}
        # df = pd.DataFrame(data)
        # df.to_csv('people_count.csv', mode='a', header=not os.path.exists('people_count.csv'), index=False)

        # Send data to URL
        api_url = "https://tn2023demo.vmukti.com/api/analytics"
        response = requests.post(api_url, json=data)
        if response.status_code == 200:
            print("Booth Data sent successfully!", img_url)
        else:
            print("Failed to send data!")

    if cv2.waitKey(1) & 0xFF == 27:
        # Clear the CSV file before exiting
        if os.path.exists('people_count.csv'):
            os.remove('people_count.csv')
        break

cap.release()
cv2.destroyAllWindows()