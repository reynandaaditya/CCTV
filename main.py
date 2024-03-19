import cv2
import numpy as np
import requests
import torch
import os

# Create 'Detected_Photos' directory if it doesn't exist
if not os.path.exists('Detected_Photos'):
    os.makedirs('Detected_Photos')

# Loading the model
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

cap = cv2.VideoCapture(1)

target_classes = ['car', 'bus', 'truck', 'person']

count = 0
number_of_photos = 3
pts = []  # Polygon points

# Function to send image via Telegram bot
def send_image_via_telegram(image_path, bot_token, chat_id):
    url = f"https://api.telegram.org/bot{bot_token}/sendPhoto"
    files = {'photo': open(image_path, 'rb')}
    data = {'chat_id': chat_id}
    requests.post(url, files=files, data=data)

# Function to draw polygon (roi)
def draw_polygon(event, x, y, flags, param):
    global pts
    if event == cv2.EVENT_LBUTTONDOWN:
        pts.append([x, y])
    elif event == cv2.EVENT_RBUTTONDOWN:
        pts = []

# Function to check if a point is inside a polygon
def inside_polygon(point, polygon):
    try:
        result = cv2.pointPolygonTest(np.array(polygon, dtype=np.int32), tuple(point), False)
        return result >= 0
    except Exception as e:
        print(f"Error in inside_polygon: {e}")
        return False

cv2.namedWindow('Video')
cv2.setMouseCallback('Video', draw_polygon)

def preprocess(img):
    height, width = img.shape[:2]
    ratio = height / width
    img = cv2.resize(img, (640, int(640 * ratio)))
    return img

while True:
    ret, frame = cap.read()
    frame_detected = frame.copy()
    frame = preprocess(frame)
    results = model(frame)

    for index, row in results.pandas().xyxy[0].iterrows():
        center_x = None
        center_y = None

        if row['name'] in target_classes:
            name = str(row['name'])
            x1, y1, x2, y2 = int(row['xmin']), int(row['ymin']), int(row['xmax']), int(row['ymax'])
            center_x = int((x1 + x2) / 2)
            center_y = int((y1 + y2) / 2)

            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 0), 3)
            cv2.putText(frame, name, (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
            cv2.circle(frame, (center_x, center_y), 5, (0, 0, 255), -1)

            if count < number_of_photos and inside_polygon((center_x, center_y), np.array(pts)) and name == 'person':
                cv2.imwrite(f"Detected_Photos/detected{count}.jpg", frame_detected)
                send_image_via_telegram(f"Detected_Photos/detected{count}.jpg", '6407086971:AAFBBkVA-y5i4kJTtiTfcFLXKKC_aQ0pNjE', '1006642950')
                count += 1

    for index, row in results.pandas().xyxy[0].iterrows():
        center_x = None
        center_y = None

        if row['name'] in target_classes:
            name = str(row['name'])
            x1, y1, x2, y2 = int(row['xmin']), int(row['ymin']), int(row['xmax']), int(row['ymax'])
            center_x = int((x1 + x2) / 2)
            center_y = int((y1 + y2) / 2)

            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 0), 3)
            cv2.putText(frame, name, (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
            cv2.circle(frame, (center_x, center_y), 5, (0, 0, 255), -1)

        if len(pts) >= 4:
            frame_copy = frame.copy()
            cv2.fillPoly(frame_copy, np.array([pts]), (0, 255, 0))
            frame = cv2.addWeighted(frame_copy, 0.1, frame, 0.9, 0)
            if center_x is not None and center_y is not None:
                if inside_polygon((center_x, center_y), np.array([pts])) and name == 'person':
                    mask = np.zeros_like(frame_detected)
                    points = np.array([[x1, y1], [x1, y2], [x2, y2], [x2, y1]])
                    points = points.reshape((-1, 1, 2))
                    mask = cv2.fillPoly(mask, [points], (255, 255, 255))
                    frame_detected = cv2.bitwise_and(frame_detected, mask)
                    if count < number_of_photos:
                        cv2.imwrite(f"Detected_Photos/detected{count}.jpg", frame_detected)

    cv2.imshow("Video", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
