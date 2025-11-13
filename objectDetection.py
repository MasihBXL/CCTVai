from ultralytics import YOLO
import cv2

# Use a valid model name or path (change if you have a different .pt)
model = YOLO("yolov8n.pt")

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise RuntimeError("Cannot open webcam (device 0)")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame)[0]            # run inference on the frame
    annotated = results.plot()          # annotated BGR numpy image
    cv2.imshow("webcam", annotated)
    intery = cv2.waitKey(1)
 
    if cv2.waitKey(1) & 0xFF == 27:     # press ESC to quit
        break

cap.release()
cv2.destroyAllWindows()

