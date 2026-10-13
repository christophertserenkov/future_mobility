import cv2
import threading
from picamera2 import Picamera2
from ultralytics import YOLO

model = YOLO("best.onnx")

camera = Picamera2()
config = camera.create_preview_configuration(
    raw={"size": (1920, 1080)},
    main={"size": (1440, 720)},
)
config["main"]["format"] = "RGB888"
camera.configure(config)
camera.start()

latest_frame = camera.capture_array()

def captureFrames():
    global latest_frame
    while True:
        frame = camera.capture_array()
        latest_frame = frame

def getFrame():
    return latest_frame.copy()

capture_thread = threading.Thread(target=captureFrames, daemon=True)
capture_thread.start()


while True:
    frame = getFrame()
    results = model.predict(frame, imgsz=(160,320), conf=0.65)[0]
    if len(results.boxes) > 0:
        detected_signs = [results.names[int(box.cls[0])] for box in results.boxes]
        print(detected_signs)
    else:
        detected_signs = []
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
        
cv2.destroyAllWindows()
