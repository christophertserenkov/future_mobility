import cv2
import threading
from time import sleep, monotonic
from flask import Flask, Response
from picamera2 import Picamera2
from scipy.signal import find_peaks
from ultralytics import YOLO
from hardware import HardWare

# Degub mode disables motors and enables cv image show
DEBUG_MODE = False
MOTOR_ENABLE = True
MOTOR_BASE_SPEED = 100
MOTOR_CROSWALK_SPEED = int(MOTOR_BASE_SPEED / 2)

motor_speed = 0

# Yolo model
model = YOLO("best.onnx")

# Estimated track width
APPROX_TRACK_WIDTH = 500

# PID Cooficents
KP = 0.8
KI = 0.1
KD = 0

# Hardware pins
SERVO = 18
MOTOR_IN1 = 20
MOTOR_IN2 = 21
HALL_PIN = 22

# Y coordinate to cut off
CROP_Y_COORDINATE = 520

# Y coordinate of track detection line
TRACK_LINE_1 = 10

# Flask streaming configuration
FLASK_HOST = "0.0.0.0"
FLASK_PORT = 5000
JPEG_QUALITY = 70

# PID memory variables
previous_error = 0
integral_sum = 0

# PID Time variable
last_PID_time = monotonic()

# No lines variables
tries = 0

# Flask shared frames
app = Flask(__name__)

# Variabled for steaming frame and image detection frame
latest_frame = None
display_frame = None

# Lock frame so no conflicts between read and write
frame_lock = threading.Lock()

# Stop the capture thread
stop_event = threading.Event()

# Init hardware
hw = HardWare(MOTOR_IN1, MOTOR_IN2, SERVO, HALL_PIN)

# Camera config
camera = Picamera2()
config = camera.create_preview_configuration(
    raw={"size": (1920, 1080)},
    main={"size": (1440, 720)},
)
config["main"]["format"] = "RGB888"
camera.configure(config)


# Stream the newest annotated frame using MJPEG
def generate_frames():
    while not stop_event.is_set():
        with frame_lock:
            frame = display_frame.copy() if display_frame is not None else None

        if frame is None:
            sleep(0.01)
            continue

        success, buffer = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])

        if not success:
            continue

        yield (
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n\r\n"
            + buffer.tobytes()
            + b"\r\n"
        )

# Publish frame to streaming
def publishFrame(frame):
    global display_frame

    with frame_lock:
        display_frame = frame.copy()

# Flask server main page
@app.route("/")
def index():
    return """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Robot Camera</title>
        <meta name="viewport" content="width=device-width, initial-scale=1">
        <style>
            body {
                background: #181818;
                color: white;
                font-family: sans-serif;
                text-align: center;
                margin: 20px;
            }
            img {
                width: 100%;
                max-width: 1200px;
                height: auto;
                border: 1px solid #555;
            }
        </style>
    </head>
    <body>
        <h2>Robot Camera Feed</h2>
        <img src="/video_feed" alt="Camera stream">
    </body>
    </html>
    """

# Flask server video backend thingy
@app.route("/video_feed")
def video_feed():
    return Response(generate_frames(), mimetype="multipart/x-mixed-replace; boundary=frame")

# Run flask 
def run_flask():
    app.run(host=FLASK_HOST, port=FLASK_PORT, threaded=True, debug=False, use_reloader=False)

def captureFrames():
    global latest_frame

    while not stop_event.is_set():
        frame = camera.capture_array()

        with frame_lock:
            latest_frame = frame


def getFrame():
    with frame_lock:
        if latest_frame is None:
            return None

        return latest_frame.copy()

def signDetection():
    global motor_speed;

    while True:
        # Get frame 
        frame = getFrame()

        # Run model
        results = model.predict(frame, imgsz=(160,320), conf=0.65)[0]

        # Detect crosswalk
        if len(results.boxes) > 0:
            detected_signs = [results.names[int(box.cls[0])] for box in results.boxes]
            if 'Crosswalk' in detected_signs:
                motor_speed = MOTOR_CROSWALK_SPEED
                sleep(2)
            else:
                motor_speed = MOTOR_BASE_SPEED
            print(f"Detected signs: {detected_signs}")
        else:
            motor_speed = MOTOR_BASE_SPEED
        
        print(f"Motor Speed: {motor_speed}")


try:
    camera.start()

    # Set turn angle to 0
    hw.setServo(0)

    # Start Flask in a background thread
    server_thread = threading.Thread(target=run_flask, daemon=True)
    server_thread.start()

    # Start camera capture thread
    capture_thread = threading.Thread(target=captureFrames, daemon=True)
    capture_thread.start()

    # Start sign detection thread 
    sign_detection_thread = threading.Thread(target=signDetection, daemon=True)
    sign_detection_thread.start()

    print(f"Camera server starting on port {FLASK_PORT}")
    print("Open http://<raspberry-pi-ip>:5000")

    # Main loop
    while True:
        # Get newest camera frame
        frame = getFrame()

        if frame is None:
            sleep(0.01)
            continue

        # Crop image
        h, w, _ = frame.shape
        frame = frame[CROP_Y_COORDINATE:CROP_Y_COORDINATE + (h - CROP_Y_COORDINATE)]

        # Lower resolution
        roi_h, roi_w, _ = frame.shape
        frame = cv2.resize(frame, (roi_w // 2, roi_h // 2))

        # Convert to greyscale (Picamera2 RGB888 output)
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)

        # Reset per-frame detection results
        midpoint = None
        line_midpoints = []

        # Find lines and midpoint
        row = gray[TRACK_LINE_1, :]
        _, props = find_peaks(row, width=[10, 100], prominence=100, distance=100)
        line_midpoints = (props["left_ips"] + props["right_ips"]) / 2

        # Define camera center
        camera_center = gray.shape[1] / 2

        if len(line_midpoints) == 0:
            tries += 1

            # Stop after prolonged loss of the track
            if tries > 100 and MOTOR_ENABLE:
                hw.setMotor(0)
        else:
            # Reset tries
            tries = 0

            # Enable motors
            if MOTOR_ENABLE:
                hw.setMotor(motor_speed)

            # State machine
            if len(line_midpoints) == 2:
                # Get midpoint
                left = line_midpoints[0]
                right = line_midpoints[1]
                if left < camera_center and right > camera_center:
                    midpoint = (left + right) / 2
                else:
                    mid_of_incorrect_points = (left + right) / 2 

                    if mid_of_incorrect_points < camera_center:
                        midpoint = mid_of_incorrect_points + APPROX_TRACK_WIDTH
                    else:
                        midpoint = mid_of_incorrect_points - APPROX_TRACK_WIDTH

            elif len(line_midpoints) > 2:
                # Store best point pairs
                best_pair = None

                # Store best line
                best_difference = float("inf")

                for i in range(len(line_midpoints) - 1):
                    left_candidate = line_midpoints[i]
                    right_candidate = line_midpoints[i + 1]

                    # Get distance between two points
                    width = right_candidate - left_candidate

                    # Difference from predicted width
                    difference = abs(APPROX_TRACK_WIDTH - width)

                    # Update best if current diff is smaller than best
                    if difference < best_difference:
                        best_difference = difference
                        best_pair = (left_candidate, right_candidate)

                if best_pair is not None:
                    midpoint = (best_pair[0] + best_pair[1]) / 2

            elif len(line_midpoints) == 1:
                # Estimate the missing boundary using the previous steering error
                previous_midpoint = camera_center + previous_error

                candidate_left = line_midpoints[0] - APPROX_TRACK_WIDTH / 2
                candidate_right = line_midpoints[0] + APPROX_TRACK_WIDTH / 2

                if abs(candidate_left - previous_midpoint) < abs(candidate_right - previous_midpoint):
                    midpoint = candidate_left
                else:
                    midpoint = candidate_right

            if midpoint is not None:
                # Calculate error
                error = int(midpoint - camera_center)

                # Calculate dt for PID and set last time to now
                now = monotonic()
                dt = now - last_PID_time
                last_PID_time = now

                if dt > 0:
                    # Calculate integral
                    if abs(error) <= 20:
                        integral_sum = 0
                    else:
                        integral_sum += error * dt

                    # Calculate derivative
                    derivative = (error - previous_error) / dt

                    # PID calculation
                    correction = KP * error + KI * integral_sum + KD * derivative
                    correction = max(-100, min(100, correction))

                    previous_error = error

                    hw.setServo(correction * 2)

                    # print(f"Error: {error}, Correction: {correction:.2f}")

        # Annotate frame for stream and debug
        display_frame = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        height, width = gray.shape

        # Camera center: white vertical line
        cv2.line(display_frame, (int(camera_center), 0), (int(camera_center), height - 1), (255, 255, 255), thickness=2)

        # Detection row: green horizontal line
        cv2.line(display_frame, (0, TRACK_LINE_1), (width - 1, TRACK_LINE_1), (0, 255, 0), thickness=2)

        # Detected boundary centers: blue dots
        for point in line_midpoints:
            cv2.circle(display_frame, (int(point), TRACK_LINE_1), 5, (255, 0, 0), thickness=-1)

        # Estimated track center: red dot
        if midpoint is not None:
            cv2.circle(display_frame, (int(midpoint), TRACK_LINE_1), 6, (0, 0, 255), thickness=-1)

        # Display status on the image
        status = "TRACK LOST" if len(line_midpoints) == 0 else "TRACK DETECTED"
        status_color = (0, 0, 255) if midpoint is None else (0, 255, 0)
        cv2.putText(display_frame, status, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, status_color, 2)

        # Publish frame to Flask
        publishFrame(display_frame)

        if DEBUG_MODE:
            # Show image
            cv2.imshow("Robot Camera", display_frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

except KeyboardInterrupt:
    print("Stopping robot...")

finally:
    # Signal threads to stop
    stop_event.set()

    # Stop motors
    try:
        hw.setMotor(0)
        hw.setServo(0)
        hw.stop()
    except Exception as exc:
        print(f"Hardware shutdown error: {exc}")

    # Stop camera and release resources
    try:
        camera.stop()
    except Exception:
        pass

    # Wait for camera thread
    if 'capture_thread' in locals() and capture_thread.is_alive():
        capture_thread.join(timeout=2)

    try:
        camera.close()
    except Exception:
        pass

    cv2.destroyAllWindows()