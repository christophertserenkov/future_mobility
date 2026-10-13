
import cv2
import threading
from time import sleep, monotonic

from flask import Flask, Response
from picamera2 import Picamera2
from scipy.signal import find_peaks
from ultralytics import YOLO

from hardware import HardWare


# =========================
# Configuration
# =========================

DEBUG_MODE = False
MOTOR_ENABLE = True
MOTOR_BASE_SPEED = 60

APPROX_TRACK_WIDTH = 580

KP = 1.2
KI = 2
KD = 0.5

SERVO = 18
MOTOR_IN1 = 20
MOTOR_IN2 = 21
HALL_PIN = 22

CROP_Y_COORDINATE = 520
TRACK_LINE_1 = 10

# Flask streaming configuration
FLASK_HOST = "0.0.0.0"
FLASK_PORT = 5000
JPEG_QUALITY = 70


# =========================
# PID state
# =========================

previous_error = 0
integral_sum = 0
last_PID_time = monotonic()

tries = 0


# =========================
# Flask shared frame
# =========================

app = Flask(__name__)

latest_frame = None
frame_lock = threading.Lock()


def generate_frames():
    """Stream the newest annotated frame using MJPEG."""
    while True:
        with frame_lock:
            frame = (
                latest_frame.copy()
                if latest_frame is not None
                else None
            )

        if frame is None:
            sleep(0.01)
            continue

        success, buffer = cv2.imencode(
            ".jpg",
            frame,
            [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY]
        )

        if not success:
            continue

        yield (
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n\r\n"
            + buffer.tobytes()
            + b"\r\n"
        )


@app.route("/")
def index():
    return """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Robot Camera</title>
        <meta name="viewport"
              content="width=device-width, initial-scale=1">
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


@app.route("/video_feed")
def video_feed():
    return Response(
        generate_frames(),
        mimetype="multipart/x-mixed-replace; boundary=frame"
    )


def run_flask():
    app.run(
        host=FLASK_HOST,
        port=FLASK_PORT,
        threaded=True,
        debug=False,
        use_reloader=False
    )


# =========================
# Initialize hardware
# =========================

hw = HardWare(
    MOTOR_IN1,
    MOTOR_IN2,
    SERVO,
    HALL_PIN
)

camera = Picamera2()

config = camera.create_preview_configuration(
    raw={"size": (1920, 1080)},
    main={"size": (1440, 720)}
)

config["main"]["format"] = "RGB888"

camera.configure(config)


# =========================
# Main program
# =========================

try:
    camera.start()
    hw.setServo(0)

    # Start Flask in a background thread
    server_thread = threading.Thread(
        target=run_flask,
        daemon=True
    )
    server_thread.start()

    print(
        f"Camera server starting on port {FLASK_PORT}"
    )
    print("Open http://<raspberry-pi-ip>:5000")

    while True:

        # -------------------------
        # Capture and preprocess
        # -------------------------

        frame = camera.capture_array()

        h, w, _ = frame.shape

        frame = frame[
            CROP_Y_COORDINATE:
            CROP_Y_COORDINATE + (h - CROP_Y_COORDINATE)
        ]

        roi_h, roi_w, _ = frame.shape

        frame = cv2.resize(
            frame,
            (roi_w // 2, roi_h // 2)
        )

        # Picamera2 RGB888 output -> grayscale
        gray = cv2.cvtColor(
            frame,
            cv2.COLOR_RGB2GRAY
        )

        # Initialize per-frame detection results
        midpoint = None
        line_midpoints = []

        # -------------------------
        # Detect track boundaries
        # -------------------------

        row = gray[TRACK_LINE_1, :]

        _, props = find_peaks(
            row,
            width=[10, 100],
            prominence=100,
            distance=100
        )

        line_midpoints = (
            props["left_ips"] + props["right_ips"]
        ) / 2

        camera_center = gray.shape[1] / 2

        # -------------------------
        # Track detection state
        # -------------------------

        if len(line_midpoints) == 0:

            tries += 1

            # Stop after prolonged loss of the track
            if tries > 100 and MOTOR_ENABLE:
                hw.setMotor(0)

        else:
            tries = 0

            if MOTOR_ENABLE:
                hw.setMotor(MOTOR_BASE_SPEED)

            if len(line_midpoints) == 2:
                left = line_midpoints[0]
                right = line_midpoints[1]
                midpoint = (left + right) / 2

            elif len(line_midpoints) > 2:
                # Find adjacent boundaries closest to
                # the expected track width
                best_pair = None
                best_difference = float("inf")

                for i in range(len(line_midpoints) - 1):

                    left_candidate = line_midpoints[i]
                    right_candidate = line_midpoints[i + 1]

                    width = right_candidate - left_candidate

                    difference = abs(
                        APPROX_TRACK_WIDTH - width
                    )

                    if difference < best_difference:
                        best_difference = difference
                        best_pair = (
                            left_candidate,
                            right_candidate
                        )

                if best_pair is not None:
                    midpoint = (
                        best_pair[0] + best_pair[1]
                    ) / 2

            elif len(line_midpoints) == 1:

                # Estimate the missing boundary using
                # the previous steering error
                previous_midpoint = (
                    camera_center + previous_error
                )

                candidate_left = (
                    line_midpoints[0]
                    - APPROX_TRACK_WIDTH / 2
                )

                candidate_right = (
                    line_midpoints[0]
                    + APPROX_TRACK_WIDTH / 2
                )

                if abs(
                    candidate_left - previous_midpoint
                ) < abs(
                    candidate_right - previous_midpoint
                ):
                    midpoint = candidate_left
                else:
                    midpoint = candidate_right

            # -------------------------
            # PID steering
            # -------------------------

            if midpoint is not None:

                error = int(midpoint - camera_center)

                now = monotonic()
                dt = now - last_PID_time
                last_PID_time = now

                if dt > 0:

                    if abs(error) <= 20:
                        integral_sum = 0
                    else:
                        integral_sum += error * dt

                    derivative = (error - previous_error) / dt

                    correction = (
                        KP * error
                        + KI * integral_sum
                        + KD * derivative
                    )

                    correction = max(
                        -100,
                        min(100, correction)
                    )

                    previous_error = error

                    hw.setServo(correction * 2)

                    print(
                        f"Error: {error}, "
                        f"Correction: {correction:.2f}"
                    )

        # -------------------------
        # Annotate camera frame
        # -------------------------

        display_frame = cv2.cvtColor(
            gray,
            cv2.COLOR_GRAY2BGR
        )

        height, width = gray.shape

        # Camera center: white vertical line
        cv2.line(
            display_frame,
            (int(camera_center), 0),
            (int(camera_center), height - 1),
            (255, 255, 255),
            2
        )

        # Detection row: green horizontal line
        cv2.line(
            display_frame,
            (0, TRACK_LINE_1),
            (width - 1, TRACK_LINE_1),
            (0, 255, 0),
            2
        )

        # Detected boundary centers: blue dots
        for point in line_midpoints:
            cv2.circle(
                display_frame,
                (int(point), TRACK_LINE_1),
                5,
                (255, 0, 0),
                -1
            )

        # Estimated track center: red dot
        if midpoint is not None:
            cv2.circle(
                display_frame,
                (int(midpoint), TRACK_LINE_1),
                6,
                (0, 0, 255),
                -1
            )

        # Display status on the image
        status = (
            "TRACK LOST"
            if len(line_midpoints) == 0
            else "TRACK DETECTED"
        )

        cv2.putText(
            display_frame,
            status,
            (10, 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 0, 255) if midpoint is None
            else (0, 255, 0),
            2
        )

        # -------------------------
        # Publish frame to Flask
        # -------------------------

        with frame_lock:
            latest_frame = display_frame.copy()

        # -------------------------
        # Optional local debugging
        # -------------------------

        if DEBUG_MODE:

            cv2.imshow("Robot Camera", display_frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

except KeyboardInterrupt:
    print("Stopping robot...")

finally:
    # Stop the motors even if an exception occurs
    try:
        hw.setMotor(0)
        hw.setServo(0)
        hw.stop()
    except Exception as exc:
        print(f"Hardware shutdown error: {exc}")

    try:
        camera.stop()
        camera.close()
    except Exception:
        pass

    cv2.destroyAllWindows()
