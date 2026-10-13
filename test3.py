import cv2
import pigpio
import threading
from time import sleep, monotonic
from picamera2 import Picamera2
from scipy.signal import find_peaks
from hardware import HardWare
from ultralytics import YOLO

# Degub mode disables motors and enables cv image show
DEBUG_MODE = False
MOTOR_ENABLE = True
MOTOR_SPEED = 80

# Pedestrian sign slowdown
SLOW_SPEED = 10
SLOW_TIME = 4
PEDESTRIAN_SIGN = "pedestrian"
slow_until = 0

# Crosswalk (more than 3 lines)
CROSSWALK_SPEED = 50
CROSSWALK_TIME = 2
CROSSWALK_COOLDOWN = 3
CROSSWALK_FRAMES = 3
crosswalk_frames = 0
crosswalk_until = 0
crosswalk_ignore_until = 0

# Yolo model
model = YOLO("best.onnx")

# Prints all sign names the model knows, check the pedestrian sign name from here
print(model.names)

# I have slept for two hours tonight I have no comment on these variables
APPROX_TRACK_WIDTH = 500
WIDTH_ADAPT = 0.1

# PID Cooficents
KP = 0.8
KI = 0.1
KD = 0

# Smoothing (ERROR_FILTER = 1, DEADBAND = 0 and MAX_STEP = 200 means off)
ERROR_FILTER = 1
DEADBAND = 0
MAX_STEP = 200

# PID memory variables
previous_error = 0
filtered_error = 0
integral_sum = 0
previous_correction = 0

# PID Time variable
last_PID_time = monotonic()

# No lines variables
tries = 0

# Hardware pins
SERVO = 18 
MOTOR_IN1 = 20
MOTOR_IN2 = 21
HALL_PIN = 22

# Y coordinate to cut off
CROP_Y_COORDINATE = 520

# Y coordinates of trach detection lines
TRACK_LINES = [30, 40, 50]

# Peak settings
PEAK_WIDTH = [10, 250]
PEAK_PROMINENCE = 80
PEAK_DISTANCE = 30

# How far a peak can be from the old line position to still count as that line
MATCH_GATE = 180

# Line memory variables
left_line = None
right_line = None
track_width = APPROX_TRACK_WIDTH

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
camera.start()

# Set turn angle to 0
hw.setServo(0)

latest_frame = camera.capture_array()
running = True

# Thread 1 - Capture image
def camera_thread():
    global latest_frame

    while running:
        frame = camera.capture_array()
        latest_frame = frame


# Helper function to return frame
def getFrame():
    return latest_frame.copy()


# Thread 2 - Find signs
# YOLO is slow so it runs in its own thread and does not slow down the line following
def sign_thread():
    global slow_until

    while running:
        results = model.predict(getFrame(), imgsz=(160, 320), conf=0.65, verbose=False)[0]
        detected_signs = [results.names[int(box.cls[0])] for box in results.boxes]

        # Stay slow for SLOW_TIME seconds after seeing the pedestrian sign
        if PEDESTRIAN_SIGN in detected_signs:
            slow_until = monotonic() + SLOW_TIME
            print("Pedestrian sign")


# Start threads
threading.Thread(target=camera_thread, daemon=True).start()
threading.Thread(target=sign_thread, daemon=True).start()

try:
    # Main loop
    while True:
        # Get frame
        frame = getFrame()

        # Crop image 
        frame = frame[CROP_Y_COORDINATE:]

        # Lower resolution
        roi_h, roi_w, _ = frame.shape
        frame = cv2.resize(frame, (roi_w // 2, roi_h // 2))

        # Convert to greyscale
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        camera_center = (frame.shape[1] / 2)

        # Start by assuming the robot is in the middle of the track
        if left_line is None:
            left_line = camera_center - track_width / 2
            right_line = camera_center + track_width / 2

        # Find lines on every track line
        all_peaks = []

        for y in TRACK_LINES:
            row = frame[y, :]
            _, props = find_peaks(row, width=PEAK_WIDTH, prominence=PEAK_PROMINENCE, distance=PEAK_DISTANCE)
            all_peaks.extend((props["left_ips"] + props["right_ips"]) / 2)

        # Group peaks that are close to each other
        all_peaks.sort()
        groups = []

        for peak in all_peaks:
            if len(groups) > 0 and peak - groups[-1][-1] < 60:
                groups[-1].append(peak)
            else:
                groups.append([peak])

        # Only keep lines that were seen on at least 2 track lines
        line_midpoints = [sum(group) / len(group) for group in groups if len(group) >= 2]
        print(f"Line midpoints {len(line_midpoints)}")

        # Crosswalk, more than 3 lines for CROSSWALK_FRAMES frames in a row
        if len(line_midpoints) > 3:
            crosswalk_frames += 1
        else:
            crosswalk_frames = 0

        # Slow down for CROSSWALK_TIME seconds and then ignore crosswalk for a while
        # so we do not trigger again on the same crosswalk
        if crosswalk_frames >= CROSSWALK_FRAMES and monotonic() > crosswalk_ignore_until:
            crosswalk_until = monotonic() + CROSSWALK_TIME
            crosswalk_ignore_until = crosswalk_until + CROSSWALK_COOLDOWN
            print("Crosswalk")

        # Match lines to the old left and right line
        left_match = None
        right_match = None

        # Look wider if the lines were lost on the last frame
        gate = MATCH_GATE
        if tries > 0:
            gate = MATCH_GATE * 2

        best_left = gate
        best_right = gate

        for line in line_midpoints:
            # Distance from the old positions
            distance_left = abs(line - left_line)
            distance_right = abs(line - right_line)

            if distance_left <= distance_right:
                if distance_left < best_left:
                    best_left = distance_left
                    left_match = line
            else:
                if distance_right < best_right:
                    best_right = distance_right
                    right_match = line

        # State machine
        if left_match is not None and right_match is not None:
            left_line = left_match
            right_line = right_match

            # Slowly learn the real track width
            width = right_line - left_line

            if 0.7 * APPROX_TRACK_WIDTH < width < 1.3 * APPROX_TRACK_WIDTH:
                track_width += WIDTH_ADAPT * (width - track_width)

        elif left_match is not None:
            left_line = left_match
            right_line = left_line + track_width

        elif right_match is not None:
            right_line = right_match
            left_line = right_line - track_width

        # Get midpoint
        midpoint = (left_line + right_line) / 2

        if left_match is None and right_match is None:
            # Add one try and keep the old steering
            tries += 1
            hw.setServo(previous_correction * 2)

            if tries > 100:
                hw.setMotor(0)
        else:
            # Reset tries
            tries = 0

            # Slow down if we saw the pedestrian sign lately
            speed = MOTOR_SPEED

            if monotonic() < slow_until:
                speed = SLOW_SPEED

            # Slow down on crosswalk (the slower one of the two wins)
            if monotonic() < crosswalk_until:
                speed = min(speed, CROSSWALK_SPEED)

            if MOTOR_ENABLE:
                hw.setMotor(speed)

            # Calculate error and smooth it
            raw_error = midpoint - camera_center
            filtered_error = ERROR_FILTER * raw_error + (1 - ERROR_FILTER) * filtered_error
            error = filtered_error

            if abs(error) < DEADBAND:
                error = 0

            # Calculate dt for PID and set last time to now
            now = monotonic()
            dt = max(now - last_PID_time, 0.001)
            last_PID_time = now

            # Calculate integral
            if abs(error) <= 20:
                integral_sum = 0
            else:
                integral_sum += error * dt
                integral_sum = max(-100, min(100, integral_sum))

            # Calculate derivative
            derivative = (error - previous_error) / dt

            correction = KP * error + KI * integral_sum + KD * derivative
            correction = max(-100, min(100, correction))

            # Limit how fast the steering can change
            step = max(-MAX_STEP, min(MAX_STEP, correction - previous_correction))
            correction = previous_correction + step

            previous_error = error
            previous_correction = correction

            hw.setServo(correction * 2)

            print(f"Error: {error}")
            print(f"Correction: {correction}")


        if DEBUG_MODE:
            # Debug stuff
            cv2.line(frame, (int(frame.shape[1] / 2), 0), (int(frame.shape[1] / 2), frame.shape[0]), 255, thickness=2)

            for y in TRACK_LINES:

                cv2.line(frame, (0, y), (frame.shape[1], y), 0, thickness=1)

            for line in line_midpoints:

                cv2.circle(frame, (int(line), TRACK_LINES[0]), 3, 128, thickness=2)

            # Where the robot thinks the lines are
            cv2.circle(frame, (int(left_line), TRACK_LINES[0]), 8, 255, thickness=2)

            cv2.circle(frame, (int(right_line), TRACK_LINES[0]), 8, 255, thickness=2)

            cv2.circle(frame, (int(midpoint), TRACK_LINES[0]), 5, 255, thickness=-1)

            # Show image
            cv2.imshow('Frame', frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):

                break

except KeyboardInterrupt:
    pass

finally:
    running = False
    cv2.destroyAllWindows()
    hw.setMotor(0)