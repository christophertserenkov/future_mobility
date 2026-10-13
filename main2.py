import cv2
import pigpio
import threading
from time import sleep, monotonic
from picamera2 import Picamera2
from scipy.signal import find_peaks
from hardware import HardWare
from ultralytics import YOLO
from PIDControl import PID

# Degub mode disables motors and enables cv image show
DEBUG_MODE = False
MOTOR_ENABLE = True
MOTOR_BASE_SPEED = 80

# Yolo model
model = YOLO("best.onnx")

# I have slept for two hours tonight I have no comment on these variables
APPROX_TRACK_WIDTH = 500

# PID Cooficents
KP = 0.8
KI = 0.1
KD = 0

# PID memory variables
previous_error = 0
integral_sum = 0
previous_detected_midpoint = 0

# PID Time variable
last_PID_time = monotonic()

# No lines variables
tries = 0

motor_speed = MOTOR_BASE_SPEED

# Hardware pins
SERVO = 18 
MOTOR_IN1 = 20
MOTOR_IN2 = 21
HALL_PIN = 22

# Y coordinate to cut off
CROP_Y_COORDINATE = 520

# Y coordinate of trach detection line
TRACK_LINE_1 = 10

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

# Thread 1 - Capture image
def camera_thread():
    global latest_frame

    while True:
        frame = camera.capture_array()
        latest_frame = frame


# Helper function to return frame
def getFrame():
    return latest_frame.copy()


def sign_detection_thread():
    global motor_speed

    while True
        frame = getFrame()
        results = model.predict(frame, imgsz=(160,320), conf=0.65)[0]
        if len(results.boxes) > 0:
            detected_signs = [results.names[int(box.cls[0])] for box in results.boxes]
            if 'crosswalk' in detected_signs:
                motor_speed = int(MOTOR_BASE_SPEED / 2)
            else:
                motor_speed = MOTOR_BASE_SPEED
        else:
            motor_speed = MOTOR_BASE_SPEED
            detected_signs = []

        print(detected_signs)


def motor_speed_thread():
    global motor_speed

    while True:
        speed = control.getSpeed()
        print(f"Speed: {round(speed, 1)}")

        if speed != last_speed or time() - last_calculation > 0.25:
            duty_cycle = pid.correct(speed, motor_speed)
            print(f"Duty cycle: {round(duty_cycle)}")
            control.setMotor(duty_cycle)

        last_speed = speed



try:
    # Main loop
    while True:
        # Capture frame
        frame = camera.capture_array()

        # Crop image 
        h, w, _ = frame.shape
        frame = frame[CROP_Y_COORDINATE:CROP_Y_COORDINATE + (h - CROP_Y_COORDINATE)] 

        # Lower resolution
        roi_h, roi_w, _ = frame.shape
        frame = cv2.resize(frame, (roi_w // 2, roi_h // 2))

        # Convert to greyscale
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Find lines and midpoint
        row = frame[TRACK_LINE_1, :]
        _, props = find_peaks(row, width=[10, 100], prominence=100, distance=100)
        line_midpoints = (props["left_ips"] + props["right_ips"]) / 2
        print(f"Line midpoints {len(line_midpoints)}")

        camera_center = (frame.shape[1] / 2)

        if len(line_midpoints) == 0:
            if tries > 100:
                hw.setMotor(0)
        else:
            # Reset tries
            tries = 0

            if MOTOR_ENABLE:
                hw.setMotor(MOTOR_SPEED)

            # State machine
            if len(line_midpoints) == 2:
                # Get midpoint
                left = line_midpoints[0]
                right = line_midpoints[-1]
                midpoint = (left + right) / 2

            elif len(line_midpoints) > 2:
                # Store best point pairs
                best_pair = None

                # Store best line 
                best_difference = float("inf")

                for i in range(len(line_midpoints) - 1):
                    # Get distance between two points
                    width = line_midpoints[i + 1] - line_midpoints[i]

                    # Difference from prodictid width
                    diff_from_prediction = abs(APPROX_TRACK_WIDTH - width)

                    # Update best if current diff is smaller than best
                    if best_difference > diff_from_prediction:
                        best_pair = [line_midpoints[i + 1], line_midpoints[i]]
                        best_difference = diff_from_prediction
                
                left = best_pair[0]
                right = best_pair[-1]
                midpoint = (left + right) / 2

            elif len(line_midpoints) == 1:
                previous_midpoint = camera_center + previous_error

                midpoint_left = line_midpoints[0] + APPROX_TRACK_WIDTH / 2
                midpoint_right = line_midpoints[0] - APPROX_TRACK_WIDTH / 2

                if abs(midpoint_left - previous_midpoint) < abs(midpoint_right - previous_midpoint):
                    midpoint = midpoint_left
                else:
                    midpoint = midpoint_right
                

            # Calculate error
            error = int(midpoint - camera_center)

            # Calculate dt for PID and set last time to now
            now = monotonic()
            dt = now - last_PID_time
            last_PID_time = now

            # Calculate integral
            if abs(error) <= 20:
                integral_sum = 0
            else:
                integral_sum += error * dt

            # Calculate derivative
            derivative = (error - previous_error) / dt

            correction = KP * error + KI * integral_sum + KD * derivative
            correction = max(-100, min(100, correction))

            previous_error = error

            hw.setServo(correction * 2)

            print(f"Error: {error}")
            print(f"Correction: {correction}")


        if DEBUG_MODE:
            # Debug stuff
            cv2.line(frame, (int(frame.shape[1] / 2), 0), (int(frame.shape[1] / 2), frame.shape[0]), (255, 255, 255), thickness=2)

            cv2.line(frame, (0, TRACK_LINE_1), (frame.shape[1], TRACK_LINE_1), (0, 0, 0), thickness=2)

            for line in line_midpoints:

                cv2.circle(frame, (int(line), TRACK_LINE_1), 3, (255, 0, 0), thickness=2)

            if midpoint is not None:

                cv2.circle(frame, (int(midpoint), TRACK_LINE_1), 3, (255, 0, 0), thickness=2)

            # Show image
            cv2.imshow('Frame', frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):

                break

    cv2.destroyAllWindows()
    hw.setMotor(0)
    hw.stop()

except KeyboardInterrupt:
    cv2.destroyAllWindows()
    hw.setMotor(0)
    hw.stop()
        