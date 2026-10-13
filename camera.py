import cv2
from picamera2 import Picamera2
from scipy.signal import find_peaks

crop_y = 520

line1 = 40


camera = Picamera2()
config = camera.create_preview_configuration(
    raw={"size": (1920, 1080)},
    main={"size": (1440, 720)},
)

config["main"]["format"] = "RGB888"
camera.configure(config)
camera.start()

while True:
    frame = camera.capture_array()
    # Crop image 
    h, w, _ = frame.shape
    frame = frame[crop_y:crop_y + (h - crop_y)] 

    # Lower resolution
    roi_h, roi_w, _ = frame.shape
    frame = cv2.resize(frame, (roi_w // 2, roi_h // 2))

    # Convert to greyscale
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Find lines and midpoint
    row = frame[line1, :]
    _, props = find_peaks(row, width=[10, 100], prominence=100, distance=100)
    line_midpoints = (props["left_ips"] + props["right_ips"]) / 2

    try: 
        midpoint = int((line_midpoints[0] + line_midpoints[len(line_midpoints) - 1]) / 2)

        # Calculate error
        error = int(midpoint - (frame.shape[1] / 2))

        # Debug stuff
        cv2.line(frame, (int(frame.shape[1] / 2), 0), (int(frame.shape[1] / 2), frame.shape[0]), (255, 255, 255), thickness=2)
        cv2.line(frame, (0, line1), (frame.shape[1], line1), (0, 0, 0), thickness=2)
        cv2.circle(frame, (int(line_midpoints[0]), line1), 3, (255, 0, 0), thickness=2)
        cv2.circle(frame, (int(line_midpoints[len(line_midpoints) - 1]), line1), 3, (255, 0, 0), thickness=2)
        cv2.circle(frame, (midpoint, line1), 3, (255, 0, 0), thickness=2)
        print(error)
    except IndexError:
        pass


    # Show image
    cv2.imshow('Frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()