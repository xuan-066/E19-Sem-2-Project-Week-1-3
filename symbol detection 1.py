import numpy as np
import time
import datetime
import sys
import cv2
from picamera2 import Picamera2

# --- 1. Import TFLite Runtime ---
try:
    from tflite_runtime.interpreter import Interpreter
except ImportError:
    print("Error: TFLite Runtime not found. Run: pip install tflite-runtime")
    sys.exit()

# --- 2. Helper Functions ---
def load_labels(path):
    with open(path, 'r') as f:
        return [line.strip() for line in f.readlines()]

def set_input_tensor(interpreter, image):
    tensor_index = interpreter.get_input_details()[0]['index']
    input_tensor = interpreter.tensor(tensor_index)()[0]

    if interpreter.get_input_details()[0]['dtype'] == np.float32:
        input_tensor[:, :] = (np.float32(image) / 127.5) - 1.0
    else:
        input_tensor[:, :] = image

def classify_image(interpreter, image, top_k=1):
    set_input_tensor(interpreter, image)
    interpreter.invoke()

    output_details = interpreter.get_output_details()[0]
    output = np.squeeze(interpreter.get_tensor(output_details['index']))

    if 'quantization' in output_details:
        scale, zero_point = output_details['quantization']
        if scale > 0:
            output = scale * (output - zero_point)

    ordered = np.argpartition(-output, 1)
    return [(i, output[i]) for i in ordered[:top_k]][0]

# --- 3. Load Model and Labels ---
model_path = "/home/pi/Documents/model_unquant.tflite"
label_path = "/home/pi/Documents/labels.txt"

labels = load_labels(label_path)
interpreter = Interpreter(model_path)
interpreter.allocate_tensors()
_, height, width, _ = interpreter.get_input_details()[0]['shape']
print(f"Model loaded. Input size: {width}x{height}")

# --- 4. Safe Camera Initialization ---
MAX_RETRIES = 5
for attempt in range(MAX_RETRIES):
    try:
        print("Initializing Picamera2...")
        picam2 = Picamera2()
        config = picam2.create_preview_configuration(main={"size": (width, height)})
        picam2.configure(config)
        picam2.start()
        print("Camera initialized successfully.")
        break
    except RuntimeError as e:
        print(f"Camera init failed (attempt {attempt+1}/{MAX_RETRIES}): {e}")
        time.sleep(2)
else:
    print("Failed to initialize camera after several attempts. Exiting.")
    sys.exit(1)

time.sleep(2)
print("Starting symbol detection loop! Press Ctrl+C to stop.")

# --- 5a. Pixel-Area Method for LEFT/RIGHT Arrows ---
def detect_left_right(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 145, 255, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) == 0:
        return None
    cnt = max(contours, key=cv2.contourArea)
    if cv2.contourArea(cnt) < 150:
        return None
    x, y, w, h = cv2.boundingRect(cnt)
    roi = thresh[y:y+h, x:x+w]
    center = w // 2
    left = roi[:, :center]
    right = roi[:, center:]
    left_pixels = cv2.countNonZero(left)
    right_pixels = cv2.countNonZero(right)
    return "Right" if right_pixels > left_pixels else "Left"

# --- 5b. Original Convex-Hull Arrow Detection ---
def get_arrow_direction(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 130, 255, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) == 0:
        return None
    cnt = max(contours, key=cv2.contourArea)
    if cv2.contourArea(cnt) < 150:
        return None
    hull = cv2.convexHull(cnt)
    hull = hull.reshape(-1, 2)
    M = cv2.moments(cnt)
    if M["m00"] == 0:
        return None
    cx = int(M["m10"] / M["m00"])
    cy = int(M["m01"] / M["m00"])
    centroid = np.array([cx, cy])
    distances = np.linalg.norm(hull - centroid, axis=1)
    tip = hull[np.argmax(distances)]
    dx = tip[0] - cx
    dy = cy - tip[1]
    # --- Decide preliminary direction ---
    if abs(dy) > abs(dx):
        preliminary = "Right" if dx > 0 else "Left"
    else:
        preliminary = "Up" if dy > 0 else "Down"
    # --- If LEFT/RIGHT, override with pixel-area method ---
    if preliminary in ["Left", "Right"]:
        final = detect_left_right(frame)
        return final
    return preliminary

# --- 6. Main Detection Loop ---
try:
    while True:
        frame = picam2.capture_array()
        if frame.shape[2] == 4:
            frame = frame[:, :, :3]
        frame_resized = cv2.resize(frame, (width, height)) if (frame.shape[0] != height or frame.shape[1] != width) else frame
        label_id, prob = classify_image(interpreter, frame_resized)
        label_text = labels[label_id]

        if label_text.lower().startswith("arrow"):
            direction = get_arrow_direction(frame_resized)
            if direction:
                print(f"[{datetime.datetime.now().strftime('%H:%M:%S')}] Arrow Direction: {direction}, Confidence: {prob*100:.1f}%")
            else:
                print(f"[{datetime.datetime.now().strftime('%H:%M:%S')}] Arrow detected but direction not found")
        else:
            print(f"[{datetime.datetime.now().strftime('%H:%M:%S')}] Symbol: {label_text}, Confidence: {prob*100:.1f}%")

        time.sleep(0.05)

except KeyboardInterrupt:
    print("Detection stopped by user.")

finally:
    picam2.stop()
    print("Camera safely shut down.")
