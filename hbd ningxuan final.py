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

# --- 5a. Pixel-Area Method for LEFT/RIGHT Arrows -ction ---
def detect_arrow(image):

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray,(5,5),0)

    _, thresh = cv2.threshold(blur, 160, 255, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.imshow("grey",thresh)
    for cnt in contours:
        if cv2.contourArea(cnt) < 2000: # Ignore noise
            continue
        x,y,w,h = cv2.boundingRect(cnt)

        roi = thresh[y:y+h, x:x+w]

        cx = w//2
        cy = h//2

        left = np.sum(roi[:,0:cx])
        right = np.sum(roi[:,cx:w])
        top = np.sum(roi[0:cy,:])
        bottom = np.sum(roi[cy:h,:])

        if right > left and right > top and right > bottom:
            return "right"

        if left > right and left > top and left > bottom:
            return "left"

        if top > bottom and top > left and top > right:
            return "up"

        if bottom > top and bottom > left and bottom > right:
            return "down"

    return None


# --- 6. Main Detection Loop ---
try:
    while True:
        frame = picam2.capture_array()
        if frame.shape[2] == 4:
            frame = frame[:, :, :3]
        frame_resized = cv2.resize(frame, (width, height)) if (frame.shape[0] != height or frame.shape[1] != width) else frame
        label_id, prob = classify_image(interpreter, frame_resized)
        label_text = labels[label_id]
        
        # --- Show preview window ---
        display_frame = cv2.resize(frame, (640, 480))
        
        
        # --- Existing AI + Arrow logic ---
        if label_text.lower().startswith("5") or label_text.lower().startswith("7") or label_text.lower().startswith("13") or label_text.lower().startswith("15"):
            direction = detect_arrow(frame_resized)
            print(direction)
            cv2.putText(display_frame,
                    f"{direction}",
                    (10, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 255, 255),
                    2)
            
        else:
            print(f"[{datetime.datetime.now().strftime('%H:%M:%S')}] Symbol: {label_text}, Confidence: {prob*100:.1f}%")
            cv2.putText(display_frame,
                    f"{label_text}: {prob*100:.1f}%",
                    (10, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 255, 255),
                    2)
            
        cv2.imshow("Camera Preview", display_frame)

        if cv2.waitKey(1) & 0xFF == 27:  # ESC key to exit
            break

        time.sleep(0.05)

except KeyboardInterrupt:
    print("Detection stopped by user.")

finally:
    picam2.stop()
    cv2.destroyAllWindows()
    print("Camera safely shut down.")
