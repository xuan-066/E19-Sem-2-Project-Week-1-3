import RPi.GPIO as GPIO
import cv2
import numpy as np
from picamera2 import Picamera2
import time
import threading
import logging

# =============================================================
# CONFIG
# =============================================================
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')

IN1, IN2, IN3, IN4 = 4, 17, 27, 22
ENA, ENB = 18, 19

BASE_SPEED = 28
Kp = 0.6
shared = threading.Lock()
shared_roi = None

last_error=0

frame_lock = threading.Lock() 
state_lock = threading.Lock() 
  
shared_frame = None
running = True

status = {
    "symbol": "None",
    "direction": "None",
    "action_in_progress": False,
    "last_action_time": 0
}

# --- Global dict to share ORB stats ---
orb_debug_stats = { 
    "kp_count": 0,
    "good_matches": 0,
    "inliers": 0,
    "symbol": "None",
    "box": None
}

ACTION_COOLDOWN = 1
arrow_memory = []
ARROW_CONFIRM_FRAMES = 2

# =============================================================
# MOTOR SETUP
# =============================================================
GPIO.setmode(GPIO.BCM)
GPIO.setwarnings(False)
GPIO.setup([IN1, IN2, IN3, IN4, ENA, ENB], GPIO.OUT)

pwmLeft = GPIO.PWM(ENA, 100)
pwmRight = GPIO.PWM(ENB, 100)
pwmLeft.start(0)
pwmRight.start(0)

def set_motors(left, right):
    GPIO.output(IN1, GPIO.HIGH if left >= 0 else GPIO.LOW)
    GPIO.output(IN2, GPIO.LOW if left >= 0 else GPIO.HIGH)
    GPIO.output(IN3, GPIO.HIGH if right >= 0 else GPIO.LOW)
    GPIO.output(IN4, GPIO.LOW if right >= 0 else GPIO.HIGH)
    pwmLeft.ChangeDutyCycle(max(0, min(abs(left), 100)))
    pwmRight.ChangeDutyCycle(max(0, min(abs(right), 100)))

def stop_robot():
    set_motors(0, 0)

# =============================================================
# ARROW DETECTION
# =============================================================
def get_arrow_direction_hybrid(img):
    global shared_roi # Access the global variable
    try:
        # Define the ROI coordinates
        roi_y1, roi_y2, roi_x1, roi_x2 = 60, 180, 80, 240
        roi_frame = img
        hsv_vision = cv2.cvtColor(roi_frame, cv2.COLOR_BGR2HSV)

        mask_blue = cv2.inRange(hsv_vision, np.array([0, 80, 62]), np.array([22, 255, 255]))
        mask_green = cv2.inRange(hsv_vision, np.array([45, 51, 50]), np.array([86, 167, 255]))
        mask_darkgreen = cv2.inRange(hsv_vision, np.array([24, 55, 47]), np.array([90, 255, 255]))
        mask_danger = cv2.inRange(hsv_vision, np.array([64, 120, 159]), np.array([95, 255, 255]))
        mask_orange = cv2.inRange(hsv_vision, np.array([82 ,140, 0]), np.array([114, 255, 255]))
        mask_red = cv2.inRange(hsv_vision, np.array([116 ,80, 115]), np.array([140, 249, 255]))
        mask_red1 = cv2.inRange(hsv_vision, np.array([114 ,91, 116]), np.array([130, 255, 255]))
        
        sat_mask = mask_blue | mask_green | mask_darkgreen | mask_danger | mask_orange  | mask_red | mask_red1
        gray = sat_mask
        
        # Pre-processing
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        enhanced = clahe.apply(gray)
        blurred = cv2.GaussianBlur(enhanced, (5,5), 0)
        thresh = cv2.adaptiveThreshold(gray, 255,
                                       cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                       cv2.THRESH_BINARY_INV, 11, 2)
        conts, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        direction = "None"
        for cnt in conts:
            area = cv2.contourArea(cnt)
            
            if 3000 < area < 7000:
                cv2.drawContours(roi_frame, [cnt], -1, (0, 255, 0), 2)
                
                hull = cv2.convexHull(cnt)
                hull_area = cv2.contourArea(hull)
                if hull_area == 0: continue
                if (float(area) / hull_area) > 0.9: continue
                
                peri = cv2.arcLength(cnt, True)
                approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
                if 8<=len(approx) <=12:
                    M = cv2.moments(approx)
                    if M["m00"] != 0:
                        cx, cy = int(M["m10"]/M["m00"]), int(M["m01"]/M["m00"])
                        max_dist = 0
                        arrow_tip = None
                        for point in approx:
                            px, py = point[0]
                            dist = np.sqrt((px-cx)**2 + (py-cy)**2)
                            if dist > max_dist:
                                max_dist = dist
                                arrow_tip = (px, py)
                        
                        if arrow_tip:
                            dx, dy = arrow_tip[0] - cx, arrow_tip[1] - cy
                            if abs(dx) > abs(dy):
                                direction = "RIGHT" if dx < 0 else "LEFT"
                            else:
                                direction = "UP" if dy > 0 else "DOWN"

        with shared:
            global shared_roi
            shared_roi = sat_mask
            
        return direction
    except Exception as e:
        print(f"Arrow error: {e}")
        return "None"

# =============================================================
# DETECTION THREAD (symbols + arrow)
# =============================================================
def detection_thread():
    global shared_frame, running, arrow_memory
    
    # --- ORB LOGIC CONSTANTS ---
    MIN_GOOD_MATCHES = 6       
    MIN_INLIERS = {"QR": 4, "Fingerprint": 6, "Danger": 4, "STOP": 6, "Recycle": 3}
    MARGIN = 5
    RATIO_THRESH = 0.80         

    orb = cv2.ORB_create(nfeatures=1000, scaleFactor=1.2, nlevels=8, edgeThreshold=15, patchSize=31, fastThreshold=20)
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))

    symbol_files = { 
        "QR": "QR.jpeg", "Fingerprint": "fingerprint.jpeg",
        "Recycle": "recycle.jpeg", "Danger": "danger.jpeg", "STOP": "stop.jpeg",
    }
    
    templates = {}
    print("[INFO] Loading high-fidelity multi-scale templates locally...")
    for name, path in symbol_files.items():
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if img is not None:
            scale_data = []
            for size in [500, 300, 400]:
                resized = cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA)
                enhanced = cv2.GaussianBlur(clahe.apply(resized), (3, 3), 0)
                kp, des = orb.detectAndCompute(enhanced, None)
                if des is not None and len(kp) >= 10:
                    scale_data.append((resized, kp, des))
            if scale_data: templates[name] = scale_data

    while running:
        with frame_lock:
            if shared_frame is None:
                continue
            local_frame = shared_frame

        with state_lock:
            current_dir = status["direction"]
            last_time = status["last_action_time"]

        now = time.time()
        # If we are cooling down OR an arrow is currently being executed by the main thread, SLEEP!
        if now - last_time < ACTION_COOLDOWN or current_dir != "None":
            continue
        
            
        blurred_vision = cv2.GaussianBlur(local_frame, (5, 5), 0)
        hsv_vision = cv2.cvtColor(blurred_vision, cv2.COLOR_BGR2HSV)

        mask_blue = cv2.inRange(hsv_vision, np.array([0, 80, 62]), np.array([22, 255, 255]))
        mask_green = cv2.inRange(hsv_vision, np.array([45, 51, 50]), np.array([86, 167, 255]))
        mask_darkgreen = cv2.inRange(hsv_vision, np.array([24, 55, 47]), np.array([90, 255, 255]))
        mask_danger = cv2.inRange(hsv_vision, np.array([64, 120, 159]), np.array([95, 255, 255]))
        mask_orange = cv2.inRange(hsv_vision, np.array([82 ,140, 0]), np.array([114, 255, 255]))
        mask_fingerprint = cv2.inRange(hsv_vision, np.array([130, 32, 48]), np.array([179, 255, 255]))
        mask_red = cv2.inRange(hsv_vision, np.array([116 ,80, 115]), np.array([140, 249, 255]))
        mask_red1 = cv2.inRange(hsv_vision, np.array([114 ,91, 116]), np.array([130, 255, 255]))

       
        sat_mask = mask_blue | mask_green | mask_darkgreen | mask_danger | mask_orange | mask_fingerprint | mask_red | mask_red1 

        kernel_open = np.ones((5,5), np.uint8)
        kernel_close = np.ones((7,7), np.uint8)
        sat_mask = cv2.morphologyEx(sat_mask, cv2.MORPH_OPEN, kernel_open, iterations=1)
        sat_mask = cv2.morphologyEx(sat_mask, cv2.MORPH_CLOSE, kernel_close, iterations=1)

        clean_mask = np.zeros_like(sat_mask)
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(sat_mask, 8)
        for i in range(1, num_labels):
            if stats[i, cv2.CC_STAT_AREA] >= 400: clean_mask[labels == i] = 255
        sat_mask = clean_mask

        pixel_counts = {c_name: cv2.countNonZero(cv2.bitwise_and(c_mask, sat_mask)) for c_name, c_mask in color_masks.items()}
        blob_color = max(pixel_counts, key=pixel_counts.get)
        if pixel_counts[blob_color] == 0: blob_color = "Color"
        
        ALLOWED_SYMBOL_COLORS = {"QR": ["Blue"], "Recycle": ["Green/Cyan"], "STOP": ["Green/Cyan"], "Danger": ["Yellow", "Orange"], "Fingerprint": ["Purple"]}

        gray = cv2.cvtColor(local_frame, cv2.COLOR_BGR2GRAY)
        gray = clahe.apply(gray)
        gray = cv2.GaussianBlur(gray, (3, 3), 0)
        
        kp_frame, des_frame = orb.detectAndCompute(gray, None)
        found_symbol = False
        distorted_glitch_detected = False
        best_symbol, best_inliers, second_best_inliers = None, 0, 0
        
        # --- DEBUG TRACKERS ---
        current_kp_count = len(kp_frame) if kp_frame else 0
        all_symbols_stats = {} 

        if des_frame is not None and len(des_frame) > 2:
            for name, scale_data in templates.items():
                if blob_color not in ALLOWED_SYMBOL_COLORS.get(name, []): 
                    continue 
                
                symbol_max_inliers = 0
                symbol_max_matches = 0
                
                for (tmp_img, kp_tmp, des_tmp) in scale_data:
                    try: matches = bf.knnMatch(des_tmp, des_frame, k=2)
                    except cv2.error: continue

                    good = [m for pair in matches if len(pair) == 2 for m, n in [pair] if m.distance < RATIO_THRESH * n.distance]

                    if len(good) > symbol_max_matches:
                        symbol_max_matches = len(good)

                    if len(good) >= MIN_GOOD_MATCHES:
                        src_pts = np.float32([kp_tmp[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
                        dst_pts = np.float32([kp_frame[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
                        H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 8.0)

                        if H is not None and mask is not None:
                            inliers = int(mask.sum())
                            if inliers >= MIN_INLIERS.get(name, 6): 
                                h_tmp, w_tmp = tmp_img.shape[:2]
                                box_pts = np.float32([[0, 0], [0, h_tmp - 1], [w_tmp - 1, h_tmp - 1], [w_tmp - 1, 0]]).reshape(-1, 1, 2)
                                dst_box = cv2.perspectiveTransform(box_pts, H)
                                rx, ry, rw, rh = cv2.boundingRect(np.int32(dst_box))
                                symbol_area = rw * rh
                                
                                aspect_ratio = float(rw) / float(rh) if rh > 0 else 0.0
                                # --- CUSTOM ASPECT RATIO FILTER PER SYMBOL ---
                                #print(aspect_ratio)
                                min_ar = 0.3
                                max_ar = 1.2
                                if name == "Recycle":
                                    min_ar = 0.2
                                    max_ar = 2.0  # Give Recycle much more room to stretch
                                elif name == "Danger":
                                    min_ar = 0.7
                                    max_ar = 1.3
                                
                                if 0 < symbol_area < 150000 and min_ar < aspect_ratio < max_ar and cv2.isContourConvex(np.int32(dst_box)):
                                    if inliers > symbol_max_inliers: 
                                        symbol_max_inliers = inliers
                                    if inliers >= 12: break
                                else: distorted_glitch_detected = True 
                
                all_symbols_stats[name] = {"matches": symbol_max_matches, "inliers": symbol_max_inliers}
                                    
                if symbol_max_inliers > best_inliers: 
                    second_best_inliers, best_symbol, 
                    best_inliers = best_inliers, 
                    name, symbol_max_inliers
                elif symbol_max_inliers > second_best_inliers: 
                    second_best_inliers = symbol_max_inliers

        # --- TERMINAL DEBUGGING ---
        valid_colors = [color for colors in ALLOWED_SYMBOL_COLORS.values() for color in colors]
        if blob_color in valid_colors or current_kp_count > 50:
            debug_parts = []
            for name in templates.keys():
                if blob_color not in ALLOWED_SYMBOL_COLORS.get(name, []):
                    debug_parts.append(f"{name}: SkipColor")
                elif name in all_symbols_stats:
                    m = all_symbols_stats[name]['matches']
                    i = all_symbols_stats[name]['inliers']
                    debug_parts.append(f"{name}: M={m},I={i}")
                else:
                    debug_parts.append(f"{name}: M=0,I=0")
            
            details = " | ".join(debug_parts)
            #logging.info(f"[ORB] Color: {blob_color:10s} | KP: {current_kp_count:4d} || {details}")

        # --- SYMBOL ACTION ---
        if best_symbol and (best_inliers - second_best_inliers) >= MARGIN:
            found_symbol = True
            logging.info(f">>> SYMBOL DETECTED: {best_symbol} <<<")
            
            with state_lock:
                status["symbol"] = best_symbol
                status["direction"] = "None"
                status["action_in_progress"] = True
                
            # ACTIONS
            if best_symbol == "Danger" or best_symbol == "STOP":
                stop_robot()
                time.sleep(2)
                set_motors(40, 40)
                time.sleep(0.5)
            elif best_symbol == "Recycle":
                set_motors(-85, 85)
                time.sleep(1.8) # Blind turn to get off the current line
                
                logging.info("[RECYCLE] Scanning for line to stop turning...")
                while True:
                    with frame_lock:
                        if shared_frame is None: continue
                        f = shared_frame.copy()
                        
                    h_f, w_f, _ = f.shape
                    roi_scan = f[h_f-60:h_f, :].copy() 
                    blurred_scan = cv2.GaussianBlur(roi_scan, (5, 5), 0)
                    hsv_scan = cv2.cvtColor(blurred_scan, cv2.COLOR_BGR2HSV)
                    gray_scan = cv2.cvtColor(blurred_scan, cv2.COLOR_BGR2GRAY)


                    # Black Scan
                    _, m_blk = cv2.threshold(gray_scan, 105, 255, cv2.THRESH_BINARY_INV)
                    c_blk, _ = cv2.findContours(m_blk, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

                    # If we find ANY line area > 300, Stop turning!
                    if get_best_contour(c_blk, 300) is not None:
                        logging.info("[RECYCLE] Line Found! Stopping turn.")
                        break 
                                    
                stop_robot()
            elif best_symbol == "QR":
                print("SYMBOL: QR")
            elif best_symbol == "Fingerprint":
                print("SYMBOL: Fingerprint")
                
            with state_lock:
                status["action_in_progress"] = False
                status["last_action_time"] = time.time()

        # --- ARROW DETECTION ---
        if not found_symbol:
            direction = get_arrow_direction_hybrid(local_frame)
            arrow_memory.append(direction)
            if len(arrow_memory) > ARROW_CONFIRM_FRAMES:
                arrow_memory.pop(0)
            if arrow_memory.count(direction) >= ARROW_CONFIRM_FRAMES and direction != "None":
                logging.info(f">>> ARROW: {direction}")
                arrow_memory = []
                with state_lock:
                    status["symbol"] = "None"
                    status["direction"] = direction

# =============================================================
# CAMERA THREAD (non-blocking to reduce lag)
# =============================================================
def camera_thread():
    global shared_frame, running
    while running:
        frame = picam2.capture_array()
        with frame_lock:
            shared_frame = frame.copy()

# Helper function to find the largest valid contour
def get_best_contour(conts, min_area=300):
    if not conts: return None
    largest = max(conts, key=cv2.contourArea)
    if cv2.contourArea(largest) > min_area:
        return largest
    return None

# =============================================================
# MAIN
# =============================================================
try:
    picam2 = Picamera2()
    config = picam2.create_preview_configuration(main={"size": (320,240)})
    picam2.configure(config)
    picam2.start()

    # Start detection thread
    threading.Thread(target=detection_thread, daemon=True).start()
    # Start camera thread
    threading.Thread(target=camera_thread, daemon=True).start()

    prev_frame_time = 0 

    # --- TRACKING VARIABLES FOR DEBOUNCE & LANE MEMORY ---
    color_history = ["Black", "Black", "Black"]
    CONFIRM_FRAMES = 3
    
    lane_memory = None
    color_left_votes = 0
    color_right_votes = 0
    was_on_priority_color = False

    while True:
        with frame_lock:
            if shared_frame is None:
                continue
            frame = shared_frame.copy()

        new_frame_time = time.time()
        fps = 1 / (new_frame_time - prev_frame_time) if prev_frame_time != 0 else 0
        prev_frame_time = new_frame_time

        with state_lock:
            in_action = status["action_in_progress"]
            sym = status["symbol"]
            arr = status["direction"]

        if not in_action:
            if arr == "LEFT":
                set_motors(-50, 80)
                time.sleep(0.5)
                with state_lock:
                    status["direction"] = "None"
                    status["last_action_time"] = time.time() # <--- COOLDOWN TRIGGERED HERE
            elif arr == "RIGHT":
                set_motors(80, -50)
                time.sleep(0.5)
                with state_lock:
                    status["direction"] = "None"
                    status["last_action_time"] = time.time() # <--- COOLDOWN TRIGGERED HERE

            elif arr == "UP":
                set_motors(40, 40)
                time.sleep(0.3)
                with state_lock:
                    status["direction"] = "None"
                    status["last_action_time"] = time.time() # <--- COOLDOWN TRIGGERED HERE

            else:
                # --- COLOR PRIORITY LINE FOLLOW ---
                h, w, _ = frame.shape
                roi = frame[h-60:h, :].copy() 
                
                # Blur floor slightly to avoid reflection noise
                blurred_roi = cv2.GaussianBlur(roi, (5, 5), 0)
                hsv = cv2.cvtColor(blurred_roi, cv2.COLOR_BGR2HSV)
                gray = cv2.cvtColor(blurred_roi, cv2.COLOR_BGR2GRAY)

                # 1. Red Mask
                lower_red1 = np.array([116, 80, 115])
                upper_red1 = np.array([140, 255, 255])
                lower_red2 = np.array([114, 91, 116])
                upper_red2 = np.array([130, 255, 255])
                mask_red = cv2.bitwise_or(
                    cv2.inRange(hsv, lower_red1, upper_red1),
                    cv2.inRange(hsv, lower_red2, upper_red2)
                )
                mask_red = cv2.morphologyEx(mask_red, cv2.MORPH_OPEN, np.ones((3,3), np.uint8))
                mask_red = cv2.morphologyEx(mask_red, cv2.MORPH_CLOSE, np.ones((3,3), np.uint8))
                conts_red, _ = cv2.findContours(mask_red, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

                # 2. Yellow Mask (FIXED RANGE to actually see Yellow/Orange)
                #lower_yellow = np.array([15, 60, 60])
                #upper_yellow = np.array([45, 255, 255])
                #mask_yellow = cv2.inRange(hsv, lower_yellow, upper_yellow)
                
                mask_danger = cv2.inRange(hsv, np.array([64, 120, 159]), np.array([95, 255, 255]))
                mask_orange = cv2.inRange(hsv, np.array([82 ,140, 0]), np.array([114, 255, 255]))
                mask_yellow = mask_danger | mask_orange
                mask_yellow = cv2.morphologyEx(mask_yellow, cv2.MORPH_OPEN, np.ones((3,3), np.uint8))
                mask_yellow = cv2.morphologyEx(mask_yellow, cv2.MORPH_CLOSE, np.ones((3,3), np.uint8))
                conts_yellow, _ = cv2.findContours(mask_yellow, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

                # 3. Black Mask (Standard Fallback)
                _, mask_black = cv2.threshold(gray, 90, 255, cv2.THRESH_BINARY_INV)
                conts_black, _ = cv2.findContours(mask_black, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

                # Evaluate Contours
                best_red = get_best_contour(conts_red, min_area=300)
                best_yellow = get_best_contour(conts_yellow, min_area=300)
                best_black = get_best_contour(conts_black, min_area=300)

                # --- 1. COOLDOWN / DEBOUNCE LOGIC ---
                raw_color = "Black" # Default
                if best_red is not None: raw_color = "Red"
                elif best_yellow is not None: raw_color = "Yellow"
                elif best_black is None: raw_color = "Lost"

                color_history.append(raw_color)
                if len(color_history) > CONFIRM_FRAMES:
                    color_history.pop(0)

                # Find the most common color in the last few frames
                follow_color = max(set(color_history), key=color_history.count)

                target_contour = None
                if follow_color == "Red": target_contour = best_red
                elif follow_color == "Yellow": target_contour = best_yellow
                elif follow_color == "Black": target_contour = best_black

                # --- 2. LANE MEMORY & EXITING LOGIC ---
                if follow_color in ["Red", "Yellow"]:
                    was_on_priority_color = True
                    
                    if lane_memory is None and target_contour is not None:
                        # Find center of color line
                        M_color = cv2.moments(target_contour)
                        if M_color["m00"] > 0:
                            cx_color = int(M_color["m10"]/M_color["m00"])
                            
                            # Find center of black line (or assume middle of screen if lost)
                            cx_black = w // 2 
                            if best_black is not None:
                                M_black = cv2.moments(best_black)
                                if M_black["m00"] > 0:
                                    cx_black = int(M_black["m10"]/M_black["m00"])
                            
                            # Vote on position
                            if cx_color < cx_black:
                                color_left_votes += 1
                            else:
                                color_right_votes += 1
                                
                            if color_left_votes > 3:
                                lane_memory = "Left"
                                print("[LANE MEMORY] Locked LEFT")
                                set_motors(-50, 80)
                                time.sleep(0.5)

                            elif color_right_votes > 3:
                                lane_memory = "Right"
                                print("[LANE MEMORY] Locked RIGHT")
                                set_motors(80, -50)
                                time.sleep(0.5)

                elif follow_color == "Black":
                    if was_on_priority_color:
                        print(f"[LANE EXIT] Exiting Priority Line. Memory: {lane_memory}")
                        with state_lock:
                            if not status["action_in_progress"]:
                                # Drive forward slightly to clear the line
                                set_motors(BASE_SPEED, BASE_SPEED)
                                time.sleep(0.1)
                                
                                if lane_memory == "Left":
                                    # If color line was on Left, Black line is on Right. Turn Right!
                                    set_motors(-50, 80)
                                    time.sleep(0.5)
                                elif lane_memory == "Right":
                                    # If color line was on Right, Black line is on Left. Turn Left!
                                    set_motors(80, -50)
                                    time.sleep(0.5)
                                    
                        # Reset tracking variables completely
                        was_on_priority_color = False
                        lane_memory = None
                        color_left_votes = 0
                        color_right_votes = 0
                        color_history = ["Black"] * CONFIRM_FRAMES

                # --- 3. DRIVING LOGIC ---
                if target_contour is not None:
                    cv2.drawContours(roi, [target_contour], -1, (0, 255, 0), 2)
                    
                    M = cv2.moments(target_contour)
                    if M["m00"] > 0:
                        cx = int(M["m10"]/M["m00"])
                        error = cx - (w//2)
                        
                        if follow_color == "Black" and cv2.contourArea(target_contour) < 2000:
                            with state_lock:
                                status["direction"] = "None"
                                
                        with state_lock:
                            if not status["action_in_progress"]:
                                set_motors(BASE_SPEED + (Kp*error), BASE_SPEED - (Kp*error))
                                last_error = error
                else:
                    # Spin recovery if lost
                    with state_lock:
                        if not status["action_in_progress"]:
                            if last_error > 0: set_motors(80, -50)
                            else: set_motors(-50, 80)

        # ===== DISPLAY =====
        cv2.putText(frame, f"S:{sym} A:{arr}", (10,25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)
        cv2.putText(frame, f"FPS: {int(fps)}", (220, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

        cv2.putText(frame, f"{sym.upper()} DETECTED", (50,120), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 3)

        cv2.imshow("Robot", frame)
        cv2.imshow("Yellow Mask Debug", mask_yellow)
        
        if not in_action:
            cv2.putText(roi, f"Track: {follow_color} ({lane_memory})", (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            cv2.imshow("Line Crop Tracking", roi)

        with shared:
            if shared_roi is not None:
                cv2.imshow("cont", shared_roi)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    running = False
    stop_robot()
    GPIO.cleanup()
    picam2.stop()
    cv2.destroyAllWindows() 




    kernel = np.ones((3,3), np.uint8)
    mask_red = cv2.morphologyEx(mask_red, cv2.MORPH_OPEN, kernel)
    mask_red = cv2.morphologyEx(mask_red, cv2.MORPH_CLOSE, kernel)
    mask_yellow = cv2.morphologyEx(mask_yellow, cv2.MORPH_OPEN, kernel)
    mask_yellow = cv2.morphologyEx(mask_yellow, cv2.MORPH_CLOSE, kernel)