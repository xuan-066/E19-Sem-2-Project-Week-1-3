import RPi.GPIO as GPIO
import cv2
import numpy as np
from picamera2 import Picamera2
import time

# =========================
# MOTOR PINS
# =========================
IN1, IN2, IN3, IN4 = 4, 17, 27, 22
ENA, ENB = 18, 19

BASE_SPEED = 50
Kp = 0.9
correction = 0

# =========================
# GPIO SETUP
# =========================
GPIO.setmode(GPIO.BCM)
GPIO.setwarnings(False)
GPIO.setup([IN1, IN2, IN3, IN4, ENA, ENB], GPIO.OUT)

# PWM 1000Hz
pwmLeft = GPIO.PWM(ENA, 1000)
pwmRight = GPIO.PWM(ENB, 1000)
pwmLeft.start(0)
pwmRight.start(0)

# =========================
# MOTOR FUNCTION
# =========================
def set_motor_speed(left, right):
    # --- LEFT MOTOR ---
    if left >= 0:
        GPIO.output(IN1, GPIO.HIGH)
        GPIO.output(IN2, GPIO.LOW)
    else:
        GPIO.output(IN1, GPIO.LOW)
        GPIO.output(IN2, GPIO.HIGH)

    pwmLeft.ChangeDutyCycle(min(abs(left), 100))

    # --- RIGHT MOTOR ---
    if right >= 0:
        GPIO.output(IN3, GPIO.HIGH)
        GPIO.output(IN4, GPIO.LOW)
    else:
        GPIO.output(IN3, GPIO.LOW)
        GPIO.output(IN4, GPIO.HIGH)

    pwmRight.ChangeDutyCycle(min(abs(right), 100))


def stop_motors():
    set_motor_speed(0, 0)

# =========================
# CAMERA SETUP
# =========================
picam2 = Picamera2()
config = picam2.create_preview_configuration(
    main={"size": (320, 240), "format": "RGB888"}
)
picam2.configure(config)
picam2.start()

print("Line Follower Running... Press 'q' to stop.")

# =========================
# MAIN LOOP
# =========================
try:
    while True:
        frame = picam2.capture_array()
        height, width, _ = frame.shape

        # Bottom 80 pixels ROI
        roi = frame[height-80:height, :]

        # Image processing
        gray = cv2.cvtColor(roi, cv2.COLOR_RGB2GRAY)
        blur = cv2.GaussianBlur(gray, (5,5), 0)
        _, thresh = cv2.threshold(blur, 140, 255, cv2.THRESH_BINARY_INV)

        # Contour detection
        contours, _ = cv2.findContours(
            thresh,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )

        if contours:
            largest = max(contours, key=cv2.contourArea)
            M = cv2.moments(largest)

            if M["m00"] > 0:
                cx = int(M["m10"] / M["m00"])

                cv2.drawContours(roi, [largest], -1, (0,255,0), 2)
                cv2.circle(roi, (cx, 40), 5, (255,0,0), -1)

                center = width // 2
                error = cx - center

                # P CONTROL ONLY
                correction = Kp * error

                left_speed = BASE_SPEED + correction
                right_speed = BASE_SPEED - correction

                set_motor_speed(left_speed, right_speed)

        else:
            # Simple recovery (no delay)
            if correction > 0:
                set_motor_speed(70, -60)
            else:
                set_motor_speed(-60, 70)

        cv2.imshow("ROI", roi)
        cv2.imshow("Threshold", thresh)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    stop_motors()
    pwmLeft.stop()
    pwmRight.stop()
    GPIO.cleanup()
    picam2.stop()
    cv2.destroyAllWindows()