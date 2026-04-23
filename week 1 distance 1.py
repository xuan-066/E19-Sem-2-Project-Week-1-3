import RPi.GPIO as GPIO
from gpiozero import DigitalInputDevice
from time import sleep
from math import pi
# ================= GPIO MODE =================
GPIO.setmode(GPIO.BCM)
GPIO.setwarnings(False)

# ================= MOTOR DRIVER PINS =================
IN1 = 4
IN2 = 17
IN3 = 27
IN4 = 22
ENA = 18
ENB = 19

# ================= ENCODER PINS =================
LEFT_ENCODER_PIN = 5
RIGHT_ENCODER_PIN = 6

# ================= ENCODER PARAMETERS =================
WHEEL_DIAMETER = 0.065      # meters
PULSES_PER_REV = 20

wheel_circumference = pi * WHEEL_DIAMETER
distance_per_pulse = wheel_circumference/PULSES_PER_REV

# ================= COUNTERS =================
left_count = 0
right_count = 0

# ================= GPIO SETUP =================
GPIO.setup([IN1, IN2, IN3, IN4, ENA, ENB], GPIO.OUT)

# ================= PWM SETUP =================
pwm_left = GPIO.PWM(ENA, 1000)
pwm_right = GPIO.PWM(ENB, 1000)

pwm_left.start(0)
pwm_right.start(0)

# ================= ENCODER SETUP (gpiozero) =================
left_encoder = DigitalInputDevice(LEFT_ENCODER_PIN, pull_up=True)
right_encoder = DigitalInputDevice(RIGHT_ENCODER_PIN, pull_up=True)

def left_pulse():
    global left_count
    left_count += 1

def right_pulse():
    global right_count
    right_count += 1

left_encoder.when_activated = left_pulse
right_encoder.when_activated = right_pulse

# ================= MOTOR FUNCTIONS =================
def set_speed(left_speed, right_speed):
    pwm_left.ChangeDutyCycle(left_speed)
    pwm_right.ChangeDutyCycle(right_speed)
def reverse(speed):
    GPIO.output(IN1, GPIO.LOW)
    GPIO.output(IN2, GPIO.HIGH)
    GPIO.output(IN3, GPIO.LOW)
    GPIO.output(IN4, GPIO.HIGH)
    set_speed(speed, speed)
    
def forward(speed):
    GPIO.output(IN1, GPIO.HIGH)
    GPIO.output(IN2, GPIO.LOW)
    GPIO.output(IN3, GPIO.HIGH)
    GPIO.output(IN4, GPIO.LOW)
    set_speed(speed, speed)

def stop():
    GPIO.output(IN1, GPIO.LOW)
    GPIO.output(IN2, GPIO.LOW)
    GPIO.output(IN3, GPIO.LOW)
    GPIO.output(IN4, GPIO.LOW)
    set_speed(0, 0)

# ================= MAIN PROGRAM =================
try:
    print("Robot moving forward...")

    # Reset encoder counts
    left_count = 0
    right_count = 0

    #t=0.5*2.012
    #print("Reverse")
    reverse(75.2)
    sleep(2)
    #forward(75.2)
    #sleep(2)      # Move for 2 seconds
    stop()

    # ================= DISTANCE CALCULATION ================
    #correctionfactor=0.9605
    left_distance = left_count * distance_per_pulse
    right_distance = right_count * distance_per_pulse
    avg_distance = (left_distance + right_distance) / 2
    #distance=avg_distance*correctionfactor

    print("Left pulses:", left_count)
    print("Right pulses:", right_count)
    #print(f"Left distance: {left_distance:.3f} meters")
    #print(f"Right distance: {right_distance:.3f} meters")
    print(f"Distance travelled: {avg_distance:.3f} meters")

except KeyboardInterrupt:
    print("Program stopped by user")

finally:
    stop()
    pwm_left.stop()
    pwm_right.stop()
    GPIO.cleanup()
