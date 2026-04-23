import RPi.GPIO as GPIO
import time

# ================= GPIO MODE =================
GPIO.setmode(GPIO.BCM)
GPIO.setwarnings(False)

# ================= L298N PINS =================
IN1 = 4     # Left motor direction
IN2 = 17
IN3 = 27    # Right motor direction
IN4 = 22

ENA = 18    # PWM speed control (Left)
ENB = 19    # PWM speed control (Right)

# ================= PIN SETUP =================
motor_pins = [IN1, IN2, IN3, IN4, ENA, ENB]
for pin in motor_pins:
    GPIO.setup(pin, GPIO.OUT)

# ================= PWM SETUP =================
pwm_left = GPIO.PWM(ENA, 1000)   # 1 kHz frequency
pwm_right = GPIO.PWM(ENB, 1000)

pwm_left.start(0)
pwm_right.start(0)

# ================= MOTOR FUNCTIONS =================
def set_speed(left_speed, right_speed):
    pwm_left.ChangeDutyCycle(left_speed)
    pwm_right.ChangeDutyCycle(right_speed)


def turn_left(speed=70):
    GPIO.output(IN1, GPIO.LOW)
    GPIO.output(IN2, GPIO.HIGH)     # Left motors stop
    GPIO.output(IN3, GPIO.HIGH)
    GPIO.output(IN4, GPIO.LOW)
    set_speed(speed, speed)

def turn_right(speed=70):
    GPIO.output(IN1, GPIO.HIGH)
    GPIO.output(IN2, GPIO.LOW)
    GPIO.output(IN3, GPIO.LOW)
    GPIO.output(IN4, GPIO.HIGH)     # Right motors stop
    set_speed(speed, speed)

def stop():
    GPIO.output(IN1, GPIO.LOW)
    GPIO.output(IN2, GPIO.LOW)
    GPIO.output(IN3, GPIO.LOW)
    GPIO.output(IN4, GPIO.LOW)
    set_speed(0, 0)
    
 
    

# ================= MAIN TEST SEQUENCE =================

try:
    
    turn = int(input("Enter your choice"))
    
    if turn == 1:
        t = 45*0.0085
        print("Turn Left")
        turn_left(70)
        time.sleep(t)
    
    else:
        s =45*0.0080
        print("Turn Right")
        turn_right(70)
        time.sleep(s)

    print("Stop")
    stop()

except KeyboardInterrupt:
    print("Program interrupted")

finally:
    stop()
    pwm_left.stop()
    pwm_right.stop()
GPIO.cleanup()
