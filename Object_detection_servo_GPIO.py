import RPi.GPIO as GPIO
from time import sleep


GPIO.setmode(GPIO.BOARD)
GPIO.setwarnings(False)
GPIO.setup(3, GPIO.OUT)
pwm = GPIO.PWM(3, 50)
pwm.start(0)
GPIO.setwarnings(False)
#pwm.ChangeDutyCycle(0)

def SetPW(pulse_width):
    
    duty = pulse_width/20
    GPIO.output(3, True)
    duty = pulse_width/20 *100
    pwm.ChangeDutyCycle(duty)
    #print(str(duty))
    sleep(0.5)
    GPIO.output(3, False)
    pwm.ChangeDutyCycle(0)
    
    
"""
while True:
    #sleep(1)
    #pwm.start(0)
    SetPW(2.2) # overflow
    SetPW(0)
    print("overflow")
    sleep(1)
    SetPW(1.8)
    SetPW(0)
    print("fork")
    sleep(1)
    SetPW(1.5)
    SetPW(0)
    print("knife")
    sleep(1)
    SetPW(1.2)
    print("spoon")
    sleep(1)
"""


pwm.stop()
GPIO.cleanup()