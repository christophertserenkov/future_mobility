import pigpio
from time import sleep

HALL_PIN = 22
last_tick = 0

MOTOR_IN1 = 20
MOTOR_IN2 = 21

def hall_callback(gpio, level, tick):
    global last_tick
    dt_us = pigpio.tickDiff(last_tick, tick)
    speed = float(float(1 / float(dt_us / 60000000)) / 3)

    print(f"Pin: {gpio}, Tase: {level}, Aeg (µs): {tick}, Speed: (rpm) {speed}")
    last_tick = tick


pi = pigpio.pi()
pi.set_mode(HALL_PIN, pigpio.INPUT)
pi.callback(HALL_PIN, pigpio.FALLING_EDGE, hall_callback)

pi.set_mode(MOTOR_IN1, pigpio.OUTPUT)
pi.set_mode(MOTOR_IN2, pigpio.OUTPUT)

pi.set_PWM_frequency(MOTOR_IN1, 1000)
pi.set_PWM_frequency(MOTOR_IN2, 1000)

pi.set_PWM_dutycycle(MOTOR_IN1, 255)
pi.set_PWM_dutycycle(MOTOR_IN2, 0)

sleep(10)

pi.set_PWM_dutycycle(MOTOR_IN1, 0)
pi.set_PWM_dutycycle(MOTOR_IN2, 0)

pi.stop()