import pigpio
from time import sleep

class HardWare:
    def __init__(self, dc_in1, dc_in2, servo, hall_pin):
        self.dc_in1 = dc_in1
        self.dc_in2 = dc_in2
        self.servo = servo
        self.hall_pin = hall_pin
        self.speed = 0
        self.last_tick = 0
        self.max_pwm = 255

        self.pi = pigpio.pi()

        self.pi.set_mode(self.dc_in1, pigpio.OUTPUT)
        self.pi.set_mode(self.dc_in2, pigpio.OUTPUT)

        self.pi.set_PWM_frequency(self.dc_in1, 1000)
        self.pi.set_PWM_frequency(self.dc_in2, 1000)

        self.pi.set_mode(self.hall_pin, pigpio.INPUT)
        self.pi.callback(self.hall_pin, pigpio.FALLING_EDGE, self.__hall_callback)

    def setMotor(self, percent):
        percent = max(-100, min(100, percent))
        if percent >= 0:
            pwm_value = int(float(percent / 100) * self.max_pwm)
            self.pi.set_PWM_dutycycle(self.dc_in1, pwm_value)
            self.pi.set_PWM_dutycycle(self.dc_in2, 0)
        else:
            pwm_value = int(float(abs(percent) / 100) * self.max_pwm)
            self.pi.set_PWM_dutycycle(self.dc_in1, 0)
            self.pi.set_PWM_dutycycle(self.dc_in2, pwm_value)
    
    def setServo(self, turn_rate):
        turn_rate = max(-100, min(100, turn_rate))
        servo_signal = int(1500 + (float(turn_rate / 100) * 700))
        self.pi.set_servo_pulsewidth(self.servo, servo_signal)

    def getSpeed(self):
        return self.speed

    def __hall_callback(self, gpio, level, tick):
        dt_us = pigpio.tickDiff(self.last_tick, tick)
        self.speed = float(float(1 / float(dt_us / 60000000)) / 3)

        # print(f"Pin: {gpio}, Tase: {level}, Aeg (µs): {tick}, Speed: (rpm) {self.speed}")
        self.last_tick = tick

    def stop(self):
        self.setMotor(0)
        self.setServo(0)
        self.pi.stop()



