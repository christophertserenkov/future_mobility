from time import time, sleep
from hardware import HardWare

SERVO  = 18 
MOTOR_IN1 = 20
MOTOR_IN2 = 21
HALL_PIN = 22

class PID:
    KP = 0.2
    KI = 1.2
    KD = 0.02

    MAX_CONTROL_INF = 100

    def __init__(self):
        self.__prev_calc = time()
        self.__prev_err = 0
        self.__integral = 0

    def correct(self, input, setpoint=0) -> float:
        err = setpoint - input
        dt = self.__getPassedTime()

        self.__integral += (err + self.__prev_err) / 2 * dt  # trapetsvalem
        self.__integral = max(min(self.__integral, self.MAX_CONTROL_INF), -self.MAX_CONTROL_INF)
        derivative = (err - self.__prev_err) / dt  # tuletis sammuga taha

        self.__prev_err = err
        correction = self.KP * err + self.KI * self.__integral + self.KD * derivative
        correction = max(min(correction, self.MAX_CONTROL_INF), 0)
        return correction

    def __getPassedTime(self) -> float:
        dt = time() - self.__prev_calc
        self.__prev_calc = time()
        return dt

control = HardWare(MOTOR_IN1, MOTOR_IN2, SERVO, HALL_PIN)
pid = PID()

start_time = time()
last_calculation = time()
last_speed = 0

while True:
    speed = control.getSpeed()
    print(f"Speed: {round(speed, 1)}")
    if speed != last_speed or time() - last_calculation > 0.25:
        duty_cycle = pid.correct(speed, 150)
        print(f"Duty cycle: {round(duty_cycle)}")
        control.setMotor(duty_cycle)

    last_speed = speed
    sleep(0.05)
    if time() - start_time > 7:
        control.stop()
        break