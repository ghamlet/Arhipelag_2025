import time

import serial


class Arduino:  # класс ардуино! При создании объекта, задаёт порт и скорость обмена данными, для общения по UART
    def __init__(self, port, baudrate=9600, timeout=None, coding='utf-8'):
        self.port = port
        self.baudrate = baudrate
        self.timeout = timeout
        self.coding = coding

        self.serial_connected = False
        self.serial = serial.Serial(port, baudrate=baudrate, timeout=timeout, write_timeout=1)

        #self.serial.flush()
        self.serial.reset_input_buffer()
        self.serial.reset_output_buffer()

        self.serial_connected = True

    def waiting(self):
        return self.serial.in_waiting

    def send_data(self, data: str):  # метод класса для отправки данных через UART
        if 'DIST' in data:
            print('Sent to Arduino:', data)
        data += '\n'
        try:
            self.serial.write(data.encode(self.coding))
        except serial.SerialTimeoutException:
            self.serial.reset_input_buffer()
            self.serial.reset_output_buffer()

    def read_data(self):
        line = self.serial.read(self.waiting()).decode(self.coding).strip()
        return line

    def set_speed(self, speed):
        msg = f'*GO:{speed}|'
        self.send_data(msg)

    def set_angle(self, angle):
        msg = f'*ANGLE:{angle}|'
        self.send_data(msg)

    def dist(self, time):
        msg = f'*DIST:{time}|'
        self.send_data(msg)

    def stop(self):
        msg = '*STOP|'
        self.send_data(msg)

    def drop(self):
        msg = '*DROP|'
        self.send_data(msg)

    def check(self):
        msg = '*CHECK|'
        self.send_data(msg)

    def __del__(self):
        if self.serial_connected:
            self.stop()
            time.sleep(1)
            self.serial.close()

