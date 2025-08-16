import atexit
import time
import random
from pathlib import Path

import cv2
import numpy as np
# import yolopy

from arduino import Arduino
from func import *

from video_recorder import  VideoRecorder
# from track_bars import ColorTracker



# Добавлена логическая переменная для управления записью видео
RECORD_VIDEO = False  # Установите False для отключения записи видео

DIST_METER = 1825  # ticks to finish 1m
CAR_SPEED = 1675
THRESHOLD = 220
CAMERA_ID = '/dev/video0'
ARDUINO_PORT = '/dev/ttyUSB0' #Kvant
# ARDUINO_PORT = '/dev/ttyACM0' #Mega

SIZE = (533, 300)  # размер изображения, с которым будет работать алгоритм обнаружения дорожной разметки

# Список с точками трапеции, ограничивающей область перед колёсами, для поиска линий дорожной разметки
# Форму трапеции необходимо подбирать под конкретную камеру
TRAP = np.float32([[10, 299],
                   [523, 299],
                   [440, 200],
                   [93, 200]])
# Та же трапеция, только для функции отрисовки многоугольников. Она требует тип int.
src_draw = np.array(TRAP, dtype=np.int32)

# Список с углами прямоугольника, в который преобразуется трапеция
RECT = np.float32([[0, SIZE[1]],
                   [SIZE[0], SIZE[1]],
                   [SIZE[0], 0],
                   [0, 0]])

# Параметры ПИД-регулятора для движения по разметке
KP = 0.35
KD = 0.15

arduino = None
video_orig = None


@atexit.register
def exit_func(*args):
    if arduino is not None:
        arduino.stop()
    if video_orig is not None:
        video_orig.close()
    # cv2.destroyAllWindows()


arduino = Arduino(ARDUINO_PORT, baudrate=115_200, timeout=0.1)
# arduino = FakeArduino(debug=False)
time.sleep(2)
print("Arduino port:", arduino.port)

# cap = find_camera(fourcc="MJPG", frame_width=1280, frame_height=720)
cap = cv2.VideoCapture(CAMERA_ID, cv2.CAP_V4L2)
cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
# cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
# cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 960)

if not cap.isOpened():
    print('[ERROR] Cannot open camera ID:', CAMERA_ID)
    quit()

# Инициализация видеорекордера, если запись видео включена
if RECORD_VIDEO:
    video_orig = VideoRecorder(
        camera_id=CAMERA_ID,
        output_dir="recordings",
        codec='MJPG',
        show_preview=False
    )
    video_orig.start_recording()



# wait for stable white balance
for i in range(30):
    ret, frame = cap.read()

# arduino.set_speed(CAR_SPEED)
last_err = 0

while True:
    start_time = time.time()
    ret, frame = cap.read()
    # Для обнаружения разметки берём только нижние 720 строк кадра
    frame = frame[-720:, :]

    
    #end_frame = time.time()
    if not ret:
        break



    # Запись кадра, если запись видео включена
    if RECORD_VIDEO and video_orig is not None:
        video_orig.record_frame(frame)

    mini_frame = cv2.resize(frame, SIZE)  # Масштабируем изображение
    gray = cv2.cvtColor(mini_frame, cv2.COLOR_BGR2GRAY)  # Переводим изображение в чёрно-белое с градациями серого
    binary = cv2.inRange(gray, THRESHOLD, 255)  # Бинаризуем по порогу, должны остаться только белые линии разметки
    # cv2.imshow("binary", binary)

    # Оставляем только интересующую нас область перед колёсами автомобиля. Преобразуем её из трапеции в прямоугольник
    perspective = trans_perspective(binary, TRAP, RECT, SIZE)
    left, right = find_lines(perspective)  # Вычисляем координаты левой и правой линий дорожной разметки
    center_img = perspective.shape[1] // 2  # Х координата центра изображения
    err = 0 - ((left + right) // 2 - center_img)  # Вычисляем ошибку - отклонение центра дороги от центра кадра
    err = -err # Инвертирование направления поворота колёс

    # Определяем угол поворота колёс в зависимости от ошибки.
    angle = int(90 + KP * err + KD * (err - last_err))
    last_err = err  # Запоминаем какая была ошибка в этой итерации цикла
    angle = min(max(65, angle), 115)  # не позволяем углу принимать значения за пределами интервала [65, 115]
    # print("Angle:", angle)

    arduino.set_speed(CAR_SPEED)
    arduino.set_angle(angle)

    end_time = time.time()

    fps = 1 / (end_time - start_time)
    if fps < 10:
        print(f'[WARNING] FPS is too low! ({fps:.1f} fps)')

