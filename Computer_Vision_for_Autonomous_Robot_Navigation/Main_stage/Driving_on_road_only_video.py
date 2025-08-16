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





# Добавлена логическая переменная для управления записью видео
RECORD_VIDEO = True  # Установите False для отключения записи видео


THRESHOLD = 220
CAMERA_ID = 0

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

# cap = find_camera(fourcc="MJPG", frame_width=1280, frame_height=720)
cap = cv2.VideoCapture(CAMERA_ID)
# cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
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
        codec='avc1',
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
    
    #end_frame = time.time()
    if not ret:
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # перемотка на начало
        continue
        # break

    frame = frame[-720:, :]



    # Запись кадра, если запись видео включена
    if RECORD_VIDEO and video_orig is not None:
        video_orig.record_frame(frame)


    mini_frame = cv2.resize(frame, SIZE)  # Масштабируем изображение
    gray = cv2.cvtColor(mini_frame, cv2.COLOR_BGR2GRAY)  # Переводим изображение в чёрно-белое с градациями серого
    binary = cv2.inRange(gray, THRESHOLD, 255)  # Бинаризуем по порогу, должны остаться только белые линии разметки
    cv2.imshow("binary", binary)

    # Оставляем только интересующую нас область перед колёсами автомобиля. Преобразуем её из трапеции в прямоугольник
    perspective = trans_perspective(binary, TRAP, RECT, SIZE, d=1)
    left, right = find_lines(perspective, d=1)  # Вычисляем координаты левой и правой линий дорожной разметки
    center_img = perspective.shape[1] // 2  # Х координата центра изображения
    err = 0 - ((left + right) // 2 - center_img)  # Вычисляем ошибку - отклонение центра дороги от центра кадра
    err = -err # Инвертирование направления поворота колёс

    # Определяем угол поворота колёс в зависимости от ошибки.
    angle = int(90 + KP * err + KD * (err - last_err))
    last_err = err  # Запоминаем какая была ошибка в этой итерации цикла
    angle = min(max(65, angle), 115)  # не позволяем углу принимать значения за пределами интервала [65, 115]
    # print("Angle:", angle)


    end_time = time.time()

   
    cv2.waitKey(10)