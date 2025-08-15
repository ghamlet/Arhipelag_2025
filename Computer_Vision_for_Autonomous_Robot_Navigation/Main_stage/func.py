# coding: utf-8
import cv2
import numpy as np

"""Вспомогательные функции для алгоритма детектирования разметки.
   Рядом должен находится файл с основным алгоритмом."""

# Функция для пространственного преобразования изображения в форме трапеции в прямоугольное изображение.
# Трапеция - область перед колёсами, которая нас интересует
# Прямоугольник - та же область, но под другим углом обора.
def trans_perspective(binary, trap, rect, size, d=0):
    matrix_trans = cv2.getPerspectiveTransform(trap, rect)
    perspective = cv2.warpPerspective(binary, matrix_trans, size, flags=cv2.INTER_LINEAR)
    if d:
        cv2.imshow('Perspective', perspective)
    return perspective


# Функция для определения координат левой и правой линий разметки.
def find_lines(perspective, d=0):
    hist = np.sum(perspective, axis=0)
    h, w = perspective.shape[:2]
    if d:
        cv2.imshow("Perspektiv2in", perspective)

    mid = hist.shape[0] // 2
    i = 0
    centre = 0
    sum_mass = 0
    while (i <= mid):
        centre += hist[i] * (i + 1)
        sum_mass += hist[i]
        i += 1

    if sum_mass > 0:
        mid_mass_left = centre / sum_mass
        if abs(mid - mid_mass_left) < 0.05 * w:
            left_found = False
        else:
            left_found = True
    else:
        left_found = False

    if not left_found:
        mid_mass_left = w // 3

    i = mid
    centre = 0
    sum_mass = 0
    while i < hist.shape[0]:
        centre += hist[i] * (i + 1)
        sum_mass += hist[i]
        i += 1
    if sum_mass > 0:
        mid_mass_right = centre / sum_mass
        if abs(mid - mid_mass_right) < 0.05 * w:
            right_found = False
        else:
            right_found = True
    else:
        right_found = False

    if not right_found:
        mid_mass_right = w - 1

    mid_mass_right = min(w - 1, mid_mass_right)
    mid_mass_left = int(mid_mass_left)
    mid_mass_right = int(mid_mass_right)

    if d:
        cv2.line(perspective, (mid_mass_left, 0), (mid_mass_left, perspective.shape[1]), 50, 2)
        cv2.line(perspective, (mid_mass_right, 0), (mid_mass_right, perspective.shape[1]), 50, 2)
        # cv2.line(perspective, ((mid_mass_right + mid_mass_left) // 2, 0), ((mid_mass_right + mid_mass_left) // 2, perspective.shape[1]), 110, 3)
        cv2.imshow('CentrMass', perspective)

    return mid_mass_left, mid_mass_right
