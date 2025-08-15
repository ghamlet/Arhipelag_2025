import cv2
import math
import numpy as np
import pandas as pd
from collections import defaultdict


def create_color_masks(image):
    
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    
    # Определение диапазонов цветов в HSV
    # Красный (два диапазона из-за разрыва near 0°)
    lower_red1 = np.array([0, 100, 100])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([160, 100, 100])
    upper_red2 = np.array([180, 255, 255])
    
    # Зелёный
    lower_green = np.array([40, 50, 50])
    upper_green = np.array([80, 255, 255])
    
    # Синий
    lower_blue = np.array([90, 50, 50])
    upper_blue = np.array([130, 255, 255])
    
    # Создание масок
    mask_red1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask_red2 = cv2.inRange(hsv, lower_red2, upper_red2)
    mask_red = cv2.bitwise_or(mask_red1, mask_red2)
    
    mask_green = cv2.inRange(hsv, lower_green, upper_green)
    mask_blue = cv2.inRange(hsv, lower_blue, upper_blue)
    
    # Поиск контуров для каждого цвета
    color_masks = {
        "red": mask_red,
        "green": mask_green,
        "blue": mask_blue
    }
    

    return color_masks




def detect_color_rods(color_masks: dict):
    """
    Обнаруживает цветные стержни и возвращает их точки в формате [((x,y), 'color'), ...]
    
    Параметры:
        color_masks: словарь цветовых масок {'color_name': mask}
        image: изображение для анализа
        
    Возвращает:
        list: [((x1, y1), 'color1'), ((x2, y2), 'color2'), ...] - отсортировано по X
        image: изображение с визуализацией
    """


    # result = image.copy()
    color_points = defaultdict(list)  # Временное хранилище по цветам
    
    # Цвета для рисования контуров (BGR формат)
    contour_colors = {
        'red': (0, 0, 255),
        'green': (0, 255, 0),
        'blue': (255, 0, 0)
    }
    
    for color_name, mask in color_masks.items():
        contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        current_color = contour_colors.get(color_name, (255, 255, 255))
        
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area > 5:
                # Находим нижнюю центральную точку
                bottom_y = max(pt[0][1] for pt in cnt)
                bottom_pts = [pt[0] for pt in cnt if pt[0][1] == bottom_y]
                
                if bottom_pts:
                    avg_x = int(np.mean([pt[0] for pt in bottom_pts]))
                    bottom_center = (avg_x, bottom_y)
                    color_points[color_name].append(bottom_center)
                    
    

    # Собираем все точки в один список и сортируем по X
    all_points = []
    for color, points in color_points.items():
        for pt in sorted(points, key=lambda p: p[0]):
            all_points.append((pt, color))
    
    # Дополнительная сортировка на случай, если цвета перемешаны
    all_points.sort(key=lambda item: item[0][0])    
    
    return all_points






def find_closest_line(point, etalon_frames):
   

    frame_h, frame_v = etalon_frames
    result = {"vertical": None, "horizontal": None}
    

   
    for line_type in ["horizontal", "vertical"]:
        frame = frame_h if line_type == "horizontal" else frame_v
        
        binary = cv2.inRange(frame, (240, 240, 240), (255, 255, 255))
        contours, _ = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE) # CHAIN_APPROX_SIMPLE для прямых будет хранить только две точки в контуре 
        if not contours:
            continue

        if line_type == "vertical":
        # Сортируем контуры по X (левая граница каждого контура)
            contours = sorted(contours, key=lambda c: cv2.boundingRect(c)[0])

        elif line_type == "horizontal":
                # Сортируем контуры по Y (верхняя граница) ОТ БОЛЬШЕГО К МЕНЬШЕМУ (сверху вниз)
            contours = sorted(contours, key=lambda c: cv2.boundingRect(c)[1], reverse=True)


        
        min_distance = float('inf')
        best_idx = 0

        for (j, cnt) in enumerate(contours):
            points = cnt.reshape(-1, 2)
            dists = np.linalg.norm(points - np.array(point), axis=1)

            if dists.min() < min_distance:
                min_distance = dists.min()
                best_idx = j  # индекс контура с ближайшей точкой

        
        result[line_type] = best_idx
        
        
    return result




def get_real_distances(line_indices):
    """
    Преобразует индексы линий в реальные расстояния в см.
    
    Параметры:
        line_indices: словарь {'vertical': индекс, 'horizontal': индекс}
        
    Возвращает:
        Словарь {'vertical': расстояние_см, 'horizontal': расстояние_см}
    """


    # Параметры разметки
    VERTICAL_LINES = 21
    VERTICAL_LENGTH = 800  # см
    VERTICAL_REFERENCE = 10  # Нулевая точка для вертикальных линий (индекс 10)
    
    HORIZONTAL_LINES = 19
    HORIZONTAL_LENGTH = 600  # см
    HORIZONTAL_REFERENCE = -1  # Нулевая точка - последняя линия (низ)
    
    # Рассчитываем расстояния между линиями
    vertical_step = VERTICAL_LENGTH / VERTICAL_LINES +1
    horizontal_step = HORIZONTAL_LENGTH / HORIZONTAL_LINES +1

    sign = None


    # Вычисляем расстояния от референсных линий

    vertical_segments = line_indices['vertical'] - VERTICAL_REFERENCE
    vertical_dist = vertical_segments * vertical_step

    horizontal_segments = line_indices['horizontal'] - HORIZONTAL_REFERENCE
    horizontal_dist = horizontal_segments * horizontal_step

    # print(vertical_segments, horizontal_segments)
    # print(vertical_dist, horizontal_dist)
    # print()

    if vertical_dist < 0:
        sign = "-"
        vertical_dist = abs(vertical_dist)
    else:
        sign = "+"


    
    return {
        'vertical': round(vertical_dist, 2),
        'horizontal': round(horizontal_dist, 2),
        "sign": sign
        
    }



def calculate_angle(distances):
    """
    Вычисляет угол по двум катетам с учетом знака.
    
    Параметры:
        distances: словарь {
            'vertical': вертикальный катет (см),
            'horizontal': горизонтальный катет (см),
            'sign': знак ('+' или '-')
        }
        
    Возвращает:
        Угол в градусах (от -90 до 90) с учетом знака
    """

    vertical = distances['vertical']
    horizontal = distances['horizontal']
    sign = distances['sign']
    
    # Вычисляем тангенс угла (противоположный катет / прилежащий)
    if horizontal == 0:  # избегаем деления на 0
        angle_rad = math.pi / 2  # 90 градусов
    else:
        angle_rad = math.atan(abs(vertical) / abs(horizontal))
    
    # Переводим радианы в градусы
    angle_deg = math.degrees(angle_rad)
    
    # Применяем знак
    if sign == '-':
        angle_deg = -angle_deg
    
    return round(angle_deg)



def preliminary_operations():
    """ В этой функции вы можете выполнить любые операции, которые необходимо сделать лишь единожды.
        Например, можно создать эталонное изображение, с которым будут сравниваться изображения в основном алгоритме.

        Эта функция должна возвращать список! Этот список будет передаваться в основную функцию при каждом её вызове.
        Содержимое списка вы определяете самостоятельно.

        Если вы не собираетесь использовать эту функцию, пусть возвращает пустой список [].
    """

    # TODO: Отредактируйте функцию по своему усмотрению.
    # Изображения - эталоны и файл eval.py упакуйте в архив "*.zip" и загружайте на онлайн платформу в качестве решения.

    frame_horizontal = cv2.imread("Computer_Vision_for_Autonomous_Robot_Navigation/user_task/materials/horizontal_marking.jpg")
    frame_vertical = cv2.imread("Computer_Vision_for_Autonomous_Robot_Navigation/user_task/materials/vertical_marking.jpg")



    etalon_frames = [frame_horizontal, frame_vertical]

    return etalon_frames






def predict_color_and_angle(image, etalon_frames):
    """
         Функция, детектирующая цветные объекты и определяющая их угловое отклонение от центра кадра
         Входные данные: изображение (bgr), список из функции preliminary_operations.
         Выходные данные: список с парами «цвет-угол отклонения»
                          Пары должны быть представлены в порядке расположения объектов в кадре слева направо.


         Формат вывода: [["red", -45], ["green", 10], ["blue", 25]]

         Объекты в левой половине изображения имеют отрицательный угол отклонения, в правой половине — положительный.

    """

    color_masks = create_color_masks(image) 
    bottom_points = detect_color_rods(color_masks)

    answer = []

    for el in bottom_points:
        point_coords, color = el

        line_indices = find_closest_line(point_coords, etalon_frames)
        distances = get_real_distances(line_indices)
        angle = calculate_angle(distances)

        answer.append([color, angle])

    return answer


