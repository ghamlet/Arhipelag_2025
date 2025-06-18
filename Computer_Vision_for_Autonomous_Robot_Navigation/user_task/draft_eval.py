import cv2
import os
import numpy as np
import math
import pandas as pd


# Фиксированные точки (можно изменить)
FIXED_POINT1 = (400, 600)  # (x, y)
FIXED_POINT2 = (400, 0)  # (x, y)



def apply_perspective_transform(image)-> np.ndarray: 
    """
    Применяет перспективное преобразование к изображению, 
    преобразуя трапециевидную область в прямоугольную.
    
    Параметры:
        image (np.ndarray): Исходное изображение (BGR или Grayscale).
        
    Возвращает:
        np.ndarray: Изображение после перспективного преобразования.
    """


    NEW_SIZE_X = 800
    NEW_SIZE_Y = 600

    pts2 = np.float32([[0,0],
                    [NEW_SIZE_X,0],
                    [0,NEW_SIZE_Y],
                    [NEW_SIZE_X,NEW_SIZE_Y]
                    ])     # координаты прошлых точек на новом изображении после трансформации


    
    trapezoid_points =  [[300, 300], [500, 300], [0, NEW_SIZE_Y], [NEW_SIZE_X, NEW_SIZE_Y]]
    pts1 = np.float32(trapezoid_points)
    
    # Получаем матрицу перспективного преобразования
    perspective_matrix = cv2.getPerspectiveTransform(pts1, pts2)
    
    # Применяем преобразование
    warped_image = cv2.warpPerspective(image, perspective_matrix, (NEW_SIZE_X, NEW_SIZE_Y))
    
    return warped_image



def calculate_angles(points_list):
    """
    Вычисляет углы для каждой точки относительно статичных точек
    
    Параметры:
        points_list: список точек в формате [((x, y), color), ...]
        
    Возвращает:
        Список углов в формате [["color", angle], ...]
    """
   
    
  
    angles = []
    
    for (point, color) in points_list:
        
        
        # Вычисляем угол между static_p1, p, static_p2
        v1 = (point[0] - FIXED_POINT1[0], point[1] - FIXED_POINT1[1])
        v2 = (FIXED_POINT2[0] - FIXED_POINT1[0], FIXED_POINT2[1] - FIXED_POINT1[1])
        
        # Скалярное произведение
        dot_product = v1[0] * v2[0] + v1[1] * v2[1]
        
        # Модули векторов
        mod_v1 = math.sqrt(v1[0]**2 + v1[1]**2)
        mod_v2 = math.sqrt(v2[0]**2 + v2[1]**2)
        
        # Косинус угла
        cos_angle = dot_product / (mod_v1 * mod_v2)
        cos_angle = np.clip(cos_angle, -1.0, 1.0)
        
        # Угол в градусах
        angle = math.degrees(math.acos(cos_angle))
        
        # Эмпирическая поправка
        angle = int(angle - 2)
        
        # Определение знака угла
        if point[0] < 400:  # Точка в левой половине
            angle = -angle
        
        angles.append([color, angle])
    
    return angles




def extract_object_list(row):
    object_list = []
    for i in range(4):
        color = row[2+i*2]  # 2 4 6 8
        # print(color)
        angle = row[3+i*2]  # 3 5 7 9
        # print(angle)
        if color != "empty":
            angle = int(angle)
            object_list.append([color, angle])
        # print()

    return object_list



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
    Обнаруживает цветные стержни и отмечает их нижние центральные точки
    
    Параметры:
        color_masks: словарь цветовых масок {'color_name': mask}
        image: изображение для рисования результатов
        
    Возвращает:
        Список точек в формате [(x, y, color)]
    """

    result = warped_image.copy()
    bottom_points = []
    
    for color_name, mask in color_masks.items():
        contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area > 5:  # Фильтр по площади
                # Находим самую нижнюю точку контура
                bottom_y = max(pt[0][1] for pt in cnt)
                
                # Находим все точки с этой Y-координатой
                bottom_pts = [pt[0] for pt in cnt if pt[0][1] == bottom_y]
                
                # Вычисляем среднее значение X для этих точек
                if bottom_pts:
                    avg_x = int(np.mean([pt[0] for pt in bottom_pts]))
                    bottom_center = (avg_x, bottom_y)
                    
                    # Сохраняем точку
                    bottom_points.append((bottom_center, color_name))
                    
                    # # Рисуем точку и линию до низа (для наглядности)
                    # cv2.circle(result, bottom_center, 1, (44, 0, 255), -1)
                   
                    
                    # # Подписываем цвет
                    # cv2.putText(result, color_name, (bottom_center[0]-20, bottom_center[1]-10),
                    #            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    


    # Сортировка точек по X координате
    bottom_points.sort(key=lambda point: point[0][0])
    
    # cv2.imshow("res", result)
 
    return bottom_points


        
    



if __name__ == "__main__":
    MAIN_DIR = "Computer_Vision_for_Autonomous_Robot_Navigation/user_task/"
    data = pd.read_csv(MAIN_DIR + "annotations.csv", sep=';')
    
    DIR_IMAGES = MAIN_DIR + "images"



    for row in data.itertuples():
        #        path_to_img = os.path.join(DIR_IMAGES, img) 

        image = cv2.imread(MAIN_DIR + row[1])
        if image is None:
            print(f"Ошибка загрузки: {row[1]}")
            continue

        # правильный ответ
        true_obj_list = extract_object_list(row)
        print("True: ", true_obj_list)
        print()



        warped_image = apply_perspective_transform(image) # получаем перспективу

        color_masks = create_color_masks(warped_image)  
        bottom_points = detect_color_rods(color_masks)



        angles = calculate_angles(bottom_points)

            # Вывод результатов
        # cv2.waitKey(0) 
