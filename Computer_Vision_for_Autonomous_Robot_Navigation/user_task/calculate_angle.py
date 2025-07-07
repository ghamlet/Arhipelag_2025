import cv2
import numpy as np
import math
import pandas as pd

# Глобальные переменные для хранения точек
points = []
image = None
image_copy = None
dst = None

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


    
    trapezoid_points =  [[310, 300], [490, 300], [0, NEW_SIZE_Y], [NEW_SIZE_X, NEW_SIZE_Y]]
    pts1 = np.float32(trapezoid_points)
    
    # Получаем матрицу перспективного преобразования
    perspective_matrix = cv2.getPerspectiveTransform(pts1, pts2)
    
    # Применяем преобразование
    warped_image = cv2.warpPerspective(image, perspective_matrix, (NEW_SIZE_X, NEW_SIZE_Y))
    
    return warped_image



def calculate_angle(p1, p2, p3):
    """
    Вычисляет угол между векторами p1p2 и p2p3 в градусах
    """
    
    # Вектор 1: от p1 к p2
    v1 = (p1[0] - p2[0], p1[1] - p2[1])
    # Вектор 2: от p3 к p2
    v2 = (p3[0] - p2[0], p3[1] - p2[1])
    
    # Скалярное произведение
    dot_product = v1[0] * v2[0] + v1[1] * v2[1]
    
    # Модули векторов
    mod_v1 = math.sqrt(v1[0]**2 + v1[1]**2)
    mod_v2 = math.sqrt(v2[0]**2 + v2[1]**2)
    
    # Косинус угла
    cos_angle = dot_product / (mod_v1 * mod_v2)
    
    # Защита от ошибок округления
    cos_angle = np.clip(cos_angle, -1.0, 1.0)
    
    # Угол в градусах
    angle = math.degrees(math.acos(cos_angle))

    # опытным путем
    angle = angle -2

     # Определение знака угла
    center_x = 400
    if p1[0] < center_x:  # Точка в левой половине
        angle = -angle
    
    return angle




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



def mouse_callback(event, x, y, flags, param):
    global points, dst, image_copy
    
    if event == cv2.EVENT_LBUTTONDOWN and len(points) < 1:
        points = [(x, y)]  # Сохраняем только последнюю точку
        
        # Обновляем изображение
        dst = image_copy.copy()
        
        # Рисуем все три точки (1 ручная + 2 фиксированные)
        all_points = [points[0], FIXED_POINT1, FIXED_POINT2]
        for pt in all_points:
            cv2.circle(dst, pt, 5, (0, 0, 255), -1)
        
        # Рисуем линии и вычисляем угол
        if len(all_points) == 3:
            cv2.line(dst, all_points[0], all_points[1], (0, 255, 0), 2)
            cv2.line(dst, all_points[1], all_points[2], (0, 255, 0), 2)
            
            angle = calculate_angle(*all_points)
            cv2.putText(dst, f"Angle: {angle:.2f}", (20, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
        print(points[0])
        cv2.imshow("Image", dst)



def main():
    global image, image_copy, dst, points
    
    # Загрузка данных (ваша реализация)
    MAIN_DIR = "Computer_Vision_for_Autonomous_Robot_Navigation/user_task/"
    data = pd.read_csv(MAIN_DIR + "annotations.csv", sep=';')
    
    for row in data.itertuples():
        # image = cv2.imread(MAIN_DIR + row[1]) 
        
        image = cv2.imread("result.jpg")
        if image is None:
            print(f"Ошибка загрузки: {row[1]}")
            continue
            
        dst = apply_perspective_transform(image)
        image_copy = dst.copy()
        points = []  # Сброс точек для нового изображения
        
        cv2.namedWindow("Image")
        cv2.setMouseCallback("Image", mouse_callback)
        
        print("Кликните по изображению чтобы отметить точку, 'q' - выход, 'n' - следующее изображение")
        

        # true_obj_list = extract_object_list(row)
        # print(true_obj_list)

        while True:
            cv2.imshow("Image", dst)
            cv2.imshow("input", image)

            key = cv2.waitKey(1)
            
            if key == ord('r'):  # Сброс
                points = []
                dst = image_copy.copy()

            

            elif key == ord('q'):  # Выход
                cv2.destroyAllWindows()
                return
            
            elif key == ord('n'):  # Следующее изображение
                break
            
        
        cv2.destroyWindow("Image")



if __name__ == "__main__":
    main()