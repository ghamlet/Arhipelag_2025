import cv2
import numpy as np

# Загрузка изображения
image = cv2.imread('Main_stage/images/video_20250816_174334.png')
if image is None:
    print("Ошибка загрузки изображения")
    exit()

# 1. Предварительная обработка
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
blurred = cv2.GaussianBlur(gray, (5, 5), 0)
_, binary = cv2.threshold(blurred, 200, 255, cv2.THRESH_BINARY)

# 2. Морфологические операции для соединения пунктира
kernel = np.ones((5,5), np.uint8)
closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=2)

# 3. Поиск контуров
contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# 4. Разделение на левые и правые контуры
height, width = image.shape[:2]
left_contours = []
right_contours = []

for cnt in contours:
    # Получаем ограничивающий прямоугольник
    x, y, w, h = cv2.boundingRect(cnt)
    center_x = x + w/2
    
    # Разделяем на левые и правые
    if center_x < width/2:
        left_contours.append(cnt)
    else:
        right_contours.append(cnt)

# 5. Объединение контуров
def combine_contours(contour_list):
    if not contour_list:
        return None
    # Соединяем все точки контуров
    combined = np.vstack(contour_list)
    # Находим выпуклую оболочку
    hull = cv2.convexHull(combined)
    return hull

left_lane = combine_contours(left_contours)
right_lane = combine_contours(right_contours)

# 6. Рисование результата
result = image.copy()

# Рисуем левую линию (синий)
if left_lane is not None:
    cv2.drawContours(result, [left_lane], -1, (255, 255, 255), -1)

# Рисуем правую линию (красный)
if right_lane is not None:
    cv2.drawContours(result, [right_lane], -1, (255, 255, 255), -1)

# 7. Показ результатов
cv2.imshow('Binary', binary)
cv2.imshow('Closed', closed)
cv2.imshow('Result', result)
cv2.waitKey(0)
cv2.destroyAllWindows()

cv2.imwrite('lane_detection_result.jpg', result)
