import cv2
import numpy as np

# Загрузка изображения
image = cv2.imread('Main_stage/images/image_copy2.png')
h, w = image.shape[:2]

# Точки искаженного изображения (трапеция)
src_points = np.float32([
    [625, 154],  # верхний левый
    [730, 161],  # верхний правый
    [907, 601],  # нижний правый
    [22, 579]    # нижний левый
])

# Точки, в которые нужно преобразовать (прямоугольник)
dst_points = np.float32([
    [0, 0],      # верхний левый
    [w, 0],      # верхний правый
    [w, h],      # нижний правый
    [0, h]       # нижний левый
])

# Получаем матрицу преобразования перспективы
matrix = cv2.getPerspectiveTransform(src_points, dst_points)

# Применяем преобразование
corrected_img = cv2.warpPerspective(image, matrix, (w, h))

# Сохраняем результат
cv2.imwrite('corrected_road.jpg', corrected_img)
cv2.imshow('Corrected Image', corrected_img)
cv2.waitKey(0)
cv2.destroyAllWindows()