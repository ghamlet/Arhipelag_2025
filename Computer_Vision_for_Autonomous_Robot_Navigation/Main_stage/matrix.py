# import cv2
# import numpy as np

# # Загрузка изображения
# image = cv2.imread('Main_stage/images/image_copy2.png')
# height, width = image.shape[:2]

# # Точки исходного прямоугольника (равномерная сетка)
# src_points = np.float32([
#     [0, 0],          # верхний левый
#     [width, 0],      # верхний правый
#     [width, height], # нижний правый
#     [0, height]      # нижний левый
# ])

# # Точки трапеции (как выглядит дорога на фото)
# dst_points = np.float32([
#     [625, 154],           # верхний левый (сужение)
#    [730, 161],           # верхний правый
#      [907, 601],            # нижний правый
#    [22, 579]                # нижний левый
# ])

# # Получение матрицы перспективного преобразования
# matrix = cv2.getPerspectiveTransform(src_points, dst_points)

# # Создание равномерной сетки (белые линии)
# grid = np.zeros((height, width, 3), dtype=np.uint8)
# grid.fill(255)  # белый фон

# # Рисуем вертикальные и горизонтальные линии
# line_color = (0, 0, 0)  # черные линии (на белом фоне)
# line_thickness = 2

# # Горизонтальные линии (через каждые 50 пикселей)
# for y in range(0, height, 50):
#     cv2.line(grid, (0, y), (width, y), line_color, line_thickness)

# # Вертикальные линии (через каждые 50 пикселей)
# for x in range(0, width, 50):
#     cv2.line(grid, (x, 0), (x, height), line_color, line_thickness)

# # Применение перспективного преобразования к сетке
# warped_grid = cv2.warpPerspective(grid, matrix, (width, height))

# # Наложение сетки на исходное изображение
# alpha = 0.3  # прозрачность
# result = cv2.addWeighted(image, 1, warped_grid, alpha, 0)

# # Сохранение результата
# cv2.imwrite('road_with_grid.jpg', result)
# cv2.imshow('Result', result)
# cv2.waitKey(0)

# cv2.destroyAllWindows()


import cv2
import numpy as np

# Загрузка изображения
image = cv2.imread('/home/arrma/PROGRAMMS/Arhipelag_2025/Computer_Vision_for_Autonomous_Robot_Navigation/Main_stage/images/video_20250816_174334.png')
height, width = image.shape[:2]

# Точки исходного прямоугольника (равномерная сетка)
src_points = np.float32([
    [0, 0],          # верхний левый
    [width, 0],      # верхний правый
    [width, height], # нижний правый
    [0, height]      # нижний левый
])

# Точки трапеции (как выглядит дорога на фото)
dst_points = np.float32([
    [625, 154],      # верхний левый (сужение)
    [730, 161],      # верхний правый
    [907, 601],      # нижний правый
    [22, 579]        # нижний левый
])





# Матрица перспективного преобразования
matrix = cv2.getPerspectiveTransform(src_points, dst_points)

# Создание равномерной сетки (белые линии)
grid = np.zeros((height, width, 3), dtype=np.uint8)
grid.fill(255)  # белый фон

# Рисуем вертикальные и горизонтальные линии (черные)
line_color = (0, 0, 0)  # BGR (черный)
line_thickness = 2

# Горизонтальные линии (через каждые 50 пикселей)
for y in range(0, height, 50):
    cv2.line(grid, (0, y), (width, y), line_color, line_thickness)

# Вертикальные линии (через каждые 50 пикселей)
for x in range(0, width, 50):
    cv2.line(grid, (x, 0), (x, height), line_color, line_thickness)

# Применение перспективного преобразования к сетке
warped_grid = cv2.warpPerspective(grid, matrix, (width, height))

# Наложение сетки на исходное изображение
alpha = 0.3  # прозрачность
result = cv2.addWeighted(image, 1, warped_grid, alpha, 0)

# Сохранение результата
cv2.imwrite('road_with_grid.jpg', result)
cv2.imshow('Result', result)
cv2.waitKey(0)
cv2.destroyAllWindows()