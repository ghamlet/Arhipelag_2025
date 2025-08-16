import cv2
import numpy as np

# Загрузка изображения
img = cv2.imread("Main_stage/images/image_copy2.png")
h, w = img.shape[:2]


# Примерные параметры камеры (нужно заменить на реальные)
camera_matrix = np.array([
    [1000, 0, w/2],  # fx, 0, cx
    [0, 1000, h/2],  # 0, fy, cy
    [0, 0, 1]
], dtype=np.float32)

dist_coeffs = np.array([-0.4, 0.1, 0, 0], dtype=np.float32)  # k1, k2, p1, p2

# Коррекция искажения
undistorted_img = cv2.undistort(img, camera_matrix, dist_coeffs)

# Сохранение
cv2.imwrite('undistorted_road.jpg', undistorted_img)
cv2.imshow('Undistorted', undistorted_img)
cv2.waitKey(0)
cv2.destroyAllWindows()