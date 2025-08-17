import cv2
import numpy as np

def find_white_lines(image):
    """Находит белые линии на изображении"""
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lower_white = np.array([0, 0, 200])
    upper_white = np.array([180, 30, 255])
    white_mask = cv2.inRange(hsv, lower_white, upper_white)
    
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, bright_mask = cv2.threshold(gray, 250, 255, cv2.THRESH_BINARY)
    
    combined_mask = cv2.bitwise_or(white_mask, bright_mask)
    kernel = np.ones((3,3), np.uint8)
    combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel)
    combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN, kernel)
    
    return combined_mask

def find_road_edges(image):
    """Находит границы дороги"""
    white_lines_mask = find_white_lines(image)
    contours, _ = cv2.findContours(white_lines_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    height, width = image.shape[:2]
    min_area = 1000
    valid_contours = []
    
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > min_area:
            M = cv2.moments(cnt)
            if M['m00'] > 0:
                cy = int(M['m01']/M['m00'])
                if cy > height * 0.3:
                    valid_contours.append(cnt)
    
    valid_contours.sort(key=lambda c: cv2.boundingRect(c)[0])
    left_edge = right_edge = None
    
    if len(valid_contours) >= 2:
        left_edge, right_edge = valid_contours[0], valid_contours[-1]
    elif len(valid_contours) == 1:
        M = cv2.moments(valid_contours[0])
        cx = int(M['m10']/M['m00'])
        if cx < width/2:
            left_edge = valid_contours[0]
        else:
            right_edge = valid_contours[0]
    
    # Инициализируем точки по умолчанию
    dst_points = np.float32([
        [width*0.35, height*0.2],
        [width*0.65, height*0.2],
        [width*0.65, height*0.9],
        [width*0.35, height*0.9]
    ])
    
    viz = image.copy()
    
    if left_edge is not None:
        cv2.drawContours(viz, [left_edge], -1, (0,255,0), 2)
        left_points = left_edge.reshape(-1, 2)
        try:
            left_top_x = np.max(left_points[left_points[:,1] == np.min(left_points[:,1]), 0])
            left_bottom_x = np.max(left_points[left_points[:,1] == np.max(left_points[:,1]), 0])
            dst_points[0][0] = left_top_x
            dst_points[3][0] = left_bottom_x
            dst_points[0][1] = np.min(left_points[:,1])
            dst_points[3][1] = np.max(left_points[:,1])
        except:
            pass
    
    if right_edge is not None:
        cv2.drawContours(viz, [right_edge], -1, (0,0,255), 2)
        right_points = right_edge.reshape(-1, 2)
        try:
            right_top_x = np.min(right_points[right_points[:,1] == np.min(right_points[:,1]), 0])
            right_bottom_x = np.min(right_points[right_points[:,1] == np.max(right_points[:,1]), 0])
            dst_points[1][0] = right_top_x
            dst_points[2][0] = right_bottom_x
            dst_points[1][1] = np.min(right_points[:,1])
            dst_points[2][1] = np.max(right_points[:,1])
        except:
            pass
    
    return dst_points, white_lines_mask, viz

def extend_trapezoid(points, width, height, extend=2.0):
    """Расширяет трапецию за пределы изображения"""
    tl, tr, br, bl = points
    left_vec = bl - tl
    right_vec = br - tr
    
    new_tl = tl - left_vec * (extend - 1)
    new_bl = bl + left_vec * (extend - 1)
    new_tr = tr - right_vec * (extend - 1)
    new_br = br + right_vec * (extend - 1)
    
    return np.float32([new_tl, new_tr, new_br, new_bl])
def create_extended_perspective_grid(width, height, points, num_x=15, num_y=30, extend=2.0):
    """Создает перспективную сетку с расширенными вертикальными линиями"""
    # Создаем пустое изображение
    grid = np.zeros((height, width, 3), dtype=np.uint8)
    grid.fill(255)  # Белый фон
    
    # Точки исходного прямоугольника
    src_points = np.float32([
        [0, 0],
        [width, 0],
        [width, height],
        [0, height]
    ])
    
    # Матрица преобразования
    matrix = cv2.getPerspectiveTransform(src_points, points)
    
    # Рисуем линии сетки
    color = (0, 0, 0)
    thickness = 1
    
    # Горизонтальные линии (обычные, внутри трапеции)
    for i in range(num_y + 1):
        y = int(i * height / num_y)
        line = np.array([[0, y], [width, y]], dtype=np.float32)
        warped_line = cv2.perspectiveTransform(line.reshape(1, -1, 2), matrix).reshape(-1, 2)
        cv2.line(grid, tuple(warped_line[0].astype(int)), tuple(warped_line[1].astype(int)), color, thickness)
    
    # Вертикальные линии (расширенные за пределы трапеции)
    for i in range(num_x + 1):
        x = int(i * width / num_x)
        # Создаем удлиненную вертикальную линию (выше и ниже изображения)
        line = np.array([[x, -height*extend], [x, height*(1+extend)]], dtype=np.float32)
        warped_line = cv2.perspectiveTransform(line.reshape(1, -1, 2), matrix).reshape(-1, 2)
        cv2.line(grid, tuple(warped_line[0].astype(int)), tuple(warped_line[1].astype(int)), color, thickness)
    
    return grid



def main():
    NUM_CELLS_X = 15
    NUM_CELLS_Y = 30
    
    image = cv2.imread("/home/arrma/PROGRAMMS/Arhipelag_2025/Computer_Vision_for_Autonomous_Robot_Navigation/lane_detection_result.jpg")
    if image is None:
        print("Не удалось загрузить изображение")
        return
    
    height, width = image.shape[:2]
    dst_points, mask, contour_img = find_road_edges(image)
    
    # Создаем сетку с расширенными вертикальными линиями
    grid = create_extended_perspective_grid(
        width, height, dst_points, 
        NUM_CELLS_X, NUM_CELLS_Y,
        extend=1.5  # Линии выйдут на 1.5 высоты изображения за границы
    )
    
    # Наложение с прозрачностью
    result = cv2.addWeighted(image, 0.7, grid, 0.3, 0)
    
    # Отображение результатов
    cv2.imshow("Road with Extended Vertical Grid", result)
    cv2.imwrite("road_with_extended_vertical_grid.jpg", result)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


    # 665 112

if __name__ == "__main__":
    main()