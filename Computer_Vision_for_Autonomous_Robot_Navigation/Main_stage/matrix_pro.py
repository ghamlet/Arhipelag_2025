import cv2
import numpy as np

def find_white_lines(image):
    """Находит белые линии на изображении через различные методы"""
    # Конвертируем в HSV для лучшего выделения белых областей
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    
    # Создаем маску для белых областей
    lower_white = np.array([0, 0, 200])
    upper_white = np.array([180, 30, 255])
    white_mask = cv2.inRange(hsv, lower_white, upper_white)
    
    # Также используем простую бинаризацию по яркости
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, bright_mask = cv2.threshold(gray, 250, 255, cv2.THRESH_BINARY)
    
    # Объединяем маски
    combined_mask = cv2.bitwise_or(white_mask, bright_mask)
    
    # Морфологические операции для очистки
    kernel = np.ones((3,3), np.uint8)
    combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel)
    combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN, kernel)
    
    return combined_mask

def find_road_edges(image):
    """Находит границы дороги через поиск белых линий и контуров"""
    # Находим белые линии
    white_lines_mask = find_white_lines(image)
    
    # Находим контуры в маске белых линий
    contours, _ = cv2.findContours(white_lines_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    height, width = image.shape[:2]
    
    # Фильтруем контуры по размеру и положению
    min_area = 1000
    valid_contours = []
    
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > min_area:
            # Проверяем, что контур находится в нижней части изображения (дорога)
            M = cv2.moments(cnt)
            if M['m00'] > 0:
                cy = int(M['m01']/M['m00'])
                if cy > height * 0.3:  # Только контуры в нижних 70% изображения
                    valid_contours.append(cnt)
    
    # Сортируем контуры по положению (слева направо)
    valid_contours.sort(key=lambda c: cv2.boundingRect(c)[0])
    
    # Находим левую и правую границы дороги
    left_edge = None
    right_edge = None
    
    if len(valid_contours) >= 2:
        # Берем самый левый и самый правый контур
        left_edge = valid_contours[0]
        right_edge = valid_contours[-1]
    elif len(valid_contours) == 1:
        # Если найден только один контур, определяем его положение
        M = cv2.moments(valid_contours[0])
        cx = int(M['m10']/M['m00'])
        if cx < width/2:
            left_edge = valid_contours[0]
        else:
            right_edge = valid_contours[0]
    
    # Если не нашли контуры, используем детекцию линий
    if left_edge is None or right_edge is None:
        # Детекция линий через преобразование Хафа
        edges = cv2.Canny(white_lines_mask, 50, 150)
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, 100, minLineLength=100, maxLineGap=50)
        
        if lines is not None:
            left_lines = []
            right_lines = []
            
            for line in lines:
                x1, y1, x2, y2 = line[0]
                if y2 < y1:  # Линия идет снизу вверх
                    continue
                
                # Вычисляем угол линии
                angle = np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi
                
                # Фильтруем по углу (вертикальные линии)
                if abs(angle) < 30 or abs(angle) > 150:
                    # Определяем, левая это линия или правая
                    if x1 < width/2:
                        left_lines.append(line[0])
                    else:
                        right_lines.append(line[0])
            
            # Убираем весь блок с линиями Хафа - берем точки только от контуров
    
    # Создаем точки трапеции от контуров белых линий
    if left_edge is not None and right_edge is not None:
        # Берем точки контуров
        left_points = left_edge.reshape(-1, 2)
        right_points = right_edge.reshape(-1, 2)
        
        # От левой линии берем самые правые точки (сверху и снизу)
        left_right_x = np.max(left_points[:, 0])
        left_top_y = np.min(left_points[:, 1])
        left_bottom_y = np.max(left_points[:, 1])
        
        # От правой линии берем самые левые точки (сверху и снизу)
        right_left_x = np.min(right_points[:, 0])
        right_top_y = np.min(right_points[:, 1])
        right_bottom_y = np.max(right_points[:, 1])
        
        # Строим трапецию с учетом наклона линий
        # Верхние точки - на одной высоте (самая верхняя)
        top_y = min(left_top_y, right_top_y)
        
        # Нижние точки - на одной высоте (самая нижняя)
        bottom_y = max(left_bottom_y, right_bottom_y)
        
        # Находим x-координаты для верхних и нижних точек с учетом наклона
        # Для левой линии: находим САМУЮ ПРАВУЮ точку на высоте top_y и bottom_y (внутренняя граница)
        left_top_mask = np.abs(left_points[:, 1] - top_y) < 5  # точки вблизи top_y
        left_bottom_mask = np.abs(left_points[:, 1] - bottom_y) < 5  # точки вблизи bottom_y
        
        if np.any(left_top_mask):
            left_top_x = np.max(left_points[left_top_mask, 0])  # самая правая точка
        else:
            left_top_x = left_right_x
            
        if np.any(left_bottom_mask):
            left_bottom_x = np.max(left_points[left_bottom_mask, 0])  # самая правая точка
        else:
            left_bottom_x = left_right_x
        
        # Для правой линии: находим САМУЮ ЛЕВУЮ точку на высоте top_y и bottom_y (внутренняя граница)
        right_top_mask = np.abs(right_points[:, 1] - top_y) < 5  # точки вблизи top_y
        right_bottom_mask = np.abs(right_points[:, 1] - bottom_y) < 5  # точки вблизи bottom_y
        
        if np.any(right_top_mask):
            right_top_x = np.min(right_points[right_top_mask, 0])  # самая левая точка
        else:
            right_top_x = right_left_x
            
        if np.any(right_bottom_mask):
            right_bottom_x = np.min(right_points[right_bottom_mask, 0])  # самая левая точка
        else:
            right_bottom_x = right_left_x
        
        dst_points = np.float32([
            [left_top_x, top_y],           # верхний левый (левая линия на верхней высоте)
            [right_top_x, top_y],          # верхний правый (правая линия на верхней высоте)
            [right_bottom_x, bottom_y],    # нижний правый (правая линия на нижней высоте)
            [left_bottom_x, bottom_y]      # нижний левый (левая линия на нижней высоте)
        ])
        
        # Отладочная информация
        print(f"Точки трапеции от контуров:")
        print(f"Левая линия: верх x={left_top_x}, низ x={left_bottom_x}")
        print(f"Правая линия: верх x={right_top_x}, низ x={right_bottom_x}")
        print(f"Высота: верх y={top_y}, низ y={bottom_y}")
    else:
        # Используем значения по умолчанию (внутренние границы)
        dst_points = np.float32([
            [width * 0.35, int(height * 0.2)],       # верхний левый (внутренняя левая граница)
            [width * 0.65, int(height * 0.2)],       # верхний правый (внутренняя правая граница)
            [width * 0.65, int(height * 0.9)],       # нижний правый (внутренняя правая граница)
            [width * 0.35, int(height * 0.9)]        # нижний левый (внутренняя левая граница)
        ])
    
    # Визуализация найденных контуров
    viz = image.copy()
    if left_edge is not None:
        cv2.drawContours(viz, [left_edge], -1, (0, 255, 0), 2)
    if right_edge is not None:
        cv2.drawContours(viz, [right_edge], -1, (0, 0, 255), 2)
    
    return dst_points, white_lines_mask, viz

def calculate_grid_step(points, num_cells=15):
    """Вычисляет оптимальный шаг сетки на основе размеров трапеции"""
    # Вычисляем размеры трапеции
    top_width = np.linalg.norm(points[1] - points[0])
    bottom_width = np.linalg.norm(points[2] - points[3])
    height = np.linalg.norm(points[3] - points[0])
    
    # Вычисляем средний размер ячейки
    avg_cell_size = min(top_width, bottom_width, height) / num_cells
    
    # Округляем до ближайшего значения, кратного 10
    grid_step = max(20, int(avg_cell_size / 10) * 10)
    
    return grid_step

def create_grid_with_cells(width, height, num_cells_horizontal, num_cells_vertical):
    """Создает сетку с заданным количеством ячеек"""
    grid = np.zeros((height, width, 3), dtype=np.uint8)
    grid.fill(255)  # Белый фон
    
    # Рисуем черные линии сетки
    line_color = (0, 0, 0)
    line_thickness = 1
    
    # Вычисляем шаги сетки
    horizontal_step = width // num_cells_horizontal
    vertical_step = height // num_cells_vertical
    
    # Горизонтальные линии
    for i in range(num_cells_vertical + 1):
        y = i * vertical_step
        cv2.line(grid, (0, y), (width, y), line_color, line_thickness)
    
    # Вертикальные линии
    for i in range(num_cells_horizontal + 1):
        x = i * horizontal_step
        cv2.line(grid, (x, 0), (x, height), line_color, line_thickness)
    
    return grid

def main():
    """Основная функция"""
    # Параметры сетки - можно легко изменять
    NUM_CELLS_HORIZONTAL = 8   # Количество ячеек по горизонтали
    NUM_CELLS_VERTICAL = 12    # Количество ячеек по вертикали
    
    # Можно изменить эти значения для настройки сетки:
    # NUM_CELLS_HORIZONTAL = 10  # Больше ячеек по горизонтали
    # NUM_CELLS_VERTICAL = 15    # Больше ячеек по вертикали
    # NUM_CELLS_HORIZONTAL = 6   # Меньше ячеек по горизонтали
    # NUM_CELLS_VERTICAL = 8     # Меньше ячеек по вертикали
    
    # Загружаем изображение
    image_path = "/home/arrma/PROGRAMMS/Arhipelag_2025/Computer_Vision_for_Autonomous_Robot_Navigation/lane_detection_result.jpg"
    image = cv2.imread(image_path)
    
    if image is None:
        print(f"Ошибка загрузки изображения: {image_path}")
        return
    
    height, width = image.shape[:2]
    print(f"Размер изображения: {width}x{height}")
    
    # Находим границы дороги
    dst_points, white_lines_mask, contour_img = find_road_edges(image)
    
    # Выводим координаты трапеции для отладки
    print("Координаты трапеции:")
    for i, point in enumerate(dst_points):
        print(f"Точка {i}: ({point[0]:.1f}, {point[1]:.1f})")
    
    # Выводим параметры сетки
    print(f"Сетка: {NUM_CELLS_HORIZONTAL} x {NUM_CELLS_VERTICAL} ячеек")
    
    # Создаем сетку с заданным количеством ячеек
    grid = create_grid_with_cells(width, height, NUM_CELLS_HORIZONTAL, NUM_CELLS_VERTICAL)
    
    # Точки исходного прямоугольника
    src_points = np.float32([
        [0, 0],          # верхний левый
        [width, 0],      # верхний правый
        [width, height], # нижний правый
        [0, height]      # нижний левый
    ])
    
    # Получаем матрицу перспективного преобразования
    matrix = cv2.getPerspectiveTransform(src_points, dst_points)
    
    # Применяем перспективное преобразование к сетке
    warped_grid = cv2.warpPerspective(grid, matrix, (width, height))
    
    # Накладываем сетку на исходное изображение
    alpha = 0.4  # Прозрачность сетки
    result = cv2.addWeighted(image, 1, warped_grid, alpha, 0)
    
    # Рисуем границы трапеции
    yellow = (0, 255, 255)
    cv2.line(result, tuple(dst_points[0].astype(int)), tuple(dst_points[1].astype(int)), yellow, 2)
    cv2.line(result, tuple(dst_points[1].astype(int)), tuple(dst_points[2].astype(int)), yellow, 2)
    cv2.line(result, tuple(dst_points[2].astype(int)), tuple(dst_points[3].astype(int)), yellow, 2)
    cv2.line(result, tuple(dst_points[3].astype(int)), tuple(dst_points[0].astype(int)), yellow, 2)
    
    # Добавляем точки трапеции
    for i, point in enumerate(dst_points):
        cv2.circle(result, tuple(point.astype(int)), 5, (0, 0, 255), -1)
        cv2.putText(result, str(i), tuple(point.astype(int)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    # Сохраняем результаты
    cv2.imwrite('road_with_grid_final.jpg', result)
    
    # Показываем результаты
    cv2.imshow('White Lines Detection', white_lines_mask)
    cv2.imshow('Road Contours', contour_img)
    cv2.imshow('Final Result with Grid', result)
    
    print("Нажмите любую клавишу для закрытия окон...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()