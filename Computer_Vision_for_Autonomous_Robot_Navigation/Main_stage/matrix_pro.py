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
            
            # Если нашли линии, используем их для определения границ
            if left_lines and right_lines:
                # Берем средние позиции линий
                left_x = int(np.mean([line[0] for line in left_lines]))
                right_x = int(np.mean([line[0] for line in right_lines]))  # Исправлено: было left_lines
                
                # Находим реальную верхнюю границу дороги
                # Анализируем y-координаты найденных линий
                left_y_coords = [line[1] for line in left_lines]  # y-координаты левых линий
                right_y_coords = [line[1] for line in right_lines]  # y-координаты правых линий
                
                # Берем самую верхнюю точку среди всех линий
                all_y_coords = left_y_coords + right_y_coords
                top_y = min(all_y_coords) if all_y_coords else int(height * 0.2)
                
                # Ограничиваем верхнюю границу
                min_top_y = int(height * 0.1)
                top_y = max(top_y, min_top_y)
                
                # Создаем точки для трапеции
                dst_points = np.float32([
                    [left_x, top_y],                  # верхний левый
                    [right_x, top_y],                 # верхний правый
                    [right_x + 50, height],           # нижний правый
                    [left_x - 50, height]             # нижний левый
                ])
                return dst_points, white_lines_mask, image.copy()
    
    # Создаем точки трапеции на основе найденных контуров
    if left_edge is not None and right_edge is not None:
        # Получаем крайние точки контуров
        left_rect = cv2.boundingRect(left_edge)
        right_rect = cv2.boundingRect(right_edge)
        
        # Находим реальную верхнюю границу дороги
        # Ищем самую верхнюю точку среди левого и правого контуров
        left_top_y = left_rect[1]  # y-координата верхней границы левого контура
        right_top_y = right_rect[1]  # y-координата верхней границы правого контура
        top_y = min(left_top_y, right_top_y)  # Берем самую верхнюю точку
        
        # Ограничиваем верхнюю границу, чтобы трапеция не была слишком узкой
        min_top_y = int(height * 0.1)  # Минимум 10% от высоты
        top_y = max(top_y, min_top_y)
        
        # Создаем точки трапеции
        dst_points = np.float32([
            [left_rect[0] + left_rect[2], top_y],                # верхний левый
            [right_rect[0], top_y],                              # верхний правый
            [right_rect[0] + right_rect[2], height],             # нижний правый
            [left_rect[0], height]                               # нижний левый
        ])
    else:
        # Используем значения по умолчанию
        dst_points = np.float32([
            [width * 0.3, int(height * 0.2)],
            [width * 0.7, int(height * 0.2)],
            [width * 0.8, height],
            [width * 0.2, height]
        ])
    
    # Визуализация найденных контуров
    viz = image.copy()
    if left_edge is not None:
        cv2.drawContours(viz, [left_edge], -1, (0, 255, 0), 2)
    if right_edge is not None:
        cv2.drawContours(viz, [right_edge], -1, (0, 0, 255), 2)
    
    # Рисуем границы дороги на маске белых линий
    yellow = (0, 255, 255)  # Желтый цвет в BGR
    
    if left_edge is not None:
        # Для левой линии берем самую правую границу
        left_rect = cv2.boundingRect(left_edge)
        left_rightmost_x = left_rect[0] + left_rect[2]  # x + width
        # Рисуем вертикальную линию по самой правой границе левого контура
        cv2.line(white_lines_mask, (left_rightmost_x, 0), (left_rightmost_x, height), yellow, 3)
        # Добавляем текст с координатой
        cv2.putText(white_lines_mask, f'L:{left_rightmost_x}', (left_rightmost_x + 5, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, yellow, 2)
    
    if right_edge is not None:
        # Для правой линии берем самую левую границу
        right_rect = cv2.boundingRect(right_edge)
        right_leftmost_x = right_rect[0]  # x
        # Рисуем вертикальную линию по самой левой границе правого контура
        cv2.line(white_lines_mask, (right_leftmost_x, 0), (right_leftmost_x, height), yellow, 3)
        # Добавляем текст с координатой
        cv2.putText(white_lines_mask, f'R:{right_leftmost_x}', (right_leftmost_x - 50, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, yellow, 2)
    
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

def create_grid_from_road_edges(width, height, dst_points, num_cells_horizontal=10, num_cells_vertical=10):
    """Создает сетку от внутренних сторон линий дороги"""
    grid = np.zeros((height, width, 3), dtype=np.uint8)
    grid.fill(255)  # Белый фон
    
    # Рисуем черные линии сетки
    line_color = (0, 0, 0)
    line_thickness = 2  # Увеличиваем толщину для лучшей видимости
    
    # Получаем координаты трапеции
    top_left = dst_points[0]      # Верхний левый
    top_right = dst_points[1]     # Верхний правый
    bottom_right = dst_points[2]  # Нижний правый
    bottom_left = dst_points[3]   # Нижний левый
    
    # Рисуем вертикальные линии (параллельно границам дороги)
    for i in range(num_cells_horizontal + 1):
        # Интерполируем x-координату между левой и правой границами
        t = i / num_cells_horizontal
        
        # Верхняя точка линии
        top_x = top_left[0] + t * (top_right[0] - top_left[0])
        top_y = top_left[1] + t * (top_right[1] - top_left[1])
        
        # Нижняя точка линии
        bottom_x = bottom_left[0] + t * (bottom_right[0] - bottom_left[0])
        bottom_y = bottom_left[1] + t * (bottom_right[1] - bottom_left[1])
        
        # Рисуем линию
        cv2.line(grid, (int(top_x), int(top_y)), (int(bottom_x), int(bottom_y)), 
                line_color, line_thickness)
    
    # Рисуем горизонтальные линии (перпендикулярно границам дороги)
    for i in range(num_cells_vertical + 1):
        # Интерполируем y-координату между верхней и нижней границами
        t = i / num_cells_vertical
        
        # Левая точка линии
        left_x = top_left[0] + t * (bottom_left[0] - top_left[0])
        left_y = top_left[1] + t * (bottom_left[1] - top_left[1])
        
        # Правая точка линии
        right_x = top_right[0] + t * (bottom_right[0] - top_right[0])
        right_y = top_right[1] + t * (bottom_right[1] - top_right[1])
        
        # Рисуем линию
        cv2.line(grid, (int(left_x), int(left_y)), (int(right_x), int(right_y)), 
                line_color, line_thickness)
    
    return grid

def create_simple_grid(width, height, dst_points, num_cells_horizontal=10, num_cells_vertical=10):
    """Создает простую сетку с прямыми линиями"""
    grid = np.zeros((height, width, 3), dtype=np.uint8)
    grid.fill(255)  # Белый фон
    
    # Рисуем черные линии сетки
    line_color = (0, 0, 0)
    line_thickness = 3  # Еще больше увеличиваем толщину
    
    # Получаем границы дороги
    left_x = dst_points[0][0]   # Левая граница
    right_x = dst_points[1][0]  # Правая граница
    top_y = dst_points[0][1]    # Верхняя граница
    bottom_y = dst_points[2][1] # Нижняя граница
    
    # Рисуем вертикальные линии
    for i in range(num_cells_horizontal + 1):
        x = left_x + (right_x - left_x) * i / num_cells_horizontal
        cv2.line(grid, (int(x), int(top_y)), (int(x), int(bottom_y)), 
                line_color, line_thickness)
    
    # Рисуем горизонтальные линии
    for i in range(num_cells_vertical + 1):
        y = top_y + (bottom_y - top_y) * i / num_cells_vertical
        cv2.line(grid, (int(y), int(left_x)), (int(y), int(right_x)), 
                line_color, line_thickness)
    
    return grid

def main():
    """Основная функция"""
    # Параметры сетки - можно легко изменять
    NUM_CELLS_HORIZONTAL = 8   # Количество ячеек по горизонтали
    NUM_CELLS_VERTICAL = 12    # Количество ячеек по вертикали
    USE_SIMPLE_GRID = True     # True - простая сетка, False - сложная с перспективой
    
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
    
    # Вычисляем размеры дороги
    road_width = dst_points[1][0] - dst_points[0][0]
    road_height = dst_points[2][1] - dst_points[0][1]
    print(f"Ширина дороги: {road_width:.1f} пикселей")
    print(f"Высота дороги: {road_height:.1f} пикселей")
    
    # Выводим координаты границ дороги
    if len(dst_points) >= 2:
        print(f"Левая граница дороги (самая правая точка левого контура): {dst_points[0][0]:.1f}")
        print(f"Правая граница дороги (самая левая точка правого контура): {dst_points[1][0]:.1f}")
    
    # Создаем сетку от внутренних сторон линий дороги
    if USE_SIMPLE_GRID:
        grid = create_simple_grid(width, height, dst_points, 
                                NUM_CELLS_HORIZONTAL, NUM_CELLS_VERTICAL)
        print(f"Создана простая сетка: {NUM_CELLS_HORIZONTAL} x {NUM_CELLS_VERTICAL} ячеек")
        print(f"Толщина линий сетки: 3 пикселя")
    else:
        grid = create_grid_from_road_edges(width, height, dst_points, 
                                         NUM_CELLS_HORIZONTAL, NUM_CELLS_VERTICAL)
        print(f"Создана сложная сетка с перспективой: {NUM_CELLS_HORIZONTAL} x {NUM_CELLS_VERTICAL} ячеек")
        print(f"Толщина линий сетки: 2 пикселя")
    
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
    cv2.imwrite('grid_only.jpg', grid)
    
    # Показываем результаты
    cv2.imshow('White Lines Detection', white_lines_mask)
    cv2.imshow('Road Contours', contour_img)
    cv2.imshow('Final Result with Grid', result)
    
    # Показываем сетку отдельно для отладки
    cv2.imshow('Grid Only', grid)
    
    print("Нажмите любую клавишу для закрытия окон...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()