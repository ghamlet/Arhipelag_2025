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




def find_vanishing_point(image):
    """
    Находит точку схода для двух основных линий дороги на изображении
    Возвращает координаты точки схода (x, y) или None, если не удалось определить
    """
    # Находим границы дороги (используем вашу существующую функцию)
    dst_points, _, _ = find_road_edges(image)
    
    # Если не удалось найти границы, возвращаем None
    if dst_points is None:
        return None
    
    # Преобразуем точки трапеции в массив numpy
    pts = np.array(dst_points, dtype=np.float32)
    
    # Вычисляем уравнения линий для левой и правой границ дороги
    def line_equation(p1, p2):
        A = p2[1] - p1[1]
        B = p1[0] - p2[0]
        C = p2[0]*p1[1] - p1[0]*p2[1]
        return A, B, -C
    
    # Левая граница (точки 0 и 3)
    left_line = line_equation(pts[0], pts[3])
    
    # Правая граница (точки 1 и 2)
    right_line = line_equation(pts[1], pts[2])
    
    # Находим точку пересечения двух линий
    def intersection(line1, line2):
        D  = line1[0] * line2[1] - line1[1] * line2[0]
        Dx = line1[2] * line2[1] - line1[1] * line2[2]
        Dy = line1[0] * line2[2] - line1[2] * line2[0]
        if D != 0:
            x = Dx / D
            y = Dy / D
            return x, y
        else:
            return None
    
    vanishing_point = intersection(left_line, right_line)
    
    return vanishing_point


def draw_perspective_grid(image, vanishing_point, num_vert_lines=36, num_horiz_lines=12,
                         line_color=(255, 255, 255), line_thickness=2):
    """
    Рисует перспективную сетку с линиями, расходящимися во всех направлениях от точки схода
    
    :param image: исходное изображение
    :param vanishing_point: tuple (x, y) - координаты точки схода
    :param num_vert_lines: количество линий по кругу (рекомендуется 24-36)
    :param num_horiz_lines: количество горизонтальных линий
    :param line_color: цвет линий (BGR)
    :param line_thickness: толщина линий
    :return: изображение с нарисованной сеткой
    """
    result = image.copy()
    height, width = image.shape[:2]
    
    try:
        vp_x, vp_y = map(int, vanishing_point)
    except (TypeError, ValueError):
        print("Ошибка: некорректные координаты точки схода")
        return image

    # 1. Вертикальные линии (расходящиеся по всем направлениям)
    angles = np.linspace(0, 2*np.pi, num_vert_lines, endpoint=False)
    
    for angle in angles:
        # Вычисляем направление линии
        dir_x = np.cos(angle)
        dir_y = np.sin(angle)
        
        # Находим точки пересечения с границами изображения
        t_values = []
        
        # Проверяем все 4 границы изображения
        for border_x in [0, width]:
            if dir_x != 0:
                t = (border_x - vp_x) / dir_x
                y = vp_y + t * dir_y
                if 0 <= y <= height:
                    t_values.append(t)
                    
        for border_y in [0, height]:
            if dir_y != 0:
                t = (border_y - vp_y) / dir_y
                x = vp_x + t * dir_x
                if 0 <= x <= width:
                    t_values.append(t)
        
        # Фильтруем только положительные значения
        positive_ts = [t for t in t_values if t > 0]
        if not positive_ts:
            continue
            
        # Берем ближайшую точку пересечения
        t = min(positive_ts)
        end_x = int(vp_x + t * dir_x)
        end_y = int(vp_y + t * dir_y)
        
        # Рисуем линию
        cv2.line(result, (vp_x, vp_y), (end_x, end_y), line_color, line_thickness)

    # 2. Горизонтальные линии (с перспективным уменьшением шага)
    if vp_y < height and vp_y >= 0:
        # Экспоненциальное распределение для увеличения плотности у точки схода
        line_positions = []
        max_y = height
        min_y = max(0, vp_y + 10)  # Не подходим слишком близко к точке схода
        
        for i in range(1, num_horiz_lines + 1):
            # Плавное уменьшение шага по мере приближения к точке схода
            t = np.sqrt(i / num_horiz_lines)  # Квадратный корень для плавности
            line_y = int(max_y - (max_y - min_y) * t)
            line_positions.append(line_y)
        
        # Рисуем горизонтальные линии
        for y in line_positions:
            cv2.line(result, (0, y), (width, y), line_color, line_thickness)
    
    # # 3. Рисуем точку схода для наглядности
    # if 0 <= vp_x <= width and 0 <= vp_y <= height:
    #     cv2.circle(result, (vp_x, vp_y), 8, (0, 0, 255), -1)
    #     cv2.circle(result, (vp_x, vp_y), 4, (0, 255, 255), -1)
    
    return result



def draw_converging_lines(image, vanishing_point, num_lines=12, line_scale=1.0, line_color=(255, 255, 255), line_thickness=2):
    """
    Рисует линии, сходящиеся в точке схода, гарантируя их видимость на изображении
    
    :param image: исходное изображение (numpy array)
    :param vanishing_point: tuple (x, y) - координаты точки схода
    :param num_lines: количество линий (рекомендуется 8-16)
    :param line_scale: масштаб длины линий (0.1-2.0)
    :param line_color: цвет линий (BGR)
    :param line_thickness: толщина линий
    :return: изображение с нарисованными линиями
    """
    result = image.copy()
    height, width = image.shape[:2]
    
    try:
        vp_x, vp_y = map(int, vanishing_point)
    except (TypeError, ValueError):
        print("Ошибка: некорректные координаты точки схода")
        return image
    
    # Рассчитываем оптимальную длину линий
    max_dim = max(height, width)
    base_length = int(max_dim * 0.8 * line_scale)
    
    # Создаем точки старта по окружности вокруг точки схода
    angles = np.linspace(0, 2*np.pi, num_lines, endpoint=False)
    
    for angle in angles:
        # Вычисляем направление линии
        dir_x = np.cos(angle)
        dir_y = np.sin(angle)
        
        # Находим точку на границе изображения
        t_values = []
        # Проверяем пересечение с левой границей (x=0)
        if dir_x != 0:
            t = (0 - vp_x) / dir_x
            y = vp_y + t * dir_y
            if 0 <= y <= height:
                t_values.append(t)
        # Проверяем пересечение с правой границей (x=width)
        if dir_x != 0:
            t = (width - vp_x) / dir_x
            y = vp_y + t * dir_y
            if 0 <= y <= height:
                t_values.append(t)
        # Проверяем пересечение с верхней границей (y=0)
        if dir_y != 0:
            t = (0 - vp_y) / dir_y
            x = vp_x + t * dir_x
            if 0 <= x <= width:
                t_values.append(t)
        # Проверяем пересечение с нижней границей (y=height)
        if dir_y != 0:
            t = (height - vp_y) / dir_y
            x = vp_x + t * dir_x
            if 0 <= x <= width:
                t_values.append(t)
        
        if not t_values:
            continue  # Линия не пересекает изображение
        
        # Выбираем минимальное положительное t
        t = min(t for t in t_values if t > 0)
        
        # Вычисляем конечную точку
        end_x = int(vp_x + t * dir_x )  # Укорачиваем на 10% для гарантии видимости
        end_y = int(vp_y + t * dir_y )
        
        # Рисуем линию
        cv2.line(result, (vp_x, vp_y), (end_x, end_y), line_color, line_thickness)
    
    return result



# Пример использования
image = cv2.imread('/home/arrma/PROGRAMMS/Arhipelag_2025/Computer_Vision_for_Autonomous_Robot_Navigation/lane_detection_result.jpg')  # Загрузите ваше изображение
if image is None:
    print("Не удалось загрузить изображение")
    exit()




# vanishing_point = (665, 112)  # Точка схода
vanishing_point = find_vanishing_point(image)
print(vanishing_point)


result_image =  draw_perspective_grid(
    image=image,
    vanishing_point=vanishing_point,
    num_vert_lines=93,  # 93  
    num_horiz_lines=70,
    line_color=(0, 255, 0),  # Зеленый цвет
    line_thickness=1
)

# result_image = draw_converging_lines(
#     image,
#     vanishing_point=vanishing_point,
#     num_lines=93,        # Количество линий  31   84   87   93 best
#     line_scale=2,      # Масштаб длины (0.8-1.5)
#     line_color=(0, 0, 255),  # Красный цвет
#     line_thickness=2
# )
# Сохраняем и показываем результат
cv2.imwrite('result_with_lines.jpg', result_image)
cv2.imshow('Converging Lines', result_image)
cv2.waitKey(0)
cv2.destroyAllWindows()