import cv2
import numpy as np
import os

# Захват видео
cap = cv2.VideoCapture("Main_stage/output.mp4")  
if not cap.isOpened():
    print("Ошибка открытия видеофайла!")
    exit()

# Установка желаемого размера кадра
TARGET_WIDTH = 1280
TARGET_HEIGHT = 720
cap.set(cv2.CAP_PROP_FRAME_WIDTH, TARGET_WIDTH)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, TARGET_HEIGHT)

# Параметры перспективного преобразования
NEW_SIZE = 500
pts2 = np.float32([[0,0], [NEW_SIZE,0], [0,NEW_SIZE], [NEW_SIZE,NEW_SIZE]])

pointsList = []
minb = ming = minr = maxb = maxg = maxr = None
points_of_quadr = False
is_pressed_y = False
setting_up_trackers = True

def delete_last_added_point():
    """Удаляет последнюю добавленную точку"""
    global pointsList
    if pointsList:
        pointsList = pointsList[:-1]

def draw_infinite_line(img, pt1, pt2, color, thickness):
    """Рисует бесконечную линию через две точки"""
    height, width = img.shape[:2]
    
    # Если линия вертикальная
    if pt1[0] == pt2[0]:
        cv2.line(img, (pt1[0], 0), (pt1[0], height), color, thickness)
        return
    
    # Вычисляем коэффициенты прямой y = kx + b
    k = (pt2[1] - pt1[1]) / (pt2[0] - pt1[0])
    b = pt1[1] - k * pt1[0]
    
    # Вычисляем точки пересечения с границами изображения
    # Левая граница (x=0)
    y_left = int(b)
    left_pt = (0, y_left) if 0 <= y_left < height else None
    
    # Правая граница (x=width-1)
    y_right = int(k * (width-1) + b)
    right_pt = (width-1, y_right) if 0 <= y_right < height else None
    
    # Верхняя граница (y=0)
    x_top = int(-b/k) if k != 0 else None
    top_pt = (x_top, 0) if x_top is not None and 0 <= x_top < width else None
    
    # Нижняя граница (y=height-1)
    x_bottom = int((height-1 - b)/k) if k != 0 else None
    bottom_pt = (x_bottom, height-1) if x_bottom is not None and 0 <= x_bottom < width else None
    
    # Собираем все допустимые точки пересечения
    border_pts = []
    for pt in [left_pt, right_pt, top_pt, bottom_pt]:
        if pt is not None:
            border_pts.append(pt)
    
    # Если нашли хотя бы 2 точки пересечения, рисуем линию
    if len(border_pts) >= 2:
        cv2.line(img, border_pts[0], border_pts[1], color, thickness)

        
def render_points_on_video():
    """Отображает видео с отмеченными точками и линиями"""
    global copy_img
    
    ret, img = cap.read()
    if not ret:
        print("Не удалось получить кадр или видео закончилось")
        exit()
    
    img = cv2.resize(img, (TARGET_WIDTH, TARGET_HEIGHT))
    copy_img = img.copy()
    
    # Отрисовка точек с номерами
    for i, pt in enumerate(pointsList):
        cv2.circle(img, pt, 8, (0, 0, 255), -1)
        cv2.putText(img, str(i+1), (pt[0]+15, pt[1]+15), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    
    # Отрисовка бесконечных линий между точками 1-2 и 3-4
    if len(pointsList) >= 2:
        draw_infinite_line(img, pointsList[0], pointsList[1], (0, 255, 0), 2)
    if len(pointsList) >= 4:
        draw_infinite_line(img, pointsList[2], pointsList[3], (255, 0, 0), 2)  # Вторую линию рисуем синим
    
    cv2.imshow("Video with Perspective Points", img)

def handle_key_press(close_all_windows=False):
    """Обрабатывает нажатия клавиш"""
    global points_of_quadr, setting_up_trackers, is_pressed_y
    
    key = cv2.waitKey(10) & 0xFF
    
    if key == ord('q'):  # Выход
        exit()
    elif key == ord('y'):  # Подтверждение точек
        is_pressed_y = True
    elif key == ord('x'):  # Удаление последней точки
        if close_all_windows:
            cv2.destroyAllWindows()
        delete_last_added_point()
        points_of_quadr = False
        is_pressed_y = False
        setting_up_trackers = False
    elif key == ord('s') and all(v is not None for v in [minb, ming, minr, maxb, maxg, maxr]):
        save_color_settings()

def save_color_settings():
    """Сохраняет настройки цветов в файл"""
    with open(os.path.join(os.path.dirname(__file__), "trackbars_save.txt"), "a") as f:
        title = input("\nВведите описание настроек (или 'no' для отмены): ")
        if title.lower() not in ("n", "no"):
            f.write(f"{title}: {minb, ming, minr}, {maxb, maxg, maxr}\n")
            print("Настройки сохранены")

def mousePoints(event, x, y, flags, params):
    """Обработчик событий мыши"""
    global points_of_quadr
    if event == cv2.EVENT_LBUTTONDOWN and len(pointsList) < 4:
        pointsList.append([x, y])
        if len(pointsList) == 4:
            points_of_quadr = True

def trackbar():
    """Создает трекбары для настройки цветов"""
    cv2.namedWindow("Color Trackbars")
    cv2.createTrackbar('minb', 'Color Trackbars', 0, 255, lambda x: None)
    cv2.createTrackbar('ming', 'Color Trackbars', 0, 255, lambda x: None)
    cv2.createTrackbar('minr', 'Color Trackbars', 0, 255, lambda x: None)
    cv2.createTrackbar('maxb', 'Color Trackbars', 255, 255, lambda x: None)
    cv2.createTrackbar('maxg', 'Color Trackbars', 255, 255, lambda x: None)
    cv2.createTrackbar('maxr', 'Color Trackbars', 255, 255, lambda x: None)

def apply_perspective_transform():
    """Применяет перспективное преобразование"""
    pts1 = np.float32(pointsList)
    M = cv2.getPerspectiveTransform(pts1, pts2)
    warped = cv2.warpPerspective(copy_img, M, (NEW_SIZE, NEW_SIZE))
    return warped

def process_color_filtering(warped_img):
    """Применяет цветовые фильтры"""
    hsv = cv2.cvtColor(warped_img, cv2.COLOR_BGR2HSV)
    
    minb = cv2.getTrackbarPos('minb', 'Color Trackbars')
    ming = cv2.getTrackbarPos('ming', 'Color Trackbars')
    minr = cv2.getTrackbarPos('minr', 'Color Trackbars')
    maxb = cv2.getTrackbarPos('maxb', 'Color Trackbars')
    maxg = cv2.getTrackbarPos('maxg', 'Color Trackbars')
    maxr = cv2.getTrackbarPos('maxr', 'Color Trackbars')
    
    mask = cv2.inRange(hsv, (minb, ming, minr), (maxb, maxg, maxr))
    result = cv2.bitwise_and(warped_img, warped_img, mask=mask)
    
    return result, (minb, ming, minr, maxb, maxg, maxr)

# Основной цикл программы
print("Инструкция:")
print("1. Кликните 4 точки в следующем порядке:")
print("   - Точка 1: первая точка первой линии")
print("   - Точка 2: вторая точка первой линии (зеленая)")
print("   - Точка 3: первая точка второй линии")
print("   - Точка 4: вторая точка второй линии (синяя)")
print("2. Нажмите 'y' для подтверждения точек")
print("3. Нажмите 'x' для удаления последней точки")
print("4. Нажмите 'q' для выхода")

while True:
    # Этап 1: Выбор точек перспективы
    while not points_of_quadr:
        render_points_on_video()
        cv2.setMouseCallback("Video with Perspective Points", mousePoints)
        handle_key_press()
    
    # Этап 2: Подтверждение точек
    while not is_pressed_y:
        render_points_on_video()
        handle_key_press()
    
    # Этап 3: Настройка цветовых фильтров
    trackbar()
    setting_up_trackers = True
    
    while setting_up_trackers:
        render_points_on_video()
        
        warped = apply_perspective_transform()
        if warped is not None:
            cv2.imshow("Warped Perspective", warped)
            
            filtered, colors = process_color_filtering(warped)
            cv2.imshow("Color Filtered Result", filtered)
        
        handle_key_press(close_all_windows=True)
