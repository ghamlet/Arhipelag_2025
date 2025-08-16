import cv2
import numpy as np

# Размеры выходного изображения (в метрах и пикселях)
ROAD_LENGTH = 3.0  # 3 метра
ROAD_WIDTH = 2.0    # Ширина зоны обработки
PX_PER_METER = 500  # Плотность пикселей на метр
OUTPUT_WIDTH = int(ROAD_WIDTH * PX_PER_METER)
OUTPUT_HEIGHT = int(ROAD_LENGTH * PX_PER_METER)

# Точки для перспективного преобразования (заполнятся вручную)
src_points = []  # 4 точки на исходном изображении
dst_points = np.float32([
    [0, OUTPUT_HEIGHT],
    [OUTPUT_WIDTH, OUTPUT_HEIGHT],
    [OUTPUT_WIDTH, 0],
    [0, 0]
])

def mouse_callback(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN and len(src_points) < 4:
        src_points.append([x, y])
        print(f"Точка {len(src_points)}: ({x}, {y})")

def calibrate_perspective(frame):
    """Калибровка перспективы по 4 точкам"""
    h, w = frame.shape[:2]
    
    # Создаем окно для калибровки
    cv2.namedWindow("Calibration")
    cv2.setMouseCallback("Calibration", mouse_callback)
    
    while True:
        display = frame.copy()
        
        # Рисуем выбранные точки
        for i, pt in enumerate(src_points):
            cv2.circle(display, tuple(pt), 10, (0, 0, 255), -1)
            cv2.putText(display, str(i+1), (pt[0]+15, pt[1]+15), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        
        # Рисуем линии между точками (если есть)
        if len(src_points) >= 2:
            cv2.line(display, tuple(src_points[0]), tuple(src_points[1]), (0, 255, 0), 2)
        if len(src_points) >= 4:
            cv2.line(display, tuple(src_points[2]), tuple(src_points[3]), (0, 255, 0), 2)
        
        cv2.imshow("Calibration", display)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('r'):
            src_points.clear()
        elif len(src_points) == 4 and key == ord('c'):
            break
    
    if len(src_points) == 4:
        # Вычисляем матрицу перспективного преобразования
        M = cv2.getPerspectiveTransform(np.float32(src_points), dst_points)
        cv2.destroyWindow("Calibration")
        return M
    return None

def process_frame(frame, M):
    """Применяет перспективное преобразование"""
    warped = cv2.warpPerspective(frame, M, (OUTPUT_WIDTH, OUTPUT_HEIGHT))
    
    # Масштабируем для отображения (если нужно)
    display = cv2.resize(warped, (800, 1200))
    
    # Добавляем метрическую сетку
    for y in range(0, OUTPUT_HEIGHT, int(PX_PER_METER)):
        cv2.line(display, (0, y), (OUTPUT_WIDTH, y), (0, 255, 255), 1)
        cv2.putText(display, f"{y/PX_PER_METER:.1f}m", (10, y-10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
    
    return display

# Основной цикл обработки видео
cap = cv2.VideoCapture("/home/arrma/PROGRAMMS/Arhipelag_2025/Computer_Vision_for_Autonomous_Robot_Navigation/Main_stage/recordings/video_20250816_174334.mp4")  # Или номер камеры

# Калибровка по первому кадру
ret, frame = cap.read()
if ret:
    M = calibrate_perspective(frame)
    
    if M is not None:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            # Применяем перспективное преобразование
            result = process_frame(frame, M)
            
            cv2.imshow("Road Perspective", result)
            if cv2.waitKey(30) == 27:  # ESC для выхода
                break

cap.release()
cv2.destroyAllWindows()