import cv2
import numpy as np
from tensorflow.keras.models import load_model

# Загрузка модели
model = load_model('/home/arrma/PROGRAMMS/Arhipelag_2025/Computer_Vision_for_Autonomous_Robot_Navigation/Educational_task/traffic_sign_model.h5')
class_names = ['NoDrive', 'Stop', 'Parking', 'RoadWorks', 'PedestrianCrossing']

def predict_box(image):
    """
    Улучшенная функция для детекции и классификации дорожных знаков.
    
    Args:
        image: входное изображение в формате BGR (результат cv2.imread)
    
    Returns:
        tuple: (x, y, x2, y2) - координаты рамки знака
               или (1,1,1,1) если знак не обнаружен
    """
    # Цветовые диапазоны для разных типов знаков
    color_ranges = [
        # Красные знаки (два диапазона для красного)
        {'lower': np.array([0, 70, 50]), 'upper': np.array([10, 255, 255])},
        {'lower': np.array([170, 70, 50]), 'upper': np.array([180, 255, 255])},
        # Синие знаки
        {'lower': np.array([100, 70, 50]), 'upper': np.array([130, 255, 255])},
        # Желтые знаки
        {'lower': np.array([20, 70, 70]), 'upper': np.array([40, 255, 255])}
    ]
    
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    
    # Объединяем все маски
    combined_mask = np.zeros(image.shape[:2], dtype=np.uint8)
    
    for color in color_ranges:
        mask = cv2.inRange(hsv, color['lower'], color['upper'])
        combined_mask = cv2.bitwise_or(combined_mask, mask)
    
    # Улучшаем маску
    kernel = np.ones((5,5), np.uint8)
    combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel)
    combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN, kernel)
    
    # Поиск контуров
    contours, _ = cv2.findContours(combined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Фильтрация контуров по размеру и форме
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if 50 < area < 50000:  # Разумные пределы для размера знака
            x, y, w, h = cv2.boundingRect(cnt)
            aspect_ratio = float(w)/h
            
            # Проверяем соотношение сторон (знаки обычно близки к квадрату/кругу)
            if 0.7 < aspect_ratio < 1.5:
                # Вырезаем и обрабатываем область знака
                roi = image[y:y+h, x:x+w]
                roi = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
                roi = cv2.resize(roi, (32, 32)) / 255.0
                
                # Классификация
                pred = model.predict(np.expand_dims(roi, axis=0))
                class_id = np.argmax(pred)
                class_name = class_names[class_id]

                confidence = np.max(pred)
                print(class_name)
                
                if confidence > 0.5:  # Достаточная уверенность
                    return (x, y, x+w, y+h)
    
    return (1,1,1,1)  # Знак не обнаружен