import cv2
import numpy as np
from color_detection_system import ColorDetectionSystem


def preliminary_operations():
    """
    Инициализирует систему детектирования цветов и загружает эталонные изображения.
    """
    try:
        frame_h = cv2.imread("Computer_Vision_for_Autonomous_Robot_Navigation/user_task/materials/horizontal_marking.jpg")
        frame_v = cv2.imread("Computer_Vision_for_Autonomous_Robot_Navigation/user_task/materials/vertical_marking.jpg")
        main_net = cv2.imread("Computer_Vision_for_Autonomous_Robot_Navigation/user_task/result.jpg")
        
        if frame_h is None or frame_v is None or main_net is None:
            raise FileNotFoundError("Не удалось загрузить эталонные изображения")
            
        return [frame_h, frame_v, main_net]
    
    except Exception as e:
        print(f"Ошибка: {str(e)}")
        return []


def predict_color_and_angle(image, etalon_frames):
    """
    Основная функция для детектирования цветных объектов и определения их углов.
    """
    try:
        if not isinstance(image, np.ndarray) or len(image.shape) != 3:
            raise ValueError("Некорректное входное изображение")
            
        if len(etalon_frames) != 3:
            raise ValueError("Некорректные эталонные изображения")
        

        results = ColorDetectionSystem.process_image(image, tuple(etalon_frames))
        return results
        
    except Exception as e:
        print(f"Ошибка: {str(e)}")
        return []