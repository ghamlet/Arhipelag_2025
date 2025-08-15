import cv2
import numpy as np

def transfer_white_pixels_mask(source_img, target_img, threshold=200):
    """
    Переносит белые пиксели используя маски numpy
    """
    # Преобразуем в оттенки серого
    if len(source_img.shape) == 3:
        source_gray = cv2.cvtColor(source_img, cv2.COLOR_BGR2GRAY)
    else:
        source_gray = source_img.copy()
    
    result = target_img.copy()
    
    # Создаем маску белых пикселей
    white_mask = source_gray >= threshold
    
    # Применяем маску
    if len(result.shape) == 3:  # Цветное изображение
        result[white_mask] = [255, 255, 255]
    else:  # Градации серого
        result[white_mask] = 255
    
    return result




# Пример использования
if __name__ == "__main__":
    # Загружаем изображения
    source = cv2.imread('Computer_Vision_for_Autonomous_Robot_Navigation/user_task/materials/vertical_marking.jpg')
    target = cv2.imread('Computer_Vision_for_Autonomous_Robot_Navigation/user_task/materials/horizontal_marking.jpg')
    
    # Убеждаемся, что размеры совпадают
    if source.shape != target.shape:
        target = cv2.resize(target, (source.shape[1], source.shape[0]))
    
    # Переносим белые пиксели
    result = transfer_white_pixels_mask(source, target)
    
    # Сохраняем результат
    cv2.imwrite( "Computer_Vision_for_Autonomous_Robot_Navigation/user_task/"+'result.jpg', result)
    
    # Показываем результат
    cv2.imshow('Source', source)
    cv2.imshow('Target', target)
    cv2.imshow('Result', result)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

