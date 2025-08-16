import cv2
import numpy as np
from matplotlib import pyplot as plt

def remove_glare_with_blur(img, show_steps=False):
    """
    Уменьшает засветы методом размытия и вычитания
    :param img: исходное изображение (BGR)
    :param show_steps: показывать промежуточные шаги обработки
    :return: обработанное изображение
    """
    # 1. Преобразуем в float32 для точных вычислений
    img_float = img.astype(np.float32) / 255.0
    
    # 2. Применяем сильное размытие (захватываем только крупные засветы)
    blurred = cv2.GaussianBlur(img_float, (0, 0), 30)
    
    # 3. Вычитаем размытую версию с коэффициентами
    # Формула: result = img * 1.5 - blurred * 0.5
    result = cv2.addWeighted(img_float, 1.5, blurred, -0.5, 0)
    
    # 4. Обрезаем значения до допустимого диапазона [0,1]
    result = np.clip(result, 0, 1)
    
    # 5. Конвертируем обратно в 8-bit
    result = (result * 255).astype(np.uint8)
    
    if show_steps:
        plt.figure(figsize=(15, 5))
        
        plt.subplot(131)
        plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        plt.title('Original Image')
        
        plt.subplot(132)
        plt.imshow(cv2.cvtColor((blurred*255).astype(np.uint8), cv2.COLOR_BGR2RGB))
        plt.title('Blurred Version')
        
        plt.subplot(133)
        plt.imshow(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
        plt.title('Result After Subtraction')
        
        plt.tight_layout()
        plt.show()
    
    return result

# Пример использования
if __name__ == "__main__":
    # Загрузка изображения с засветами
    img = cv2.imread('Main_stage/images/image_copy2.png')
    
    if img is None:
        print("Ошибка загрузки изображения")
    else:
        # Обработка изображения
        result = remove_glare_with_blur(img, show_steps=True)
        
        # Сохранение результата
        cv2.imwrite('corrected_image.jpg', result)
        
        # Показ результатов
        cv2.imshow('Original', img)
        cv2.imshow('Corrected', result)
        cv2.waitKey(0)
        cv2.destroyAllWindows()