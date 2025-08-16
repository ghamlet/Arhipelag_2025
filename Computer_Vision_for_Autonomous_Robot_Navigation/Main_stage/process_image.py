import cv2
import os


class ColorTracker:
    
    def __init__(self, image_path):
        self.main_dir = os.path.dirname(os.path.abspath(__file__))
        self.image = cv2.imread(image_path)
        
        if self.image is None:
            raise FileNotFoundError(f"Image not found at {image_path}")
        
        self.minb, self.ming, self.minr = 0, 0, 0
        self.maxb, self.maxg, self.maxr = 255, 255, 255
        self.create_trackbar()


    def create_trackbar(self):
        """Создает трекбары для настройки цветовых порогов"""
        cv2.namedWindow("trackbar")
        cv2.createTrackbar('minb', 'trackbar', self.minb, 255, lambda x: None)
        cv2.createTrackbar('ming', 'trackbar', self.ming, 255, lambda x: None)
        cv2.createTrackbar('minr', 'trackbar', self.minr, 255, lambda x: None)
        cv2.createTrackbar('maxb', 'trackbar', self.maxb, 255, lambda x: None)
        cv2.createTrackbar('maxg', 'trackbar', self.maxg, 255, lambda x: None)
        cv2.createTrackbar('maxr', 'trackbar', self.maxr, 255, lambda x: None)


    def get_trackbar_positions(self):
        """Получает текущие значения трекбаров"""
        self.minb = cv2.getTrackbarPos('minb', 'trackbar')
        self.ming = cv2.getTrackbarPos('ming', 'trackbar')
        self.minr = cv2.getTrackbarPos('minr', 'trackbar')
        self.maxb = cv2.getTrackbarPos('maxb', 'trackbar')
        self.maxg = cv2.getTrackbarPos('maxg', 'trackbar')
        self.maxr = cv2.getTrackbarPos('maxr', 'trackbar')


    def process_frame(self):
        """Обрабатывает кадр, применяя маску по цвету"""
        hsv = cv2.cvtColor(self.image, cv2.COLOR_BGR2HSV)
        self.get_trackbar_positions()

        mask = cv2.inRange(hsv, (self.minb, self.ming, self.minr), (self.maxb, self.maxg, self.maxr))
        result = cv2.bitwise_and(self.image, self.image, mask=mask)

        cv2.imshow('Original', self.image)
        cv2.imshow('Mask', mask)
        cv2.imshow('Result', result)


    def save_thresholds(self):
        """Сохраняет значения порогов в текстовый файл"""
        with open(os.path.join(self.main_dir, "trackbars_save.txt"), "a") as f:
            title = input("\nEnter the description \nTo cancel, write 'no': ")
            
            if title.lower() != "no":
                f.write(f"{title}:  {self.minb, self.ming, self.minr}, {self.maxb, self.maxg, self.maxr}\n")
                print("Saved\n")


if __name__ == "__main__":
    # Укажите путь к изображению
    image_path = "Main_stage/image.png"  # Замените на свой путь
    
    tracker = ColorTracker(image_path)
    
    while True:
        tracker.process_frame()
        
        k = cv2.waitKey(1) & 0xFF
        if k == ord('q'):  # Выход
            break
        elif k == ord('s'):  # Сохранить настройки
            tracker.save_thresholds()
    
    cv2.destroyAllWindows()