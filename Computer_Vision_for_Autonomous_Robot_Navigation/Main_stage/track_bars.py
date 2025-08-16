import cv2
import os


class ColorTracker:
    
    def __init__(self):
        self.main_dir = os.path.dirname(os.path.abspath(__file__))
        
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


    def process_frame(self, frame):
        """Обрабатывает кадр, применяя маску по цвету"""
        
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        self.get_trackbar_positions()

        mask = cv2.inRange(hsv, (self.minb, self.ming, self.minr), (self.maxb, self.maxg, self.maxr))
        result = cv2.bitwise_and(frame, frame, mask=mask)

        cv2.imshow('result', result)

        k = cv2.waitKey(1)
        if k == ord('q'):
            exit()
            
        elif k == ord('s'):
            self.save_thresholds()


    def save_thresholds(self):
        """Сохраняет значения порогов в текстовый файл"""
        
        with open(os.path.join(self.main_dir, "trackbars_save.txt"), "a") as f:
            title = input("\nEnter the description \nTo cancel, write no: ")
            
            if title.lower() != "no":
                f.write(f"{title}:  {self.minb, self.ming, self.minr}, {self.maxb, self.maxg, self.maxr}\n")
                print("Saved\n")
                


