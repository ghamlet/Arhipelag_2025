#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import cv2
import numpy as np
from pioneer_sdk import Pioneer, Camera
import time

def main():
    # Инициализация дрона и камеры
    pioneer = Pioneer(name="pioneer", ip="127.0.0.1", mavlink_port=8000, connection_method="udpout", 
                           device="dev/serial0", baud=115200, logger=True, log_connection=True)
    

    camera = Camera(ip="127.0.0.1", port=18000, log_connection=True, timeout=4)

    try:
        print("Взлетаем...")
        pioneer.arm()
        pioneer.takeoff()



        # Координаты целевой точки (x, y, z в метрах)
        target_x, target_y, target_z = 0, 3, 2
        
        print(f"Летим к точке ({target_x}, {target_y}, {target_z})")
        pioneer.go_to_local_point(x=target_x, y=target_y, z=target_z, yaw=0)
        pioneer.set_manual_speed(2, 2,  0.2, 0)

        
        while not pioneer.point_reached():
            time.sleep(0.01)
            pass 


        print("Начинаем поиск маркера...")
        
        # Параметры ArUco
        aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_100)
        aruco_params = cv2.aruco.DetectorParameters()
        detector = cv2.aruco.ArucoDetector(aruco_dict, aruco_params)
        
        # Разрешение камеры
        frame_width, frame_height = 640, 480
        frame_center = (frame_width//2, frame_height//2)
        


        while True:
            # Получаем кадр с камеры
            frame = camera.get_cv_frame()
            if frame is None:
                continue
            
            # Детектим маркеры
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            corners, ids, _ = detector.detectMarkers(gray)
            
            if ids is not None:
                # Берем первый найденный маркер
                marker_corners = corners[0][0]
                center_x = int(marker_corners[:, 0].mean())
                center_y = int(marker_corners[:, 1].mean())
                center = (center_x, center_y)
                
                # Рисуем центр кадра (зеленый)
                cv2.circle(frame, frame_center, 5, (0, 255, 0), -1)
                cv2.circle(frame, center, 5, (0, 0, 255), -1)
                cv2.arrowedLine(frame, frame_center, center, (255, 0, 0), 2)
                vector = (center[0]-frame_center[0], center[1]-frame_center[1])
                cv2.putText(frame, f"Vector: ({vector[0]}, {vector[1]})", 
                           (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
            
            # Показываем кадр
            cv2.imshow("Marker Tracking", frame)
            
            # Выход по ESC
            if cv2.waitKey(1) == 27:
                break
                
    except Exception as e:
        print(f"Ошибка: {e}")
    finally:
        print("Завершение работы...")
        cv2.destroyAllWindows()
        pioneer.land()
        pioneer.close_connection()



if __name__ == "__main__":
    main()