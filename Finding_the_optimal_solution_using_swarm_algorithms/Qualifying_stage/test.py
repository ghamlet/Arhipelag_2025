from pioneer_sdk import Pioneer, Camera
import cv2
import math
import numpy as np
import time

from drone_navigation import (
    CustomPioneer,
    ArucoDetector,
    ArucoMarkerAverager,
    ArucoMarkerPathPlanner,
    FlightMissionRunner
)



flight_height = float(2)



MAP_COVERAGE_POINTS = [  (-3, 3.5), (-3, -3.5), (0, -3.5), (0, 3.5), (3, 3.5), (3, -3.5)]


POINTS_of_markers = [  
                        ["71" , (-0.19, 3.76)   ] ,
                        ["52",  (-3.07, 2.11)   ],
                        ["24",  (2.88, 2.45)    ],
                        ["63",  ( -0.95, -0.34) ],
                        ["35",  (3.03, -1.42)   ], 
                        [ "16", (-3.48, -2.11)  ], 
                        ["47",  (2.07, -3.36)   ]
                        
                    ]








if __name__ == "__main__":

    hover_start_time = None  # Время начала зависания
    hover_duration = 0    # Длительность зависания в секундах
    point_reached = False
    
    
     # Создаем миссию с настройками по умолчанию
    mission = FlightMissionRunner(MAP_COVERAGE_POINTS)

    aruco_detector = ArucoDetector()

    marker_tracker = ArucoMarkerAverager() 

    
    print(f"Всего точек в маршруте: {mission.get_total_points()}")
    print("Маршрут:", mission.points)



    pioneer = CustomPioneer(name="pioneer", ip="127.0.0.1", mavlink_port=8000, connection_method="udpout", 
                           device="dev/serial0", baud=115200, logger=True, log_connection=True)
    

    camera = Camera(ip="127.0.0.1", port=18000, log_connection=True, timeout=4)

    pioneer.arm()
    pioneer.takeoff()



    first_point = mission.get_next_point()
    x, y = first_point
    pioneer.go_to_local_point(x=x, y=y, z=flight_height, yaw=0)





    while not mission.is_flight_complete:
        
        frame = camera.get_cv_frame()
        if frame is None:
            continue
        
    

        if aruco_detector.detect_markers_presence(frame, visual=True):

            markers_global = aruco_detector.get_markers_global_positions(frame, pioneer, verbose=False)
            if markers_global:
                # Добавление измерений
                    marker_tracker.add_marker_sample(markers_global)

                   
                                                
        
               

        if pioneer.point_reached():  # если дрон долетел до нужной точки или же он только летит на первую точку, то можно давать новую команду
            pioneer.start_hover(hover_duration)  # Просто запускаем зависание

            
        if pioneer.check_hover_complete():
            next_point = mission.get_next_point()
            if next_point:
                x, y = next_point
                pioneer.go_to_local_point(x=x, y=y, z=flight_height, yaw=0)
            

            



        cv2.imshow("frame", frame)
        key = cv2.waitKey(1) 

        if key == ord('q'):  
            cv2.destroyAllWindows()

            pioneer.land()
            pioneer.close_connection()
            del pioneer
            break




    print(f"Миссия завершена! Пройдено: {mission.get_current_progress():.1f}%")
    cv2.destroyAllWindows()

    
    all_avg_coords = marker_tracker.get_all_markers_coords()
    print(f"Все усредненные координаты: ", all_avg_coords)



    planner = ArucoMarkerPathPlanner(all_avg_coords, pioneer)
    coordinates_plan = planner.get_coordinates_plan() 
    print("План полёта (координаты):", coordinates_plan)

    
    marker_tracker = ArucoMarkerAverager() 



    mission = FlightMissionRunner(coordinates_plan)
    first_point = mission.get_next_point()
    x, y = first_point
    pioneer.go_to_local_point(x=x, y=y, z=flight_height, yaw=0)

    hover_duration_over_marker = 2
    landing_pause_time = 2



    while not mission.is_flight_complete:
        
        frame = camera.get_cv_frame()
        if frame is None:
            continue

        

        if pioneer.point_reached():  # если дрон долетел до нужной точки или же он только летит на первую точку, то можно давать новую команду
            start_time = time.time()

            while time.time() - start_time < hover_duration_over_marker:
                frame = camera.get_cv_frame()
                if frame is None:
                    continue
                

                if aruco_detector.detect_markers_presence(frame, visual=True):
                    markers_ids = aruco_detector.get_detected_markers_ids(frame)
                    
                    if len(markers_ids) == 1:
                        markers_global = aruco_detector.get_markers_global_positions(frame, pioneer, verbose=False)
                        if markers_global:
                            marker_tracker.add_marker_sample(markers_global)



            marker_x, marker_y = marker_tracker.get_marker_coords_by_id(markers_ids[0])


            pioneer.go_to_local_point(x=marker_x, y=marker_y, z=1, yaw=0)
            while not pioneer.point_reached():
                pass

            pioneer.go_to_local_point(x=marker_x, y=marker_y, z=0, yaw=0)
            while not pioneer.point_reached():
                pass


            pioneer.start_hover(landing_pause_time)



        if pioneer.check_hover_complete():
            if mission.has_more_points():
                pioneer.takeoff()

                next_point = mission.get_next_point()
                if next_point:
                    x, y = next_point
                    pioneer.go_to_local_point(x=x, y=y, z=flight_height, yaw=0)
                




        cv2.imshow("frame", frame)
        key = cv2.waitKey(1) 

        if key == ord('q'):  
            cv2.destroyAllWindows()

            pioneer.land()
            pioneer.close_connection()
            del pioneer
            break

    
    pioneer.close_connection()
    del pioneer
    