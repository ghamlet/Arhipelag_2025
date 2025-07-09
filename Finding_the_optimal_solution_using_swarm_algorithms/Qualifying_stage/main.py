from pioneer_sdk import Pioneer, Camera
import cv2
import math
import numpy as np
import time

from drone_navigation import (
    CustomPioneer,
    ArucoDetector,
    ArucoMarkerAverager,
    FlightMissionPathPlanner,
    FlightMissionRunner
)



flight_height = float(2)


FULL_MAP_COVERAGE_POINTS = [  (-3, 3.5), (-3, -3.5), (0, -3.5), (0, 3.5), (3, 3.5), (3, -3.5)]

FAST_MAP_POINTS = [ (-1.7, 3.5),  (-1.7, -3.5),  (2.5, -3.5), (2.5, 2.5) ]




if __name__ == "__main__":

    hover_duration = 0   
    
    
    mission = FlightMissionRunner(FAST_MAP_POINTS)
    aruco_detector = ArucoDetector()
    marker_tracker = ArucoMarkerAverager() 

    camera = Camera(ip="127.0.0.1", port=18000, log_connection=True, timeout=4)

    pioneer = CustomPioneer(name="pioneer", ip="127.0.0.1", mavlink_port=8000, connection_method="udpout", 
                           device="dev/serial0", baud=115200, logger=True, log_connection=True)
    
    pioneer.arm()
    pioneer.takeoff()



    first_point = mission.get_next_point()
    x, y = first_point
    pioneer.go_to_local_point(x=x, y=y, z=flight_height, yaw=0)


    while not mission.is_complete():
        
        frame = camera.get_cv_frame()
        if frame is None:
            continue
        
    
        if aruco_detector.detect_markers_presence(frame):
            markers_global = aruco_detector.get_markers_global_positions(frame, pioneer)
            if markers_global:
                    marker_tracker.add_marker_sample(markers_global)

                   
        if pioneer.point_reached_by_faster_mode(threshold=0.5): 
            next_point = mission.get_next_point()
            if next_point:
                x, y = next_point
                pioneer.go_to_local_point(x=x, y=y, z=flight_height, yaw=0)
            


    
    all_avg_coords = marker_tracker.get_all_markers_coords()

    planner = FlightMissionPathPlanner(all_avg_coords, pioneer, verbose=True)
    optimal_route, distance = planner.find_optimal_route()
    coordinates_plan = planner.get_coordinates_plan(optimal_route)
        
    mission = FlightMissionRunner(coordinates_plan)



    first_point = mission.get_next_point()
    x, y = first_point
    pioneer.go_to_local_point(x=x, y=y, z=flight_height, yaw=0)

    hover_duration_over_marker = 2
    landing_pause_time = 0


    while not mission.is_complete():
        if pioneer.point_reached_by_faster_mode(threshold=0.3):  
            
            pioneer.go_to_local_point(x=x, y=y, z=0, yaw=0)
            while not pioneer.point_reached():
                pass

            if mission.has_more_points():
                next_point = mission.get_next_point()
                if next_point:
                    x, y = next_point
                    pioneer.go_to_local_point(x=x, y=y, z=flight_height, yaw=0)
                
      
    pioneer.disarm()
    pioneer.close_connection()

    del pioneer
    