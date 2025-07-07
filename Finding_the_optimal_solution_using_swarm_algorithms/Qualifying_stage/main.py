from pioneer_sdk import Pioneer, Camera
import cv2
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



if __name__ == "__main__":

    hover_start_time = None  
    hover_duration = 0    
    point_reached = False
    
    
    mission = FlightMissionRunner(MAP_COVERAGE_POINTS)
    aruco_detector = ArucoDetector()
    marker_averager = ArucoMarkerAverager() 


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
        

        if aruco_detector.detect_markers_presence(frame):
            markers_global = aruco_detector.get_markers_global_positions(frame, pioneer)
            if markers_global:
                marker_averager.add_marker_sample(markers_global)

                                                   
        if pioneer.point_reached():  
            pioneer.start_hover(hover_duration)  

        
        if pioneer.check_hover_complete():
            next_point = mission.get_next_point()
            if next_point:
                x, y = next_point
                pioneer.go_to_local_point(x=x, y=y, z=flight_height, yaw=0)

    
    all_avg_coords = marker_averager.get_all_markers_coords()

    planner = ArucoMarkerPathPlanner(all_avg_coords, pioneer)
    coordinates_plan = planner.get_coordinates_plan() 

    marker_averager = ArucoMarkerAverager() 

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

        
        if pioneer.point_reached(): 
            start_time = time.time()

            while time.time() - start_time < hover_duration_over_marker:
                frame = camera.get_cv_frame()
                if frame is None:
                    continue
                
                if aruco_detector.detect_markers_presence(frame):
                    markers_ids = aruco_detector.get_detected_markers_ids(frame)
                    
                    if len(markers_ids) == 1:
                        markers_global = aruco_detector.get_markers_global_positions(frame, pioneer)
                        if markers_global:
                            marker_averager.add_marker_sample(markers_global)



            marker_x, marker_y = marker_averager.get_marker_coords_by_id(markers_ids[0])
            pioneer.go_to_local_point(x=marker_x, y=marker_y, z=1, yaw=0)
            while not pioneer.point_reached():
                pass

            pioneer.go_to_local_point(x=marker_x, y=marker_y, z=0, yaw=0)
            while not pioneer.point_reached():
                pass

            pioneer.start_hover(landing_pause_time)


        if pioneer.check_hover_complete():
            if mission.has_more_points():

                next_point = mission.get_next_point()
                if next_point:
                    x, y = next_point
                    pioneer.go_to_local_point(x=x, y=y, z=flight_height, yaw=0)
                
    
    pioneer.close_connection()
    del pioneer
    