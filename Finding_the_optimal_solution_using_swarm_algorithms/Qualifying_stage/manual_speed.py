from pioneer_sdk import Pioneer, Camera
import time

from aruco_detector import ArucoDetector

if __name__ == "__main__":
    print("start")
    pioneer = Pioneer(name="pioneer", ip="127.0.0.1", mavlink_port=8000, connection_method="udpout", 
                           device="dev/serial0", baud=115200, logger=True, log_connection=True)
    
    aruco_detector = ArucoDetector()
    camera = Camera(ip="127.0.0.1", port=18000, log_connection=True, timeout=4)

    pioneer.arm()
    pioneer.takeoff()




    pioneer.go_to_local_point(x=0, y=0, z=2, yaw=0)
    while not pioneer.point_reached():
        pass

    
    while True:
        frame = camera.get_cv_frame()
        if frame is None:
            continue
        
    

        aruco_detector.detect_markers_presence(frame, visual=True)

    
    # high = pioneer.get_dist_sensor_data(get_last_received=True)


    # while high > 0.7:
    #     pioneer.set_manual_speed_body_fixed(vx= 0, vy=0, vz=-1, yaw_rate=0)
    #     time.sleep(0.01)
    #     high = pioneer.get_dist_sensor_data(get_last_received=True)



    # for i in range(10):
    #     pioneer.set_manual_speed_body_fixed(vx= 0, vy=0, vz=0, yaw_rate=0)

    # time.sleep(5)
   




    # while True:
    #     print(pioneer.get_dist_sensor_data(get_last_received=True))




    # while True:
    #     # по диагонали
    #     pioneer_mini.set_manual_speed_body_fixed(vx= 1, vy=-1, vz=0, yaw_rate=0)  # значения не играют роли. важен знак чисел
    #     time.sleep(0.05)

      
    #         for i in range(10):
    #             pioneer_mini.set_manual_speed_body_fixed(vx= 0, vy=0, vz=0, yaw_rate=0)
    #         break

        
    #     # через 10 секунд остановится
    #     if time.time() - t > 10:
    #         break

   

    pioneer.land()

    pioneer.close_connection()
    del pioneer
