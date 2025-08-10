from pioneer_sdk import Pioneer, Camera
import cv2
import time
from flight_utils import load_flight_coordinates, get_config
from drone_navigation import FlightMissionRunner, CustomPioneer, WebcamStream, YoloRKNN
from ultralytics import YOLO
import time  
from pathlib import Path
from threading import (
    Thread,
)

import cv2
from datetime import datetime

import numpy as np






if __name__ == "__main__":
    try:
        flight_height = 1.5

        MAP_POINTS = load_flight_coordinates()



        webcam_stream = WebcamStream(stream_id=0)
        

        webcam_stream.start()  # processing frames in input stream
        video_writer = webcam_stream.create_videowriter()
        num_frames_processed = 0


        pioneer_conf = get_config('local')  # или 'global'   local

        # Инициализация миссии
        mission = FlightMissionRunner(MAP_POINTS)
        
        # Инициализация дрона
        pioneer = CustomPioneer(
            name="pioneer",
            ip=pioneer_conf["ip"],
            mavlink_port=pioneer_conf["port"],
            connection_method="udpout",
            device="dev/serial0",
            baud=115200,
            verbose=True
        )

        

        pioneer.arm()
        pioneer.takeoff()
        
        # Начало миссии
        first_point = mission.get_next_point()
        x, y, z = first_point
        pioneer.go_to_local_point(x=x, y=y, z=flight_height, yaw=0)
        

        # Основной цикл миссии
        while not mission.is_complete():
            if webcam_stream.stopped:
                break


            _, frame = webcam_stream.read()
            
            video_writer.write(frame)


            cv2.imshow("frame", frame)
            key = cv2.waitKey(10)
            if key == ord("q"):
                break



            if pioneer.point_reached(threshold=0.3):
                # Переход к следующей точке
                next_point = mission.get_next_point()
                if next_point:
                    x, y, z = next_point
                    pioneer.go_to_local_point(x=x, y=y, z=flight_height, yaw=0)
            

           

    except Exception as e:
        print(f"Ошибка обработки: {str(e)}")
    
              

    except KeyboardInterrupt:
        print("\nПолучен сигнал прерывания (Ctrl+C)")
        video_writer.release()
        cv2.destroyAllWindows()
    
    except Exception as e:
        print(f"Критическая ошибка: {str(e)}")
    
    finally:
        print("Завершение работы")
        pioneer.land()
        pioneer.disarm()
        pioneer.close_connection()
        video_writer.release()

        cv2.destroyAllWindows()