from pioneer_sdk import Pioneer, Camera
import cv2
import time
from collections import defaultdict, deque
from flight_utils import load_flight_coordinates, get_config
from drone_navigation import FlightMissionRunner, CustomPioneer
from ultralytics import YOLO

from object_detector import ObjectDetector

# Настройки детекции
# DETECTION_THRESHOLD = 0.7  # Порог уверенности
# MIN_DETECTIONS = 5         # Минимальное количество положительных детекций
# MAX_FRAMES = 50            # Максимальное количество кадров для анализа
# HISTORY_SIZE = 10          # Размер буфера для трекинга объектов
# MIN_CONFIDENCE = 0.7       # Минимальная уверенность для учета детекции
# REQUIRED_CONSECUTIVE = 5   # Требуемое количество последовательных детекций
# MIN_CLASS_RATIO = 0.6      # Минимальный процент класса среди всех детекций


COORDS_TEST = [

    [0,0,1],
]


if __name__ == "__main__":
    try:
        model = YOLO('Training_a_neural_network_to_find_an_object_of_interest/best_loss07.pt', verbose=False)
        flight_height = 1.5

        # MAP_POINTS = load_flight_coordinates()
        MAP_POINTS = COORDS_TEST

        pioneer_conf = get_config('global')  # или 'global'

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

        # Инициализация камеры
        video_path = 'Training_a_neural_network_to_find_an_object_of_interest/videos/output_pascal_line2.mp4'
        camera = cv2.VideoCapture(video_path)

        # Инициализация
        detector = ObjectDetector(
            model=model,
            camera=camera,
            class_names=model.names  # Словарь классов из YOLO
        )

        # Взлет
        pioneer.arm()
        pioneer.takeoff()
        
        # Начало миссии
        first_point = mission.get_next_point()
        x, y, z = first_point
        pioneer.go_to_local_point(x=x, y=y, z=flight_height, yaw=0)
        

        # Основной цикл миссии
        while not mission.is_complete():
            ret, frame = camera.read()
            if not ret:
                break



            if pioneer.point_reached(threshold=0.3):


                # detected_class, best_frame = detector.analyze_point()
                # if detected_class:
                #     print(f"Обнаружен объект класса: {detected_class}")
                #     # Дополнительные действия при обнаружении
                
                

                # Переход к следующей точке
                next_point = mission.get_next_point()
                if next_point:
                    x, y, z = next_point
                    pioneer.go_to_local_point(x=x, y=y, z=flight_height, yaw=0)
            

            # Отображение основного потока
            cv2.imshow("Flight View", frame)
            if cv2.waitKey(100) & 0xFF == ord('q'):
                break




    except Exception as e:
        print(f"Ошибка обработки: {str(e)}")
    
              

    except KeyboardInterrupt:
        print("\nПолучен сигнал прерывания (Ctrl+C)")
    
    except Exception as e:
        print(f"Критическая ошибка: {str(e)}")
    
    # finally:
    #     print("Завершение работы")
    #     pioneer.land()
    #     pioneer.disarm()
    #     pioneer.close_connection()
    #     cv2.destroyAllWindows()