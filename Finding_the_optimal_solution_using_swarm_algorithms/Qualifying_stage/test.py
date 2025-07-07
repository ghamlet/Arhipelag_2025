from pioneer_sdk import Pioneer, Camera
import cv2
import math
import numpy as np
import time
from custom_pioneer import CustomPioneer

flight_height = float(2)


takeoff = True
last_point_reached = False

# полное покрытие
POINTS = [  (-3, 3.5), (-3, -3.5), (0, -3.5), (0, 3.5), (3, 3.5), (3, -3.5)]


POINTS_of_markers = [  
                        ["71" , (-0.19, 3.76)   ] ,
                        ["52",  (-3.07, 2.11)   ],
                        ["24",  (2.88, 2.45)    ],
                        ["63",  ( -0.95, -0.34) ],
                        ["35",  (3.03, -1.42)   ], 
                        [ "16", (-3.48, -2.11)  ], 
                        ["47",  (2.07, -3.36)   ]
                        
                    ]



def find_aruco_marker_coords_v1(drone_coords: list, marker_offset: list, marker_id: int):
    """
    Вычисляет глобальные координаты маркера на основе координат дрона и смещения маркера относительно дрона.
    
    Параметры:
        drone_coords: текущие координаты дрона [x, y, z]
        marker_offset: смещение маркера относительно дрона [[x], [y], [z]]
        marker_id: ID обнаруженного маркера (для информационных сообщений)
        
    Возвращает:
        tuple: (marker_x, marker_y) или None если смещение слишком большое
        
    """

     # # определяет координаты маркера относительно камеры с помощью перспективно-точечного алгоритма (Perspective-n-Point)
                # # success, rvecs, tvecs = cv2.solvePnP(points_of_marker, marker_corners, camera_matrix, dist_coeffs)
                # # print(ids, tvecs, "\n")
                # # marker_coords = find_aruco_marker_coords_v1(drone_coords, tvecs, marker_id)            
                # cv2.drawFrameAxes(frame, camera_matrix, dist_coeffs, rvecs, tvecs, 0.1)


    MARKER_OFFSET_THRESHOLD = 0.1   # порогового значения смещения маркера


    # Проверяем, что смещение по модулю меньше 0.5
    offset_x = abs(marker_offset[0][0])
    offset_y = abs(marker_offset[1][0])
    
    if offset_x > MARKER_OFFSET_THRESHOLD or offset_y > MARKER_OFFSET_THRESHOLD:
        # print(f"Маркер {marker_id} отклонен: смещение слишком большое (x={offset_x:.4f}, y={offset_y:.4f})")
        return None
    
    # Получаем координаты дрона
    drone_x, drone_y = drone_coords[0], drone_coords[1]
    
    # Получаем смещение маркера
    marker_offset_x = marker_offset[0][0]
    marker_offset_y = marker_offset[1][0]
    print(marker_offset_x, marker_offset_y)
    
    # Вычисляем глобальные координаты маркера
    marker_global_x = drone_x + marker_offset_x
    marker_global_y = drone_y + marker_offset_y
    
    print(f"Глобальные координаты маркера {marker_id} : x={marker_global_x:.4f}, y={marker_global_y:.4f}")
    

    return marker_global_x, marker_global_y





def detect_aruco_markers_presence(frame, aruco_detector: cv2.aruco.ArucoDetector, visual=False):
    """
    Определяет наличие ArUco маркеров на изображении.
    
    Args:
        frame: Входное изображение
        aruco_detector: Детектор ArUco маркеров
    
    Returns:
        bool: True если маркеры обнаружены, False в противном случае
    """

    corners, ids, _ = aruco_detector.detectMarkers(frame)
    if ids is not None:  

        if visual: 
            frame_copy = frame.copy()
            cv2.aruco.drawDetectedMarkers(frame_copy, corners, ids)
            cv2.imshow("DetectedMarkers", frame_copy)
            cv2.waitKey(1)

        return True
    
    elif visual:
        cv2.imshow("DetectedMarkers", frame)
        cv2.waitKey(1)


    return False 
    
    



def get_aruco_markers_image_coordinates(frame, aruco_detector):
    """
    Находит координаты центров ArUco маркеров на изображении.
    
    Args:
        frame: Входное изображение
        aruco_detector: Детектор ArUco маркеров
    
    Returns:
        dict: {marker_id: (center_x, center_y)} или пустой словарь, если маркеры не найдены
    """

    markers = {}
    corners, ids, _ = aruco_detector.detectMarkers(frame)
    
    if ids is not None:
        for i in range(len(ids)):
            marker_id = ids[i][0]
            marker_corners = corners[i][0]
            center_x = int(marker_corners[:, 0].mean())
            center_y = int(marker_corners[:, 1].mean())
            markers[marker_id] = (center_x, center_y)
    
    return markers




def get_aruco_markers_global_positions(frame, aruco_detector, pioneer, verboose= False):
    """
    Получает глобальные координаты всех обнаруженных ArUco маркеров, используя вспомогательные функции.
    
    Args:
        frame: Входное изображение
        aruco_detector: Детектор ArUco маркеров
        pioneer: Объект для получения позиции дрона
        image_size: Размеры изображения (width, height) (по умолчанию 640x480)
        ground_cover: Размер покрываемой местности в метрах (width, height) (по умолчанию 4.4x3.4)
    
    Returns:
        dict: Словарь {marker_id: (global_x, global_y)} или пустой словарь, если маркеры не найдены
    """

    global_markers = {}


    image_size=(640, 480)
    ground_cover=(4.4, 3.4)
    
    # Проверяем наличие маркеров через вспомогательную функцию
    if not detect_aruco_markers_presence(frame, aruco_detector):
        # if verboose:
        #     print("Маркеры не обнаружены")

        return None
    
    # Получаем координаты центров через вспомогательную функцию
    image_markers = get_aruco_markers_image_coordinates(frame, aruco_detector)
    
    # Получаем позицию дрона
    drone_pos = pioneer.get_local_position_lps(get_last_received=True)
    drone_x, drone_y = drone_pos[:2]
    
    # Параметры преобразования
    img_width, img_height = image_size
    ground_width, ground_height = ground_cover
    scale_x = ground_width / img_width
    scale_y = ground_height / img_height
    img_center_x = img_width / 2
    img_center_y = img_height / 2
    
    # Рассчитываем глобальные координаты
    for marker_id, (marker_x, marker_y) in image_markers.items():
        global_x = drone_x + (marker_x - img_center_x) * scale_x
        global_y = drone_y - (marker_y - img_center_y) * scale_y
        global_markers[marker_id] = (global_x, global_y)


    if verboose:
        if global_markers:
            for marker_id, coords in global_markers.items():
                print(f"Маркер {marker_id}: {coords}")


    return global_markers





# def load_coefficients(path):
#     cv_file = cv2.FileStorage(path, cv2.FILE_STORAGE_READ)
#     if not cv_file.isOpened():
#         raise ValueError(f"Не удалось открыть файл {path}")

#     camera_matrix = cv_file.getNode("mtx").mat()
#     dist_coeffs = cv_file.getNode("dist").mat()

#     cv_file.release()

#     return camera_matrix, dist_coeffs



# Dictionary of aruco-markers
aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_100)  # 100 mm
# Parameters for marker detection (in this case, default parameters)
aruco_params = cv2.aruco.DetectorParameters()
# Create instance of ArucoDetector.
# Required starting from version opencv 4.7.0
aruco_detector = cv2.aruco.ArucoDetector(aruco_dict, aruco_params)


# Load camera matrix and distortion coefficients from file
# camera_matrix, dist_coeffs = load_coefficients("optimized_camera_params.yml")

# size_of_marker = 0.1  # side length in meters

# # Coordinates of marker corners
# points_of_marker = np.array(
#     [
#         (size_of_marker / 2, -size_of_marker / 2, 0),
#         (-size_of_marker / 2, -size_of_marker / 2, 0),
#         (-size_of_marker / 2, size_of_marker / 2, 0),
#         (size_of_marker / 2, size_of_marker / 2, 0),
#     ]
# )



class MarkerTracker:
    def __init__(self, min_samples=10, max_samples=100):
        """
        Инициализация трекера для хранения и усреднения координат маркеров.
        
        Параметры:
            min_samples: минимальное количество измерений для усреднения (по умолчанию 10)
            max_samples: максимальное количество хранимых измерений (по умолчанию 100)
        """
        self.marker_data = {}  # {id: {'samples': [(x1,y1), ...], 'avg_coords': (x,y)}}
        self.min_samples = min_samples
        self.max_samples = max_samples
    
    def add_marker_sample(self, markers_dict):
        """
        Добавляет новые координаты для всех маркеров из словаря.
        
        Параметры:
            markers_dict: словарь {marker_id: (x, y)} с текущими координатами маркеров
                         или None/пустой словарь, если маркеры не обнаружены
        """
        if not markers_dict:
            return
        
        for marker_id, coords in markers_dict.items():
            if coords is None:
                continue
                
            # Инициализация записи для нового маркера
            if marker_id not in self.marker_data:
                self.marker_data[marker_id] = {
                    'samples': [],
                    'avg_coords': None
                }
            
            # Добавление новых координат
            samples = self.marker_data[marker_id]['samples']
            samples.append(coords)
            
            # Ограничение истории измерений
            if len(samples) > self.max_samples:
                samples.pop(0)
            
            # Вычисление среднего при достаточном количестве образцов
            if len(samples) >= self.min_samples:
                avg_x = round(sum(x for x, y in samples) / len(samples), 2)  # Округление
                avg_y = round(sum(y for x, y in samples) / len(samples), 2)  # Округление
                self.marker_data[marker_id]['avg_coords'] = (avg_x, avg_y)
                
                # print(f"Усредненные координаты маркера {marker_id}: "
                #       f"({avg_x:.2f}, {avg_y:.2f}) на основе {len(samples)} измерений")

    def get_marker_coords_by_id(self, marker_id):
        """
        Возвращает усредненные координаты для указанного маркера по его ID.
        
        Параметры:
            marker_id: ID маркера (целое число)
            
        Возвращает:
            tuple: (avg_x, avg_y) средние координаты (округленные до 2 знаков) 
                  или None, если данных недостаточно
        """
        marker = self.marker_data.get(marker_id)
        if marker and marker['avg_coords']:
            return (round(marker['avg_coords'][0], 2), 
                    round(marker['avg_coords'][1], 2))
        return None


    def get_all_markers_coords(self):
        """
        Возвращает усредненные координаты всех отслеживаемых маркеров.
        
        Возвращает:
            dict: {marker_id: (avg_x, avg_y)} словарь с координатами (округленными до 2 знаков)
                  только тех маркеров, для которых есть усредненные данные
        """
        return {
            marker_id: (round(data['avg_coords'][0], 2), 
                      round(data['avg_coords'][1], 2))
            for marker_id, data in self.marker_data.items()
            if data['avg_coords'] is not None
        }
   




class FlightPlanner:
    def __init__(self, all_avg_coords, pioneer_instance:Pioneer):
        """
        Инициализация планировщика полета.
        
        Параметры:
            all_avg_coords: словарь {marker_id: (x, y)} с координатами маркеров
            pioneer: экземпляр класса Pioneer (опционально)
            start_position: начальная позиция дрона (если None, получаем от дрона)
        """
        self.all_avg_coords = all_avg_coords
        self.pioneer = pioneer_instance
        self.start_position = self._get_current_position_drone()
        self.marker_graph = self._build_marker_graph()
        self._flight_plan = self._generate_flight_plan()  # Автоматически генерируем план при создании
    

    def _get_current_position_drone(self):
        """Получает текущую позицию дрона через класс Pioneer."""
        return self.pioneer.get_local_position_lps(get_last_received=True)[:2]
    

    def _build_marker_graph(self):
        """Строит граф переходов между маркерами на основе их ID."""
        graph = {}
        for marker_id in self.all_avg_coords:
            first_digit = marker_id // 10  # Первая цифра - текущий маркер
            second_digit = marker_id % 10  # Вторая цифра - следующий маркер
            graph[first_digit] = second_digit
        return graph
    
    def _find_closest_marker(self, position):
        """Находит маркер, ближайший к указанной позиции."""
        closest_id = None
        min_distance = float('inf')
        
        for marker_id, coords in self.all_avg_coords.items():
            distance = math.dist(position, coords)
            if distance < min_distance:
                min_distance = distance
                closest_id = marker_id
        return closest_id
    


    def _generate_flight_plan(self):
        """Генерирует план полёта по маркерам (вызывается автоматически при инициализации)."""
        current_position = self._get_current_position_drone()
        visited = set()
        flight_plan = []
        
        # Находим стартовый маркер
        current_marker_id = self._find_closest_marker(current_position)
        flight_plan.append(current_marker_id)
        visited.add(current_marker_id // 10)  # Добавляем первую цифру маркера
        
        # Строим маршрут по графу
        while True:
            current_node = current_marker_id // 10
            next_node = self.marker_graph.get(current_node)
            
            # Проверяем условия завершения
            if next_node is None or next_node in visited:
                break
                
            # Находим маркер, соответствующий следующему узлу
            next_marker_candidates = [
                marker_id for marker_id in self.all_avg_coords 
                if marker_id // 10 == next_node and marker_id not in visited
            ]
            
            if not next_marker_candidates:
                break
                
            next_marker_id = next_marker_candidates[0]
            flight_plan.append(next_marker_id)
            visited.add(next_node)
            current_marker_id = next_marker_id
            
        return flight_plan
    
    def get_flight_plan(self):
        """Возвращает сгенерированный план полёта (ID маркеров)."""
        return self._flight_plan
    
    def get_coordinates_plan(self, flight_plan=None):
        """
        Возвращает план полёта в виде координат.
        
        Параметры:
            flight_plan: если None, используется сохранённый план
        """
        if flight_plan is None:
            flight_plan = self._flight_plan
        return [self.all_avg_coords[marker_id] for marker_id in flight_plan]




class FlightMission:
    def __init__(self, points=None, start_x=-4, start_y=4, field_size=10, drone_cover=2):
        """Инициализация миссии с параметрами поля и обзора дрона"""
        self.start_x = start_x
        self.start_y = start_y
        self.field_size = field_size
        self.drone_cover = drone_cover
        self.is_flight_complete = False

        self.points = points if points else self._generate_flight_path()
        self.current_index = 0
    

    def _generate_flight_path(self):
        """Генерирует маршрут 'змейкой' с учётом параметров поля"""
        step = self.drone_cover
        half_field = self.field_size // 2
        points = []
        y = self.start_y
        
        for row in range(int(self.field_size / self.drone_cover)):
            if row % 2 == 0:
                # Движение вправо
                x_range = range(-half_field + 1, half_field, step)
            else:
                # Движение влево
                x_range = range(half_field - 1, -half_field, -step)
            
            for x in x_range:
                points.append((x, y))
            
            y -= step  # Переход на следующую строку
        
        return points

    
    def has_more_points(self):
        """Проверяет, остались ли точки для посещения"""
        if self.current_index < len(self.points):
            return True
        else:
            self.is_flight_complete = True
            return False
        
    
    def get_next_point(self):
        """
        Возвращает следующую точку маршрута
        Returns:
            tuple|None: (x, y) или None если маршрут завершён
        """
        if self.has_more_points():
            point = self.points[self.current_index]
            self.current_index += 1
            return point 
        
        return None
    
    
    def get_total_points(self):
        """Возвращает общее количество точек в маршруте"""
        return len(self.points)
    
    def get_current_progress(self):
        """Возвращает прогресс выполнения миссии в процентах"""
        return (self.current_index / len(self.points)) * 100 if self.points else 0







if __name__ == "__main__":
    print("start")



    hover_start_time = None  # Время начала зависания
    hover_duration = 3    # Длительность зависания в секундах
    point_reached = False
    
    
     # Создаем миссию с настройками по умолчанию
    mission = FlightMission(POINTS)
    marker_tracker = MarkerTracker()  # Создаем трекер маркеров

    
    print(f"Всего точек в маршруте: {mission.get_total_points()}")
    print("Маршрут:", mission.points)



    pioneer = CustomPioneer(name="pioneer", ip="127.0.0.1", mavlink_port=8000, connection_method="udpout", 
                           device="dev/serial0", baud=115200, logger=True, log_connection=True)
    

    camera = Camera(ip="127.0.0.1", port=18000, log_connection=True, timeout=4)

    pioneer.arm()
    pioneer.takeoff()
    # time.sleep(3)



    first_point = mission.get_next_point()
    x, y = first_point
    pioneer.go_to_local_point(x=x, y=y, z=flight_height, yaw=0)





    while not mission.is_flight_complete:
        
        frame = camera.get_cv_frame()
        if frame is None:
            continue
        
    

        if detect_aruco_markers_presence(frame, aruco_detector, visual=True):

            markers_global = get_aruco_markers_global_positions(frame, aruco_detector, pioneer, verboose=False)
            if markers_global:
                # Добавление измерений
                    marker_tracker.add_marker_sample(markers_global)

                   
                                                
        
               

        if pioneer.point_reached(custom_behaviour=False):  # если дрон долетел до нужной точки или же он только летит на первую точку, то можно давать новую команду
            next_point = mission.get_next_point()
            if next_point:
                x, y = next_point
                pioneer.go_to_local_point(x=x, y=y, z=flight_height, yaw=0)
            

            

                # pioneer.land()
                # pioneer.close_connection()
                # del pioneer
                # break



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



    planner = FlightPlanner(all_avg_coords, pioneer)
    coordinates_plan = planner.get_coordinates_plan() 
    print("План полёта (координаты):", coordinates_plan)




    mission = FlightMission(coordinates_plan)
    first_point = mission.get_next_point()
    x, y = first_point
    pioneer.go_to_local_point(x=x, y=y, z=flight_height, yaw=0)

    hover_duration = 4


    while not mission.is_flight_complete:
        
        frame = camera.get_cv_frame()
        if frame is None:
            continue


        if pioneer.point_reached(custom_behaviour=False):  # если дрон долетел до нужной точки или же он только летит на первую точку, то можно давать новую команду

            pioneer.go_to_local_point(x=x, y=y, z=0, yaw=0)
            while not pioneer.point_reached(custom_behaviour=False):
                frame = camera.get_cv_frame()
                if frame is None:
                    continue
                cv2.imshow("frame", frame)
                key = cv2.waitKey(1) 


            pioneer.start_hover(hover_duration)  # Просто запускаем зависание




        if pioneer.check_hover_complete():  # Проверяем завершение зависания
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