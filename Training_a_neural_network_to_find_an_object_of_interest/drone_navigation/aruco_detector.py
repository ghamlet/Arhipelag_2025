import cv2

class ArucoDetector:
    def __init__(self, dictionary_type=cv2.aruco.DICT_4X4_100):
        """
        Инициализация детектора ArUco маркеров.
        
        Args:
            dictionary_type: Тип словаря ArUco маркеров (по умолчанию DICT_4X4_100)
        """
        self.aruco_dict = cv2.aruco.getPredefinedDictionary(dictionary_type)
        self.aruco_params = cv2.aruco.DetectorParameters()
        self.detector = cv2.aruco.ArucoDetector(self.aruco_dict, self.aruco_params)
        self.image_size = (640, 480)
        self.ground_cover = (4.4, 3.4)
    

    def detect_markers_presence(self, frame, visual=False):
        """
        Определяет наличие ArUco маркеров на изображении.
        
        Args:
            frame: Входное изображение
            visual: Режим визуализации
            
        Returns:
            bool: True если маркеры обнаружены, False в противном случае
        """

        corners, ids, _ = self.detector.detectMarkers(frame)
        
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
    
    
    def get_markers_image_coordinates(self, frame):
        """
        Находит координаты центров ArUco маркеров на изображении.
        
        Args:
            frame: Входное изображение
            
        Returns:
            dict: {marker_id: (center_x, center_y)} или пустой словарь
        """
        markers = {}
        corners, ids, _ = self.detector.detectMarkers(frame)
        
        if ids is not None:
            for i in range(len(ids)):
                marker_id = ids[i][0]
                marker_corners = corners[i][0]
                center_x = int(marker_corners[:, 0].mean())
                center_y = int(marker_corners[:, 1].mean())
                markers[marker_id] = (center_x, center_y)
        
        return markers
    

    def get_markers_global_positions(self, frame, pioneer, verbose=False):
        """
        Получает глобальные координаты всех обнаруженных ArUco маркеров.
        
        Args:
            frame: Входное изображение
            pioneer: Объект для получения позиции дрона
            verbose: Режим подробного вывода
            
        Returns:
            dict: {marker_id: (global_x, global_y)} или None если маркеры не найдены
        """
        
        global_markers = {}
        
        if not self.detect_markers_presence(frame):
            if verbose:
                print("Маркеры не обнаружены")
            return None
        
        image_markers = self.get_markers_image_coordinates(frame)
        drone_pos = pioneer.get_local_position_lps(get_last_received=True)
        drone_x, drone_y = drone_pos[:2]
        
        img_width, img_height = self.image_size
        ground_width, ground_height = self.ground_cover
        scale_x = ground_width / img_width
        scale_y = ground_height / img_height
        img_center_x = img_width / 2
        img_center_y = img_height / 2
        
        for marker_id, (marker_x, marker_y) in image_markers.items():
            global_x = drone_x + (marker_x - img_center_x) * scale_x
            global_y = drone_y - (marker_y - img_center_y) * scale_y
            global_markers[marker_id] = (global_x, global_y)

        if verbose and global_markers:
            for marker_id, coords in global_markers.items():
                print(f"Маркер {marker_id}: {coords}")

        return global_markers
    

    def get_detected_markers_ids(self, frame):
        """
        Возвращает список ID всех обнаруженных маркеров на кадре.
        
        Args:
            frame: Входное изображение
            
        Returns:
            list: Список ID маркеров или пустой список, если маркеры не найдены
        """
        
        _, ids, _ = self.detector.detectMarkers(frame)
        return ids.flatten().tolist() if ids is not None else []
    