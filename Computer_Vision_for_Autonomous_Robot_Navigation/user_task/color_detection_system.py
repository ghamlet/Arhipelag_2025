import cv2
import math
import numpy as np
from collections import defaultdict
from typing import Dict, List, Tuple, Optional, Union

class ColorDetectionSystem:
    """
    Система для детектирования цветных объектов и вычисления их углового положения
    относительно эталонной разметки.
    """
    
    VERTICAL_LINES_COUNT = 21
    HORIZONTAL_LINES_COUNT = 19
    VERTICAL_LENGTH_CM = 800
    HORIZONTAL_LENGTH_CM = 600
    VERTICAL_REFERENCE_IDX = 10
    HORIZONTAL_REFERENCE_IDX = -1  # Последняя линия (низ)

    @staticmethod
    def create_color_masks(bgr_image: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Генерирует бинарные маски для основных цветов в HSV пространстве.
        """
        hsv_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2HSV)
        
        color_ranges = {
            'red': [
                (np.array([0, 100, 100]), np.array([10, 255, 255])),    # Нижний диапазон красного
                (np.array([160, 100, 100]), np.array([180, 255, 255]))  # Верхний диапазон красного
            ],
            'green': [(np.array([40, 50, 50]), np.array([80, 255, 255]))],
            'blue': [(np.array([90, 50, 50]), np.array([130, 255, 255]))]
        }
        
        masks = {}
        for color, ranges in color_ranges.items():
            if color == 'red':
                mask1 = cv2.inRange(hsv_image, ranges[0][0], ranges[0][1])
                mask2 = cv2.inRange(hsv_image, ranges[1][0], ranges[1][1])
                masks[color] = cv2.bitwise_or(mask1, mask2)
            else:
                masks[color] = cv2.inRange(hsv_image, ranges[0][0], ranges[0][1])
                
        return masks

    @staticmethod
    def detect_color_objects(
        color_masks: Dict[str, np.ndarray], 
        visualization_image: np.ndarray
    ) -> List[Tuple[Tuple[int, int], str]]:
        """
        Детектирует цветные объекты и находит их нижние центральные точки.
        """
        color_contours = defaultdict(list)
        contour_colors = {'red': (0, 0, 255), 'green': (0, 255, 0), 'blue': (255, 0, 0)}
        
        for color, mask in color_masks.items():
            contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            
            for contour in contours:
                if cv2.contourArea(contour) > 5:
                    bottom_y = max(pt[0][1] for pt in contour)
                    bottom_pts = [pt[0] for pt in contour if pt[0][1] == bottom_y]
                    
                    if bottom_pts:
                        avg_x = int(np.mean([pt[0] for pt in bottom_pts]))
                        center = (avg_x, bottom_y)
                        color_contours[color].append(center)
                        cv2.drawContours(visualization_image, [contour], -1, contour_colors[color], -1)
        
        sorted_points = []
        for color, points in color_contours.items():
            sorted_points.extend([(pt, color) for pt in sorted(points, key=lambda p: p[0])])
        
        return sorted(sorted_points, key=lambda item: item[0][0])

    @classmethod
    def find_closest_reference_lines(
        cls,
        point: Tuple[int, int],
        reference_frames: Tuple[np.ndarray, np.ndarray, np.ndarray]
    ) -> Dict[str, Optional[int]]:
        
        """
        Находит ближайшие эталонные линии к заданной точке с полной обработкой размерностей.
        
        Args:
            point: Координаты точки (x, y) в формате (int, int)
            reference_frames: Кортеж из (горизонтальная разметка, вертикальная разметка, изображение для визуализации)
            
        Returns:
            Словарь с индексами ближайших линий {'horizontal': index, 'vertical': index}
        """
        frame_h, frame_v, vis_frame = reference_frames
        result = {"horizontal": None, "vertical": None}
        point_x, point_y = point
        point_np = np.array([[point_x, point_y]], dtype=np.float32)  # Формат (1, 2)

        for line_type in ["horizontal", "vertical"]:
            frame = frame_h if line_type == "horizontal" else frame_v
            binary = cv2.inRange(frame, (240, 240, 240), (255, 255, 255))
            contours, _ = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
            
            if not contours:
                continue

            # Сортировка контуров
            sort_key = lambda c: cv2.boundingRect(c)[1] if line_type == "horizontal" else cv2.boundingRect(c)[0]
            sorted_contours = sorted(contours, key=sort_key, reverse=line_type == "horizontal")

            min_dist = float('inf')
            best_contour_idx = None

            for idx, contour in enumerate(sorted_contours):
                # Преобразуем контур в массив точек формата (N, 1, 2) -> (N, 2)
                contour_pts = contour.reshape(-1, 2).astype(np.float32)
                
                # Вычисляем расстояния от всех точек контура до целевой точки
                diff = contour_pts - point_np
                dists = np.linalg.norm(diff, axis=1)
                current_min_dist = np.min(dists)
                
                if current_min_dist < min_dist:
                    min_dist = current_min_dist
                    best_contour_idx = idx

            if best_contour_idx is not None:
                result[line_type] = best_contour_idx
                cv2.drawContours(vis_frame, [sorted_contours[best_contour_idx]], -1, (255, 0, 255), 2)

        cv2.imshow("Closest Reference Lines", vis_frame)
        cv2.waitKey(1)
        return result


    @classmethod
    def calculate_real_distances(cls, line_indices: Dict[str, int]) -> Dict[str, Union[float, str]]:
        """
        Преобразует индексы линий в реальные расстояния в сантиметрах.
        """
        vertical_step = cls.VERTICAL_LENGTH_CM / (cls.VERTICAL_LINES_COUNT - 1)
        horizontal_step = cls.HORIZONTAL_LENGTH_CM / (cls.HORIZONTAL_LINES_COUNT - 1)
        
        vertical_segments = line_indices['vertical'] - cls.VERTICAL_REFERENCE_IDX
        vertical_dist = vertical_segments * vertical_step
        
        horizontal_segments = line_indices['horizontal'] - cls.HORIZONTAL_REFERENCE_IDX
        horizontal_dist = horizontal_segments * horizontal_step
        
        sign = "-" if vertical_dist < 0 else "+"
        
        return {
            'vertical': round(abs(vertical_dist), 2),
            'horizontal': round(abs(horizontal_dist), 2),
            'sign': sign
        }

    @staticmethod
    def calculate_angle(distances: Dict[str, Union[float, str]]) -> int:
        """
        Вычисляет угол наклона по расстояниям до эталонных линий.
        """
        vertical = distances['vertical']
        horizontal = distances['horizontal']
        
        if horizontal == 0:
            angle_rad = math.pi / 2
        else:
            angle_rad = math.atan(vertical / horizontal)
            
        angle_deg = int(round(math.degrees(angle_rad)))
        angle_deg = -angle_deg if distances['sign'] == '-' else angle_deg
        return angle_deg


    

    @classmethod
    def process_image(
        cls,
        input_image: np.ndarray,
        reference_frames: Tuple[np.ndarray, np.ndarray, np.ndarray]
    ) -> List[Tuple[str, int]]:
        """
        Основной метод обработки изображения.
        """
        
            
        color_masks = cls.create_color_masks(input_image)
        detected_objects = cls.detect_color_objects(color_masks, reference_frames[2])
        
        results = []
        for (point, color) in detected_objects:
            line_indices = cls.find_closest_reference_lines(point, reference_frames)
            distances = cls.calculate_real_distances(line_indices)
            angle = cls.calculate_angle(distances)
            results.append([color, angle])
            
        return results