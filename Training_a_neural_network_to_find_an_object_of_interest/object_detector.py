import cv2
from collections import defaultdict
from typing import List, Tuple, Dict, Optional
import numpy as np

class ObjectDetector:
    def __init__(self, model, camera, class_names: Dict[int, str]):
        """
        Инициализация детектора объектов
        
        :param model: Обученная модель YOLO
        :param camera: Объект камеры
        :param class_names: Словарь {class_id: class_name}
        """
        self.model = model
        self.camera = camera
        self.class_names = class_names
        
        # Параметры по умолчанию
        self.min_confidence = 0.5
        self.min_detections = 3
        self.analysis_frames = 50
        self.display_duration = 150  # мс


    def analyze_point(self) -> Tuple[Optional[str], Optional[np.ndarray]]:
        """
        Анализирует объекты в текущей точке
        
        :return: (main_class, best_frame) или (None, None) если ничего не обнаружено
        """
        class_counts = defaultdict(int)
        detection_samples = []
        
        try:
            for frame_num in range(self.analysis_frames):
                ret, frame = self.camera.read()
                if not ret:
                    print("Ошибка чтения кадра")
                    break
                
                results = self._process_frame(frame)
                if results is not None:
                    detection_samples.append((frame, results))
                    self._update_counts(results, class_counts)
                
                self._display_progress(frame, frame_num)
                
                if self._check_early_exit():
                    break
            
            return self._analyze_results(class_counts, detection_samples)
            
        except Exception as e:
            print(f"Ошибка анализа: {str(e)}")
            return None, None
        
        finally:
            cv2.destroyWindow('Detection Progress')

    def _process_frame(self, frame: np.ndarray):
        """Обрабатывает один кадр и возвращает результаты детекции"""
        try:
            return self.model.predict(
                frame, 
                imgsz=224, 
                conf=self.min_confidence, 
                verbose=False
            )
        except Exception as e:
            print(f"Ошибка детекции: {str(e)}")
            return None

    def _update_counts(self, results, counts: Dict[str, int]):
        """Обновляет счетчик обнаружений по классам"""
        if results[0].boxes:
            for box in results[0].boxes:
                class_name = self.class_names[int(box.cls)]
                counts[class_name] += 1

    def _display_progress(self, frame: np.ndarray, frame_num: int):
        """Отображает прогресс анализа"""
        status_frame = frame.copy()
        cv2.putText(
            status_frame, 
            f"Анализ: {frame_num+1}/{self.analysis_frames}", 
            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2
        )
        cv2.imshow('Detection Progress', status_frame)
        cv2.waitKey(50)

    def _check_early_exit(self) -> bool:
        """Проверяет необходимость досрочного прерывания"""
        return cv2.waitKey(50) & 0xFF == ord('q')

    def _analyze_results(self, counts: Dict[str, int], 
                        samples: List[Tuple[np.ndarray, any]]) -> Tuple[Optional[str], Optional[np.ndarray]]:
        """Анализирует и визуализирует результаты"""
        if not counts:
            print("Ни одного объекта не обнаружено")
            return None, None
            
        main_class = max(counts, key=counts.get)
        total = sum(counts.values())
        
        if counts[main_class] < self.min_detections:
            print(f"Недостаточно детекций. Лучший класс: {main_class} ({counts[main_class]}/{total})")
            return None, None
        
        print(f"Подтвержден класс: {main_class} ({counts[main_class]}/{total} детекций)")
        best_frame, best_results = self._get_best_detection(samples, main_class)
        self._display_result(best_frame, main_class)
        
        return main_class, best_frame

    def _get_best_detection(self, samples: List[Tuple[np.ndarray, any]], 
                           target_class: str) -> Tuple[np.ndarray, any]:
        """Находит кадр с лучшей детекцией указанного класса"""
        filtered_samples = []
        for frame, results in samples:
            if results[0].boxes:
                for box in results[0].boxes:
                    if self.class_names[int(box.cls)] == target_class:
                        filtered_samples.append((frame, results, box.conf))
        
        # Возвращаем кадр с максимальным confidence
        return max(filtered_samples, key=lambda x: x[2])[:2] if filtered_samples else (None, None)


    def _display_result(self, frame: np.ndarray, class_name: str):
        """Отображает лучший результат"""
        if frame is None:
            return
            
        cv2.putText(
            frame, 
            f"Confirmed: {class_name}", 
            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,0), 2
        )
        cv2.imshow('Best Detection', frame)
        cv2.waitKey(self.display_duration)
        cv2.destroyWindow('Best Detection')