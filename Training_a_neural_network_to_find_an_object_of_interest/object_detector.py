import os
import cv2
import numpy as np
from datetime import datetime
from collections import defaultdict
from typing import List, Tuple, Dict, Optional

class ObjectDetectionSaver:
    def __init__(self, pioneer, model, class_names: Dict[int, str], camera_index=0):
        self.pioneer = pioneer
        self.model = model
        self.class_names = class_names
        
        # Инициализация камеры
        self.cap = cv2.VideoCapture(camera_index)
        if not self.cap.isOpened():
            raise RuntimeError("Не удалось открыть камеру")
        
        # Настройки детекции
        self.min_confidence = 0.5
        self.min_detections = 5
        self.analysis_frames = 40
        self.display_duration = 150
        
        # Настройки сохранения
        current_dir = os.path.dirname(os.path.abspath(__file__))
        self.save_dir = os.path.join(current_dir, 'save_foto')
        self.current_point = 0
        self.create_save_directory()
    
    def create_save_directory(self):
        """Создает директорию для сохранения фото"""
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)
    
    def create_point_directory(self):
        """Создает папку для текущей точки"""
        self.current_point += 1
        point_dir = os.path.join(self.save_dir, f"point_{self.current_point}")
        os.makedirs(point_dir, exist_ok=True)
        return point_dir
    
    def analyze_and_save(self):
        """Анализирует точку и сохраняет результаты"""
        point_dir = self.create_point_directory()
        print(f"\nДостигнута точка {self.current_point}. Анализ объектов...")
        
        # Анализ объектов - теперь получаем список классов
        confirmed_classes, best_frame = self._analyze_objects()
        
        # Сохранение фотографий для всех подтвержденных классов
        self._save_detection_results(point_dir, confirmed_classes, best_frame)
        return bool(confirmed_classes)  # Возвращаем True, если есть подтвержденные классы
        
    def _analyze_objects(self) -> Tuple[Optional[str], Optional[np.ndarray]]:
        """Анализирует объекты в текущей точке"""
        class_counts = defaultdict(int)
        detection_samples = []
        
        for frame_num in range(self.analysis_frames):
            ret, frame = self.cap.read()
            if not ret:
                print("Ошибка чтения кадра")
                break
            
            # Детекция объектов
            results = self._process_frame(frame)
            if results:
                detection_samples.append((frame, results))
                self._update_counts(results, class_counts)
            
            # Отображение прогресса
            self._display_progress(frame, frame_num)
            
            if self._check_early_exit():
                break
        
        return self._analyze_results(class_counts, detection_samples)
    
    def _process_frame(self, frame: np.ndarray):
        """Обрабатывает кадр для детекции объектов"""
        try:
            return self.model.predict(frame, imgsz=224, conf=self.min_confidence, verbose=False)
        except Exception as e:
            print(f"Ошибка детекции: {str(e)}")
            return None
    
    def _update_counts(self, results, counts: Dict[str, int]):
        """Обновляет счетчики обнаруженных классов"""
        if results[0].boxes:
            for box in results[0].boxes:
                class_name = self.class_names[int(box.cls)]
                counts[class_name] += 1
    
    def _analyze_results(self, counts: Dict[str, int], samples: List[Tuple[np.ndarray, any]]) -> Tuple[List[str], Optional[np.ndarray]]:
        """Анализирует результаты детекции и возвращает все подтвержденные классы"""
        if not counts:
            print("[ДЕТЕКЦИЯ] Объекты не обнаружены")
            return [], None
            
        total = sum(counts.values())
        confirmed_classes = []
        
        # Сортируем классы по количеству детекций (от большего к меньшему)
        sorted_classes = sorted(counts.items(), key=lambda x: x[1], reverse=True)
        
        print("\n[ДЕТЕКЦИЯ] Статистика обнаружения:")
        print("--------------------------------")
        print(f"Всего кадров обработано: {self.analysis_frames}")
        print(f"Всего обнаружений: {total}")
        print("\nДетали по классам:")
        
        for class_name, count in sorted_classes:
            status = "ПОДТВЕРЖДЕН" if count >= self.min_detections else "НЕДОСТАТОЧНО"
            print(f"- {class_name.upper():<15}: {count:>3} детекций ({status})")
            
            if count >= self.min_detections:
                confirmed_classes.append(class_name)
        
        print("--------------------------------")
        
        if not confirmed_classes:
            best_class, best_count = sorted_classes[0]
            print(f"[ДЕТЕКЦИЯ] Основной класс: {best_class} ({best_count}/{total}), но недостаточно детекций (требуется {self.min_detections})")
            return [], None
        
        print(f"[ДЕТЕКЦИЯ] Подтвержденные классы: {', '.join(confirmed_classes)}")
        best_frame, _ = self._get_best_detection(samples, confirmed_classes[0])
        return confirmed_classes, best_frame



    def _get_best_detection(self, samples, target_class: str):
        """Находит лучшую детекцию указанного класса"""
        filtered = []
        for frame, results in samples:
            if results[0].boxes:
                for box in results[0].boxes:
                    if self.class_names[int(box.cls)] == target_class:
                        filtered.append((frame, results, box.conf))
        
        return max(filtered, key=lambda x: x[2])[:2] if filtered else (None, None)
    
    def _save_detection_results(self, point_dir: str, confirmed_classes: List[str], best_frame: Optional[np.ndarray]):
        """Сохраняет результаты детекции для всех подтвержденных классов"""
        print("\n[СОХРАНЕНИЕ] Начато сохранение результатов...")
        
        # Сохраняем лучший кадр для каждого подтвержденного класса
        for i, class_name in enumerate(confirmed_classes):
            if best_frame is not None:
                best_path = os.path.join(point_dir, f"best_{class_name}_{i+1}.jpg")
                cv2.imwrite(best_path, best_frame)
                print(f"- Лучший кадр для '{class_name}' сохранен: {best_path}")
        
        # Сохраняем серию кадров
        print(f"\nСохранение серии из 50 кадров в {point_dir}")
        for i in range(50):
            ret, frame = self.cap.read()
            if ret:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                path = os.path.join(point_dir, f"frame_{timestamp}_{i}.jpg")
                cv2.imwrite(path, frame)
                if i % 10 == 0:  # Выводим прогресс каждые 10 кадров
                    print(f"- Сохранен кадр {i+1}/50")
        
        print("[СОХРАНЕНИЕ] Все данные успешно сохранены\n")


    def _display_progress(self, frame: np.ndarray, frame_num: int):
        """Отображает прогресс анализа"""
        status_frame = frame.copy()
        cv2.putText(status_frame, f"Анализ: {frame_num+1}/{self.analysis_frames}", 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
        cv2.imshow('Detection Progress', status_frame)
        cv2.waitKey(50)
    
    def _check_early_exit(self) -> bool:
        """Проверяет необходимость досрочного выхода"""
        return cv2.waitKey(50) & 0xFF == ord('q')
    
    def release(self):
        """Освобождает ресурсы"""
        self.cap.release()
        cv2.destroyAllWindows()
