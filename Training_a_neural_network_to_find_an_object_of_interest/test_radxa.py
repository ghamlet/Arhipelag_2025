import time
import os
import cv2
import numpy as np
from pathlib import Path
from threading import Thread
from datetime import datetime
from collections import defaultdict
from typing import List, Tuple, Dict, Optional
import logging

from ultralytics import YOLO
from ultralytics.engine.results import Results

logger = logging.getLogger(__name__)

class YoloRKNN:
    def __init__(self, model_path: Path):
        self.__model = self.__setup_model(model_path)

    def __setup_model(self, path: Path):
        model = YOLO(path.as_posix(), task="detect")
        return model

    def predict(self, image, **kwargs) -> list[Results]:
        return self.__model.predict(image, **kwargs)

class WebcamStream:
    def __init__(self, stream_id: str | int = 0, model: YoloRKNN | None = None, verbose: bool = False):
        self.stream_id = stream_id
        self.__model = model
        self.__writer = None
        self.__res = None
        self.verbose = verbose
        self.vcap = cv2.VideoCapture(self.stream_id)

        width, height = (640, 640)
        ret = self.vcap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        if not ret and self.verbose:
            logger.error(f"Can't set camera frame width -> {width}")

        ret = self.vcap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        if not ret and self.verbose:
            logger.error(f"Can't set camera frame height -> {height}")

        if ret and self.verbose:
            print(f"Camera resolution: {self.vcap.get(cv2.CAP_PROP_FRAME_WIDTH)}x{self.vcap.get(cv2.CAP_PROP_FRAME_HEIGHT)}")

        if not self.vcap.isOpened():
            print("[Exiting]: Error accessing webcam stream.")
            exit(0)

        self.succsess, self.frame = self.vcap.read()
        if not self.succsess:
            print("[Exiting] No more frames to read on init")
            exit(0)

        if self.verbose:
            print("Init camera")
        self.stopped = True
        self.t = Thread(target=self.update, args=())
        self.t.daemon = True

    def start(self):
        self.stopped = False
        self.t.start()

    def update(self):
        while True:
            if self.stopped:
                break

            self.succsess, self.frame = self.vcap.read()
            if not self.succsess:
                if self.verbose:
                    print("[Exiting] No more frames to read in update")
                self.stopped = True
                break

            if self.__model is not None:
                self.__res = self.__model.predict(self.frame, verbose=False)

        self.vcap.release()

    def read(self):
        res = self.__res or None
        return res, self.frame

    def stop(self):
        self.stopped = True

    @property
    def resolution(self):
        return np.array([
            int(self.vcap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            int(self.vcap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        ])

    def create_videowriter(self) -> cv2.VideoWriter:
        current_dir = os.path.dirname(os.path.abspath(__file__))
        output_dir = os.path.join(current_dir, 'outputs')
        os.makedirs(output_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_filename = f"output_{timestamp}.mp4"
        output_path = os.path.join(output_dir, output_filename)
        
        frame_width = int(self.vcap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(self.vcap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        return cv2.VideoWriter(
            output_path,
            cv2.VideoWriter.fourcc(*"mp4v"),
            30,
            (frame_width, frame_height)
        )

class ObjectDetectionSaver:
    def __init__(self, model, class_names: Dict[int, str], camera_stream, verbose: bool = False):
        self.model = model
        self.class_names = class_names
        self.camera_stream = camera_stream
        self.verbose = verbose
        
        # Настройки детекции
        self.min_confidence = 0.5
        self.min_detections = 10
        self.analysis_frames = 30
        self.display_duration = 150
        
        # Настройки сохранения
        current_dir = os.path.dirname(os.path.abspath(__file__))
        self.save_dir = os.path.join(current_dir, 'save_foto')
        self.current_point = 0
        self.create_save_directory()
    
    def create_save_directory(self):
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)
    
    def create_point_directory(self):
        self.current_point += 1
        point_dir = os.path.join(self.save_dir, f"point_{self.current_point}")
        os.makedirs(point_dir, exist_ok=True)
        return point_dir
    
    def analyze_and_save(self):
        point_dir = self.create_point_directory()
        if self.verbose:
            print(f"\nДостигнута точка {self.current_point}. Анализ объектов...")
        
        confirmed_classes, best_frame = self._analyze_objects()
        self._save_detection_results(point_dir, confirmed_classes, best_frame)
        return bool(confirmed_classes)
        
    def _analyze_objects(self) -> Tuple[List[str], Optional[np.ndarray]]:
        class_counts = defaultdict(int)
        detection_samples = []
        
        for frame_num in range(self.analysis_frames):
            if self.camera_stream.stopped:
                break
                
            _, frame = self.camera_stream.read()
            if frame is None:
                if self.verbose:
                    print("Ошибка чтения кадра")
                break
            
            results = self._process_frame(frame)
            if results:
                detection_samples.append((frame, results))
                self._update_counts(results, class_counts)
            
            # Отображаем текущий кадр с детекциями
            if results and results[0].boxes:
                annotated_frame = results[0].plot()
                cv2.imshow("Live Detection", annotated_frame)
            else:
                cv2.imshow("Live Detection", frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        return self._analyze_results(class_counts, detection_samples)
    
    def _process_frame(self, frame: np.ndarray):
        try:
            return self.model.predict(frame, imgsz=640, conf=self.min_confidence, verbose=False)
        except Exception as e:
            if self.verbose:
                print(f"Ошибка детекции: {str(e)}")
            return None
    
    def _update_counts(self, results, counts: Dict[str, int]):
        if results[0].boxes:
            for box in results[0].boxes:
                class_name = self.class_names[int(box.cls)]
                counts[class_name] += 1
    
    def _analyze_results(self, counts: Dict[str, int], samples: List[Tuple[np.ndarray, any]]) -> Tuple[List[str], Optional[np.ndarray]]:
        if not counts:
            if self.verbose:
                print("[ДЕТЕКЦИЯ] Объекты не обнаружены")
            return [], None
            
        total = sum(counts.values())
        confirmed_classes = []
        sorted_classes = sorted(counts.items(), key=lambda x: x[1], reverse=True)
        
        if self.verbose:
            print("\n[ДЕТЕКЦИЯ] Статистика обнаружения:")
            print("--------------------------------")
            print(f"Всего кадров обработано: {self.analysis_frames}")
            print(f"Всего обнаружений: {total}")
            print("\nДетали по классам:")
            
            for class_name, count in sorted_classes:
                status = "ПОДТВЕРЖДЕН" if count >= self.min_detections else "НЕДОСТАТОЧНО"
                print(f"- {class_name.upper():<15}: {count:>3} детекций ({status})")
            
            print("--------------------------------")
        
        for class_name, count in sorted_classes:
            if count >= self.min_detections:
                confirmed_classes.append(class_name)
        
        if not confirmed_classes:
            best_class, best_count = sorted_classes[0]
            if self.verbose:
                print(f"[ДЕТЕКЦИЯ] Основной класс: {best_class} ({best_count}/{total}), но недостаточно детекций (требуется {self.min_detections})")
            return [], None
        
        # Всегда выводим подтвержденные классы, независимо от verbose
        print(f"[ДЕТЕКЦИЯ] Подтвержденные классы: {', '.join(confirmed_classes)}")
        
        best_frame, _ = self._get_best_detection(samples, confirmed_classes[0])
        return confirmed_classes, best_frame

    def _get_best_detection(self, samples, target_class: str):
        filtered = []
        for frame, results in samples:
            if results[0].boxes:
                for box in results[0].boxes:
                    if self.class_names[int(box.cls)] == target_class:
                        filtered.append((frame, results, box.conf))
        
        return max(filtered, key=lambda x: x[2])[:2] if filtered else (None, None)
    
    def _save_detection_results(self, point_dir: str, confirmed_classes: List[str], best_frame: Optional[np.ndarray]):
        if self.verbose:
            print("\n[СОХРАНЕНИЕ] Начато сохранение результатов...")
        
        for i, class_name in enumerate(confirmed_classes):
            if best_frame is not None:
                # Получаем результаты детекции для лучшего кадра
                results = self.model.predict(best_frame, imgsz=640, conf=self.min_confidence, verbose=False)
                if results and results[0].boxes:
                    # Рисуем bounding boxes на кадре перед сохранением
                    annotated_frame = results[0].plot()
                    best_path = os.path.join(point_dir, f"best_{class_name}_{i+1}.jpg")
                    cv2.imwrite(best_path, annotated_frame)
                    # print(f"- Лучший кадр для '{class_name}' с детекциями сохранен: {best_path}")
        
        if self.verbose:
            print("[СОХРАНЕНИЕ] Все данные успешно сохранены\n")



if __name__ == "__main__":
    # Параметр verbose (True - подробный вывод, False - только подтвержденные классы)
    VERBOSE = False
    
    # Инициализация модели
    model_path = Path("/home/arrma/PROGRAMMS/Arhipelag_2025/Training_a_neural_network_to_find_an_object_of_interest/weights/best_sana_v2.pt")
    model = YoloRKNN(model_path)
    
    # Инициализация видеопотока
    webcam_stream = WebcamStream(
        stream_id="Training_a_neural_network_to_find_an_object_of_interest/videos/output_pascal_line2.mp4",
        model=model,
        verbose=VERBOSE
    )
    webcam_stream.start()
    
    # Инициализация видеозаписи с улучшенными параметрами
    video_writer = None
    try:
        current_dir = os.path.dirname(os.path.abspath(__file__))
        output_dir = os.path.join(current_dir, 'outputs')
        os.makedirs(output_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_filename = f"output_{timestamp}.avi"  # Используем AVI для большей надежности
        output_path = os.path.join(output_dir, output_filename)
        
       
        # Пробуем разные кодеки
        codecs = [
            ('MJPG', '.avi'),  # Первый вариант - самый надежный
            ('XVID', '.avi'),
            ('mp4v', '.mp4'),
        ]
        
        for codec, ext in codecs:
            output_path = os.path.join(output_dir, f"output_{timestamp}{ext}")
            fourcc = cv2.VideoWriter_fourcc(*codec)
            video_writer = cv2.VideoWriter(output_path, fourcc, 30, (640, 640))
            if video_writer.isOpened():
                print(f"Успешно инициализирован VideoWriter с кодеком {codec}")
                break
            video_writer.release()
        
        if video_writer is None or not video_writer.isOpened():
            raise RuntimeError("Не удалось инициализировать VideoWriter ни с одним кодеком")
    
        # Классы для детекции
        class_names = {
            0: "black",
            1: "brown",
            2: "white",
            3: "wolf"
        }

        colors ={
            "black": "green",
            "white": "green",
            "brown": "yellow",
            "wolf": "red"
        }
        

        # Инициализация системы сохранения детекций
        detection_saver = ObjectDetectionSaver(model, class_names, webcam_stream, verbose=VERBOSE)
        
        num_frames_processed = 0
        start = time.time()
        last_log_time = time.time()
        
        while True:
            if webcam_stream.stopped:
                break

            results, frame = webcam_stream.read()
            if frame is None:
                break
            
            # Обработка и отображение кадра
            if results and results[0].boxes:
                annotated_frame = results[0].plot()
            else:
                annotated_frame = frame.copy()
            
            cv2.imshow("YOLO Detection", annotated_frame)
            
            # # Запись кадра
            # try:
            #     video_writer.write(annotated_frame)
            #     num_frames_processed += 1
            # except Exception as e:
            #     print(f"Ошибка записи кадра: {str(e)}")
            #     break
            
          
            # Проверяем наличие объектов и сохраняем при необходимости
            if results and results[0].boxes and len(results[0].boxes) > 0:
                detection_saver.analyze_and_save()
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break




    except Exception as e:
        print(f"Произошла ошибка: {str(e)}")
    finally:
        # Гарантированное освобождение ресурсов
        if video_writer is not None:
            video_writer.release()
            print(f"Видео сохранено в: {output_path}")
            print(f"Размер файла: {os.path.getsize(output_path)/1024/1024:.2f} MB")
        
        webcam_stream.stop()
        cv2.destroyAllWindows()
        
        # Вывод статистики
        end = time.time()
        duration = end - start
        print(f"\nИтоговая статистика:")
        print(f"Обработано кадров: {num_frames_processed}")
        print(f"Время работы: {duration:.2f} секунд")
        print(f"Средний FPS: {num_frames_processed/max(1, duration):.2f}")
        if duration > 0:
            print(f"Записанное видео: {duration/60:.2f} минут")