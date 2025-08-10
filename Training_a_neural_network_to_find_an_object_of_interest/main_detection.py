from pioneer_sdk import Pioneer, Camera
import cv2
import time
from flight_utils import load_flight_coordinates, get_config
from drone_navigation import FlightMissionRunner, CustomPioneer
import os
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
    def __init__(self, model, class_names: Dict[int, str], camera_stream, pioneer: Pioneer = None, verbose: bool = False):
        self.model = model
        self.class_names = class_names
        self.camera_stream = camera_stream
        self.pioneer = pioneer
        self.verbose = verbose
        
        # Настройки детекции
        self.min_confidence = 0.5
        self.min_detections = 10
        self.analysis_frames = 30
        self.display_duration = 150
        
        # Настройки фильтрации по площади
        self.min_object_area = 1000
        self.max_object_area = 50000
        
        # Цвета для разных классов объектов
        self.colors = {
            "black": (0, 255, 0),    # green
            "white": (0, 255, 0),    # green
            "brown": (255, 255, 0),  # yellow
            "wolf": (255, 0, 0)      # red
        }
        
        # Настройки сохранения
        current_dir = os.path.dirname(os.path.abspath(__file__))
        self.save_dir = os.path.join(current_dir, 'save_foto')
        self.current_point = 0
        self.create_save_directory()

    def create_save_directory(self):
        """Создает директорию для сохранения результатов"""
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)
    
    def create_point_directory(self):
        """Создает поддиректорию для текущей точки анализа"""
        self.current_point += 1
        point_dir = os.path.join(self.save_dir, f"point_{self.current_point}")
        os.makedirs(point_dir, exist_ok=True)
        return point_dir

    def _analyze_objects(self) -> Tuple[List[Dict], Optional[np.ndarray]]:
        """
        Анализирует объекты в последовательности кадров
        
        Возвращает:
            Кортеж (список подтвержденных классов, лучший кадр)
        """
        class_counts = defaultdict(int)
        class_confidences = defaultdict(list)
        class_areas = defaultdict(list)
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
                self._update_counts_and_confidences(results, class_counts, class_confidences, class_areas)
            
            # Отображаем текущий кадр с детекциями
            if results and results[0].boxes:
                annotated_frame = results[0].plot()
                cv2.imshow("Live Detection", annotated_frame)
            else:
                cv2.imshow("Live Detection", frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        return self._analyze_results(class_counts, class_confidences, class_areas, detection_samples)
    
    def _process_frame(self, frame: np.ndarray) -> Optional[Results]:
        """Обрабатывает кадр и возвращает результаты детекции"""
        try:
            return self.model.predict(frame, imgsz=640, conf=self.min_confidence, verbose=False)
        except Exception as e:
            if self.verbose:
                print(f"Ошибка детекции: {str(e)}")
            return None
    
    def _update_counts_and_confidences(self, results: Results, 
                                     counts: Dict[str, int], 
                                     confidences: Dict[str, List[float]],
                                     areas: Dict[str, List[float]]):
        """Обновляет счетчики и уверенности с учетом площади объектов"""
        if results[0].boxes:
            for box in results[0].boxes:
                class_name = self.class_names[int(box.cls)]
                # Рассчитываем площадь объекта
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                area = (x2 - x1) * (y2 - y1)
                
                # Проверяем площадь объекта
                if self.min_object_area <= area <= self.max_object_area:
                    counts[class_name] += 1
                    confidences[class_name].append(float(box.conf))
                    areas[class_name].append(area)
                elif self.verbose:
                    print(f"Объект {class_name} пропущен из-за недопустимой площади: {area:.1f} px")

    def _analyze_results(self, counts, confidences, areas, samples):
        if not counts:
            if self.verbose:
                print("[ДЕТЕКЦИЯ] Объекты не обнаружены")
            return [], None
            
        total = sum(counts.values())
        confirmed_classes = []
        
        for class_name, count in counts.items():
            if count >= self.min_detections:
                confirmed_classes.append({
                    'class_name': class_name,
                    'count': count,
                    'max_confidence': max(confidences[class_name]),
                    'avg_area': np.mean(areas[class_name]) if class_name in areas else 0,
                    'min_area': min(areas[class_name]) if class_name in areas else 0,
                    'max_area': max(areas[class_name]) if class_name in areas else 0
                })
        
        if not confirmed_classes:
            if self.verbose:
                print("[ДЕТЕКЦИЯ] Недостаточно детекций для подтверждения")
            return [], None
        
        best_class = confirmed_classes[0]['class_name']
        best_frame, _ = self._get_best_detection(samples, best_class)
        
        return confirmed_classes, best_frame

    def analyze_and_save(self):
        point_dir = self.create_point_directory()
        if self.verbose:
            print(f"\nАнализ объектов в точке {self.current_point}...")
        
        confirmed_classes, best_frame = self._analyze_objects()
        saved_files = self._save_detection_results(point_dir, confirmed_classes, best_frame)
        
        detection_results = []
        for class_info in confirmed_classes:
            result = {
                'class_name': class_info['class_name'],
                'detection_count': class_info['count'],
                'confidence': class_info['max_confidence'],
                'best_frame_path': saved_files.get(class_info['class_name'], None)
            }
            
            # Добавляем информацию о площади, если она есть
            if 'min_area' in class_info and 'max_area' in class_info:
                result.update({
                    'min_area': class_info['min_area'],
                    'max_area': class_info['max_area'],
                    'avg_area': class_info.get('avg_area', 0)
                })
            
            detection_results.append(result)
        
        self._control_leds(detection_results)
        return detection_results

    def _get_best_detection(self, samples: List[Tuple[np.ndarray, Results]], target_class: str) -> Tuple[Optional[np.ndarray], Optional[Results]]:
        """Находит лучший кадр для заданного класса с учетом площади"""
        filtered = []
        for frame, results in samples:
            if results[0].boxes:
                for box in results[0].boxes:
                    if self.class_names[int(box.cls)] == target_class:
                        # Проверяем площадь объекта
                        x1, y1, x2, y2 = box.xyxy[0].tolist()
                        area = (x2 - x1) * (y2 - y1)
                        if self.min_object_area <= area <= self.max_object_area:
                            filtered.append((frame, results, box.conf, area))
                        elif self.verbose:
                            print(f"Лучший объект {target_class} пропущен из-за площади: {area:.1f} px")
        
        # Возвращаем кадр с максимальной уверенностью, учитывая площадь
        return max(filtered, key=lambda x: x[2])[:2] if filtered else (None, None)
    
    def _save_detection_results(self, point_dir: str, confirmed_classes: List[Dict], best_frame: Optional[np.ndarray]) -> Dict[str, str]:
        """Сохраняет результаты детекции в файлы"""
        saved_files = {}
        if self.verbose:
            print("\n[СОХРАНЕНИЕ] Начато сохранение результатов...")
        
        for i, class_info in enumerate(confirmed_classes):
            class_name = class_info['class_name']
            if best_frame is not None:
                results = self.model.predict(best_frame, imgsz=640, conf=self.min_confidence, verbose=False)
                if results and results[0].boxes:
                    annotated_frame = results[0].plot()
                    best_path = os.path.join(point_dir, f"best_{class_name}_{i+1}.jpg")
                    cv2.imwrite(best_path, annotated_frame)
                    saved_files[class_name] = best_path
        
        if self.verbose:
            print("[СОХРАНЕНИЕ] Все данные успешно сохранены\n")
        
        return saved_files

    def _control_leds(self, detections: List[Dict[str, any]]):
        """Управляет подсветкой дрона на основе обнаруженных объектов"""
        if not detections or not self.pioneer:
            return
            
        wolf_detected = any(d['class_name'] == 'wolf' for d in detections)
        brown_detected = any(d['class_name'] == 'brown' for d in detections)
        black_white_detected = any(d['class_name'] in ['black', 'white'] for d in detections)
        
        if wolf_detected:
            self._blink_red_alarm()
        elif brown_detected:
            self._set_led_color(*self.colors['brown'], duration=2)
        elif black_white_detected:
            self._set_led_color(*self.colors['black'], duration=2)

    def _set_led_color(self, r, g, b, duration=0):
        def set_color():
            self.pioneer.led_control(led_id=255, r=r, g=g, b=b)
            if duration > 0:
                time.sleep(duration)
                self.pioneer.led_control(led_id=255, r=0, g=0, b=0)
        
        Thread(target=set_color).start()


    def _blink_red_alarm(self, times=3, interval=0.5):
        def blink():
            for _ in range(times):
                self.pioneer.led_control(led_id=255, r=255, g=0, b=0)
                time.sleep(interval)
                self.pioneer.led_control(led_id=255, r=0, g=0, b=0)
                time.sleep(interval)
        
        Thread(target=blink).start()



if __name__ == "__main__":
    try:
        # Параметр verbose (True - подробный вывод, False - только подтвержденные классы)
        VERBOSE = False
        
        # Инициализация модели
        model_path = Path("Training_a_neural_network_to_find_an_object_of_interest/best_sana_v2_rknn_model/best_sana_v2-rk3566.rknn")
        model = YoloRKNN(model_path)
        
        # Инициализация видеопотока
        webcam_stream = WebcamStream(
            stream_id=0,
            model=model,
            verbose=VERBOSE
        )
        webcam_stream.start()
        
        # Инициализация видеозаписи
        video_writer = None
        current_dir = os.path.dirname(os.path.abspath(__file__))
        output_dir = os.path.join(current_dir, 'outputs')
        os.makedirs(output_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_filename = f"output_{timestamp}.avi"
        output_path = os.path.join(output_dir, output_filename)
        
        # Пробуем разные кодеки
        codecs = [
            ('MJPG', '.avi'),
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

        # Получение конфигурации для пионера
        pioneer_conf = get_config('local')
        
        # Инициализация дрона
        pioneer = CustomPioneer(
            name="pioneer",
            ip=pioneer_conf["ip"],
            mavlink_port=pioneer_conf["port"],
            connection_method="udpout",
            device="dev/serial0",
            baud=115200,
            verbose=False
        )

        # Инициализация системы сохранения детекций
        detection_saver = ObjectDetectionSaver(model, class_names, webcam_stream, pioneer, verbose=VERBOSE)

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
            
            # Запись кадра
            try:
                video_writer.write(annotated_frame)
                num_frames_processed += 1
            except Exception as e:
                print(f"Ошибка записи кадра: {str(e)}")
                break
            
            # Проверяем наличие объектов и сохраняем при необходимости
            if results and results[0].boxes and len(results[0].boxes) > 0:
                detections = detection_saver.analyze_and_save()

                for detection in detections:
                    print(f"Обнаружен: {detection['class_name']}")
                    # print(f"Количество детекций: {detection['detection_count']}")
                    # print(f"Максимальная уверенность: {detection['confidence']:.2f}")
                print("\n\n")

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            # Получение текущих координат дрона



    except Exception as e:
        print(f"Ошибка обработки: {str(e)}")
    except KeyboardInterrupt:
        print("\nПолучен сигнал прерывания (Ctrl+C)")
    finally:
        print("Завершение работы")
        if 'pioneer' in locals():
            pioneer.close_connection()
        if 'webcam_stream' in locals():
            webcam_stream.stop()
        if 'video_writer' in locals():
            video_writer.release()
        cv2.destroyAllWindows()