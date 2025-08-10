import cv2
import os
from pathlib import Path
from datetime import datetime
from collections import defaultdict
from ultralytics.engine.results import Results
from ultralytics import YOLO




class YoloRKNN:
    def __init__(self, model_path: Path):
        self.__model = self.__setup_model(model_path)

    def __setup_model(self, path: Path):
        model = YOLO(path.as_posix(), task="detect")
        return model

    def predict(self, image, **kwargs) -> list[Results]:
        return self.__model.predict(image, **kwargs)


def log_detected_classes(results):
    """
    Логирует классы объектов, обнаруженных на текущем кадре.
    Преобразует '9' в '6' для учёта особенностей поля.
    """
    if not results:
        print("Объекты не обнаружены.")
        return []
    result = results[0]
    if hasattr(result, 'names') and hasattr(result, 'boxes'):
        class_ids = result.boxes.cls.cpu().numpy().astype(int)
        class_names = [result.names[cid] for cid in class_ids]
        # Преобразуем '9' в '6'
        class_names = ['6' if name == '9' else name for name in class_names]
        if class_names:
            print(f"Обнаружены классы: {', '.join(class_names)}")
        else:
            print("Объекты не обнаружены.")
        return class_names
    else:
        print("Нет информации о классах объектов.")
        return []




def process_video(input_video_path: str, model_path: str, min_detections: int = 10):
    
    model = YoloRKNN(Path(model_path))

    # Открытие видеофайла
    cap = cv2.VideoCapture(input_video_path)
    if not cap.isOpened():
        print(f"Не удалось открыть видео: {input_video_path}")
        return

    # Получение параметров видео
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)


    # Подготовка выходного файла
    output_dir = os.path.join(os.path.dirname(__file__), 'outputs')
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = os.path.join(output_dir, f"processed_{timestamp}.mp4")
   
   
    class_frame_counts = defaultdict(int)  # Счётчик: класс -> количество кадров, где он был замечен


    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Предсказание и визуализация
        results = model.predict(frame, verbose=False)
        class_names = log_detected_classes(results)  # Логируем классы объектов

        # Для каждого класса, замеченного на кадре, увеличиваем счётчик
        for cname in set(class_names):
            class_frame_counts[cname] += 1

        if results:
            annotated_frame = results[0].plot()
            cv2.imshow("YOLO RKNN Detection", annotated_frame)
        else:
            cv2.imshow("YOLO RKNN Detection", frame)

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

    # Фильтрация по количеству появлений на кадрах
    filtered_classes = [c for c, count in class_frame_counts.items() if count >= min_detections]

    print(f"Уникальные найденные цифры (после фильтрации, min {min_detections} кадров): {sorted(filtered_classes)}")
    # print("Статистика по всем классам (класс: количество кадров):")
    # for c, count in sorted(class_frame_counts.items()):
    #     print(f"{c}: {count}")



if __name__ == "__main__":
    # Укажите путь к видеофайлу и модели
    input_video = 0
    model_path = "Training_a_neural_network_to_find_an_object_of_interest/best_digits_rknn_model/best_digits-rk3566.rknn"
    
    process_video(input_video, model_path,min_detections=10)