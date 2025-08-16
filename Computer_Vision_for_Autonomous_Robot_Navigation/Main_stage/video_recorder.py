import cv2
import os
import time
from datetime import datetime
from pathlib import Path

class VideoRecorder:
    def __init__(self, camera_id=0, output_dir="recordings", codec='avc1', show_preview=False):
        """
        Класс для записи видео с камеры в формате MP4
        
        Args:
            camera_id: ID камеры или путь к видеофайлу
            output_dir: Папка для сохранения видео (относительно расположения скрипта)
            codec: Кодек для записи ('avc1' для H.264, 'mp4v' для MPEG-4)
            show_preview: Показывать превью во время записи
        """
        self.camera_id = camera_id
        script_dir = Path(__file__).parent
        self.output_dir = script_dir / output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.codec = codec
        self.show_preview = show_preview
        self.is_recording = False
        self.cap = None
        self.out = None
        self.frame_count = 0
        self.start_time = 0
        self.frame_width = 0
        self.frame_height = 0
        self.fps = 20

    def start_recording(self, filename=None):
        """Начать запись видео"""
        if self.is_recording:
            print("Запись уже идет!")
            return False

        # Генерация имени файла с расширением .mp4
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"video_{timestamp}.mp4"
        self.output_file = self.output_dir / filename

        # Инициализация видеозахвата
        self.cap = cv2.VideoCapture(self.camera_id, cv2.CAP_V4L2)

        if not self.cap.isOpened():
            print("Ошибка: Не удалось открыть камеру!")
            return False

        # Установка параметров камеры
        self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 960)

        # Получение параметров видео
        self.frame_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.frame_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.fps = self.fps if self.fps > 0 else 30

        # Инициализация VideoWriter для MP4
        fourcc = cv2.VideoWriter_fourcc(*self.codec)
        self.out = cv2.VideoWriter(
            str(self.output_file), 
            fourcc, 
            self.fps, 
            (self.frame_width, self.frame_height),
            isColor=True
        )

        self.is_recording = True
        self.frame_count = 0
        self.start_time = time.time()
        
        print(f"Начата запись: {self.output_file}")
        print(f"Разрешение: {self.frame_width}x{self.frame_height}, FPS: {self.fps:.1f}")
        return True

    def record_frame(self, frame=None):
        """Записать кадр"""
        if not self.is_recording:
            return False

        if frame is None:
            ret, frame = self.cap.read()
            if not ret:
                print("Ошибка: Не удалось получить кадр")
                return False

        self.out.write(frame)
        self.frame_count += 1

        if self.show_preview:
            cv2.imshow('Video Recorder (ESC to stop)', frame)
            if cv2.waitKey(1) == 27:  # ESC
                self.stop_recording()
                return False

        return True

    def stop_recording(self):
        """Остановить запись"""
        if not self.is_recording:
            return

        self.is_recording = False
        
        if self.out:
            self.out.release()
        if self.cap:
            self.cap.release()
        if self.show_preview:
            cv2.destroyAllWindows()

        duration = time.time() - self.start_time
        print(f"\nЗапись завершена. Статистика:")
        print(f"Файл: {self.output_file}")
        print(f"Длительность: {duration:.2f} сек")
        print(f"Кадров записано: {self.frame_count}")
        if self.output_file.exists():
            print(f"Размер файла: {os.path.getsize(self.output_file)/1024/1024:.2f} MB")

    def close(self):
        """Алиас для stop_recording"""
        self.stop_recording()

    def __del__(self):
        """Деструктор для автоматической остановки записи"""
        self.stop_recording()


def standalone_recording(camera_id):
    """Функция для автономной записи видео"""
    recorder = VideoRecorder(
        camera_id=camera_id,
        output_dir="recordings",
        codec='avc1',  # Используем H.264 кодек
        show_preview=True
    )
    
    try:
        if recorder.start_recording():
            while recorder.record_frame():
                pass
    except KeyboardInterrupt:
        print("\nЗапись прервана пользователем")
    finally:
        recorder.stop_recording()


if __name__ == "__main__":
    CAMERA_ID = "/dev/video0"  # Или 0 для встроенной камеры
    standalone_recording(CAMERA_ID)