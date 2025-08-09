Датасет от разрабов где фото 224 на 224



from ultralytics import YOLO

# Load a model
model = YOLO("yolo11n.pt")  # load a pretrained model (recommended for training)

# Train the model
results = model.train(data="/content/gdrive/MyDrive/Arhipelag2025/cow_herd_detection_dataset_yolov11/data.yaml",
                       epochs=400,
    imgsz=320,
    workers=4,          # Оптимально для 4-ядерного CPU
    optimizer='AdamW',  # Лучше для небольших датасетов
    lr0=0.001,         # Среднее значение для AdamW
    augment=True,      # Всегда включать аугментацию
    patience=50,       # Остановка при отсутствии улучшений
    batch=16,          # Максимально возможный для GPU
    hsv_h=0.015,       # Доп. аугментация - HSV
    flipud=0.5 ,      # Вертикальное отражение (50%)
    verbose=True
                      )


## в тестах показала хороший результат детекции в моментах когда дрон зависал над обьектами 


from ultralytics import YOLO
import cv2

model = YOLO('Training_a_neural_network_to_find_an_object_of_interest/best_many_param.pt')

video_path = 'Training_a_neural_network_to_find_an_object_of_interest/videos/output_pascal_line2.mp4'
cap = cv2.VideoCapture(video_path)


while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
   
    results = model.predict(frame, imgsz=320)  # Обрабатываем уменьшенный кадр
    annotated_frame = results[0].plot()  # Результаты на уменьшенном кадре
    
    cv2.imshow('YOLO Detection', annotated_frame)
    
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()