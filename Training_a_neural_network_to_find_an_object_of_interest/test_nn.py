from ultralytics import YOLO
import cv2

model = YOLO('Training_a_neural_network_to_find_an_object_of_interest/best_loss07.pt')

video_path = 'Training_a_neural_network_to_find_an_object_of_interest/videos/output_pascal_line2.mp4'
cap = cv2.VideoCapture(video_path)


while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
   
    results = model.predict(frame, imgsz=224)  # Обрабатываем уменьшенный кадр
    annotated_frame = results[0].plot()  # Результаты на уменьшенном кадре
    



    
    cv2.imshow('YOLO Detection', annotated_frame)
    
    if cv2.waitKey(100) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()