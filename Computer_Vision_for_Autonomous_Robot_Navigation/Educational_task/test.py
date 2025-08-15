import torch
import torchvision
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator
import cv2
import numpy as np
import pandas as pd
import pytorch_lightning as pl
from torchvision import transforms as T
from torchvision.models import mobilenet_v2, MobileNet_V2_Weights

# Конфигурация
CLASSES = ['NoDrive', 'Stop', 'Parking', 'RoadWorks', 'PedestrianCrossing']
NUM_CLASSES = len(CLASSES)

class DetectionModel(pl.LightningModule):
    def __init__(self):
        super().__init__()
        backbone = mobilenet_v2(weights=MobileNet_V2_Weights.DEFAULT).features
        backbone.out_channels = 1280
        
        anchor_generator = AnchorGenerator(
            sizes=((32, 64, 128, 256, 512),),
            aspect_ratios=((0.5, 1.0, 2.0),)
        )
        
        roi_pooler = torchvision.ops.MultiScaleRoIAlign(
            featmap_names=['0'],
            output_size=7,
            sampling_ratio=2
        )
        
        self.model = FasterRCNN(
            backbone,
            num_classes=NUM_CLASSES + 1,
            rpn_anchor_generator=anchor_generator,
            box_roi_pool=roi_pooler
        )
    
    def forward(self, x):
        return self.model(x)

def predict_traffic_sign(image, model_path='traffic_sign_detection.pth', confidence_threshold=0.3):
    print("\n=== Начало обработки изображения ===")
    
    # 1. Загрузка модели
    print(f"Загрузка модели из {model_path}...")
    try:
        model = DetectionModel()
        model.load_state_dict(torch.load(model_path))
        model.eval()
        print("Модель успешно загружена!")
    except Exception as e:
        print(f"Ошибка загрузки модели: {e}")
        return None
    
    # 2. Подготовка изображения
    print("Подготовка изображения...")
    try:
        img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        img_tensor = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0)
        print(f"Размер изображения: {img.shape} -> Тензор: {img_tensor.shape}")
    except Exception as e:
        print(f"Ошибка обработки изображения: {e}")
        return None
    
    # 3. Выполнение предсказания
    print("Выполнение предсказания...")
    try:
        with torch.no_grad():
            predictions = model(img_tensor)
        print(f"Получено {len(predictions[0]['boxes'])} обнаружений")

        print(predictions[0]['labels'])
    except Exception as e:
        print(f"Ошибка предсказания: {e}")
        return None
    
    # 4. Визуализация результатов
    print("Визуализация результатов...")
    try:
        img_display = (img * 255).astype(np.uint8)
        
        for i, (box, label, score) in enumerate(zip(predictions[0]['boxes'], 
                                                 predictions[0]['labels'], 
                                                 predictions[0]['scores'])):
            if score > confidence_threshold:
                x1, y1, x2, y2 = map(int, box)
                print(f"Обнаружение {i+1}: {CLASSES[label-1]} (score: {score:.2f}), bbox: [{x1}, {y1}, {x2}, {y2}]")
                
                cv2.rectangle(img_display, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(img_display, f"{CLASSES[label-1]} {score:.2f}", 
                           (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)
        
        cv2.imshow('Traffic Sign Detection', cv2.cvtColor(img_display, cv2.COLOR_RGB2BGR))
        print("Результаты отображены. Нажмите 'q' для закрытия окна.")
        return img_display
    except Exception as e:
        print(f"Ошибка визуализации: {e}")
        return None

if __name__ == "__main__":
    csv_file = "/home/arrma/PROGRAMMS/Arhipelag_2025/Computer_Vision_for_Autonomous_Robot_Navigation/Educational_task/annotations.csv"
    model_path = "Computer_Vision_for_Autonomous_Robot_Navigation/Educational_task/traffic_sign_detection.pth"
    
    print("=== Начало обработки датасета ===")
    data = pd.read_csv(csv_file, sep=';')
    print(f"Загружено {len(data)} записей из CSV")

    for row in data.itertuples():
        img_path = "/home/arrma/PROGRAMMS/Arhipelag_2025/Computer_Vision_for_Autonomous_Robot_Navigation/Educational_task/" + row[1]
        print(f"\nОбработка изображения: {img_path}")
        
        img = cv2.imread(img_path)
        if img is None:
            print(f"Ошибка: не удалось загрузить изображение {img_path}")
            continue
        
        result = predict_traffic_sign(img, model_path)
        
        if result is None:
            print("Пропускаем изображение из-за ошибки")
            continue
            
        while True:
            key = cv2.waitKey(1)
            if key == ord("q"):
                cv2.destroyAllWindows()
                break