import torch
import torchvision
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator
from torchvision.transforms import functional as F
import cv2
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from sklearn.model_selection import train_test_split
from torchvision import transforms as T

# Конфигурация
CLASSES = ['NoDrive', 'Stop', 'Parking', 'RoadWorks', 'PedestrianCrossing']
NUM_CLASSES = len(CLASSES)
BATCH_SIZE = 4
EPOCHS = 20
IMG_SIZE = (512, 512)

# 1. Кастомный Dataset
class TrafficSignDataset(Dataset):
    def __init__(self, df, transforms=None):
        self.df = df
        self.transforms = transforms
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = "Computer_Vision_for_Autonomous_Robot_Navigation/Educational_task/" + row['img_name']
        
        # Загрузка изображения и конвертация в тензор
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        img = torch.from_numpy(img).permute(2, 0, 1)  # HWC to CHW
        
        # Боксы в формате [x1, y1, x2, y2]
        boxes = torch.tensor([
            [row['x'], row['y'], row['x2'], row['y2']]
        ], dtype=torch.float32)
        
        # Метки класса (1-based индексация)
        labels = torch.tensor([CLASSES.index(row['class_name']) + 1], dtype=torch.int64)
        
        target = {
            'boxes': boxes,
            'labels': labels,
            'image_id': torch.tensor([idx]),
            'area': (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0]),
            'iscrowd': torch.zeros((1,), dtype=torch.int64)
        }
        
        if self.transforms:
            img = self.transforms(img)
            
        return img, target

# 2. DataLoader с кастомными коллациями
def collate_fn(batch):
    images = [item[0] for item in batch]
    targets = [item[1] for item in batch]
    return images, targets

# 3. Модель Lightning
class DetectionModel(pl.LightningModule):
    def __init__(self):
        super().__init__()
        
        # Backbone с MobileNetV2
        backbone = torchvision.models.mobilenet_v2(pretrained=True).features
        backbone.out_channels = 1280
        
        # Anchor Generator
        anchor_generator = AnchorGenerator(
            sizes=((32, 64, 128, 256, 512),),
            aspect_ratios=((0.5, 1.0, 2.0),)
        )
        
        # ROI Pooler
        roi_pooler = torchvision.ops.MultiScaleRoIAlign(
            featmap_names=['0'],
            output_size=7,
            sampling_ratio=2
        )
        
        # Модель Faster R-CNN
        self.model = FasterRCNN(
            backbone,
            num_classes=NUM_CLASSES + 1,  # +1 для фона
            rpn_anchor_generator=anchor_generator,
            box_roi_pool=roi_pooler
        )
    
    def forward(self, x):
        return self.model(x)
    
    def training_step(self, batch, batch_idx):
        images, targets = batch
        
        # Конвертация targets в нужный формат
        targets = [{k: v.to(self.device) for k, v in t.items()} for t in targets]
        
        # Forward pass
        loss_dict = self.model(images, targets)
        losses = sum(loss for loss in loss_dict.values())
        
        self.log('train_loss', losses, prog_bar=True)
        return losses
    
    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=1e-4)

# 4. Загрузка данных и подготовка
df = pd.read_csv('Computer_Vision_for_Autonomous_Robot_Navigation/Educational_task/annotations.csv', sep=';')

# Трансформации
transform = T.Compose([
    T.Resize(IMG_SIZE),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Разделение данных
train_df, val_df = train_test_split(df, test_size=0.2, random_state=42)

# Создание Dataset и DataLoader
train_dataset = TrafficSignDataset(train_df, transforms=transform)
val_dataset = TrafficSignDataset(val_df, transforms=transform)

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    collate_fn=collate_fn,
    num_workers=4
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    collate_fn=collate_fn,
    num_workers=4
)

# 5. Обучение
checkpoint_callback = ModelCheckpoint(
    dirpath='checkpoints',
    filename='best-model-{epoch:02d}-{train_loss:.2f}',
    monitor='train_loss',
    save_top_k=1
)


trainer = pl.Trainer(
    max_epochs=EPOCHS,
    callbacks=[checkpoint_callback],
    accelerator='auto',
    devices="auto"  # Исправлено здесь
)

model = DetectionModel()
trainer.fit(model, train_loader, val_loader)

# 6. Сохранение модели
torch.save(model.state_dict(), 'traffic_sign_detection.pth')
print("Обучение завершено! Модель сохранена в 'traffic_sign_detection.pth'")