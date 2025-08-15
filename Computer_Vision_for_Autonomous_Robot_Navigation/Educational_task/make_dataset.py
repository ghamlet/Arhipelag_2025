import pandas as pd
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical  # type: ignore

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout



# Загрузка аннотаций
df = pd.read_csv('annotations.csv', sep=';')

# Параметры
IMG_SIZE = (32, 32)  # Размер изображений для модели
CLASSES = {
    'NoDrive': 0,
    'Stop': 1,
    'Parking': 2,
    'RoadWorks': 3,
    'PedestrianCrossing': 4
}

# Загрузка и предобработка изображений
X = []
y = []

for idx, row in df.iterrows():
    img = cv2.imread(row['img_name'])
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Вырезаем ROI (знак) по координатам из аннотации
    x1, y1, x2, y2 = row['x'], row['y'], row['x2'], row['y2']
    roi = img[y1:y2, x1:x2]
    
    # Ресайз и нормализация
    roi = cv2.resize(roi, IMG_SIZE)
    X.append(roi / 255.0)
    y.append(CLASSES[row['class_name']])

X = np.array(X)
y = to_categorical(y, num_classes=len(CLASSES))

# Разделение на train/test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)





model = Sequential([
    Conv2D(16, (3,3), activation='relu', input_shape=(32, 32, 3)),
    MaxPooling2D(2,2),
    Dropout(0.25),
    
    Conv2D(32, (3,3), activation='relu'),
    MaxPooling2D(2,2),
    Dropout(0.25),
    
    Flatten(),
    Dense(64, activation='relu'),
    Dropout(0.5),
    Dense(5, activation='softmax')
])

model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Обучение
history = model.fit(
    X_train, y_train,
    validation_data=(X_test, y_test),
    epochs=60,
    batch_size=32
)

# Сохранение модели (должно быть < 2 МБ)
model.save('traffic_sign_model.h5')