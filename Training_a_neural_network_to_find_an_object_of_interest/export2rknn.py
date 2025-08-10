import os
from ultralytics import YOLO

# Убедитесь, что пути существуют
# os.makedirs('/home/arrma/PROGRAMMS/Arhipelag_2025/Training_a_neural_network_to_find_an_object_of_interest/best_cows', exist_ok=True)

# Загрузка модели
model = YOLO('/home/arrma/PROGRAMMS/Arhipelag_2025/Training_a_neural_network_to_find_an_object_of_interest/weights/best_sana_v2.pt')


# Экспорт в RKNN
try:
    model.export(
        format="rknn",
        name='rk3566',
        # device='cpu',  # Используем CPU для конвертации 

    )
    print("Экспорт успешно завершен!")
except Exception as e:
    print(f"Ошибка экспорта: {e}")
