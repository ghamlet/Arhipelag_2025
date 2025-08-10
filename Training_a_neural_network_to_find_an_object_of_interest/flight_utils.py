import json
from pathlib import Path
from typing import List, Tuple



def load_flight_coordinates() -> List[Tuple[float, float, float]]:
    """
    Загружает координаты полета из файла flight_path.json
    
    Returns:
        List[Tuple[float, float, float]]: Список координат в формате [(x, y, z), ...]
    
    Raises:
        FileNotFoundError: Если файл не найден
        json.JSONDecodeError: Если файл имеет неверный формат
        ValueError: Если координаты не прошли валидацию
    """
    # Получаем путь к директории, где находится текущий скрипт
    script_dir = Path(__file__).parent.absolute()
    
    # Формируем полный путь к файлу JSON
    json_path = script_dir / 'flight_path_120.json'
    
    # Проверяем существование файла
    if not json_path.exists():
        raise FileNotFoundError(f"Файл координат не найден: {json_path}")
    
    # Загружаем координаты
    with open(json_path, 'r') as f:
        coordinates = json.load(f)
    
    # Валидируем координаты
    validate_coordinates(coordinates)
    
    return coordinates


def validate_coordinates(coords: List[List[float]]) -> None:
    """
    Проверяет корректность формата координат
    
    Args:
        coords: Список координат для проверки
        
    Raises:
        ValueError: Если формат координат неверный
    """
    if not isinstance(coords, list):
        raise ValueError("Координаты должны быть списком")
    
    for i, point in enumerate(coords):
        if len(point) != 3:
            raise ValueError(f"Точка {i} должна содержать 3 значения (x, y, z), получено: {point}")
        
        if not all(isinstance(coord, (int, float)) for coord in point):
            raise ValueError(f"Точка {i} содержит нечисловые значения: {point}")



def get_config(env='local'):
    config_path = Path(__file__).parent / 'config.json'
    with open(config_path) as f:
        data = json.load(f)
        return data['environments'][env]



if __name__ == "__main__":
    try:
        coordinates = load_flight_coordinates()
        print(f"Успешно загружено {len(coordinates)} точек маршрута")
        print("Первые 5 точек:", coordinates[:5])

    except FileNotFoundError as e:
        print(f"Ошибка: {e}")
        print("Пожалуйста, убедитесь что файл flight_path.json находится в той же папке, что и скрипт")
    except json.JSONDecodeError:
        print("Ошибка: Некорректный формат JSON файла")
    except ValueError as e:
        print(f"Ошибка валидации координат: {e}")