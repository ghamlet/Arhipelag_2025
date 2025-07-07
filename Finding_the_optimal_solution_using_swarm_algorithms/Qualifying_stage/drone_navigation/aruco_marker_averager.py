class ArucoMarkerAverager:
    def __init__(self, min_samples=10, max_samples=100):
        """
        Инициализация трекера для хранения и усреднения координат маркеров.
        
        Параметры:
            min_samples: минимальное количество измерений для усреднения (по умолчанию 10)
            max_samples: максимальное количество хранимых измерений (по умолчанию 100)
        """
        self.marker_data = {}  
        self.min_samples = min_samples
        self.max_samples = max_samples
    
    
    def add_marker_sample(self, markers_dict):
        """
        Добавляет новые координаты для всех маркеров из словаря.
        
        Параметры:
            markers_dict: словарь {marker_id: (x, y)} с текущими координатами маркеров
                         или None/пустой словарь, если маркеры не обнаружены
        """
        if not markers_dict:
            return
        
        for marker_id, coords in markers_dict.items():
            if coords is None:
                continue
                
            if marker_id not in self.marker_data:
                self.marker_data[marker_id] = {
                    'samples': [],
                    'avg_coords': None
                }
            
            samples = self.marker_data[marker_id]['samples']
            samples.append(coords)
            
            if len(samples) > self.max_samples:
                samples.pop(0)
            
            if len(samples) >= self.min_samples:
                avg_x = round(sum(x for x, y in samples) / len(samples), 2)  # Округление
                avg_y = round(sum(y for x, y in samples) / len(samples), 2)  # Округление
                self.marker_data[marker_id]['avg_coords'] = (avg_x, avg_y)
                
               
    def get_marker_coords_by_id(self, marker_id):
        """
        Возвращает усредненные координаты для указанного маркера по его ID.
        
        Параметры:
            marker_id: ID маркера (целое число)
            
        Возвращает:
            tuple: (avg_x, avg_y) средние координаты (округленные до 2 знаков) 
                  или None, если данных недостаточно
        """
        marker = self.marker_data.get(marker_id)
        if marker and marker['avg_coords']:
            return (round(marker['avg_coords'][0], 2), 
                    round(marker['avg_coords'][1], 2))
        return None


    def get_all_markers_coords(self):
        """
        Возвращает усредненные координаты всех отслеживаемых маркеров.
        
        Возвращает:
            dict: {marker_id: (avg_x, avg_y)} словарь с координатами (округленными до 2 знаков)
                  только тех маркеров, для которых есть усредненные данные
        """
        return {
            marker_id: (round(data['avg_coords'][0], 2), 
                      round(data['avg_coords'][1], 2))
            for marker_id, data in self.marker_data.items()
            if data['avg_coords'] is not None
        }
   