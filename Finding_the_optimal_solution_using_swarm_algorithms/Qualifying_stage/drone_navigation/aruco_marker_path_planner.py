import math
from pioneer_sdk import Pioneer

class ArucoMarkerPathPlanner :
    def __init__(self, all_avg_coords, pioneer_instance:Pioneer):
        """
        Инициализация планировщика полета.
        
        Параметры:
            all_avg_coords: словарь {marker_id: (x, y)} с координатами маркеров
            pioneer: экземпляр класса Pioneer (опционально)
            start_position: начальная позиция дрона (если None, получаем от дрона)
        """
        self.all_avg_coords = all_avg_coords
        self.pioneer = pioneer_instance
        self.start_position = self._get_current_position_drone()
        self.marker_graph = self._build_marker_graph()
        self._flight_plan = self._generate_flight_plan() 
    

    def _get_current_position_drone(self):
        """Получает текущую позицию дрона через класс Pioneer."""
        return self.pioneer.get_local_position_lps(get_last_received=True)[:2]
    

    def _build_marker_graph(self):
        """Строит граф переходов между маркерами на основе их ID."""
        graph = {}
        for marker_id in self.all_avg_coords:
            first_digit = marker_id // 10  
            second_digit = marker_id % 10  
            graph[first_digit] = second_digit
        return graph
    
    def _find_closest_marker(self, position):
        """Находит маркер, ближайший к указанной позиции."""
        closest_id = None
        min_distance = float('inf')
        
        for marker_id, coords in self.all_avg_coords.items():
            distance = math.dist(position, coords)
            if distance < min_distance:
                min_distance = distance
                closest_id = marker_id
        return closest_id
    


    def _generate_flight_plan(self):
        """Генерирует план полёта по маркерам (вызывается автоматически при инициализации)."""
        current_position = self._get_current_position_drone()
        visited = set()
        flight_plan = []
        
        current_marker_id = self._find_closest_marker(current_position)
        flight_plan.append(current_marker_id)
        visited.add(current_marker_id // 10)  
        
        while True:
            current_node = current_marker_id // 10
            next_node = self.marker_graph.get(current_node)
            
            if next_node is None or next_node in visited:
                break
                
            next_marker_candidates = [
                marker_id for marker_id in self.all_avg_coords 
                if marker_id // 10 == next_node and marker_id not in visited
            ]
            
            if not next_marker_candidates:
                break
                
            next_marker_id = next_marker_candidates[0]
            flight_plan.append(next_marker_id)
            visited.add(next_node)
            current_marker_id = next_marker_id
            
        return flight_plan
    
    def get_flight_plan(self):
        """Возвращает сгенерированный план полёта (ID маркеров)."""
        return self._flight_plan
    
    def get_coordinates_plan(self, flight_plan=None):
        """
        Возвращает план полёта в виде координат.
        
        Параметры:
            flight_plan: если None, используется сохранённый план
        """
        if flight_plan is None:
            flight_plan = self._flight_plan
        return [self.all_avg_coords[marker_id] for marker_id in flight_plan]
