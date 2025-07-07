import time
from pioneer_sdk import Pioneer


class CustomPioneer(Pioneer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._hover_start_time = None
        self._hover_duration = 0
        self._point_reached_flag = False
        self._is_moving = False


    def point_reached(self, custom_behaviour=False):
        """
        Возвращает True если точка достигнута.
        
        Параметры:
            custom_behaviour: False - работает как базовый метод (по умолчанию)
                           True - сохраняет True до следующего движения
        
        Поведение:
        - При custom_behaviour=False: работает как базовый метод
        - При custom_behaviour=True: сохраняет True до следующего вызова go_to_local_point()
        """
        base_reached = super().point_reached()
        
        if not custom_behaviour:
            return base_reached
            
        # Кастомное поведение
        if not self._is_moving and base_reached:
            self._point_reached_flag = True
        return self._point_reached_flag



    def go_to_local_point(self, x, y, z, yaw):
        """Переопределенный метод движения, сбрасывающий флаг достижения точки"""
        self._point_reached_flag = False
        self._is_moving = True
        super().go_to_local_point(x, y, z, yaw)
        self._is_moving = False

    def start_hover(self, duration):
        """Начинает процесс зависания на указанное время"""
        if not self.is_hovering():
            self._hover_start_time = time.time()
            self._hover_duration = duration
            print(f"[Pioneer] Начато зависание на {duration} сек")

    def is_hovering(self):
        """Проверяет, выполняется ли зависание"""
        return self._hover_start_time is not None

    def check_hover_complete(self):
        """Проверяет завершение зависания и возвращает True по его окончании"""
        if not self.is_hovering():
            return False
            
        if time.time() - self._hover_start_time >= self._hover_duration:
            self._hover_start_time = None
            print("[Pioneer] Зависание завершено")
            return True
            
        return False