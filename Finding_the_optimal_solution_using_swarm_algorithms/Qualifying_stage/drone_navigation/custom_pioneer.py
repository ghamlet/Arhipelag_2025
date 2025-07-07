import time
from pioneer_sdk import Pioneer


class CustomPioneer(Pioneer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._hover_start_time = None
        self._hover_duration = 0


    def start_hover(self, duration):
        """Начинает процесс зависания на указанное время"""
        if not self.is_hovering():
            self._hover_start_time = time.time()
            self._hover_duration = duration
            # print(f"[Pioneer] Начато зависание на {duration} сек")


    def is_hovering(self):
        """Проверяет, выполняется ли зависание"""
        return self._hover_start_time is not None


    def check_hover_complete(self):
        """Проверяет завершение зависания и возвращает True по его окончании"""
        if not self.is_hovering():
            return False
            
        if time.time() - self._hover_start_time >= self._hover_duration:
            self._hover_start_time = None
            # print("[Pioneer] Зависание завершено")
            return True
            
        return False