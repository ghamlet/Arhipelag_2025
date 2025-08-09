import time
from typing import Optional, Tuple  # Аннотации типов для статической проверки кода
from pioneer_sdk import Pioneer  # Базовый класс управления дроном Pioneer
import logging  # Модуль для логирования событий и ошибок
from dataclasses import dataclass  # Декоратор для создания классов данных
from enum import Enum, auto  # Создание перечислений для состояний системы



class NavigationState(Enum):
    """Перечисление состояний навигации дрона (конечный автомат)
    
    Состояния:
        IDLE - режим ожидания (дрон не выполняет команд)
        MOVING - режим перемещения к целевой точке
        HOVERING - режим зависания на месте
    """
    IDLE = auto()      # Дрон в режиме ожидания команд
    MOVING = auto()    # Дрон выполняет перемещение к цели
    HOVERING = auto()  # Дрон завис в воздухе на заданной позиции

@dataclass
class NavigationTarget:
    """Класс для хранения информации о целевой точке навигации
    
    Атрибуты:
        position (Tuple[float, float, float]): Координаты цели (x, y, z) в метрах
        yaw (float): Угол ориентации дрона (рысканье) в радианах, по умолчанию 0
        reached (bool): Флаг достижения цели, по умолчанию False
    """
    position: Tuple[float, float, float]  # Целевые координаты (x, y, z)
    yaw: float = 0.0  # Угол рысканья (направление носа дрона)
    reached: bool = False  # Флаг, указывающий достигнута ли цель

class CustomPioneer(Pioneer):
    """Расширенный контроллер дрона Pioneer с улучшенной навигацией
    
    Наследует функциональность базового класса Pioneer и добавляет:
    - Точное отслеживание позиции с использованием фильтрации
    - Контроль времени зависания с автоматическим переходом между состояниями
    - Улучшенное определение достижения точки с настраиваемой погрешностью
    
    Атрибуты:
        _navigation_state (NavigationState): Текущее состояние навигации
        _target (Optional[NavigationTarget]): Текущая целевая точка
        _hover_start_time (Optional[float]): Время начала зависания (timestamp)
        _hover_duration (float): Заданная длительность зависания в секундах
    """
    
    def __init__(self, name='pioneer', ip='192.168.4.1', mavlink_port=8001, 
                 connection_method='udpout', device='/dev/serial0', baud=115200, 
                 verbose=False):
        """Инициализация контроллера дрона с расширенной навигацией
        
        Параметры:
            name (str): Идентификатор дрона, по умолчанию 'pioneer'
            ip (str): IP-адрес для UDP подключения, по умолчанию '192.168.4.1'
            mavlink_port (int): Порт для MAVLink протокола, по умолчанию 8001
            connection_method (str): Метод подключения ('udpout' или 'serial')
            device (str): Путь к последовательному порту для serial подключения
            baud (int): Скорость передачи для serial подключения
            verbose (bool): Флаг подробного логирования, по умолчанию False
            
        Исключения:
            ConnectionError: При неудачном подключении к дрону
            ValueError: При недопустимых параметрах подключения
        """
        # Инициализация базового класса Pioneer
        super().__init__(
            name=name,
            ip=ip,
            mavlink_port=mavlink_port,
            connection_method=connection_method,
            device=device,
            baud=baud,
            logger=verbose,
            log_connection=verbose
        )
        
        # Инициализация состояния навигации
        self._navigation_state = NavigationState.IDLE  # Начальное состояние
        self._target: Optional[NavigationTarget] = None  # Целевая точка
        self._hover_start_time: Optional[float] = None  # Время начала зависания
        self._hover_duration: float = 0.0  # Длительность зависания

        self.logger = logging.getLogger(f"{name}")  
        self.logger.info(f"Инициализирован контроллер дрона {name}")


    def go_to_local_point(self, x: float, y: float, z: float, yaw: float = 0.0) -> None:
        """Команда дрону переместиться в указанную локальную точку
        
        Параметры:
            x (float): Координата X в метрах (вперед/назад)
            y (float): Координата Y в метрах (влево/вправо)
            z (float): Координата Z в метрах (вверх/вниз)
            yaw (float): Угол рысканья в радианах, по умолчанию 0
            
        Исключения:
            ValueError: Если координаты не являются числами
            
        Логика работы:
            1. Проверка корректности координат
            2. Установка новой цели навигации
            3. Переход в состояние MOVING
            4. Вызов родительского метода перемещения
        """
        # Проверка типа координат
        if not all(isinstance(coord, (int, float)) for coord in (x, y, z)):
            self.logger.error("Некорректные координаты: должны быть числами")
            raise ValueError("Coordinates must be numeric")
            
        # Установка новой цели
        self._target = NavigationTarget(position=(x, y, z), yaw=yaw)
        self._navigation_state = NavigationState.MOVING
        self.logger.info(f"Начато перемещение к точке: X={x:.2f}, Y={y:.2f}, Z={z:.2f}")
        
        # Вызов родительского метода
        super().go_to_local_point(x, y, z, yaw)

    def start_hover(self, duration: float) -> None:
        """Начать процедуру зависания на заданное время
        
        Параметры:
            duration (float): Длительность зависания в секундах (должна быть > 0)
            
        Исключения:
            ValueError: Если длительность не положительная
            
        Логика работы:
            1. Проверка корректности длительности
            2. Запоминание времени начала
            3. Установка длительности
            4. Переход в состояние HOVERING
        """
        if duration <= 0:
            self.logger.error("Длительность зависания должна быть положительной")
            raise ValueError("Hover duration must be positive")
            
        if self._navigation_state != NavigationState.HOVERING:
            self._hover_start_time = time.time()  # Фиксируем время начала
            self._hover_duration = duration  # Устанавливаем длительность
            self._navigation_state = NavigationState.HOVERING  # Меняем состояние
            self.logger.info(f"Начато зависание на {duration:.1f} секунд")

    def is_hovering(self) -> bool:
        """Проверка, находится ли дрон в режиме зависания
        
        Возвращает:
            bool: True если дрон в состоянии HOVERING, иначе False
            
        Пример использования:
            if drone.is_hovering():
                print("Дрон завис")
        """
        return self._navigation_state == NavigationState.HOVERING

    def check_hover_complete(self) -> bool:
        """Проверка завершения времени зависания
        
        Возвращает:
            bool: True если время зависания истекло, иначе False
            
        Логика работы:
            1. Проверка текущего состояния
            2. Расчет прошедшего времени
            3. Сравнение с заданной длительностью
            4. Сброс состояния при завершении
        """
        if not self.is_hovering():
            self.logger.debug("Дрон не в режиме зависания")
            return False
            
        elapsed = time.time() - self._hover_start_time
        if elapsed >= self._hover_duration:
            self._navigation_state = NavigationState.IDLE  # Возврат в ожидание
            self._hover_start_time = None  # Сброс таймера
            self.logger.info("Зависание завершено")
            return True
            
        self.logger.debug(f"Осталось времени зависания: {self._hover_duration - elapsed:.1f} сек")
        return False


    def point_reached(self, threshold: float = None) -> bool:
        """Проверка достижения целевой точки
        
        Параметры:
            fast_mode (bool): Использовать оптимизированную проверку (по умолчанию False)
            threshold (float): Допустимая погрешность в метрах (по умолчанию 0.1)
            
        Возвращает:
            bool: True если точка достигнута (однократно)
            
        Исключения:
            RuntimeError: Если целевая точка не установлена
            
        Логика работы:
            1. Проверка наличия цели
            2. Проверка флага достижения
            3. Выбор метода проверки (базовый/оптимизированный)
            4. Проверка попадания в радиус threshold
        """
        if self._target is None:
            self.logger.error("Целевая точка не установлена")
            raise RuntimeError("No target position set")
            
        if self._target.reached:
            self.logger.debug("Целевая точка уже была достигнута")
            return False
            
        # Использование базового метода при отключенном fast_mode
        if threshold is None:
            return super().point_reached()
            
        # Оптимизированная проверка положения
        current_pos = self.get_local_position_lps(get_last_received=True)
        if current_pos is None:
            self.logger.warning("Не удалось получить текущую позицию")
            return False
            
        # Извлечение координат X,Y (Z игнорируется для упрощения)
        current_x, current_y, current_z = current_pos
        target_x, target_y, target_z = self._target.position
        
        # Проверка попадания в радиус threshold для каждой оси
        x_reached = abs(current_x - target_x) < threshold
        y_reached = abs(current_y - target_y) < threshold
        # z_reached = abs(current_z - target_z) < threshold

        
        # Если достигли цели по обеим осям
        if x_reached and y_reached:
            self._target.reached = True  # Устанавливаем флаг
            self._navigation_state = NavigationState.IDLE  # Переходим в ожидание
            self.logger.info(f"Целевая точка достигнута (погрешность {threshold} м)")
            return True
            
        return False
    


    def land_on_marker(self, threshold=0.3):
         
        if self._target is None:
            self.logger.error("Целевая точка не установлена")
            raise RuntimeError("No target position set")
            
        reached = False
        
        target_x, target_y, target_z = self._target.position

        self.go_to_local_point(x=target_x, y=target_y, z=0, yaw=0)


        while not reached:
            # Оптимизированная проверка положения
            current_pos = self.get_local_position_lps(get_last_received=True)
            if current_pos is None:
                self.logger.warning("Не удалось получить текущую позицию")
                return False
                
            # Извлечение координат X,Y (Z игнорируется для упрощения)
            current_x, current_y, current_z = current_pos
            target_x, target_y, target_z = self._target.position
            
            # Проверка попадания в радиус threshold для каждой оси
            x_reached = abs(current_x - target_x) < threshold
            y_reached = abs(current_y - target_y) < threshold
            # z_reached = abs(current_z - target_z) < threshold

            
            # Если достигли цели по обеим осям
            if x_reached and y_reached and current_z < 0.1:
                self._target.reached = True  # Устанавливаем флаг
                self._navigation_state = NavigationState.IDLE  # Переходим в ожидание
                self.logger.info(f"Целевая точка достигнута (погрешность {threshold} м)")
                reached = True
                return 
        
