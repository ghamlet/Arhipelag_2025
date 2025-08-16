import cv2

def get_camera_resolution(camera_index=0):
    """
    Определяет разрешение и параметры видеопотока камеры
    :param camera_index: индекс камеры (по умолчанию 0)
    :return: словарь с параметрами камеры
    """
    cap = cv2.VideoCapture(camera_index)
    
    if not cap.isOpened():
        print(f"Не удалось открыть камеру с индексом {camera_index}")
        return None
    
    # Получаем параметры видеопотока
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    codec = int(cap.get(cv2.CAP_PROP_FOURCC))
    codec_str = "".join([chr((codec >> 8 * i) & 0xFF) for i in range(4)])
    
    # Получаем список поддерживаемых разрешений (экспериментально)
    test_resolutions = [
        (640, 480),   # VGA
        (1280, 720),  # HD
        (1920, 1080), # Full HD
        (3840, 2160)  # 4K
    ]
    supported_resolutions = []
    
    for w, h in test_resolutions:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, w)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
        actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        if (actual_w, actual_h) == (w, h):
            supported_resolutions.append((w, h))
    
    # Возвращаем камеру к исходному разрешению
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    
    # Читаем один кадр для проверки
    ret, frame = cap.read()
    if ret:
        actual_resolution = (frame.shape[1], frame.shape[0])
    else:
        actual_resolution = (width, height)
    
    cap.release()
    
    return {
        'current_resolution': actual_resolution,
        'reported_resolution': (width, height),
        'fps': fps,
        'codec': codec_str,
        'supported_resolutions': supported_resolutions,
        'is_resolution_matching': actual_resolution == (width, height)
    }

if __name__ == "__main__":
    # Проверяем все доступные камеры

    camera_info = get_camera_resolution(1)
    
    if camera_info:
        print(f"Текущее разрешение: {camera_info['current_resolution']}")
        print(f"Заявленное разрешение: {camera_info['reported_resolution']}")
        print(f"FPS: {camera_info['fps']:.2f}")
        print(f"Кодек: {camera_info['codec']}")
        print("Поддерживаемые разрешения:", camera_info['supported_resolutions'])
        print("Совпадение заявленного и фактического разрешения:", 
                "Да" if camera_info['is_resolution_matching'] else "Нет")
        
        # Проверяем, может ли камера изменять разрешение
        if len(camera_info['supported_resolutions']) > 1:
            print("Камера поддерживает несколько разрешений")
        else:
            print("Камера, вероятно, имеет фиксированное разрешение")
    else:
        print("Камера не найдена или недоступна")