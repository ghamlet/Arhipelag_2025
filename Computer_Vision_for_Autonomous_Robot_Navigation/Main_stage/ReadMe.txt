На Айкаре нужно запускать "Driving_on_road.py".
func - вспомогательный для Driving_on_road.py. Он должен находиться в той-же директории, что и Driving_on_road.py.

Файл ForArduino предоставлен вам для ознакомпления, он загружен в память микроконтроллера на Айкаре. Не рекомендуется изменять его и перепрошивать контроллер.



Инструкция по настройке и использованию скрипта синхронизации Rsync
Содержание
Необходимые условия


Настройка SSH-ключей
1. Генерация пары SSH-ключей (на локальной машине)
bash
ssh-keygen -t ed25519 -C "dmtr"
При запросе:

Оставьте стандартный путь (~/.ssh/id_ed25519)

Установите парольную фразу (необязательно, но рекомендуется для безопасности)

2. Копирование публичного ключа на удаленный сервер
bash
ssh-copy-id -p 22 avt_user@192.168.1.4
Введите пароль от удаленного сервера при запросе.

3. Проверка входа без пароля
bash
ssh -p 22 avt_user@192.168.1.4


Установка Rsync
На обеих машинах (локальной и удаленной):
Для Debian/Ubuntu:

bash
sudo apt update && sudo apt install rsync openssh-client -y


Настройка скрипта
Сохраните скрипт как remote_sync.sh

Сделайте его исполняемым:

bash
chmod +x remote_sync.sh
Отредактируйте переменные конфигурации:

bash
nano remote_sync.sh
Замените следующие параметры:

bash
LOCAL_DIR="/реальный/путь/к/локальной/папке"
REMOTE_USER="ваш_пользователь"
REMOTE_HOST="ваш.сервер.com"
REMOTE_DIR="/путь/на/сервере"
SSH_PORT="22"  # Если используете нестандартный порт



Длина прерывистой линии 10 см
Расстояние между ними 5 см
Высота цилиндра 15 см

scp -r avt_user@192.168.0.58:~/Arhipelag_2025/Computer_Vision_for_Autonomous_Robot_Navigation/Main_stage/recordings/* C:\Users\User.DESKTOP-JG5N3Q2\Desktop\Arhipelag_2025\Computer_Vision_for_Autonomous_Robot_Navigation\Main_stage\recordings