#!/bin/bash

# Конфигурация
LOCAL_DIR="/home/arrma/PROGRAMMS/Arhipelag_2025/Computer_Vision_for_Autonomous_Robot_Navigation/Main_stage"
REMOTE_USER="avt_user"
REMOTE_HOST="192.168.1.4"
REMOTE_DIR=""
SSH_PORT="22"
LOG_FILE="/var/log/sync_$(date +%Y%m%d).log"

echo "=== Начало синхронизации: $(date) ===" | tee -a "$LOG_FILE"
echo "Локальная папка: $LOCAL_DIR" | tee -a "$LOG_FILE"
echo "Удаленный сервер: $REMOTE_USER@$REMOTE_HOST:$REMOTE_DIR" | tee -a "$LOG_FILE"

# Создаём удалённую папку (если не существует)
ssh -p "$SSH_PORT" "$REMOTE_USER@$REMOTE_HOST" "mkdir -p '$REMOTE_DIR'"

# Запуск синхронизации
rsync -avz --delete --progress -e "ssh -p $SSH_PORT" \
    "$LOCAL_DIR/" "$REMOTE_USER@$REMOTE_HOST:$REMOTE_DIR/" \
    | tee -a "$LOG_FILE"

# Проверка статуса
if [ $? -eq 0 ]; then
    echo "=== Синхронизация успешно завершена: $(date) ===" | tee -a "$LOG_FILE"
else
    echo "!!! Ошибка синхронизации: $(date) !!!" | tee -a "$LOG_FILE"
    exit 1
fi