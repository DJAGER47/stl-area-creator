#!/bin/bash

# Скрипт для пакетной генерации STL файлов из списка регионов
# Использование: ./batch_generate.sh <шаг_генерации_в_метрах>
# Пример: ./batch_generate.sh 1000

# Проверка наличия аргумента
if [ -z "$1" ]; then
    echo "Ошибка: Не указан шаг генерации"
    echo "Использование: $0 <шаг_генерации_в_метрах>"
    echo "Пример: $0 1000"
    exit 1
fi

STEP=$1
LIST_FILE="1.txt"

# Проверка существования файла со списком
if [ ! -f "$LIST_FILE" ]; then
    echo "Ошибка: Файл $LIST_FILE не найден"
    exit 1
fi

# Проверка существования скрипта генерации
if [ ! -f "full_generate_utm.py" ]; then
    echo "Ошибка: Скрипт full_generate_utm.py не найден"
    exit 1
fi

echo "Начинаем пакетную генерацию с шагом $STEP метров"
echo "Читаем список регионов из $LIST_FILE"
echo "========================================"

# Счётчик обработанных регионов
TOTAL=0
SUCCESS=0
FAILED=0

# Чтение файла построчно и обработка каждого региона
while IFS= read -r line; do
    # Пропускаем пустые строки
    if [ -z "$line" ]; then
        continue
    fi
    
    # Извлекаем название региона (убираем номер в начале и пробелы)
    REGION=$(echo "$line" | sed 's/^[0-9]*[[:space:]]*//' | xargs)
    
    # Пропускаем пустые строки после обработки
    if [ -z "$REGION" ]; then
        continue
    fi
    
    TOTAL=$((TOTAL + 1))
    echo ""
    echo "[$TOTAL] Генерация региона: $REGION"
    echo "----------------------------------------"
    
    # Запуск генерации для текущего региона
    if python3 full_generate_utm.py -s "$STEP" -n "$REGION"; then
        echo "✓ Успешно: $REGION"
        SUCCESS=$((SUCCESS + 1))
    else
        echo "✗ Ошибка: $REGION"
        FAILED=$((FAILED + 1))
    fi
    
    echo "----------------------------------------"
    
done < "$LIST_FILE"

echo ""
echo "========================================"
echo "Генерация завершена!"
echo "Всего регионов: $TOTAL"
echo "Успешно: $SUCCESS"
echo "С ошибками: $FAILED"
echo "========================================"

# Выход с кодом ошибки, если были неудачные попытки
if [ $FAILED -gt 0 ]; then
    exit 1
fi

exit 0
