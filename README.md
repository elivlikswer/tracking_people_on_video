# MOT17 Multi-Object Tracking (YOLO + ByteTrack)

Проект по дисциплине "Компьютерное зрение". Реализация базового пайплайна для отслеживания пешеходов на датасете MOT17.

## Стек
- Детектор: YOLO11 (Ultralytics)
- Трекер: ByteTrack
- Формат вывода: MOT Challenge (txt)

## Инструкция по запуску
1. Скачайте датасет MOT17 по ссылке: https://motchallenge.net/data/MOT17/
2. Распакуйте архивы в папку `data/MOT17/train/`. 
3. Установите зависимости:
   ```bash
   pip install -r requirements.txt