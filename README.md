# Local LTX-Video Gradio

Простой локальный image-to-video интерфейс для LTX-Video.

## Структура

```text
main.py
requirements.txt
models/ltx-video/
inputs/
outputs/
```

## Подготовка

1. Создайте виртуальное окружение:

```powershell
python -m venv .venv
.\.venv\Scripts\activate
```

2. Установите зависимости:

```powershell
pip install -r requirements.txt
```

3. Положите локальный single-file checkpoint LTX-Video сюда:

```text
models/ltx-video/ltx-video-2b-v0.9.safetensors
```

Приложение загружает этот файл через `from_single_file()` и не скачивает модель автоматически.

## Запуск

```powershell
python main.py
```

После запуска Gradio откроет web-интерфейс в браузере.

## Использование

1. Загрузите картинку.
2. Введите prompt / сценарий движения.
3. Выберите длительность: `3` или `5` секунд.
4. Выберите разрешение: `512x512` или `768x512`.
5. Нажмите `Generate`.

Входные картинки сохраняются в `inputs/`, готовые mp4-видео сохраняются в `outputs/`.

## Test mode

В корне проекта есть `config.py`:

```python
TEST_MODE = True
```

Когда `TEST_MODE = True`, интерфейс проверяет загрузку картинки, prompt и папки `inputs/` / `outputs/`, но модель не запускается и видео не генерируется.

Для настоящей генерации поставьте:

```python
TEST_MODE = False
```

Если файла `config.py` нет или переменная `TEST_MODE` не указана, приложение считает `TEST_MODE = False`.

## RTX 3060 Ti 8GB

Для меньшей нагрузки на VRAM используйте:

- `512x512`
- `3 seconds`
- `8-30` inference steps

Если появится ошибка нехватки VRAM, приложение очистит CUDA cache и покажет понятное сообщение.
