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
~~~
cd C:\Users\Mane\Desktop\Python\AI_Animepython -m venv .venv.\.venv\Scripts\Activate.ps1python -m pip install --upgrade pippip install huggingface_hub@'from pathlib import Pathfrom huggingface_hub import snapshot_downloadrepo_id = "Lightricks/LTX-Video"local_dir = Path("models/ltx-video")local_dir.mkdir(parents=True, exist_ok=True)snapshot_download(    repo_id=repo_id,    local_dir=str(local_dir),    allow_patterns=[        "ltx-video-2b-v0.9.safetensors",        "model_index.json",        "tokenizer/*",        "text_encoder/*",    ],)print("DONE: model files downloaded to", local_dir.resolve())'@ | Set-Content -Encoding UTF8 .\download_ltx_models.pypython .\download_ltx_models.py
Потом проверка:
Test-Path .\models\ltx-video\ltx-video-2b-v0.9.safetensorsTest-Path .\models\ltx-video\model_index.jsonTest-Path .\models\ltx-video\tokenizerTest-Path .\models\ltx-video\text_encoder
Должно быть:
TrueTrueTrueTrue
Если скачивание медленное или будет warning про unauthenticated requests:
hf auth loginpython .\download_ltx_models.py
После этого уже ставишь зависимости проекта:
pip install -r requirements.txt
И запускаешь:
python main.py


~~~
