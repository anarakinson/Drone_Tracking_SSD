# Подготовка:
---

Перед работой нужно скачать архив с сохраненными моделями workspace.zip по ссылке:

https://drive.google.com/drive/folders/15SZZ0nlqZLGlFPS573M1izXg0--Y37pf?usp=sharing

и распаковать в директорию Tensorflow
---

# Object detection and tracking with ssd-mobilenet-320
---

Для установки среды:
```shell
# windows
python -m venv venv
#linux
python -m virtualenv venv
```

Установка необходимых библиотек:
```shell
pip install -r requirements.txt
python setup.py
```

Проверка установки библиотек для детекции:
```shell
python verification.py
```
---
Детекция и трекинг производятся с вебкамеры
---
Детекция:
```shell
python detect.py
```

Трекинг:
```shell
python object_trecking.py
```
