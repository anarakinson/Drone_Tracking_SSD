# Object detection and tracking with ssd-mobilenet-320
***

# Создание модели:
***
Необходимо запустить ноутбук **Training_Model.ipynb**, добавить снимки для обучения в нужные директории (согласно комментариям в ноутбуке) и сохранить модель для дальнейшего использования. В директории workspace/models/ уже есть сохраненные модели, обученные на [датасете с сайта kaggle.com](https://www.kaggle.com/datasets/dasmehdixtr/drone-dataset-uav)


### Подготовка:
***

Перед работой нужно скачать архив с сохраненными моделями **workspace.zip** и распаковать в директорию Tensorflow

Скачать можно по ссылке:

'https://drive.google.com/drive/folders/15SZZ0nlqZLGlFPS573M1izXg0--Y37pf?usp=sharing'

***

### Detection and tracking
***

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
***

Детекция и трекинг производятся с веб-камеры
***

Детекция:
```shell
python detect.py
```

Трекинг:
```shell
python object_trecking.py
```
