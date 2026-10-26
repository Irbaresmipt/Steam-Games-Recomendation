# Рекомендательная система на данных Steam

ФИО: Поплевин Никита Дмитриевич (Б05-326)

**Датасет:** Game Recommendations on Steam (Anton Kozyriev, Kaggle)
**Ссылка:** https://www.kaggle.com/datasets/antonkozyriev/game-recommendations-on-steam

## Запуск
1. `python -m venv .venv && source .venv/bin/activate`
2. `pip install -r requirements.txt`
3. токен Kaggle в `~/.kaggle/kaggle.json`
4. Можно запускать ноутбук

## Модели (`models/`)
Item-kNN, iALS, EASE, FM с метапризнаками, iALS с сигналом по часам.
