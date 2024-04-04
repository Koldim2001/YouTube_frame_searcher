## 🌟 YouTube Frame Searcher 🌟 - Ваш личный поисковик по YouTube! 🧭

Представляем вашему вниманию программу, которая позволит вам быстро и легко найти нужный момент в видео на YouTube! 🎥🔍

С помощью YouTube Frame Searcher, вы можете вводить список YouTube-ссылок на интересующие вас видео. Затем, вы можете передать либо текстовый запрос, либо изображение. На основе вашего запроса, программа будет искать самые релевантные кадры и предоставит вам ссылки на интересные моменты. 🔗

__Этот проект - это настоящая находка для тех, кто хочет быстро найти нужный момент в видео без необходимости просматривать всю его длительность. 🕒__


![example-of-web](configs/example.gif)

---

## __Локальная установка:__

### __Используя Python:__
Необходимо иметь установленный python 3.10 или более новой версии. \
Данные команды требуется запускать последовательно в терминале:
1. Склонируйте к себе этот репозиторий 

2. Перейдите с помощью команды cd в созданную папку 

3. Загрузите все необходимые библиотеки:
```
pip install --upgrade --force-reinstall "git+https://github.com/ytdl-org/youtube-dl.git"
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt
```
4. Запустите streamlit сервер:
```
streamlit run web.py
```
Для запуска веб-приложения надо перейти по адресу http://localhost:8501