import os
import youtube_dl
import cv2

from PIL import Image
import cv2
import os
import numpy as np

from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import euclidean_distances
import matplotlib.pyplot as plt


# Функция для извлечения кадров из видео каждые N секунд
def extract_frames(video_url, interval_sec=10):
    ydl_opts = {}
    ydl = youtube_dl.YoutubeDL(ydl_opts)
    info_dict = ydl.extract_info(video_url, download=False)
    formats = info_dict.get('formats', None)

    # Ищем формат 360p
    desired_format = next((f for f in formats if f.get('format_note') == '360p'), None)

    if desired_format:
        url = desired_format.get('url', None)
        cap = cv2.VideoCapture(url)

        # Получаем общее количество кадров 
        frame_count_max = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        # Устанавливаем частоту кадров видео
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        time_max = frame_count_max * (1/fps) * 1000 # в мс

        # Устанавливаем начальное время итерации
        time_stamp = 0

        # Устанавливаем путь для сохранения кадров
        save_path = 'images'
        frame_count = 0

        while cap.isOpened(): 
            # Устанавливаем позицию видео на текущее время
            cap.set(cv2.CAP_PROP_POS_MSEC, time_stamp)

            # Считываем кадр
            ret, frame = cap.read()

            # выход по превышению времени:
            if time_stamp >= time_max:
                break

            # Если кадр считывается успешно
            if ret:
                # Сохраняем кадр в папку images
                frame_count += 1
                cv2.imwrite(os.path.join(save_path, f"frame_{frame_count}_{video_url.split('=')[-1]}&t={round(time_stamp/1000)}s.jpg"), frame)

                # Увеличиваем время на interval_sec секунд для следующего кадра
                time_stamp += interval_sec * 1000

            else:
                break

        cap.release()
        cv2.destroyAllWindows()



def compute_distances(query_code, image_codes, method='cosine'):
    """
    Вычисляет меру близости между опорным кодом и массивом кодов изображений.

    Параметры:
    - query_code: Опорный код (размером 512).
    - image_codes: Массив кодов изображений (форма: (количество изображений, 512)).
    - method: Метод вычисления близости ('cosine' или 'euclidean').

    Возвращает:
    - Массив близости (форма: (количество изображений)).
    """

    if method == 'cosine':
        # Вычислите косинусное сходство между опорным кодом и всеми кодами изображений
        similarities = cosine_similarity(query_code.reshape(1, -1), image_codes)
        # Преобразуйте в косинусное расстояние (чем меньше, тем лучше)
        distances = 1 - similarities

    elif method == 'euclidean':
        # Вычислите евклидово расстояние между опорным кодом и всеми кодами изображений
        distances = euclidean_distances(query_code.reshape(1, -1), image_codes)
        
    return distances[0]


def display_top_k_images(list_of_files, distances, k=5):
    """
    Отображает top K фотографий с наименьшими расстояниями и выводит список ссылок на видео этих моментов

    Параметры:
    - list_of_files: Список путей к файлам изображений.
    - distances: Массив близости - расстояний (форма: (количество изображений)).
    - k: Количество фотографий для отображения (по умолчанию 5).
    """

    # Получите индексы top K фотографий с наименьшими расстояниями
    top_k_indices = distances.argsort(axis=0)
    #print(top_k_indices)
    image_path_list = []

    j = 0 # номер графика
    i = 0 # номер фотки в сортированном списке

    images_out = []

    while j<k:
        # Загрузите изображение и отобразите его на соответствующем субграфике
        image_path = list_of_files[top_k_indices[i]]
        
        # Проверяем, является ли изображение дубликатом, проверяя разницу между соседними значениями
        if i > 0 and abs(distances[top_k_indices[i]] - distances[top_k_indices[i - 1]]) < 0.00025:
            i += 1
            continue

        image_path_list.append(image_path)
        image = Image.open(image_path)
        images_out.append(image)

        j += 1
        i += 1

    return images_out, ['https://www.youtube.com/watch?v=' + image_path.split('_')[-1][:-4] for image_path in image_path_list]