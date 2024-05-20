import os
import glob
import sys
import shutil

import streamlit as st
import torch
from PIL import Image
from tqdm import tqdm
import numpy as np

from func import *

if not sys.warnoptions:
    import warnings
    warnings.simplefilter("ignore")


@st.cache_resource
def upload_models():
    # Кэшируем все модели и обработчики CLIP
    from transformers import CLIPTokenizerFast, CLIPProcessor, CLIPModel 

    print('Начался процесс подгрузки модели:')
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(device)

    model_id = 'openai/clip-vit-base-patch32'
    model = CLIPModel.from_pretrained(model_id).to(device) 
    tokenizer = CLIPTokenizerFast.from_pretrained(model_id) 
    processor = CLIPProcessor.from_pretrained(model_id)
    return model, tokenizer, processor, device


def set_page_static_info():
    st.set_page_config(
        page_title="YouTube-searcher",
        page_icon="configs/logo.png",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    st.title("Поисковик по ютуб видео")


def make_images_and_embedding(video_urls, seconds_step=10):
    # Удаляем папку images, если она уже существует
    if os.path.exists('images'):
        shutil.rmtree('images')

    # Создаем папку images
    os.makedirs('images')

    for video_url in video_urls:
        extract_frames(video_url, seconds_step)

    image_folder = 'images'

    # Пустой массив для хранения всех эмбеддингов изображений
    image_embeddings = []
    list_of_files = []

    print('Начался процесс получения эмбеддингов:')

    # Пройдитесь по всем файлам изображений в папке
    for filename in tqdm(os.listdir(image_folder)):
        if filename.endswith('.jpg') or filename.endswith('.png') or filename.endswith('.JPEG') :

            # Загрузите изображение с использованием PIL
            image = Image.open(os.path.join(image_folder, filename))
            list_of_files.append(os.path.join(image_folder, filename))

            # Получите батч изображений и выполните преобразования
            inputs = processor(text=None, images=image, return_tensors="pt")
            pixel_values = inputs["pixel_values"].to(device)

            # Получите эмбеддинг изображения
            image_features = model.get_image_features(pixel_values=pixel_values)
            image_features = image_features.squeeze(0)
            image_features = image_features.cpu().detach().numpy()

            # Добавьте эмбеддинг 
            image_embeddings.append(image_features)


        # Преобразуйте список эмбеддингов в массив NumPy
        image_arr = np.vstack(image_embeddings)

        # Сохраняем массив NumPy в файл
        np.save('image_embeddings.npy', image_arr)


def compute_k_nearest_imaget_to_text_prompt(text_imput, top_k):
    prompt = "a photo of " + text_imput
    # tokenize the prompt
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    text_emb = model.get_text_features(**inputs) 
    # Получение латентного кода текстового описания
    query_code = text_emb.squeeze(0).cpu().detach().numpy()

    # Загружаем массив NumPy из файла (в нем латентные коды всех фоток из папки)
    image_arr = np.load('image_embeddings.npy')

    # Получение мер близости
    distances_cosine = compute_distances(query_code, image_arr, method='cosine')

    # Получаем список всех файлов изображений в папке images
    list_of_files = glob.glob('images/*.jpg')

    # Пример использования функции
    images_out, list_of_links = display_top_k_images(list_of_files, distances_cosine, k=top_k)

    return images_out, list_of_links


def compute_k_nearest_imaget_to_image_prompt(image_array, top_k):
    image = processor(text=None, images=image_array, return_tensors="pt")['pixel_values'].to(device) 
    image_features = model.get_image_features(pixel_values=image)

    # Получение латентного кода 
    query_code = image_features.squeeze(0).cpu().detach().numpy()

    # Загружаем массив NumPy из файла (в нем латентные коды всех фоток из папки)
    image_arr = np.load('image_embeddings.npy')

    # Получение мер близости
    distances_cosine = compute_distances(query_code, image_arr, method='cosine')

    # Получаем список всех файлов изображений в папке images
    list_of_files = glob.glob('images/*.jpg')

    # Пример использования функции
    images_out, list_of_links = display_top_k_images(list_of_files, distances_cosine, k=top_k)

    return images_out, list_of_links


def main():
    st.sidebar.title("Загрузчик ютуб видео")
    text_input_url = st.sidebar.text_area("Введите URL-адреса видео (каждый youtube url на новой строке):")
    seconds_step = st.sidebar.number_input("Введите шаг нарезки в секундах:", min_value=1, value=10, step=1)

    if st.sidebar.button("Поделить видео на кадры"):
        video_urls = [url.strip() for url in text_input_url.split('\n') if url.strip()]
        with st.spinner('Обработка видео...'):
            if os.path.exists('image_embeddings.npy'):
                os.remove('image_embeddings.npy')
            print(video_urls)
            make_images_and_embedding(video_urls, seconds_step=seconds_step)
        st.success('Видео с ютуба загружены')

    st.sidebar.markdown('---')

    search_type = st.sidebar.radio("Выберите тип поиска:", ("По тексту", "По изображению"))
    
    if search_type == "По тексту":
        text_input = st.text_input("Введите тектовый запрос на поиск")
    else:
        image_file = st.file_uploader("Загрузите изображение для поиска схожих", type=["jpg", "png", "jpeg"])
    
    top_k = st.number_input("Введите размер top k найденных моментов:", min_value=1, value=5, step=1)

    if st.button(":red[Найти по тексту момент из видео]"):
        if os.path.exists('images') and os.path.exists('image_embeddings.npy'):
            if search_type == "По тексту":
                images_out, list_of_links = compute_k_nearest_imaget_to_text_prompt(text_input, top_k)
                for image, link in zip(images_out, list_of_links):
                    col_first, col_second, _  = st.columns(3)
                    with col_first:
                        st.image(image)
                    with col_second:
                        st.markdown(f'<a href="{link}" target="_blank">{link}</a>', unsafe_allow_html=True)
            else:
                if image_file is not None:
                    image = Image.open(image_file)
                    image_array = np.array(image.convert('RGB'))
                    images_out, list_of_links = compute_k_nearest_imaget_to_image_prompt(image_array, top_k)
                    for image, link in zip(images_out, list_of_links):
                        col_first, col_second, _  = st.columns(3)
                        with col_first:
                            st.image(image)
                        with col_second:
                            st.markdown(f'<a href="{link}" target="_blank">{link}</a>', unsafe_allow_html=True)
                else: 
                    st.error('Не удалось загрузить изображение')
        else:
            st.error('Нет доступных кадров для анализа из видео. Сначала произведите загрузку видеороликов.')



if __name__ == "__main__":
    set_page_static_info()
    model, tokenizer, processor, device = upload_models()
    main()
