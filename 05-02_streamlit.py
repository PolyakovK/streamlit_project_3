import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import io
from skimage import io as skio
from skimage.color import rgb2gray

st.title('Сжимаем изображение с помощью SVD')
st.subheader('Загрузи свое изображение, используя варианты в Sidebar')

uploaded_file = st.sidebar.file_uploader("Загрузи изображение")
image_URL = st.sidebar.text_input("Или вставь ссылку")

top_k = st.sidebar.slider("Выбери количество компонентов", 1, 500, 100)

def process_image(image):
    # Преобразуем изображение в оттенки серого, если оно цветное
    if image.ndim == 3:
        image = rgb2gray(image)
    return image

def svd_compression(image, top_k):
    U, sing_values, Vt = np.linalg.svd(image, full_matrices=False)
    sigma = np.diag(sing_values[:top_k])
    truncated_image = U[:, :top_k] @ sigma @ Vt[:top_k, :]
    return truncated_image

if uploaded_file is not None:
    image_bytes = uploaded_file.read()
    image = skio.imread(io.BytesIO(image_bytes))
    image = process_image(image)

    truncated_image = svd_compression(image, top_k)

    fig, ax = plt.subplots(1, 2, figsize=(15, 10))
    ax[0].imshow(image, cmap='gray')
    ax[0].set_title("Исходное изображение")
    ax[1].imshow(truncated_image, cmap='gray')
    ax[1].set_title('Сжатое изображение')
    st.pyplot(fig)

elif image_URL:
    try:
        image = skio.imread(image_URL)
        image = process_image(image)

        truncated_image = svd_compression(image, top_k)

        fig, ax = plt.subplots(1, 2, figsize=(15, 10))
        ax[0].imshow(image, cmap='gray')
        ax[0].set_title("Исходное изображение")
        ax[1].imshow(truncated_image, cmap='gray')
        ax[1].set_title('Сжатое изображение')
        st.pyplot(fig)

    except Exception as e:
        st.error(f"Ошибка загрузки изображения: {e}")

else:
    st.warning('Загрузи изображение или укажи ссылку, чтобы продолжить.')
