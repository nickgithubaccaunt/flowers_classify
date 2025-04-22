import streamlit as st
import requests
from PIL import Image
import matplotlib.pyplot as plt
from io import BytesIO
from streamlit_drawable_canvas import st_canvas

# URL до вашего API
API_URL = "https://classify-flowers.onrender.com/predict/"

# Имена классов цветов
CLASS_NAMES = {
    "Lilly": "🌼 Лилия",
    "Lotus": "🌼 Лотус",
    "Orchid": "🌹 Орхидея",
    "Sunflower": "🌻 Подсолнух",
    "Tulip": "🌷 Тюльпан"
}

# Настройки страницы
st.set_page_config(page_title="Классификация цветов", page_icon="🌸")

st.title("🌸 Классификация изображений цветов")

# Выбор режима ввода
tab = st.radio(
    "Выберите режим:",
    ["📷 Загрузить изображение", "✏️ Нарисовать изображение"],
    horizontal=True
)

image = None

if tab == "📷 Загрузить изображение":
    uploaded_file = st.file_uploader(
        "Загрузите изображение цветка",
        type=["jpg", "jpeg", "png"],
        help="Поддерживаются изображения в форматах JPG, JPEG и PNG"
    )
    if uploaded_file:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Загруженное изображение", use_column_width=True)

elif tab == "✏️ Нарисовать изображение":
    st.write("Нарисуйте цветок:")
    canvas_result = st_canvas(
        fill_color="#ffffff",
        stroke_width=15,
        stroke_color="#000000",
        background_color="#ffffff",
        width=300,
        height=300,
        drawing_mode="freedraw",
        key="canvas",
        update_streamlit=True
    )

    if canvas_result.image_data is not None:
        image = Image.fromarray(canvas_result.image_data.astype("uint8")).convert("RGB")
        st.image(image, caption="Ваш рисунок", use_column_width=True)

# Обработка изображения
if image and st.button("Классифицировать цветок", type="primary"):
    with st.spinner("Анализируем изображение..."):
        try:
            # Подготовка изображения
            img_resized = image.resize((128, 128))
            buffered = BytesIO()
            img_resized.save(buffered, format="JPEG")
            img_bytes = buffered.getvalue()

            # Отправка на API
            files = {"file": ("flower.jpg", img_bytes, "image/jpeg")}
            response = requests.post(API_URL, files=files)

            if response.status_code == 200:
                result = response.json()

                # Отображение результатов
                predicted_class = result.get("predicted_class", "unknown")
                human_readable = CLASS_NAMES.get(predicted_class, predicted_class)

                st.success(f"**Результат:** {human_readable}")

                # Визуализация вероятностей
                probs = result.get("probabilities", {})
                if probs:
                    st.subheader("Вероятности по классам:")

                    # Сортируем по убыванию вероятности
                    sorted_probs = dict(sorted(probs.items(), key=lambda item: item[1], reverse=True))

                    fig, ax = plt.subplots(figsize=(10, 5))
                    colors = ['#ff9999', '#66b3ff', '#99ff99', '#ffcc99', '#c2c2f0']
                    bars = ax.bar(
                        [CLASS_NAMES.get(k, k) for k in sorted_probs.keys()],
                        sorted_probs.values(),
                        color=colors
                    )

                    # Добавляем значения на столбцы
                    for bar in bars:
                        height = bar.get_height()
                        ax.text(bar.get_x() + bar.get_width() / 2., height,
                                f'{height:.2f}',
                                ha='center', va='bottom')

                    ax.set_ylim(0, 1.1)
                    plt.xticks(rotation=45)
                    st.pyplot(fig)

            else:
                st.error("Ошибка при обращении к API")
                st.text(response.text)

        except Exception as e:
            st.error(f"Произошла ошибка: {str(e)}")

# Информация о приложении
st.markdown("---")
st.markdown("""
**О приложении:**
- Использует нейросетевую модель для классификации цветов
- Для работы требуется доступ к API: `flower-classification-api.onrender.com`
""")