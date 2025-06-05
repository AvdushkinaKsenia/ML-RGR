import streamlit as st
from PIL import Image

def show():
    st.title("Информация о разработчике моделей ML")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        image = Image.open("assets/developer_photo.jpg")
        st.image(image, caption="Фото разработчика", width=200)
    
    with col2:
        st.write("**ФИО:** Авдюшкина Ксения Сергеевна")
        st.write("**Номер учебной группы:** ФИТ-231")
        st.write("**Тема РГР:** Разработка Web-приложения (дашборда) для инференса (вывода) моделей ML и анализа данных")

# Вызов функции отображения
show()