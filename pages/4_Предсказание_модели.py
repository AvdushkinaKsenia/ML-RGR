import streamlit as st
import pandas as pd
import numpy as np
from utils.data_loader import load_data
from utils.model_loader import load_ml_models, predict_quality

# Загрузка моделей один раз при старте
@st.cache_resource
def load_models():
    return load_ml_models()

def show_model_info(model_name):
    """Отображение информации о модели"""
    info = {
        'polynomial_reg': {
            'name': 'Полиномиальная регрессия',
            'description': 'Классическая модель с полиномиальными признаками 2-й степени'
        },
        'boosting': {
            'name': 'Градиентный бустинг',
            'description': 'Ансамблевая модель XGBoost с 174 деревьями'
        },
        'catboost': {
            'name': 'CatBoost',
            'description': 'Продвинутый градиентный бустинг от Yandex'
        },
        'bagging': {
            'name': 'Бэггинг',
            'description': 'Ансамбль из 116 случайных лесов'
        },
        'stacking': {
            'name': 'Стэкинг',
            'description': 'Двухуровневая модель с 3 алгоритмами: Градиентный бустинг, catboost и Бэггинг'
        },
        'neural_network': {
            'name': 'Нейронная сеть',
            'description': 'Полносвязная сеть с 3 скрытыми слоями по 64 нейрона'
        }
    }

    st.sidebar.markdown(f"### {info[model_name]['name']}")
    st.sidebar.markdown(info[model_name]['description'])

def show():
    st.title("Предсказание качества вина")
    wine = load_data()
    models = load_models()

    # Выбор модели в сайдбаре
    model_options = {
        'polynomial_reg': '1. Полиномиальная регрессия',
        'boosting': '2. Градиентный бустинг',
        'catboost': '3. CatBoost',
        'bagging': '4. Бэггинг',
        'stacking': '5. Стэкинг',
        'neural_network': '6. Нейронная сеть'
    }

    selected_model = st.sidebar.selectbox(
        "Выберите модель",
        list(model_options.keys()),
        format_func=lambda x: model_options[x]
    )

    # Показать информацию о выбранной модели
    show_model_info(selected_model)

    st.markdown("""
    ### Инструкция
    1. Выберите модель в меню слева
    2. Введите все параметры вина вручную или загрузите файл
    """)

    tab1, tab2 = st.tabs(["Ручной ввод", "Пакетное предсказание"])

    with tab1:
        st.subheader("Ручной ввод всех параметров вина")

        with st.form("wine_params"):
            col1, col2, col3 = st.columns(3)

            with col1:
                fixed_acidity = st.number_input("Фиксированная кислотность (г/л)",
                                             min_value=4.0, max_value=16.0, value=7.0, step=0.1)
                volatile_acidity = st.number_input("Летучая кислотность (г/л)",
                                                min_value=0.1, max_value=1.5, value=0.5, step=0.01)
                citric_acid = st.number_input("Лимонная кислота (г/л)",
                                           min_value=0.0, max_value=1.5, value=0.3, step=0.01)
                residual_sugar = st.number_input("Остаточный сахар (г/л)",
                                              min_value=0.5, max_value=65.0, value=5.0, step=0.1)

            with col2:
                chlorides = st.number_input("Хлориды (г/л)",
                                         min_value=0.01, max_value=0.5, value=0.08, step=0.001)
                free_sulfur_dioxide = st.number_input("Свободный SO₂ (мг/л)",
                                                   min_value=1, max_value=300, value=30, step=1)
                total_sulfur_dioxide = st.number_input("Общий SO₂ (мг/л)",
                                                    min_value=5, max_value=400, value=100, step=1)
                non_free_sulfur_dioxide = st.number_input("Non-free SO₂ (мг/л)",
                                                    min_value=5, max_value=400, value=100, step=1)

            with col3:
                density = st.number_input("Плотность (г/см³)",
                                       min_value=0.98, max_value=1.01, value=0.995, step=0.0001)
                ph = st.number_input("Уровень pH",
                                   min_value=2.7, max_value=4.0, value=3.2, step=0.01)
                sulphates = st.number_input("Сульфаты (г/л)",
                                         min_value=0.3, max_value=2.0, value=0.5, step=0.01)
                alcohol = st.number_input("Алкоголь (% об.)",
                                       min_value=5.0, max_value=20.0, value=10.0, step=0.1)
                wine_type = st.selectbox("Тип вина", ['red', 'white'])

            submitted = st.form_submit_button("Предсказать качество")

            if submitted:
                input_data = {
                    'fixed acidity': fixed_acidity,
                    'volatile acidity': volatile_acidity,
                    'citric acid': citric_acid,
                    'residual sugar': residual_sugar,
                    'chlorides': chlorides,
                    'free sulfur dioxide': free_sulfur_dioxide,
                    'non-free sulfur dioxide': non_free_sulfur_dioxide,
                    'total sulfur dioxide': total_sulfur_dioxide,
                    'density': density,
                    'pH': ph,
                    'sulphates': sulphates,
                    'alcohol': alcohol,
                    'type': 0 if wine_type == 'red' else 1
                }

                
                quality = predict_quality(models, selected_model, input_data)

                # Отображение результатов
                st.success(f"Предсказанная оценка качества: {quality}/10")

                # Детализация результатов
                with st.expander("Показать детали предсказания"):
                    st.write("### Входные параметры")
                    st.json(input_data)

                    st.write("### Интерпретация результата")
                    if quality <= 4:
                        st.warning("Низкое качество")
                    elif quality <= 6:
                        st.info("Среднее качество")
                    else:
                        st.success("Высокое качество")

                if quality >= 7:
                    st.balloons()


    with tab2:
        st.subheader("Пакетное предсказание из файла")

        uploaded_file = st.file_uploader("Загрузите CSV файл с параметрами вин", type="csv")

        if uploaded_file is not None:
            try:
                df = pd.read_csv(uploaded_file)
                st.write("Первые 5 строк файла:")
                st.write(df.head())

                required_cols = {
                    'fixed acidity', 'volatile acidity', 'citric acid',
                    'residual sugar', 'chlorides', 'free sulfur dioxide',
                    'total sulfur dioxide', 'non-free sulfur dioxide', 'density', 'pH', 'sulphates',
                    'alcohol', 'type'
                }

                if not required_cols.issubset(df.columns):
                    missing_cols = required_cols - set(df.columns)
                    st.error(f"Отсутствуют обязательные колонки: {missing_cols}")
                else:
                    with st.spinner("Выполняется предсказание..."):
                        # Преобразуем тип вина если нужно
                        if df['type'].dtype == 'object':
                            df['type'] = df['type'].apply(lambda x: 0 if x == 'red' else 1)

                        # Делаем предсказания
                        predictions = []
                        for _, row in df.iterrows():
                            try:
                                pred = predict_quality(models, selected_model, row.to_dict())
                                predictions.append(pred)
                            except Exception as e:
                                predictions.append(None)
                                st.warning(f"Ошибка предсказания для строки {_}: {str(e)}")

                        df['predicted_quality'] = predictions

                    st.success("Предсказания выполнены!")

                    # Показать результаты
                    st.write("Результаты предсказаний:")
                    st.dataframe(df.head())

                    # Визуализация
                    col1, col2 = st.columns(2)

                    with col1:
                        st.subheader("Распределение предсказанных оценок")
                        st.bar_chart(df['predicted_quality'].value_counts().sort_index())

                    with col2:
                        st.subheader("Среднее качество по типам")
                        if 'type' in df:
                            avg_quality = df.groupby('type')['predicted_quality'].mean()
                            avg_quality.index = avg_quality.index.map({0: 'Красное', 1: 'Белое'})
                            st.bar_chart(avg_quality)

                    # Кнопка скачивания
                    csv = df.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        "Скачать результаты",
                        csv,
                        f"wine_predictions_{selected_model}.csv",
                        "text/csv",
                        help="Скачать CSV файл с результатами предсказаний"
                    )

            except Exception as e:
                st.error(f"Ошибка при обработке файла: {e}")

show()
