import pandas as pd
import streamlit as st

@st.cache_data
def load_data():
    wine = pd.read_csv('data/winequality-combined.csv')
    wine['type'] = wine['type'].map({1: 'red', 0: 'white'})
    return wine

def show_data_preview(data):
    st.write("### Предпросмотр данных")
    st.write(data.head())
    
    st.write("### Основные статистики")
    st.write(data.describe())
