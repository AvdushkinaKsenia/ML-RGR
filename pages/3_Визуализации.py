import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from utils.data_loader import load_data

# Розовая цветовая палитра
PINK_PALETTE = ["#FFC8DB", "#FF9EBD", "#FF85B3", "#FF6BA9", "#FF519F", "#FF3896"]
PINK_CMAP = sns.light_palette("#FF85B3", as_cmap=True)
BACKGROUND_COLOR = "#FFEAF1"

def setup_pink_theme():
    """Настройка розовой темы для графиков"""
    plt.rcParams['figure.facecolor'] = BACKGROUND_COLOR
    plt.rcParams['axes.facecolor'] = BACKGROUND_COLOR
    plt.rcParams['axes.edgecolor'] = "#FF9EBD"
    plt.rcParams['axes.labelcolor'] = "#000000"
    plt.rcParams['xtick.color'] = "#000000"
    plt.rcParams['ytick.color'] = "#000000"
    plt.rcParams['grid.color'] = "#FFC8DB"

def show():
    st.title("Визуализации зависимостей в наборе данных о вине")
    wine = load_data()
    setup_pink_theme()
    
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "🍷 Распределение качества", 
        "📊 Корреляции", 
        "🔍 Сравнение типов", 
        "🍸 Алкоголь vs Качество", 
        "🧪 Кислотность"
    ])
    
    with tab1:
        st.subheader("1. Распределение качества вин")
        fig1, ax1 = plt.subplots(figsize=(10, 6))
        sns.countplot(data=wine, x='quality', hue='type', 
                     palette=PINK_PALETTE[:2], ax=ax1)
        ax1.set_xlabel("Оценка качества", fontsize=12)
        ax1.set_ylabel("Количество", fontsize=12)
        ax1.set_facecolor(BACKGROUND_COLOR)
        plt.title("Распределение качества по типам вин", 
                 fontsize=14, pad=20, color="#FF519F")
        st.pyplot(fig1)
    
    with tab2:
        st.subheader("2. Корреляция признаков")
        numeric_cols = wine.select_dtypes(include='number').columns
        fig2, ax2 = plt.subplots(figsize=(12, 8))
        sns.heatmap(wine[numeric_cols].corr(), annot=True, 
                   cmap=PINK_CMAP, center=0, ax=ax2,
                   linewidths=0.5, linecolor="#FF9EBD")
        ax2.set_facecolor(BACKGROUND_COLOR)
        plt.title("Матрица корреляций", 
                 fontsize=14, pad=20, color="#FF519F")
        st.pyplot(fig2)
    
    with tab3:
        st.subheader("3. Сравнение красных и белых вин")
        fig3, axes = plt.subplots(1, 2, figsize=(14, 6))
        
        # Алкоголь
        sns.boxplot(data=wine, x='type', y='alcohol', 
                   palette=PINK_PALETTE[:2], ax=axes[0])
        axes[0].set_title("Содержание алкоголя", 
                         fontsize=12, color="#FF519F")
        axes[0].set_facecolor(BACKGROUND_COLOR)
        
        # Качество
        sns.boxplot(data=wine, x='type', y='quality', 
                   palette=PINK_PALETTE[2:4], ax=axes[1])
        axes[1].set_title("Оценка качества", 
                         fontsize=12, color="#FF519F")
        axes[1].set_facecolor(BACKGROUND_COLOR)
        
        fig3.patch.set_facecolor(BACKGROUND_COLOR)
        st.pyplot(fig3)
    
    with tab4:
        st.subheader("4. Зависимость качества от алкоголя")
        fig4, ax4 = plt.subplots(figsize=(10, 6))
        sns.regplot(data=wine, x='alcohol', y='quality', 
                   scatter_kws={'alpha':0.5, 'color': '#FF85B3'}, 
                   line_kws={'color': '#FF519F', 'linewidth': 3},
                   ax=ax4)
        ax4.set_xlabel("Алкоголь (% об.)", fontsize=12)
        ax4.set_ylabel("Оценка качества", fontsize=12)
        ax4.set_facecolor(BACKGROUND_COLOR)
        plt.title("Влияние алкоголя на качество", 
                 fontsize=14, pad=20, color="#FF519F")
        st.pyplot(fig4)
    
    with tab5:
        st.subheader("5. Распределение кислотности")
        fig5, axes = plt.subplots(1, 3, figsize=(16, 5))
        
        # Фиксированная кислотность
        sns.histplot(wine['fixed acidity'], bins=30, kde=True, 
                    color=PINK_PALETTE[0], ax=axes[0])
        axes[0].set_title("Фиксированная кислотность", 
                         fontsize=12, color="#FF519F")
        axes[0].set_facecolor(BACKGROUND_COLOR)
        
        # Летучая кислотность
        sns.histplot(wine['volatile acidity'], bins=30, kde=True, 
                    color=PINK_PALETTE[2], ax=axes[1])
        axes[1].set_title("Летучая кислотность", 
                         fontsize=12, color="#FF519F")
        axes[1].set_facecolor(BACKGROUND_COLOR)
        
        # Лимонная кислота
        sns.histplot(wine['citric acid'], bins=30, kde=True, 
                    color=PINK_PALETTE[4], ax=axes[2])
        axes[2].set_title("Лимонная кислота", 
                         fontsize=12, color="#FF519F")
        axes[2].set_facecolor(BACKGROUND_COLOR)
        
        fig5.patch.set_facecolor(BACKGROUND_COLOR)
        st.pyplot(fig5)

show()