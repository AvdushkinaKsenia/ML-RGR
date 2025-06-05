import joblib
import numpy as np
import pandas as pd
from catboost import CatBoostRegressor

def load_ml_models():
    """Загрузка всех моделей"""
    models = {
        'polynomial_reg': joblib.load('models/polynomial_reg.pkl'),
        'boosting': joblib.load('models/gradient_boosting_reg.pkl'),
        'catboost': CatBoostRegressor().load_model('models/catboost_reg.cbm'),
        'bagging': joblib.load('models/bagging_reg.pkl'),
        'stacking': joblib.load('models/stacking_reg.pkl'),
        'neural_network': joblib.load('models/mlp_reg.pkl')
    }
    return models

def prepare_features(input_data, model_type):
    """Подготовка фичей для разных моделей"""
    # Базовые преобразования
    features = pd.DataFrame([input_data])
    
    return features

def predict_quality(models, model_type, input_data):
    """Предсказание с использованием выбранной модели"""
    features = prepare_features(input_data, model_type)
    model = models[model_type]
    
    prediction = model.predict(features)[0]
    
    return np.clip(round(prediction), 0, 10)