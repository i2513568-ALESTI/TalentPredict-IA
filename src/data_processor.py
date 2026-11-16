"""
Módulo de procesamiento y preparación de datos para TalentPredict AI.
Responsable de:
- Carga de datos (CSV / Supabase)
- Identificación de tipos de variables
- Filtrado y exclusión de constantes / identificadores
- Separación estratificada de entrenamiento y prueba (anti-data leakage)
"""

import os
from pathlib import Path
from typing import Tuple, List, Dict
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from dotenv import load_dotenv
from sqlalchemy import create_engine

BASE_DIR = Path(__file__).resolve().parent.parent
load_dotenv(BASE_DIR / ".env")

# Variables excluidas por ser constantes o identificadores sin valor predictivo
EXCLUDED_COLUMNS = ['EmployeeCount', 'Over18', 'StandardHours', 'EmployeeNumber']
TARGET_COLUMN = 'Attrition'

def get_feature_columns() -> Tuple[List[str], List[str]]:
    """
    Retorna las listas de variables predictoras numéricas y categóricas.
    """
    numeric_features = [
        'Age', 'DailyRate', 'DistanceFromHome', 'Education',
        'EnvironmentSatisfaction', 'HourlyRate', 'JobInvolvement',
        'JobLevel', 'JobSatisfaction', 'MonthlyIncome', 'MonthlyRate',
        'NumCompaniesWorked', 'PercentSalaryHike', 'PerformanceRating',
        'RelationshipSatisfaction', 'StockOptionLevel', 'TotalWorkingYears',
        'TrainingTimesLastYear', 'WorkLifeBalance', 'YearsAtCompany',
        'YearsInCurrentRole', 'YearsSinceLastPromotion', 'YearsWithCurrManager'
    ]
    
    categorical_features = [
        'BusinessTravel', 'Department', 'EducationField',
        'Gender', 'JobRole', 'MaritalStatus', 'OverTime'
    ]
    
    return numeric_features, categorical_features

def load_data_from_csv(csv_path: str = None) -> pd.DataFrame:
    """
    Carga el dataset de Kaggle desde el archivo CSV local.
    """
    if csv_path is None:
        csv_path = BASE_DIR / "data" / "WA_Fn-UseC_-HR-Employee-Attrition.csv"
    
    df = pd.read_csv(csv_path)
    return df

def load_data_from_supabase() -> pd.DataFrame:
    """
    Carga los datos directamente desde la base de datos Supabase PostgreSQL.
    """
    db_host = os.getenv("DB_HOST")
    db_port = os.getenv("DB_PORT", "5432")
    db_name = os.getenv("DB_NAME", "postgres")
    db_user = os.getenv("DB_USER")
    db_password = os.getenv("DB_PASSWORD")
    db_sslmode = os.getenv("DB_SSLMODE", "require")
    
    url = f"postgresql+psycopg2://{db_user}:{db_password}@{db_host}:{db_port}/{db_name}?sslmode={db_sslmode}"
    engine = create_engine(url)
    
    # Extraer los datos desde la columna JSONB 'data' para reconstruir el DataFrame completo
    query = "SELECT data FROM employees ORDER BY employee_number ASC;"
    with engine.connect() as conn:
        records = conn.execute(query).fetchall()
        data_list = [r[0] for r in records]
        
    return pd.DataFrame(data_list)

def analyze_dataset(df: pd.DataFrame) -> Dict:
    """
    Genera un análisis estadístico y de distribución del dataset.
    """
    numeric_cols, categorical_cols = get_feature_columns()
    
    attrition_counts = df[TARGET_COLUMN].value_counts().to_dict()
    attrition_rates = df[TARGET_COLUMN].value_counts(normalize=True).to_dict()
    
    analysis = {
        "total_records": len(df),
        "total_features": len(numeric_cols) + len(categorical_cols),
        "numeric_features_count": len(numeric_cols),
        "categorical_features_count": len(categorical_cols),
        "null_values_total": int(df.isnull().sum().sum()),
        "duplicates_count": int(df.duplicated(subset=['EmployeeNumber']).sum()),
        "attrition_distribution": {
            "counts": attrition_counts,
            "percentages": {k: round(v * 100, 2) for k, v in attrition_rates.items()}
        },
        "excluded_columns": EXCLUDED_COLUMNS
    }
    return analysis

def prepare_train_test_data(
    df: pd.DataFrame,
    test_size: float = 0.2,
    random_state: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Prepara la matriz X y el vector y, realizando la partición estratificada.
    Garantiza que no haya fuga de información (data leakage).
    """
    numeric_cols, categorical_cols = get_feature_columns()
    all_features = numeric_cols + categorical_cols
    
    # Separación estricta de X (solo features autorizadas) e y (target)
    X = df[all_features].copy()
    
    # Mapeo binario de Attrition: Yes = 1, No = 0
    y = df[TARGET_COLUMN].map({'Yes': 1, 'No': 0}).astype(int)
    
    # División estratificada según la clase objetivo
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=y
    )
    
    return X_train, X_test, y_train, y_test
