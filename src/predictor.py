"""
Módulo de Inferencia y Servicio de Predicciones de Machine Learning.
Proyecto: TalentPredict AI
"""

from pathlib import Path
from typing import Dict, Any, Tuple
import pandas as pd
import joblib

from src.data_processor import get_feature_columns
from src.database import save_prediction

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "models" / "talent_predict_rf.joblib"
MODEL_VERSION = "RandomForest-v1.0-Balanced"

# Variable global cacheada del modelo
_PIPELINE = None

def get_model():
    """
    Carga el pipeline entrenado desde disco (patrón singleton / caché en memoria).
    Si el archivo no existe (por ejemplo, en un despliegue limpio donde .joblib se ignoró en git),
    lo entrena y genera automáticamente.
    """
    global _PIPELINE
    if _PIPELINE is None:
        if not MODEL_PATH.exists():
            print("⚠️ Modelo no encontrado en disco. Entrenando automáticamente...")
            from src.model_trainer import train_and_save_model
            train_and_save_model()
        _PIPELINE = joblib.load(MODEL_PATH)
    return _PIPELINE

def determine_risk_level(prob_attrition: float) -> Tuple[str, str, str]:
    """
    Determina el nivel de riesgo de deserción y el color del semáforo.
    Basado en el análisis de umbrales para tasa base del 16.12%:
    - BAJO (< 20%)
    - MODERADO (20% - 35%)
    - ALTO (>= 35%)
    """
    if prob_attrition >= 0.35:
        return "ALTO", "danger", "Alta probabilidad de rotación laboral. Requiere plan de retención inmediato."
    elif prob_attrition >= 0.20:
        return "MODERADO", "warning", "Riesgo intermedio. Conviene revisar factores de satisfacción y compensación."
    else:
        return "BAJO", "success", "Bajo riesgo de rotación. El colaborador presenta estabilidad laboral."

def generate_insights(data: Dict[str, Any], prob: float) -> list:
    """
    Genera explicaciones de factores clave que influyen en el resultado de este empleado.
    """
    insights = []
    
    if str(data.get("OverTime")).lower() == "yes":
        insights.append({"factor": "Horas Extras", "impact": "Negativo", "desc": "Trabaja horas extras con frecuencia, lo que aumenta la fatiga y probabilidad de renuncia."})
    
    job_sat = int(data.get("JobSatisfaction", 3))
    if job_sat <= 2:
        insights.append({"factor": "Satisfacción Laboral", "impact": "Negativo", "desc": f"Nivel de satisfacción bajo ({job_sat}/4), factor crítico en deserción."})
        
    env_sat = int(data.get("EnvironmentSatisfaction", 3))
    if env_sat <= 2:
        insights.append({"factor": "Clima Laboral", "impact": "Negativo", "desc": f"Insatisfacción con el ambiente de trabajo ({env_sat}/4)."})
        
    income = float(data.get("MonthlyIncome", 6000))
    if income < 3000:
        insights.append({"factor": "Ingreso Mensual", "impact": "Negativo", "desc": f"Sueldo (${income:,.0f}) en el quintil inferior del mercado interno."})
        
    years_role = int(data.get("YearsInCurrentRole", 2))
    years_promo = int(data.get("YearsSinceLastPromotion", 1))
    if years_promo >= 4 and years_role >= 4:
        insights.append({"factor": "Estancamiento Profesional", "impact": "Negativo", "desc": f"{years_promo} años sin ascenso en el mismo puesto."})
        
    stock = int(data.get("StockOptionLevel", 0))
    if stock == 0:
        insights.append({"factor": "Sin Opciones de Acciones", "impact": "Negativo", "desc": "Nivel 0 de stock options; menor anclaje financiero con la empresa."})
        
    if not insights:
        insights.append({"factor": "Condiciones Generales", "impact": "Positivo", "desc": "El colaborador mantiene métricas balanceadas de estabilidad, compensación y satisfacción."})
        
    return insights

def predict_employee_attrition(employee_data: Dict[str, Any], persist_db: bool = True) -> Dict[str, Any]:
    """
    Recibe los datos del empleado, valida, ejecuta inferencia real y guarda en Supabase Gold.
    """
    pipeline = get_model()
    num_cols, cat_cols = get_feature_columns()
    all_cols = num_cols + cat_cols
    
    # Construir DataFrame con las 30 columnas esperadas
    row_dict = {}
    for col in num_cols:
        val = employee_data.get(col)
        if val is None or val == "":
            val = 0
        row_dict[col] = [float(val)]
        
    for col in cat_cols:
        val = employee_data.get(col)
        if val is None or val == "":
            val = "Unknown"
        row_dict[col] = [str(val)]
        
    df_sample = pd.DataFrame(row_dict)[all_cols]
    
    # Inferencia con Pipeline
    prob_yes = float(pipeline.predict_proba(df_sample)[0, 1])
    prediction_label = "Yes" if prob_yes >= 0.35 else "No"
    
    risk_level, badge_color, risk_text = determine_risk_level(prob_yes)
    insights = generate_insights(employee_data, prob_yes)
    
    emp_num = employee_data.get("EmployeeNumber")
    if emp_num is not None:
        try:
            emp_num = int(emp_num)
        except (ValueError, TypeError):
            emp_num = None
            
    saved_id = None
    if persist_db:
        try:
            saved_id = save_prediction(
                employee_number=emp_num,
                prediction=prediction_label,
                probability=round(prob_yes, 4),
                risk_level=risk_level,
                model_version=MODEL_VERSION,
                input_features=employee_data
            )
        except Exception as e:
            print(f"⚠️ Advertencia al persistir en Supabase: {e}")
            
    return {
        "prediction_id": saved_id,
        "employee_number": emp_num,
        "prediction": prediction_label,
        "probability_yes": round(prob_yes, 4),
        "probability_pct": round(prob_yes * 100, 1),
        "risk_level": risk_level,
        "badge_color": badge_color,
        "risk_description": risk_text,
        "model_version": MODEL_VERSION,
        "insights": insights
    }
