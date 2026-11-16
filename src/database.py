"""
Módulo de conexión y repositorio de datos para Supabase PostgreSQL.
Arquitectura Medallion:
- Bronze: Acceso a datos crudos (public.employees)
- Silver: Lectura de datos depurados (public.silver_employees)
- Gold: Almacenamiento y consulta de predicciones (public.gold_predictions) y KPIs
"""

import os
import json
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime, timezone
from dotenv import load_dotenv
from sqlalchemy import create_engine, text

BASE_DIR = Path(__file__).resolve().parent.parent
load_dotenv(BASE_DIR / ".env")

DB_HOST = os.getenv("DB_HOST")
DB_PORT = os.getenv("DB_PORT", "5432")
DB_NAME = os.getenv("DB_NAME", "postgres")
DB_USER = os.getenv("DB_USER")
DB_PASSWORD = os.getenv("DB_PASSWORD")
DB_SSLMODE = os.getenv("DB_SSLMODE", "require")

DATABASE_URL = f"postgresql+psycopg2://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}?sslmode={DB_SSLMODE}"

# Motor de conexión con pool configurado
engine = create_engine(
    DATABASE_URL,
    pool_size=5,
    max_overflow=10,
    pool_recycle=1800,
    pool_pre_ping=True
)

def get_kpis_summary() -> Dict[str, Any]:
    """
    Obtiene los KPIs globales del personal desde la capa Silver/Gold.
    """
    sql = text("""
        SELECT 
            COUNT(*) AS total_employees,
            SUM(attrition_binary) AS total_attrition,
            COUNT(*) - SUM(attrition_binary) AS total_retained,
            ROUND((AVG(attrition_binary) * 100)::NUMERIC, 2) AS attrition_rate_pct,
            ROUND(AVG(monthly_income)::NUMERIC, 2) AS avg_monthly_income,
            ROUND(AVG(age)::NUMERIC, 1) AS avg_age
        FROM public.silver_employees;
    """)
    with engine.connect() as conn:
        row = conn.execute(sql).fetchone()
        if row:
            m = row._mapping
            return {
                "total_employees": int(m["total_employees"] or 0),
                "total_attrition": int(m["total_attrition"] or 0),
                "total_retained": int(m["total_retained"] or 0),
                "attrition_rate_pct": float(m["attrition_rate_pct"] or 0.0),
                "avg_monthly_income": float(m["avg_monthly_income"] or 0.0),
                "avg_age": float(m["avg_age"] or 0.0)
            }
    return {}

def get_analytics_charts_data() -> Dict[str, Any]:
    """
    Extrae agregaciones analíticas para alimentar los gráficos de Chart.js en el dashboard:
    1. Rotación por departamento
    2. Rotación por puesto (JobRole)
    3. Rotación según horas extras (OverTime)
    4. Rotación según satisfacción laboral (JobSatisfaction)
    5. Distribución de edades
    6. Comparación de ingresos mensuales por departamento
    """
    charts = {}
    with engine.connect() as conn:
        # 1. Por departamento
        res_dept = conn.execute(text("""
            SELECT department, 
                   COUNT(*) AS total, 
                   SUM(attrition_binary) AS attrition_count,
                   ROUND((AVG(attrition_binary) * 100)::NUMERIC, 2) AS attrition_pct
            FROM public.silver_employees
            GROUP BY department
            ORDER BY total DESC;
        """)).fetchall()
        charts["by_department"] = [dict(r._mapping) for r in res_dept]

        # 2. Por JobRole
        res_role = conn.execute(text("""
            SELECT job_role, 
                   COUNT(*) AS total, 
                   SUM(attrition_binary) AS attrition_count,
                   ROUND((AVG(attrition_binary) * 100)::NUMERIC, 2) AS attrition_pct
            FROM public.silver_employees
            GROUP BY job_role
            ORDER BY attrition_pct DESC;
        """)).fetchall()
        charts["by_job_role"] = [dict(r._mapping) for r in res_role]

        # 3. Horas extras
        res_ot = conn.execute(text("""
            SELECT overtime, 
                   COUNT(*) AS total, 
                   SUM(attrition_binary) AS attrition_count,
                   ROUND((AVG(attrition_binary) * 100)::NUMERIC, 2) AS attrition_pct
            FROM public.silver_employees
            GROUP BY overtime;
        """)).fetchall()
        charts["by_overtime"] = [dict(r._mapping) for r in res_ot]

        # 4. Satisfacción laboral (1 a 4)
        res_sat = conn.execute(text("""
            SELECT job_satisfaction, 
                   COUNT(*) AS total, 
                   SUM(attrition_binary) AS attrition_count,
                   ROUND((AVG(attrition_binary) * 100)::NUMERIC, 2) AS attrition_pct
            FROM public.silver_employees
            GROUP BY job_satisfaction
            ORDER BY job_satisfaction ASC;
        """)).fetchall()
        charts["by_satisfaction"] = [dict(r._mapping) for r in res_sat]

        # 5. Distribución de edades (rangos)
        res_age = conn.execute(text("""
            SELECT 
                CASE 
                    WHEN age < 25 THEN '< 25'
                    WHEN age BETWEEN 25 AND 34 THEN '25 - 34'
                    WHEN age BETWEEN 35 AND 44 THEN '35 - 44'
                    WHEN age BETWEEN 45 AND 54 THEN '45 - 54'
                    ELSE '55+'
                END AS age_group,
                COUNT(*) AS total,
                SUM(attrition_binary) AS attrition_count
            FROM public.silver_employees
            GROUP BY age_group
            ORDER BY MIN(age) ASC;
        """)).fetchall()
        charts["by_age_group"] = [dict(r._mapping) for r in res_age]

        # 6. Ingresos promedio por departamento
        res_inc = conn.execute(text("""
            SELECT department, 
                   ROUND(AVG(monthly_income)::NUMERIC, 2) AS avg_income,
                   ROUND(AVG(CASE WHEN attrition = 'Yes' THEN monthly_income ELSE NULL END)::NUMERIC, 2) AS avg_income_attrition,
                   ROUND(AVG(CASE WHEN attrition = 'No' THEN monthly_income ELSE NULL END)::NUMERIC, 2) AS avg_income_retained
            FROM public.silver_employees
            GROUP BY department;
        """)).fetchall()
        charts["income_by_dept"] = [dict(r._mapping) for r in res_inc]

    return charts

def get_employees(limit: int = 50, offset: int = 0, search: Optional[str] = None) -> List[Dict[str, Any]]:
    """
    Consulta la lista de empleados desde la capa Silver para mostrarlos en el frontend.
    """
    sql = """
        SELECT employee_number, attrition, department, job_role, age, monthly_income, overtime, total_working_years, job_satisfaction
        FROM public.silver_employees
    """
    params = {"limit": limit, "offset": offset}
    if search:
        sql += " WHERE department ILIKE :search OR job_role ILIKE :search OR employee_number::TEXT LIKE :search"
        params["search"] = f"%{search}%"
        
    sql += " ORDER BY employee_number ASC LIMIT :limit OFFSET :offset;"
    
    with engine.connect() as conn:
        rows = conn.execute(text(sql), params).fetchall()
        return [dict(r._mapping) for r in rows]

def get_employee_raw_data(employee_number: int) -> Optional[Dict[str, Any]]:
    """
    Recupera el diccionario completo de las 35 variables de un empleado (desde Bronze data JSONB).
    """
    sql = text("SELECT data FROM public.employees WHERE employee_number = :emp_id LIMIT 1;")
    with engine.connect() as conn:
        row = conn.execute(sql, {"emp_id": employee_number}).fetchone()
        if row:
            return row[0]
    return None

def save_prediction(
    employee_number: Optional[int],
    prediction: str,
    probability: float,
    risk_level: str,
    model_version: str,
    input_features: Dict[str, Any]
) -> int:
    """
    Guarda la predicción generada en la capa Gold (public.gold_predictions).
    """
    sql = text("""
        INSERT INTO public.gold_predictions (
            employee_number, prediction, probability, risk_level, model_version, input_features, created_at
        ) VALUES (
            :emp_num, :pred, :prob, :risk, :ver, :features, :created_at
        ) RETURNING id;
    """)
    with engine.begin() as conn:
        new_id = conn.execute(sql, {
            "emp_num": employee_number,
            "pred": prediction,
            "prob": probability,
            "risk": risk_level,
            "ver": model_version,
            "features": json.dumps(input_features),
            "created_at": datetime.now(timezone.utc)
        }).scalar()
        return new_id

def get_predictions_history(limit: int = 50) -> List[Dict[str, Any]]:
    """
    Consulta el historial auditable de predicciones desde la capa Gold.
    """
    sql = text("""
        SELECT id, employee_number, prediction, probability, risk_level, model_version, input_features, created_at
        FROM public.gold_predictions
        ORDER BY created_at DESC
        LIMIT :limit;
    """)
    with engine.connect() as conn:
        rows = conn.execute(sql, {"limit": limit}).fetchall()
        results = []
        for r in rows:
            m = dict(r._mapping)
            if isinstance(m["created_at"], datetime):
                m["created_at_str"] = m["created_at"].strftime("%Y-%m-%d %H:%M:%S")
            else:
                m["created_at_str"] = str(m["created_at"])
            results.append(m)
        return results

def get_risk_distribution() -> Dict[str, int]:
    """
    Obtiene la distribución de predicciones por nivel de riesgo desde la capa Gold.
    """
    sql = text("""
        SELECT risk_level, COUNT(*) AS count
        FROM public.gold_predictions
        GROUP BY risk_level;
    """)
    dist = {"BAJO": 0, "MODERADO": 0, "ALTO": 0}
    with engine.connect() as conn:
        rows = conn.execute(sql).fetchall()
        for r in rows:
            m = r._mapping
            if m["risk_level"] in dist:
                dist[m["risk_level"]] = int(m["count"])
    return dist
