"""
Script para la configuración de la Arquitectura Medallion en Supabase PostgreSQL:
- Bronze: public.employees (existente, datos crudos con JSONB original).
- Silver: public.silver_employees (datos limpios, tipados, sin variables constantes, listos para ML y análisis).
- Gold:
    - public.gold_predictions (tabla auditable de predicciones de rotación de talento).
    - public.gold_department_kpis (agregaciones analíticas de negocio para el Dashboard).
"""

import os
from pathlib import Path
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
engine = create_engine(DATABASE_URL)

SQL_SETUP = """
-- ========================================================
-- MEDALLION ARCHITECTURE - TALENTPREDICT AI
-- ========================================================

-- 1. CAPA SILVER: silver_employees
-- Datos limpios, normalizados y tipados a partir de Bronze (public.employees)
CREATE TABLE IF NOT EXISTS public.silver_employees (
    employee_number INTEGER PRIMARY KEY,
    attrition TEXT NOT NULL,
    attrition_binary INTEGER NOT NULL,
    age INTEGER NOT NULL,
    business_travel TEXT,
    daily_rate INTEGER,
    department TEXT,
    distance_from_home INTEGER,
    education INTEGER,
    education_field TEXT,
    environment_satisfaction INTEGER,
    gender TEXT,
    hourly_rate INTEGER,
    job_involvement INTEGER,
    job_level INTEGER,
    job_role TEXT,
    job_satisfaction INTEGER,
    marital_status TEXT,
    monthly_income NUMERIC,
    monthly_rate INTEGER,
    num_companies_worked INTEGER,
    overtime TEXT,
    percent_salary_hike INTEGER,
    performance_rating INTEGER,
    relationship_satisfaction INTEGER,
    stock_option_level INTEGER,
    total_working_years INTEGER,
    training_times_last_year INTEGER,
    work_life_balance INTEGER,
    years_at_company INTEGER,
    years_in_current_role INTEGER,
    years_since_last_promotion INTEGER,
    years_with_curr_manager INTEGER,
    processed_at TIMESTAMPTZ DEFAULT NOW()
);

-- Poblar / sincronizar capa Silver desde Bronze
INSERT INTO public.silver_employees (
    employee_number, attrition, attrition_binary, age, business_travel,
    daily_rate, department, distance_from_home, education, education_field,
    environment_satisfaction, gender, hourly_rate, job_involvement, job_level,
    job_role, job_satisfaction, marital_status, monthly_income, monthly_rate,
    num_companies_worked, overtime, percent_salary_hike, performance_rating,
    relationship_satisfaction, stock_option_level, total_working_years,
    training_times_last_year, work_life_balance, years_at_company,
    years_in_current_role, years_since_last_promotion, years_with_curr_manager
)
SELECT 
    employee_number,
    attrition,
    CASE WHEN attrition = 'Yes' THEN 1 ELSE 0 END AS attrition_binary,
    age,
    (data->>'BusinessTravel') AS business_travel,
    (data->>'DailyRate')::INTEGER AS daily_rate,
    department,
    (data->>'DistanceFromHome')::INTEGER AS distance_from_home,
    (data->>'Education')::INTEGER AS education,
    (data->>'EducationField') AS education_field,
    (data->>'EnvironmentSatisfaction')::INTEGER AS environment_satisfaction,
    (data->>'Gender') AS gender,
    (data->>'HourlyRate')::INTEGER AS hourly_rate,
    (data->>'JobInvolvement')::INTEGER AS job_involvement,
    (data->>'JobLevel')::INTEGER AS job_level,
    job_role,
    (data->>'JobSatisfaction')::INTEGER AS job_satisfaction,
    (data->>'MaritalStatus') AS marital_status,
    monthly_income,
    (data->>'MonthlyRate')::INTEGER AS monthly_rate,
    (data->>'NumCompaniesWorked')::INTEGER AS num_companies_worked,
    overtime,
    (data->>'PercentSalaryHike')::INTEGER AS percent_salary_hike,
    (data->>'PerformanceRating')::INTEGER AS performance_rating,
    (data->>'RelationshipSatisfaction')::INTEGER AS relationship_satisfaction,
    (data->>'StockOptionLevel')::INTEGER AS stock_option_level,
    (data->>'TotalWorkingYears')::INTEGER AS total_working_years,
    (data->>'TrainingTimesLastYear')::INTEGER AS training_times_last_year,
    (data->>'WorkLifeBalance')::INTEGER AS work_life_balance,
    (data->>'YearsAtCompany')::INTEGER AS years_at_company,
    (data->>'YearsInCurrentRole')::INTEGER AS years_in_current_role,
    (data->>'YearsSinceLastPromotion')::INTEGER AS years_since_last_promotion,
    (data->>'YearsWithCurrManager')::INTEGER AS years_with_curr_manager
FROM public.employees
ON CONFLICT (employee_number) DO UPDATE SET
    attrition = EXCLUDED.attrition,
    attrition_binary = EXCLUDED.attrition_binary,
    department = EXCLUDED.department,
    job_role = EXCLUDED.job_role,
    age = EXCLUDED.age,
    monthly_income = EXCLUDED.monthly_income,
    overtime = EXCLUDED.overtime,
    processed_at = NOW();

-- 2. CAPA GOLD: gold_predictions
-- Tabla oficial requerida para almacenar predicciones de ML
CREATE TABLE IF NOT EXISTS public.gold_predictions (
    id SERIAL PRIMARY KEY,
    employee_number INTEGER,
    prediction TEXT NOT NULL,
    probability NUMERIC(6, 4) NOT NULL,
    risk_level TEXT NOT NULL,
    model_version TEXT NOT NULL,
    input_features JSONB,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_gold_pred_created_at ON public.gold_predictions(created_at DESC);
CREATE INDEX IF NOT EXISTS idx_gold_pred_emp ON public.gold_predictions(employee_number);

-- 3. CAPA GOLD: gold_department_kpis (Vista analítica de métricas agregadas)
CREATE OR REPLACE VIEW public.gold_department_kpis AS
SELECT 
    department,
    COUNT(*) AS total_employees,
    SUM(attrition_binary) AS total_attrition,
    COUNT(*) - SUM(attrition_binary) AS total_retained,
    ROUND((AVG(attrition_binary) * 100)::NUMERIC, 2) AS attrition_rate_pct,
    ROUND(AVG(monthly_income)::NUMERIC, 2) AS avg_monthly_income,
    ROUND(AVG(age)::NUMERIC, 1) AS avg_age,
    ROUND(AVG(job_satisfaction)::NUMERIC, 2) AS avg_job_satisfaction
FROM public.silver_employees
GROUP BY department;
"""

def setup_medallion():
    print("==================================================")
    print("🏛️ Configurando Arquitectura Medallion en Supabase")
    print("==================================================")
    print("1. Bronze: public.employees (ya existente y cargada)")
    print("2. Silver: public.silver_employees (limpieza y tipado)")
    print("3. Gold:   public.gold_predictions (tabla de inferencias)")
    print("           public.gold_department_kpis (vista de KPIs)")
    
    with engine.begin() as conn:
        conn.execute(text(SQL_SETUP))
        
    with engine.connect() as conn:
        bronze_count = conn.execute(text("SELECT count(*) FROM public.employees;")).scalar()
        silver_count = conn.execute(text("SELECT count(*) FROM public.silver_employees;")).scalar()
        pred_count = conn.execute(text("SELECT count(*) FROM public.gold_predictions;")).scalar()
        kpis = conn.execute(text("SELECT * FROM public.gold_department_kpis;")).fetchall()
        
    print(f"\n✅ Verificación de capas:")
    print(f"   🥉 BRONZE (public.employees):        {bronze_count} filas")
    print(f"   🥈 SILVER (public.silver_employees): {silver_count} filas")
    print(f"   🥇 GOLD   (public.gold_predictions): {pred_count} registros históricos")
    print(f"\n📊 Capa Gold - KPIs por Departamento:")
    for k in kpis:
        m = k._mapping
        print(f"   - {m['department']}: {m['total_employees']} empleados | Rotación: {m['attrition_rate_pct']}% | Sueldo promedio: ${m['avg_monthly_income']}")
    print("\n🎉 Arquitectura Medallion configurada exitosamente en Supabase.")

if __name__ == "__main__":
    setup_medallion()
