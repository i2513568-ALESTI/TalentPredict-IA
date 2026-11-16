"""
Script para la importación y sincronización de datos desde Kaggle (CSV) hacia Supabase PostgreSQL.
Proyecto: TalentPredict AI
"""

import os
import sys
import json
from pathlib import Path
from datetime import datetime, timezone
import pandas as pd
from dotenv import load_dotenv
from sqlalchemy import create_engine, text, Table, MetaData
from sqlalchemy.dialects.postgresql import insert

# 1. Cargar variables de entorno desde .env
BASE_DIR = Path(__file__).resolve().parent.parent
dotenv_path = BASE_DIR / ".env"
load_dotenv(dotenv_path=dotenv_path)

DB_HOST = os.getenv("DB_HOST")
DB_PORT = os.getenv("DB_PORT", "5432")
DB_NAME = os.getenv("DB_NAME", "postgres")
DB_USER = os.getenv("DB_USER")
DB_PASSWORD = os.getenv("DB_PASSWORD")
DB_SSLMODE = os.getenv("DB_SSLMODE", "require")

if not all([DB_HOST, DB_USER, DB_PASSWORD]):
    print("❌ Error: Faltan variables de entorno requeridas (DB_HOST, DB_USER, DB_PASSWORD) en .env.")
    sys.exit(1)

DATABASE_URL = f"postgresql+psycopg2://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}?sslmode={DB_SSLMODE}"

# 35 columnas oficiales del dataset IBM HR Analytics (Kaggle)
EXPECTED_COLUMNS = [
    'Age', 'Attrition', 'BusinessTravel', 'DailyRate', 'Department',
    'DistanceFromHome', 'Education', 'EducationField', 'EmployeeCount',
    'EmployeeNumber', 'EnvironmentSatisfaction', 'Gender', 'HourlyRate',
    'JobInvolvement', 'JobLevel', 'JobRole', 'JobSatisfaction',
    'MaritalStatus', 'MonthlyIncome', 'MonthlyRate', 'NumCompaniesWorked',
    'Over18', 'OverTime', 'PercentSalaryHike', 'PerformanceRating',
    'RelationshipSatisfaction', 'StandardHours', 'StockOptionLevel',
    'TotalWorkingYears', 'TrainingTimesLastYear', 'WorkLifeBalance',
    'YearsAtCompany', 'YearsInCurrentRole', 'YearsSinceLastPromotion',
    'YearsWithCurrManager'
]

def run_import():
    csv_file = BASE_DIR / "data" / "WA_Fn-UseC_-HR-Employee-Attrition.csv"
    print(f"==================================================")
    print(f"🚀 TalentPredict AI - Importador Kaggle -> Supabase")
    print(f"==================================================")
    print(f"📂 Verificando archivo CSV en: {csv_file}")
    
    if not csv_file.exists():
        print(f"❌ Error: No se encontró el archivo CSV en {csv_file}")
        sys.exit(1)
        
    # Paso 1: Leer CSV
    df = pd.read_csv(csv_file)
    print(f"📊 Registros leídos del CSV: {len(df)}")
    
    # Paso 2: Validar 35 columnas esperadas
    missing_cols = [c for c in EXPECTED_COLUMNS if c not in df.columns]
    if missing_cols:
        print(f"❌ Error: El dataset no tiene todas las columnas esperadas. Faltan: {missing_cols}")
        sys.exit(1)
    if len(df.columns) != 35:
        print(f"⚠️ Advertencia: El dataset contiene {len(df.columns)} columnas (se esperaban 35).")
    else:
        print(f"✅ Validación de estructura: 35 columnas confirmadas.")
        
    # Paso 3: Verificar unicidad de EmployeeNumber
    duplicates = df['EmployeeNumber'].duplicated().sum()
    if duplicates > 0:
        print(f"❌ Error: Se encontraron {duplicates} registros duplicados en EmployeeNumber.")
        sys.exit(1)
    print(f"✅ Validación de identificadores: 0 duplicados en EmployeeNumber ({df['EmployeeNumber'].nunique()} únicos).")
    
    # Paso 4: Conectar a Supabase
    print(f"🔌 Conectando a Supabase PostgreSQL...")
    engine = create_engine(DATABASE_URL)
    metadata = MetaData()
    
    try:
        employees_table = Table("employees", metadata, autoload_with=engine)
        print("✅ Conexión establecida y tabla 'employees' reflejada correctamente.")
    except Exception as e:
        print(f"❌ Error al conectar o inspeccionar tabla 'employees': {e}")
        sys.exit(1)
        
    # Paso 5 & 6: Preparar registros y realizar upsert en lotes
    now_utc = datetime.now(timezone.utc)
    batch_size = 250
    total_records = len(df)
    imported_count = 0
    
    print(f"📦 Iniciando carga con Upsert en lotes de {batch_size} registros...")
    
    records = []
    for _, row in df.iterrows():
        record = {
            "employee_number": int(row["EmployeeNumber"]),
            "attrition": str(row["Attrition"]),
            "department": str(row["Department"]),
            "job_role": str(row["JobRole"]),
            "age": int(row["Age"]),
            "monthly_income": float(row["MonthlyIncome"]),
            "overtime": str(row["OverTime"]),
            "data": json.loads(row.to_json()),
            "imported_at": now_utc
        }
        records.append(record)
        
    with engine.begin() as conn:
        for i in range(0, total_records, batch_size):
            batch = records[i:i + batch_size]
            stmt = insert(employees_table).values(batch)
            upsert_stmt = stmt.on_conflict_do_update(
                index_elements=['employee_number'],
                set_={
                    'attrition': stmt.excluded.attrition,
                    'department': stmt.excluded.department,
                    'job_role': stmt.excluded.job_role,
                    'age': stmt.excluded.age,
                    'monthly_income': stmt.excluded.monthly_income,
                    'overtime': stmt.excluded.overtime,
                    'data': stmt.excluded.data,
                    'imported_at': stmt.excluded.imported_at
                }
            )
            conn.execute(upsert_stmt)
            imported_count += len(batch)
            pct = (imported_count / total_records) * 100
            print(f"   -> Progreso: {imported_count}/{total_records} registros procesados ({pct:.1f}%)")
            
    # Paso 7: Verificar total de registros almacenados en Supabase
    with engine.connect() as conn:
        count_db = conn.execute(text("SELECT count(*) FROM employees;")).scalar()
        print(f"==================================================")
        print(f"🎯 Total de registros verificados en Supabase: {count_db}")
        print(f"==================================================")
        
        if count_db == total_records:
            print(f"🎉 ÉXITO: Los {total_records} registros fueron importados y validados.")
        else:
            print(f"⚠️ Nota: Hay {count_db} registros en Supabase (esperados {total_records}).")

if __name__ == "__main__":
    run_import()
