# TalentPredict AI — People Analytics & Retention Intelligence

Plataforma integral de **People Analytics** que utiliza **Machine Learning** y **Arquitectura Medallion** sobre **Supabase PostgreSQL** para predecir, auditar y prevenir el riesgo de rotación laboral (*Employee Attrition*).

---

## 🏛️ Arquitectura Medallion (Data Engineering)

La persistencia de datos en Supabase PostgreSQL está organizada bajo el patrón Medallion:

```
[ Kaggle CSV / Formulario ]
           │
           ▼
   🥉 CAPA BRONZE (Raw / Ingesta)
      • Tabla: public.employees
      • 1,470 registros crudos con auditoría y payload íntegro en data JSONB.
           │
           ▼
   🥈 CAPA SILVER (Depuración & Feature Engineering)
      • Tabla: public.silver_employees
      • 30 variables depuradas, normalizadas, sin columnas constantes (EmployeeCount, Over18, StandardHours).
      • Identificador EmployeeNumber preservado como Primary Key.
           │
           ▼
   🥇 CAPA GOLD (Serving, Inferencia & KPIs)
      • Vista: public.gold_department_kpis (KPIs precalculados para el Dashboard).
      • Tabla: public.gold_predictions (Trazabilidad e historial auditable de inferencias).
```

---

## 🧠 Pipeline de Machine Learning

* **Algoritmo:** `RandomForestClassifier` (150 estimadores, profundidad máxima 10, ponderación analítica `class_weight='balanced'`).
* **Preprocesamiento:** `ColumnTransformer` con `StandardScaler` (23 variables numéricas) y `OneHotEncoder(handle_unknown='ignore')` (7 variables categóricas).
* **Partición estratificada:** 80% Entrenamiento (1,176 registros) / 20% Prueba (294 registros) implementada **antes** del ajuste del pipeline para evitar fuga de información (*anti-data leakage*).
* **Métricas en Conjunto de Prueba:**
  * **ROC-AUC:** `0.7959` (~80% de discriminación global vs 50% del Dummy Baseline).
  * **Accuracy:** `83.33%`.
  * **Semáforo de Riesgo (Calibrado para People Analytics):**
    * 🟢 **Bajo Riesgo:** Probabilidad $< 20\%$
    * 🟡 **Riesgo Moderado:** Probabilidad entre $20\%$ y $35\%$
    * 🔴 **Alto Riesgo:** Probabilidad $\ge 35\%$ (Recall $\ge 55.3\%$, capturando efectivamente a los empleados propensos a deserción).

---

## 📁 Estructura del Proyecto

```
flask-ejemplo-main/
├── .env.example                     # Plantilla de variables de entorno
├── .gitignore                       # Filtro de seguridad (.env, *.pkl, *.joblib ignorados)
├── requirements.txt                 # Dependencias oficiales de producción
├── main.py                          # Servidor de aplicación Flask y endpoints REST
│
├── data/
│   └── WA_Fn-UseC_-HR-Employee-Attrition.csv  # Dataset original de Kaggle
│
├── models/
│   ├── talent_predict_rf.joblib     # Pipeline de Machine Learning entrenado
│   └── model_metrics.json           # Registro de métricas y matriz de confusión
│
├── src/
│   ├── data_processor.py            # Limpieza, tipos y partición estratificada
│   ├── model_trainer.py             # Entrenamiento del Pipeline y comparativa Dummy
│   ├── database.py                  # Repositorio Medallion Supabase PostgreSQL
│   └── predictor.py                 # Servicio de inferencia, insights y semáforos
│
├── scripts/
│   ├── import_kaggle_supabase.py    # Ingesta Kaggle -> Supabase con Upsert
│   ├── setup_medallion_architecture.py # Creación de capas Silver y Gold en DB
│   └── train_model.py               # Reentrenamiento del modelo ML
│
├── templates/
│   ├── layout.html                  # Plantilla base responsive (Bootstrap 5.3 + Icons)
│   ├── index.html                   # Dashboard principal con 6 gráficos Chart.js
│   ├── controles.html               # Gestión interactiva de personal (Capa Silver)
│   ├── predict.html                 # Simulador interactivo de evaluación con IA
│   └── historial.html               # Registro auditable de predicciones (Capa Gold)
│
└── tests/
    └── test_app.py                  # Suite de pruebas automatizadas (8/8 OK)
```

---

## 🚀 Instalación y Ejecución Local

### 1. Clonar y configurar entorno
```bash
git clone <repo-url>
cd flask-ejemplo-main
python -m venv entorno
source entorno/bin/activate   # En Linux/Mac
entorno\Scripts\activate      # En Windows
pip install -r requirements.txt
```

### 2. Configurar variables de entorno
Copiar `.env.example` a `.env` y configurar las credenciales:
```env
DB_HOST=aws-1-us-east-1.pooler.supabase.com
DB_PORT=5432
DB_NAME=postgres
DB_USER=postgres.cfuypsphjqpmakrvdvxw
DB_PASSWORD=tu_password_secreto
DB_SSLMODE=require

FLASK_SECRET_KEY=clave-secreta-de-produccion
```

### 3. Sincronizar capas Medallion y entrenar modelo
```bash
# 1. Ingesta a Bronze
python scripts/import_kaggle_supabase.py

# 2. Configuración y sincronización de Silver y Gold
python scripts/setup_medallion_architecture.py

# 3. Entrenamiento del Pipeline ML
python scripts/train_model.py
```

### 4. Ejecutar suite de pruebas
```bash
python tests/test_app.py
```

### 5. Iniciar servidor web
```bash
python main.py
```
Abrir en el navegador: `http://localhost:5000`

---

## ☁️ Despliegue en Render

1. Crear un nuevo **Web Service** conectado a este repositorio.
2. Configurar los parámetros de ejecución:
   * **Runtime:** `Python 3`
   * **Build Command:** `pip install -r requirements.txt`
   * **Start Command:** `gunicorn main:app`
3. En la pestaña **Environment**, añadir las variables:
   * `DB_HOST`
   * `DB_PORT`
   * `DB_NAME`
   * `DB_USER`
   * `DB_PASSWORD`
   * `DB_SSLMODE`
   * `FLASK_SECRET_KEY`
4. Render desplegará la aplicación automáticamente con HTTPS activado.
