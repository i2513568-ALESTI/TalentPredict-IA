"""
Servidor de Aplicación Web Flask - TalentPredict AI.
Plataforma de People Analytics con Machine Learning y Arquitectura Medallion en Supabase.
"""

import os
from pathlib import Path
from dotenv import load_dotenv
from flask import Flask, jsonify, request, render_template, abort

# Cargar variables de entorno antes de importar módulos internos
BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env")

from src.database import (
    get_kpis_summary,
    get_analytics_charts_data,
    get_employees,
    get_employee_raw_data,
    get_predictions_history,
    get_risk_distribution
)
from src.predictor import predict_employee_attrition, get_model

app = Flask(__name__)
app.config["SECRET_KEY"] = os.getenv("FLASK_SECRET_KEY", "talentpredict-secret-dev-key")

# Precarga del modelo en memoria durante el inicio del servidor
try:
    get_model()
    print("✅ Pipeline RandomForest cargado en memoria exitosamente.")
except Exception as e:
    print(f"⚠️ Advertencia al cargar el modelo: {e}")

# ---------------------------------------------------------------------------
# VISTAS PRINCIPALES (FRONTEND)
# ---------------------------------------------------------------------------

@app.route("/")
def home():
    """
    Dashboard principal con métricas ejecutivas, KPIs globales y gráficos analíticos.
    """
    try:
        kpis = get_kpis_summary()
        risk_dist = get_risk_distribution()
    except Exception as e:
        print(f"Error consultando Supabase en home: {e}")
        kpis = {
            "total_employees": 1470,
            "total_attrition": 237,
            "total_retained": 1233,
            "attrition_rate_pct": 16.12,
            "avg_monthly_income": 6502.93,
            "avg_age": 36.9
        }
        risk_dist = {"BAJO": 0, "MODERADO": 0, "ALTO": 0}
        
    return render_template("index.html", kpis=kpis, risk_dist=risk_dist)

@app.route("/controles")
def controles():
    """
    Módulo de Controles y Gestión de Personal (Capa Silver de Supabase).
    Muestra la tabla de colaboradores y permite ejecutar análisis predictivo.
    """
    try:
        employees = get_employees(limit=50)
    except Exception as e:
        print(f"Error consultando empleados: {e}")
        employees = []
        
    return render_template("controles.html", employees=employees)

@app.route("/predict")
def predict_view():
    """
    Módulo interactivo de predicción y simulación de escenarios de retención.
    """
    return render_template("predict.html")

@app.route("/historial")
def historial():
    """
    Historial auditable de inferencias almacenadas en la Capa Gold (public.gold_predictions).
    """
    try:
        history = get_predictions_history(limit=100)
    except Exception as e:
        print(f"Error consultando historial: {e}")
        history = []
        
    return render_template("historial.html", history=history)

# ---------------------------------------------------------------------------
# ENDPOINTS DE API REST (JSON)
# ---------------------------------------------------------------------------

@app.route("/api/predict", methods=["POST"])
def api_predict():
    """
    Endpoint para inferencia en tiempo real.
    Recibe las características del colaborador y retorna la probabilidad, riesgo e insights.
    """
    data = request.get_json()
    if not data:
        return jsonify({"error": "No se recibieron datos JSON válidos."}), 400
        
    try:
        result = predict_employee_attrition(data, persist_db=True)
        return jsonify(result), 200
    except Exception as e:
        print(f"Error durante inferencia: {e}")
        return jsonify({"error": f"Fallo al procesar la predicción: {str(e)}"}), 500

@app.route("/api/employee/<int:emp_id>", methods=["GET"])
def api_employee_data(emp_id):
    """
    Retorna el diccionario de atributos completo de un empleado desde la Capa Bronze.
    """
    data = get_employee_raw_data(emp_id)
    if not data:
        return jsonify({"error": f"Empleado con ID {emp_id} no encontrado."}), 404
    return jsonify(data), 200

@app.route("/api/analytics", methods=["GET"])
def api_analytics():
    """
    Retorna las agregaciones estructuradas para alimentar Chart.js.
    """
    try:
        charts_data = get_analytics_charts_data()
        return jsonify(charts_data), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/api/kpis", methods=["GET"])
def api_kpis():
    """
    Retorna los KPIs ejecutivos en formato JSON.
    """
    try:
        return jsonify(get_kpis_summary()), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# ---------------------------------------------------------------------------
# MANEJO DE ERRORES HTTP
# ---------------------------------------------------------------------------

@app.errorhandler(404)
def not_found(e):
    return render_template("layout.html"), 404

@app.errorhandler(500)
def server_error(e):
    return jsonify({"error": "Error interno del servidor", "details": str(e)}), 500

# ---------------------------------------------------------------------------
# INICIO LOCAL
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)
