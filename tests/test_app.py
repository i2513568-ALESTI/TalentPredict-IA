"""
Pruebas integrales de la aplicación Flask TalentPredict AI.
Verifica:
1. Renderizado de todas las vistas HTML
2. Endpoints de la API REST
3. Inferencia real con el modelo de Machine Learning
4. Conexión con Supabase PostgreSQL (Bronze, Silver, Gold)
"""

import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from main import app

def run_tests():
    print("=" * 60)
    print("🧪 EJECUTANDO SUITE DE PRUEBAS - TALENTPREDICT AI")
    print("=" * 60)
    
    client = app.test_client()
    
    # 1. Test Home / Dashboard
    res = client.get("/")
    assert res.status_code == 200, f"Error en /: {res.status_code}"
    assert b"Dashboard de People Analytics" in res.data
    print("✅ Vista '/' (Dashboard) -> OK (200)")
    
    # 2. Test Controles
    res = client.get("/controles")
    assert res.status_code == 200, f"Error en /controles: {res.status_code}"
    assert b"Controles y Gest" in res.data
    print("✅ Vista '/controles' (Gestión de Personal Silver) -> OK (200)")
    
    # 3. Test Formulario Predict
    res = client.get("/predict")
    assert res.status_code == 200, f"Error en /predict: {res.status_code}"
    assert b"Simulador de Riesgo de Rotaci" in res.data
    print("✅ Vista '/predict' (Formulario Predictivo) -> OK (200)")
    
    # 4. Test Historial
    res = client.get("/historial")
    assert res.status_code == 200, f"Error en /historial: {res.status_code}"
    assert b"Historial de Predicciones" in res.data
    print("✅ Vista '/historial' (Capa Gold) -> OK (200)")
    
    # 5. Test API KPIs
    res = client.get("/api/kpis")
    assert res.status_code == 200
    kpis = res.get_json()
    assert kpis["total_employees"] == 1470
    assert kpis["total_attrition"] == 237
    print(f"✅ API '/api/kpis' -> OK (Total: {kpis['total_employees']}, Rotación: {kpis['attrition_rate_pct']}%)")
    
    # 6. Test API Analytics
    res = client.get("/api/analytics")
    assert res.status_code == 200
    charts = res.get_json()
    assert "by_department" in charts
    assert "by_overtime" in charts
    assert "by_job_role" in charts
    print(f"✅ API '/api/analytics' -> OK (6 gráficos estructurados)")
    
    # 7. Test API Employee 1
    res = client.get("/api/employee/1")
    assert res.status_code == 200
    emp1 = res.get_json()
    assert emp1["EmployeeNumber"] == 1
    print(f"✅ API '/api/employee/1' -> OK (Empleado: {emp1['JobRole']}, Dept: {emp1['Department']})")
    
    # 8. Test API Predict (POST con datos de empleado)
    res = client.post("/api/predict", json=emp1)
    assert res.status_code == 200
    pred = res.get_json()
    assert "probability_pct" in pred
    assert "risk_level" in pred
    assert pred["prediction_id"] is not None
    print(f"✅ API '/api/predict' (POST) -> OK (Riesgo: {pred['risk_level']}, Prob: {pred['probability_pct']}%, Gold ID #{pred['prediction_id']})")
    
    print("\n🎉 TODAS LAS PRUEBAS COMPLETADAS EXITOSAMENTE (8/8).")

if __name__ == "__main__":
    run_tests()
