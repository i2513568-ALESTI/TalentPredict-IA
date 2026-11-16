"""
Módulo de entrenamiento, evaluación y serialización del modelo de Machine Learning.
Proyecto: TalentPredict AI
"""

import os
import json
from pathlib import Path
from typing import Dict, Any, List
import numpy as np
import pandas as pd
import joblib

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.dummy import DummyClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report
)

from src.data_processor import (
    load_data_from_csv,
    get_feature_columns,
    prepare_train_test_data,
    analyze_dataset
)

BASE_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = BASE_DIR / "models"

def build_pipeline(numeric_features: List[str], categorical_features: List[str], class_weight='balanced', random_state: int = 42) -> Pipeline:
    """
    Construye el Pipeline de Scikit-learn integrando preprocesamiento y RandomForest.
    """
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numeric_features),
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_features)
        ],
        remainder='drop'
    )
    
    classifier = RandomForestClassifier(
        n_estimators=150,
        max_depth=10,
        min_samples_split=5,
        min_samples_leaf=2,
        class_weight=class_weight,
        random_state=random_state,
        n_jobs=-1
    )
    
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', classifier)
    ])
    
    return pipeline

def evaluate_thresholds(y_test: pd.Series, y_prob: np.ndarray, thresholds: List[float] = [0.20, 0.25, 0.30, 0.35, 0.40, 0.50]) -> List[Dict[str, Any]]:
    """
    Evalúa el trade-off de Precision y Recall a diferentes puntos de corte de probabilidad.
    """
    results = []
    for t in thresholds:
        preds = (y_prob >= t).astype(int)
        rec = float(recall_score(y_test, preds, zero_division=0))
        prec = float(precision_score(y_test, preds, zero_division=0))
        f1 = float(f1_score(y_test, preds, zero_division=0))
        cm = confusion_matrix(y_test, preds)
        tn, fp, fn, tp = cm.ravel()
        results.append({
            "threshold": t,
            "recall_yes": round(rec, 4),
            "precision_yes": round(prec, 4),
            "f1_score_yes": round(f1, 4),
            "true_positives": int(tp),
            "false_positives": int(fp),
            "false_negatives": int(fn),
            "true_negatives": int(tn)
        })
    return results

def evaluate_model(model, X_test: pd.DataFrame, y_test: pd.Series, model_name: str = "Model") -> Dict[str, Any]:
    """
    Calcula métricas de evaluación exhaustivas sobre el conjunto de prueba.
    """
    y_pred = model.predict(X_test)
    
    threshold_eval = []
    if hasattr(model, "predict_proba"):
        y_prob = model.predict_proba(X_test)[:, 1]
        auc_val = float(roc_auc_score(y_test, y_prob))
        threshold_eval = evaluate_thresholds(y_test, y_prob)
    else:
        auc_val = 0.5
        
    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()
    
    metrics = {
        "model_name": model_name,
        "accuracy": round(float(accuracy_score(y_test, y_pred)), 4),
        "precision_yes": round(float(precision_score(y_test, y_pred, pos_label=1, zero_division=0)), 4),
        "recall_yes": round(float(recall_score(y_test, y_pred, pos_label=1, zero_division=0)), 4),
        "f1_score_yes": round(float(f1_score(y_test, y_pred, pos_label=1, zero_division=0)), 4),
        "roc_auc": round(auc_val, 4),
        "confusion_matrix": {
            "true_negatives": int(tn),
            "false_positives": int(fp),
            "false_negatives": int(fn),
            "true_positives": int(tp)
        },
        "threshold_analysis": threshold_eval,
        "classification_report": classification_report(y_test, y_pred, target_names=["No", "Yes"], output_dict=True, zero_division=0)
    }
    return metrics

def train_and_save_model() -> Dict[str, Any]:
    """
    Orquesta la carga de datos, partición estratificada, entrenamiento comparativo y serialización.
    """
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    
    print("=" * 60)
    print("🧠 TalentPredict AI - Pipeline de Machine Learning")
    print("=" * 60)
    
    # 1. Cargar y analizar datos
    df = load_data_from_csv()
    analysis = analyze_dataset(df)
    print(f"📊 Dataset cargado: {analysis['total_records']} registros.")
    print(f"   - Clase 'No' (permanecen): {analysis['attrition_distribution']['counts']['No']} ({analysis['attrition_distribution']['percentages']['No']}%)")
    print(f"   - Clase 'Yes' (rotación):   {analysis['attrition_distribution']['counts']['Yes']} ({analysis['attrition_distribution']['percentages']['Yes']}%)")
    
    # 2. Separación estratificada antes de cualquier ajuste (anti data leakage)
    X_train, X_test, y_train, y_test = prepare_train_test_data(df, test_size=0.2, random_state=42)
    print(f"📦 Partición estratificada (80/20):")
    print(f"   - Train: {X_train.shape[0]} muestras (Positivas: {y_train.sum()})")
    print(f"   - Test:  {X_test.shape[0]} muestras (Positivas: {y_test.sum()})")
    
    num_cols, cat_cols = get_feature_columns()
    
    # 3. Modelo Línea Base (DummyClassifier)
    print("\n📏 Entrenando modelo de línea base (DummyClassifier - Most Frequent)...")
    dummy = DummyClassifier(strategy='most_frequent')
    dummy.fit(X_train, y_train)
    dummy_metrics = evaluate_model(dummy, X_test, y_test, model_name="DummyClassifier (Baseline)")
    
    # 4. Pipeline Principal (RandomForestClassifier con class_weight='balanced')
    print("🌲 Entrenando Pipeline RandomForestClassifier (class_weight='balanced')...")
    pipeline = build_pipeline(num_cols, cat_cols, class_weight='balanced', random_state=42)
    pipeline.fit(X_train, y_train)
    rf_metrics = evaluate_model(pipeline, X_test, y_test, model_name="RandomForestClassifier (Balanced)")
    
    # 5. Comparativa de resultados
    print("\n" + "=" * 60)
    print("📈 RESULTADOS COMPARATIVOS EN CONJUNTO DE PRUEBA (TEST SET)")
    print("=" * 60)
    print(f"{'Métrica':<25} | {'Dummy (Línea Base)':<20} | {'RandomForest (Balanced)':<20}")
    print("-" * 72)
    print(f"{'Accuracy':<25} | {dummy_metrics['accuracy']:<20} | {rf_metrics['accuracy']:<20}")
    print(f"{'ROC-AUC':<25} | {dummy_metrics['roc_auc']:<20} | {rf_metrics['roc_auc']:<20}")
    print(f"{'Recall (Clase Yes)':<25} | {dummy_metrics['recall_yes']:<20} | {rf_metrics['recall_yes']:<20}")
    print(f"{'Precision (Clase Yes)':<25} | {dummy_metrics['precision_yes']:<20} | {rf_metrics['precision_yes']:<20}")
    print(f"{'F1-Score (Clase Yes)':<25} | {dummy_metrics['f1_score_yes']:<20} | {rf_metrics['f1_score_yes']:<20}")
    print("-" * 72)
    print(f"Matriz de Confusión RF: TN={rf_metrics['confusion_matrix']['true_negatives']}, FP={rf_metrics['confusion_matrix']['false_positives']}, FN={rf_metrics['confusion_matrix']['false_negatives']}, TP={rf_metrics['confusion_matrix']['true_positives']}")
    
    print("\n🎯 Análisis de Recall y Precisión por Umbral de Probabilidad:")
    for th in rf_metrics.get("threshold_analysis", []):
        print(f"   - Umbral {th['threshold']:.2f} -> Recall: {th['recall_yes']*100:.1f}%, Precision: {th['precision_yes']*100:.1f}%, F1: {th['f1_score_yes']:.4f}")
    
    # 6. Guardar artefactos
    model_path = MODELS_DIR / "talent_predict_rf.joblib"
    metrics_path = MODELS_DIR / "model_metrics.json"
    
    joblib.dump(pipeline, model_path)
    print(f"\n💾 Modelo guardado exitosamente en: {model_path}")
    
    output_info = {
        "dataset_analysis": analysis,
        "dummy_baseline": dummy_metrics,
        "random_forest": rf_metrics,
        "model_file": str(model_path.name),
        "features": {
            "numeric": num_cols,
            "categorical": cat_cols
        }
    }
    
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(output_info, f, indent=4)
    print(f"📄 Métricas exportadas a: {metrics_path}")
    
    return output_info

if __name__ == "__main__":
    train_and_save_model()
