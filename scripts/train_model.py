"""
Script de ejecución para entrenamiento del modelo TalentPredict AI.
"""
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from src.model_trainer import train_and_save_model

if __name__ == "__main__":
    train_and_save_model()
