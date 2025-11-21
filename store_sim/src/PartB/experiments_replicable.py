"""
Script para ejecutar experimentos replicables con diferentes semillas
"""
import sys
sys.path.insert(0, 'store_sim/src/PartB')

from data_processor import DataProcessor
from lstm_model import LSTMTrainer, set_seed
import torch
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error
import pandas as pd


def run_replicable_experiment(category, epochs=2000, seed=42):
    """
    Ejecuta un experimento replicable con una semilla específica
    
    Args:
        category: Categoría de producto a analizar
        epochs: Número de épocas de entrenamiento
        seed: Semilla para reproducibilidad
        
    Returns:
        Dict con resultados del experimento
    """
    print(f"\n{'='*70}")
    print(f"Experimento Replicable - Categoría: {category}")
    print(f"Épocas: {epochs} | Semilla: {seed}")
    print(f"{'='*70}\n")
    
    # Fijar semilla
    set_seed(seed)
    
    # Cargar datos
    print("Cargando datos...")
    data_processor = DataProcessor()
    data_processor.load_and_prepare()
    
    # Verificar que la categoría existe
    if category not in data_processor.get_categories():
        print(f"Error: Categoría '{category}' no encontrada")
        print(f"Categorías disponibles: {data_processor.get_categories()}")
        return None
    
    # Obtener datos
    data_info = data_processor.get_data_for_category(category)
    X_train = data_info["X_train"]
    X_test = data_info["X_test"]
    y_train = data_info["y_train"]
    y_test = data_info["y_test"]
    
    print(f"Datos de entrenamiento: {len(X_train)}")
    print(f"Datos de prueba: {len(X_test)}")
    
    # Device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    
    # Convertir a tensores
    X_train_t = torch.tensor(X_train, dtype=torch.float32, device=device)
    y_train_t = torch.tensor(y_train.reshape(-1, 1), dtype=torch.float32, device=device)
    X_test_t = torch.tensor(X_test, dtype=torch.float32, device=device)
    
    # Crear y entrenar modelo
    print("\nEntrenando modelo...")
    trainer = LSTMTrainer(device=device, seed=seed)
    trainer.create_model(input_size=1, hidden_size=64, output_size=1, lr=1e-3)
    
    # Entrenar
    losses = trainer.train(X_train_t, y_train_t, epochs=epochs, 
                          callback=lambda e, l: print(f"Época {e+1}/{epochs} - Loss: {l:.6f}") 
                                  if (e+1) % 100 == 0 else None)
    
    # Evaluar
    print("\nEvaluando modelo...")
    y_pred_scaled = trainer.predict(X_test_t)
    
    # Desescalar
    y_true = data_processor.inverse_transform(category, y_test)
    y_pred = data_processor.inverse_transform(category, y_pred_scaled)
    
    # Métricas
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    
    # Predicción próxima semana
    last_window = data_processor.get_last_window(category)
    next_scaled = trainer.forecast_next(last_window)
    next_value = data_processor.inverse_transform(category, np.array([next_scaled]))[0]
    
    results = {
        "category": category,
        "epochs": epochs,
        "seed": seed,
        "mae": mae,
        "rmse": rmse,
        "next_forecast": next_value,
        "train_size": len(X_train),
        "test_size": len(X_test),
        "final_loss": losses[-1],
        "device": device
    }
    
    print(f"\n{'='*70}")
    print("RESULTADOS:")
    print(f"  MAE: {mae:.2f}")
    print(f"  RMSE: {rmse:.2f}")
    print(f"  Predicción próxima semana: ${next_value:.2f}")
    print(f"  Pérdida final: {losses[-1]:.6f}")
    print(f"{'='*70}\n")
    
    return results


def compare_seeds(category, epochs=2000, seeds=[42, 123, 456, 789, 2024]):
    """
    Compara resultados con diferentes semillas para verificar reproducibilidad
    
    Args:
        category: Categoría de producto
        epochs: Número de épocas
        seeds: Lista de semillas a probar
        
    Returns:
        DataFrame con resultados comparativos
    """
    print(f"\n{'='*70}")
    print(f"COMPARACIÓN DE SEMILLAS - Categoría: {category}")
    print(f"{'='*70}\n")
    
    results = []
    
    for seed in seeds:
        print(f"\n--- Probando semilla: {seed} ---")
        result = run_replicable_experiment(category, epochs, seed)
        if result:
            results.append(result)
    
    # Crear DataFrame
    df = pd.DataFrame(results)
    
    print(f"\n{'='*70}")
    print("TABLA COMPARATIVA:")
    print(f"{'='*70}")
    print(df.to_string(index=False))
    print(f"\n{'='*70}")
    print("ESTADÍSTICAS:")
    print(f"  MAE promedio: {df['mae'].mean():.2f} ± {df['mae'].std():.2f}")
    print(f"  RMSE promedio: {df['rmse'].mean():.2f} ± {df['rmse'].std():.2f}")
    print(f"{'='*70}\n")
    
    return df


if __name__ == "__main__":
    # Ejemplo 1: Experimento simple replicable
    print("\n🔬 EJEMPLO 1: Experimento Simple Replicable")
    result = run_replicable_experiment(
        category="Electronics",  # Ajusta según tus categorías
        epochs=1000,
        seed=42
    )
    
    # Ejemplo 2: Comparar múltiples semillas
    print("\n🔬 EJEMPLO 2: Comparación de Semillas")
    df_comparison = compare_seeds(
        category="Electronics",  # Ajusta según tus categorías
        epochs=1000,
        seeds=[42, 123, 456]
    )
    
    print("\n✅ Experimentos completados!")
    print("\nPara replicar estos resultados, usa:")
    print("  python experiments_replicable.py")
