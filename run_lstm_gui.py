#!/usr/bin/env python
"""
Script para ejecutar la interfaz gráfica de entrenamiento LSTM
Ejecutar desde la raíz del proyecto: python run_lstm_gui.py
"""

if __name__ == "__main__":
    import sys
    sys.path.insert(0, 'store_sim/src/PartB')
    
    print("="*60)
    print("LSTM Sales Forecasting - Interfaz Gráfica")
    print("="*60)
    print("\nCaracterísticas:")
    print("  ✓ Selección de categoría de producto")
    print("  ✓ Control de épocas de entrenamiento (1000-5000)")
    print("  ✓ Visualización en tiempo real")
    print("  ✓ Gráficos de predicción con bandas de confianza")
    print("  ✓ Métricas MAE y RMSE")
    print("  ✓ Predicción próxima semana")
    print("\nIniciando aplicación...\n")
    
    from store_sim.src.PartB.lstm_gui import LSTMApp
    
    # Iniciar en pantalla completa para mejor calidad
    app = LSTMApp(fullscreen=True)
    app.run()
