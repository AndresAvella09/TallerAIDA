#!/usr/bin/env python
"""
Script de ejecución para la simulación visual (Parte A)
Ejecutar desde la raíz del proyecto: python run_simulation.py
"""

if __name__ == "__main__":
    from store_sim.src.PartA.main import GameInteractive
    
    print("="*60)
    print("Iniciando Simulación Visual de Tienda (Parte A)")
    print("="*60)
    print("\nControles:")
    print("  - Cierra la ventana para terminar la simulación")
    print("  - Al finalizar se generará un mapa de calor\n")
    
    game = GameInteractive(customers = 100, categories = 7)
    game.run()
