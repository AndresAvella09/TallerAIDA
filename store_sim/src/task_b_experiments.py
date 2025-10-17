from store import Store
from simulation import Simulation
from analytics import StoreAnalytics
from heatmap import create_heatmap
import matplotlib.pyplot as plt


def run_experiment(num_customers=100, algorithm='bfs', seed=1230):
    """
    Ejecuta un experimento con un algoritmo específico.
    
    Args:
        num_customers: Número de clientes a simular
        algorithm: Algoritmo a usar ('bfs' o 'astar')
        seed: Semilla para reproducibilidad
        
    Returns:
        Tuple (simulation, analytics)
    """
    print(f"\n{'='*60}")
    print(f"Ejecutando experimento con algoritmo: {algorithm.upper()}")
    print(f"Número de clientes: {num_customers}")
    print(f"{'='*60}\n")
    
    # Crear tienda y simulación
    store = Store(rows=8, cols=6, seed=seed)
    sim = Simulation(store, num_customers=num_customers, algorithm_mix={algorithm: 1.0})
    
    # Ejecutar simulación
    steps = 0
    max_steps = 1000
    
    while sim.customers and steps < max_steps:
        sim.update()
        sim.customers = [c for c in sim.customers if not c.finished]
        steps += 1
    
    print(f"Simulación completada en {steps} pasos")
    print(f"Total de clientes procesados: {len(sim.all_customers)}")
    
    # Analizar resultados
    analytics = StoreAnalytics(sim, store)
    
    # Calcular todas las métricas
    traffic_stats = analytics.calculate_traffic_stats()
    path_metrics = analytics.calculate_path_metrics(sim.all_customers)
    bottlenecks = analytics.identify_bottlenecks()
    efficiency = analytics.calculate_efficiency_score()
    
    print(f"\nResultados del experimento:")
    print(f"  - Longitud promedio de camino: {path_metrics['avg_path_length']:.2f}")
    print(f"  - Score de eficiencia: {efficiency:.2f}/100")
    print(f"  - Cuellos de botella: {len(bottlenecks)}")
    
    return sim, analytics


def run_comparison_experiments(num_customers=100, seed=122):
    """
    Ejecuta experimentos comparativos entre BFS y A*.
    
    Args:
        num_customers: Número de clientes por experimento
        seed: Semilla para reproducibilidad
    """
    print("\n" + "="*60)
    print("COMPARACIÓN DE ALGORITMOS: BFS vs A*")
    print("="*60)
    
    # Experimento 1: BFS
    sim_bfs, analytics_bfs = run_experiment(num_customers, 'bfs', seed)
    
    # Experimento 2: A*
    sim_astar, analytics_astar = run_experiment(num_customers, 'astar', seed)
    
    # Generar reportes
    print("\n\nGenerando reportes...")
    analytics_bfs.generate_report("bfs_report.txt")
    analytics_astar.generate_report("astar_report.txt")
    
    # Generar mapas de calor
    create_heatmap(sim_bfs.traffic, sim_bfs.store, "heatmap_bfs.png")
    create_heatmap(sim_astar.traffic, sim_astar.store, "heatmap_astar.png")
    
    # Generar gráfico comparativo
    generate_comparison_chart(analytics_bfs, analytics_astar)
    
    # Generar reporte combinado
    generate_combined_report(analytics_bfs, analytics_astar)
    
    print("\n" + "="*60)
    print("EXPERIMENTOS COMPLETADOS")
    print("="*60)
    print("\nArchivos generados:")
    print("  - bfs_report.txt")
    print("  - astar_report.txt")
    print("  - heatmap_bfs.png")
    print("  - heatmap_astar.png")
    print("  - algorithm_comparison.png")
    print("  - combined_report.txt")


def generate_comparison_chart(analytics_bfs, analytics_astar):
    """
    Genera un gráfico comparativo entre BFS y A*.
    
    Args:
        analytics_bfs: Análisis del experimento BFS
        analytics_astar: Análisis del experimento A*
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Comparación de Algoritmos: BFS vs A*', fontsize=16, fontweight='bold')
    
    # Gráfico 1: Longitud promedio de camino
    ax1 = axes[0, 0]
    algorithms = ['BFS', 'A*']
    avg_lengths = [
        analytics_bfs.metrics['path_metrics']['avg_path_length'],
        analytics_astar.metrics['path_metrics']['avg_path_length']
    ]
    bars1 = ax1.bar(algorithms, avg_lengths, color=['#3498db', '#e74c3c'])
    ax1.set_ylabel('Longitud Promedio')
    ax1.set_title('Longitud Promedio de Camino')
    ax1.grid(axis='y', alpha=0.3)
    
    # Añadir valores sobre las barras
    for bar in bars1:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.2f}',
                ha='center', va='bottom')
    
    # Gráfico 2: Score de eficiencia
    ax2 = axes[0, 1]
    efficiency_scores = [
        analytics_bfs.metrics['efficiency_score'],
        analytics_astar.metrics['efficiency_score']
    ]
    bars2 = ax2.bar(algorithms, efficiency_scores, color=['#3498db', '#e74c3c'])
    ax2.set_ylabel('Score (0-100)')
    ax2.set_title('Score de Eficiencia')
    ax2.set_ylim([0, 100])
    ax2.grid(axis='y', alpha=0.3)
    
    for bar in bars2:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.2f}',
                ha='center', va='bottom')
    
    # Número de cuellos de botella
    ax3 = axes[1, 0]
    bottlenecks = [
        len(analytics_bfs.metrics['bottlenecks']),
        len(analytics_astar.metrics['bottlenecks'])
    ]
    bars3 = ax3.bar(algorithms, bottlenecks, color=['#3498db', '#e74c3c'])
    ax3.set_ylabel('Número de Cuellos de Botella')
    ax3.set_title('Cuellos de Botella Identificados')
    ax3.grid(axis='y', alpha=0.3)
    
    for bar in bars3:
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(height)}',
                ha='center', va='bottom')
    
    # Gráfico 4: Tráfico total
    ax4 = axes[1, 1]
    total_traffic = [
        analytics_bfs.metrics['traffic_stats']['total_visits'],
        analytics_astar.metrics['traffic_stats']['total_visits']
    ]
    bars4 = ax4.bar(algorithms, total_traffic, color=['#3498db', '#e74c3c'])
    ax4.set_ylabel('Total de Visitas')
    ax4.set_title('Tráfico Total')
    ax4.grid(axis='y', alpha=0.3)
    
    for bar in bars4:
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(height)}',
                ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig('algorithm_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("Gráfico de comparación generado: algorithm_comparison.png")


def generate_combined_report(analytics_bfs, analytics_astar):
    """
    Genera un reporte combinado comparando BFS y A*.
    
    Args:
        analytics_bfs: Análisis del experimento BFS
        analytics_astar: Análisis del experimento A*
    """
    with open('combined_report.txt', 'w', encoding='utf-8') as f:
        f.write("="*70 + "\n")
        f.write("REPORTE COMPARATIVO: BFS vs A*\n")
        f.write("Tarea B - Análisis de Algoritmos de Búsqueda\n")
        f.write("="*70 + "\n\n")
        
        # Resumen ejecutivo
        f.write("1. RESUMEN EJECUTIVO\n")
        f.write("-"*70 + "\n\n")
        
        bfs_metrics = analytics_bfs.metrics['path_metrics']
        astar_metrics = analytics_astar.metrics['path_metrics']
        
        f.write(f"Clientes procesados: {bfs_metrics['total_customers']}\n\n")
        
        f.write("Comparación de Longitud de Camino:\n")
        f.write(f"  BFS:  {bfs_metrics['avg_path_length']:.2f} pasos promedio\n")
        f.write(f"  A*:   {astar_metrics['avg_path_length']:.2f} pasos promedio\n")
        
        improvement = ((bfs_metrics['avg_path_length'] - astar_metrics['avg_path_length']) 
                      / bfs_metrics['avg_path_length'] * 100)
        
        if improvement > 0:
            f.write(f"  A* es {improvement:.1f}% más eficiente en longitud de camino\n\n")
        elif improvement < 0:
            f.write(f"  BFS es {-improvement:.1f}% más eficiente en longitud de camino\n\n")
        else:
            f.write(f"  Ambos algoritmos tienen la misma eficiencia\n\n")
        
        f.write("Comparación de Eficiencia Global:\n")
        f.write(f"  BFS:  {analytics_bfs.metrics['efficiency_score']:.2f}/100\n")
        f.write(f"  A*:   {analytics_astar.metrics['efficiency_score']:.2f}/100\n\n")
        
        # Análisis detallado
        f.write("\n2. ANÁLISIS DETALLADO\n")
        f.write("-"*70 + "\n\n")
        
        f.write("2.1 Estadísticas de Tráfico\n\n")
        f.write("BFS:\n")
        bfs_traffic = analytics_bfs.metrics['traffic_stats']
        f.write(f"  - Total de visitas: {bfs_traffic['total_visits']}\n")
        f.write(f"  - Zona más visitada: {bfs_traffic['most_visited_zone']}\n")
        f.write(f"  - Tráfico promedio: {bfs_traffic['mean_traffic']:.2f}\n\n")
        
        f.write("A*:\n")
        astar_traffic = analytics_astar.metrics['traffic_stats']
        f.write(f"  - Total de visitas: {astar_traffic['total_visits']}\n")
        f.write(f"  - Zona más visitada: {astar_traffic['most_visited_zone']}\n")
        f.write(f"  - Tráfico promedio: {astar_traffic['mean_traffic']:.2f}\n\n")
        
        f.write("2.2 Cuellos de Botella\n\n")
        f.write(f"BFS:  {len(analytics_bfs.metrics['bottlenecks'])} cuellos de botella\n")
        f.write(f"A*:   {len(analytics_astar.metrics['bottlenecks'])} cuellos de botella\n\n")
        
        # Conclusiones y recomendaciones
        f.write("\n3. CONCLUSIONES Y RECOMENDACIONES\n")
        f.write("-"*70 + "\n\n")
        
        if astar_metrics['avg_path_length'] < bfs_metrics['avg_path_length']:
            f.write("A* demostró ser más eficiente para encontrar caminos óptimos.\n")
            f.write("Se recomienda usar A* para minimizar la distancia recorrida.\n\n")
        else:
            f.write("BFS demostró ser suficientemente eficiente para este caso.\n")
            f.write("A* no ofrece mejoras significativas en este layout.\n\n")
        
        # Calcular cuál tiene mejor score
        if analytics_astar.metrics['efficiency_score'] > analytics_bfs.metrics['efficiency_score']:
            winner = "A*"
            score_diff = analytics_astar.metrics['efficiency_score'] - analytics_bfs.metrics['efficiency_score']
        else:
            winner = "BFS"
            score_diff = analytics_bfs.metrics['efficiency_score'] - analytics_astar.metrics['efficiency_score']
        
        f.write(f"Algoritmo recomendado: {winner}\n")
        f.write(f"Ventaja en score de eficiencia: {score_diff:.2f} puntos\n\n")
        
        f.write("Recomendaciones para optimización del layout:\n")
        f.write("  1. Redistribuir secciones de alto tráfico para balancear el flujo\n")
        f.write("  2. Considerar ampliar pasillos en zonas de cuello de botella\n")
        f.write("  3. Monitorear patrones de tráfico regularmente\n")
        f.write("  4. Evaluar la colocación de productos complementarios cerca\n\n")
        
        f.write("="*70 + "\n")
        f.write("FIN DEL REPORTE COMPARATIVO\n")
        f.write("="*70 + "\n")
    
    print("Reporte combinado generado: combined_report.txt")


if __name__ == "__main__":
    run_comparison_experiments(num_customers=50, seed=128)
    
    print("\n\n¡Análisis de Tarea B completado exitosamente!")
    print("Revisa los archivos generados para ver los resultados detallados.")
