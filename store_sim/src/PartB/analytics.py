"""
Módulo de análisis de datos para la simulación de tienda.
Contiene funciones para calcular métricas y generar insights.
"""
import numpy as np
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt


class StoreAnalytics:
    """Clase para analizar el rendimiento de la tienda."""
    
    def __init__(self, simulation, store):
        """
        Inicializa el analizador con los datos de la simulación.
        
        Args:
            simulation: Objeto Simulation con los datos de tráfico
            store: Objeto Store con la configuración de la tienda
        """
        self.sim = simulation
        self.store = store
        self.metrics = {}
    
    def calculate_traffic_stats(self) -> Dict:
        """
        Calcula estadísticas básicas del tráfico.
        
        Returns:
            Dict con estadísticas de tráfico
        """
        traffic = self.sim.traffic
        
        stats = {
            'total_visits': np.sum(traffic),
            'max_traffic': np.max(traffic),
            'min_traffic': np.min(traffic),
            'mean_traffic': np.mean(traffic),
            'std_traffic': np.std(traffic),
            'hotspot': np.unravel_index(np.argmax(traffic), traffic.shape)
        }
        
        # Calcular tráfico por zona
        zone_traffic = {}
        for name, pos in self.store.sections.items():
            row, col = pos
            zone_traffic[name] = int(traffic[row, col])
        
        stats['zone_traffic'] = zone_traffic
        stats['most_visited_zone'] = max(zone_traffic, key=zone_traffic.get) if zone_traffic else None
        
        self.metrics['traffic_stats'] = stats
        return stats
    
    def calculate_path_metrics(self, customers: List) -> Dict:
        """
        Calcula métricas relacionadas con los caminos de los clientes.
        
        Args:
            customers: Lista de objetos Customer
            
        Returns:
            Dict con métricas de caminos
        """
        path_lengths = []
        algorithms_used = {}
        
        for customer in customers:
            if hasattr(customer, 'path') and customer.path:
                path_lengths.append(len(customer.path))
                
                algo = customer.algorithm
                if algo not in algorithms_used:
                    algorithms_used[algo] = {'count': 0, 'total_length': 0}
                
                algorithms_used[algo]['count'] += 1
                algorithms_used[algo]['total_length'] += len(customer.path)
        
        metrics = {
            'total_customers': len(customers),
            'avg_path_length': np.mean(path_lengths) if path_lengths else 0,
            'max_path_length': max(path_lengths) if path_lengths else 0,
            'min_path_length': min(path_lengths) if path_lengths else 0,
            'algorithms': algorithms_used
        }
        
        # Calcular promedio por algoritmo
        for algo, data in algorithms_used.items():
            if data['count'] > 0:
                data['avg_length'] = data['total_length'] / data['count']
        
        self.metrics['path_metrics'] = metrics
        return metrics
    
    def identify_bottlenecks(self, threshold_percentile: float = 75) -> List[Tuple[int, int]]:
        """
        Identifica cuellos de botella en la tienda.
        
        Args:
            threshold_percentile: Percentil para considerar una zona como cuello de botella
            
        Returns:
            Lista de posiciones (row, col) que son cuellos de botella
        """
        traffic = self.sim.traffic
        threshold = np.percentile(traffic[traffic > 0], threshold_percentile)
        
        bottlenecks = []
        rows, cols = traffic.shape
        
        for r in range(rows):
            for c in range(cols):
                if traffic[r, c] >= threshold:
                    bottlenecks.append((r, c))
        
        self.metrics['bottlenecks'] = bottlenecks
        return bottlenecks
    
    def calculate_efficiency_score(self) -> float:
        """
        Calcula un score de eficiencia general de la tienda.
        
        Returns:
            Score de eficiencia (0-100)
        """
        # Factores a considerar:
        # 1. Distribución uniforme del tráfico (mejor si está balanceado)
        # 2. Longitud promedio de caminos (mejor si es corta)
        # 3. Número de cuellos de botella (mejor si hay menos)
        
        traffic = self.sim.traffic[self.sim.traffic > 0]
        
        # Score de distribución (basado en coeficiente de variación invertido)
        cv = np.std(traffic) / np.mean(traffic) if np.mean(traffic) > 0 else 1
        distribution_score = max(0, 100 - (cv * 50))
        
        # Score de longitud de caminos (normalizado)
        if 'path_metrics' in self.metrics:
            avg_length = self.metrics['path_metrics']['avg_path_length']
            max_possible_length = self.store.rows + self.store.cols
            path_score = max(0, 100 - (avg_length / max_possible_length * 100))
        else:
            path_score = 50
        
        # Score de cuellos de botella
        if 'bottlenecks' in self.metrics:
            bottleneck_ratio = len(self.metrics['bottlenecks']) / (self.store.rows * self.store.cols)
            bottleneck_score = max(0, 100 - (bottleneck_ratio * 200))
        else:
            bottleneck_score = 50
        
        # Promedio ponderado
        efficiency = (distribution_score * 0.3 + path_score * 0.5 + bottleneck_score * 0.2)
        
        self.metrics['efficiency_score'] = efficiency
        return efficiency
    
    def generate_report(self, output_path: str = "analytics_report.txt"):
        """
        Genera un reporte completo de análisis.
        
        Args:
            output_path: Ruta donde guardar el reporte
        """
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write("=" * 60 + "\n")
            f.write("REPORTE DE ANÁLISIS DE TIENDA URBANSTYLE\n")
            f.write("=" * 60 + "\n\n")
            
            # Estadísticas de tráfico
            if 'traffic_stats' in self.metrics:
                f.write("1. ESTADÍSTICAS DE TRÁFICO\n")
                f.write("-" * 60 + "\n")
                stats = self.metrics['traffic_stats']
                f.write(f"Total de visitas: {stats['total_visits']}\n")
                f.write(f"Tráfico máximo en una celda: {stats['max_traffic']}\n")
                f.write(f"Tráfico promedio: {stats['mean_traffic']:.2f}\n")
                f.write(f"Desviación estándar: {stats['std_traffic']:.2f}\n")
                f.write(f"Punto más concurrido: {stats['hotspot']}\n")
                f.write(f"Zona más visitada: {stats['most_visited_zone']}\n\n")
                
                f.write("Tráfico por zona:\n")
                for zone, traffic in sorted(stats['zone_traffic'].items(), 
                                           key=lambda x: x[1], reverse=True):
                    f.write(f"  - {zone}: {traffic} visitas\n")
                f.write("\n")
            
            # Métricas de caminos
            if 'path_metrics' in self.metrics:
                f.write("2. MÉTRICAS DE CAMINOS\n")
                f.write("-" * 60 + "\n")
                metrics = self.metrics['path_metrics']
                f.write(f"Total de clientes: {metrics['total_customers']}\n")
                f.write(f"Longitud promedio de camino: {metrics['avg_path_length']:.2f}\n")
                f.write(f"Camino más largo: {metrics['max_path_length']}\n")
                f.write(f"Camino más corto: {metrics['min_path_length']}\n\n")
                
                f.write("Comparación de algoritmos:\n")
                for algo, data in metrics['algorithms'].items():
                    f.write(f"  - {algo.upper()}:\n")
                    f.write(f"    Usos: {data['count']}\n")
                    f.write(f"    Longitud promedio: {data.get('avg_length', 0):.2f}\n")
                f.write("\n")
            
            # Cuellos de botella
            if 'bottlenecks' in self.metrics:
                f.write("3. CUELLOS DE BOTELLA\n")
                f.write("-" * 60 + "\n")
                bottlenecks = self.metrics['bottlenecks']
                f.write(f"Número de cuellos de botella: {len(bottlenecks)}\n")
                f.write(f"Posiciones: {bottlenecks[:10]}\n")  # Mostrar solo los primeros 10
                f.write("\n")
            
            # Score de eficiencia
            if 'efficiency_score' in self.metrics:
                f.write("4. SCORE DE EFICIENCIA\n")
                f.write("-" * 60 + "\n")
                score = self.metrics['efficiency_score']
                f.write(f"Score general: {score:.2f}/100\n")
                
                if score >= 80:
                    rating = "Excelente"
                elif score >= 60:
                    rating = "Bueno"
                elif score >= 40:
                    rating = "Regular"
                else:
                    rating = "Necesita mejoras"
                
                f.write(f"Calificación: {rating}\n\n")
            
            f.write("=" * 60 + "\n")
            f.write("FIN DEL REPORTE\n")
            f.write("=" * 60 + "\n")
        
        print(f"Reporte generado en: {output_path}")
    
    def plot_comparison(self, output_path: str = "algorithm_comparison.png"):
        """
        Genera gráficos de comparación entre algoritmos.
        
        Args:
            output_path: Ruta donde guardar el gráfico
        """
        if 'path_metrics' not in self.metrics:
            print("No hay métricas de caminos para graficar.")
            return
        
        algorithms = self.metrics['path_metrics']['algorithms']
        
        if not algorithms:
            print("No hay datos de algoritmos para graficar.")
            return
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Gráfico 1: Longitud promedio por algoritmo
        algos = list(algorithms.keys())
        avg_lengths = [algorithms[a].get('avg_length', 0) for a in algos]
        
        ax1.bar(algos, avg_lengths, color=['#3498db', '#e74c3c', '#2ecc71'][:len(algos)])
        ax1.set_xlabel('Algoritmo')
        ax1.set_ylabel('Longitud Promedio de Camino')
        ax1.set_title('Comparación de Algoritmos: Longitud de Camino')
        ax1.grid(axis='y', alpha=0.3)
        
        # Gráfico 2: Número de usos por algoritmo
        counts = [algorithms[a]['count'] for a in algos]
        
        ax2.bar(algos, counts, color=['#3498db', '#e74c3c', '#2ecc71'][:len(algos)])
        ax2.set_xlabel('Algoritmo')
        ax2.set_ylabel('Número de Clientes')
        ax2.set_title('Comparación de Algoritmos: Frecuencia de Uso')
        ax2.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_path)
        plt.close()
        
        print(f"Gráfico de comparación guardado en: {output_path}")
