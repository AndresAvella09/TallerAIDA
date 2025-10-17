import numpy as np
from .customer import Customer
import random

class Simulation:
    def __init__(self, store, num_customers=3, algorithm_mix=None):
        """
        Inicializa la simulación.
        
        Args:
            store: Objeto Store
            num_customers: Número de clientes a crear
            algorithm_mix: Dict con proporción de algoritmos, ej: {'bfs': 0.5, 'astar': 0.5}
                          Si es None, todos usan 'bfs'
        """
        self.store = store
        self.customers = []
        self.all_customers = []  # Guarda todos los clientes creados (incluso los terminados)
        self.traffic = np.zeros((store.rows, store.cols), dtype=int)
        self.algorithm_mix = algorithm_mix or {'bfs': 1.0}
        self.create_customers(num_customers)
    
    def create_customers(self, n):
        sections = list(self.store.sections.keys())
        
        # Preparar lista de algoritmos según la mezcla
        algorithms = []
        for algo, proportion in self.algorithm_mix.items():
            count = int(n * proportion)
            algorithms.extend([algo] * count)
        
        # Ajustar si hay diferencia por redondeo
        while len(algorithms) < n:
            algorithms.append(list(self.algorithm_mix.keys())[0])
        
        random.shuffle(algorithms)

        for i in range(n):
            section = random.choice(sections)
            algorithm = algorithms[i] if i < len(algorithms) else 'bfs'
            customer = Customer(self.store, section, algorithm=algorithm)
            self.customers.append(customer)
            self.all_customers.append(customer)
    
    def update(self):
        for c in self.customers:
            if not c.finished:
                c.update()
                x, y = c.position
                self.traffic[x, y] += 1
    
    def add_customer(self, section=None, algorithm='bfs'):
        """
        Añade un nuevo cliente a la simulación.
        
        Args:
            section: Sección objetivo (si es None, se elige al azar)
            algorithm: Algoritmo de búsqueda a usar
        """
        if section is None:
            sections = list(self.store.sections.keys())
            section = random.choice(sections)
        
        customer = Customer(self.store, section, algorithm=algorithm)
        self.customers.append(customer)
        self.all_customers.append(customer)
    
    def reset_traffic(self):
        """Resetea la matriz de tráfico."""
        self.traffic = np.zeros((self.store.rows, self.store.cols), dtype=int)
   
