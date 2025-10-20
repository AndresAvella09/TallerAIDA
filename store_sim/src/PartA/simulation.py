import numpy as np
from .customer import Customer
import random
from collections import deque

class Simulation:
    def __init__(self, store, num_customers=3, algorithm_mix=None, spawn_interval:int = 2):
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
        self.pending = deque()
        self.tick:int = 0
        self.spawn_interval = spawn_interval
        self.traffic = np.zeros((store.rows, store.cols), dtype=int)
        self.traffic_by_algo = {
            "bfs":   np.zeros_like(self.traffic),
            "astar": np.zeros_like(self.traffic),
        }
        self.algorithm_mix = algorithm_mix or {'bfs': 1.0, 'astar': 0}
        self._prepare_pending(num_customers)
        
    def _increment_traffic(self, position, algo_name: str):
        r, c = position
        self.traffic[r, c] += 1

        # Normaliza el nombre del algoritmo a las claves del dict
        algo = algo_name.lower()
        key = "astar" if ("astar" in algo or "a*" in algo) else "bfs"
        self.traffic_by_algo[key][r, c] += 1
        print(f"[DEBUG] traffic cell=({r},{c}) total={self.traffic[r,c]} {key}={self.traffic_by_algo[key][r,c]}")

        
    def generar_vector(proporciones: dict[str, float], total: int) -> list[str]:
        # calcula cuántos de cada uno
        conteos = {k: int(round(total * v)) for k, v in proporciones.items()}

        # ajusta si por redondeo no suma el total exacto
        diferencia = total - sum(conteos.values())
        if diferencia != 0:
            # corrige añadiendo o quitando unidades del más representado
            clave_max = max(conteos, key=conteos.get)
            conteos[clave_max] += diferencia

        # crea la lista con las repeticiones
        etiquetas = [k for k, n in conteos.items() for _ in range(n)]

        # mezcla aleatoriamente
        random.shuffle(etiquetas)
        return etiquetas
    
    def _prepare_pending(self, n:int) -> None:
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
            algo = algorithms[i] if i < len(algorithms) else 'bfs'
            self.pending.append((section, algo))
    
    def _spawn_if_needed(self) -> None:
        if self.pending and (self.tick % self.spawn_interval == 0):
            section, algo = self.pending.popleft()
            customer = Customer(self.store, section, algorithm=algo)
            self.customers.append(customer)
            self.all_customers.append(customer)

        
    
    def update(self) -> None:
        self._spawn_if_needed()
        for c in self.customers:
            if not c.finished:
                c.update()
                x, y = c.position
                self.traffic[x, y] += 1
                self._increment_traffic(c.position, c.get_algorithm())
        self.tick +=1
    
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
   
