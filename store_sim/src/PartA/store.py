from typing import List, Tuple, TYPE_CHECKING
from pathlib import Path
import csv
import random
import networkx as nx
import pandas as pd
if TYPE_CHECKING:
    from .customer import Customer

class Store:
    def __init__(self,
                 rows:int = 8,
                 cols:int=6,
                 seed: int | None = None,
                 categories:int = 4):

        self.rows = rows
        self.cols = cols
        self.graph = nx.grid_2d_graph(rows, cols)
        self.entry = (0, 0)
        self.exit = (rows - 1, cols - 1)
        reserved = {self.entry, self.exit}
        self.categories:list = list(range(categories))

        # Usar ruta relativa al archivo actual
        #csv_path = Path(__file__).parent.parent.parent / "data" / "retail_sales_dataset.csv"
        #df = pd.read_csv(csv_path)
        # = df['Product Category'].unique().tolist()
        #seen = set()

        available_positions = [(r, c) for r in range(rows) for c in range(cols) if (r, c) not in reserved]

        if seed is not None:
            random.seed(seed)

        n_to_place = min(len(self.categories), len(available_positions))
        
        chosen_positions = random.sample(available_positions, k=n_to_place)

        self.sections = {}
        for cat, pos in zip(self.categories[:n_to_place], chosen_positions):
            key = str(cat)
            self.sections[key] = pos
            self.graph.nodes[pos]["zone"] = key
            self.graph.nodes[pos]["label"] = cat

        self._init_sections()

    def _init_sections(self):
        for node in self.graph.nodes:
            self.graph.nodes[node]["zone"] = "pasillo"
        for name, pos in self.sections.items():
            self.graph.nodes[pos]["zone"] = name
            
    


    def _resolve_goal_pos(self, goal:str) -> Tuple[int, int]:
        if isinstance(goal, tuple):
            return goal
        if goal in self.sections:
            return self.sections[goal]
        for k, pos in self.sections.items():
            if self.graph.nodes[pos].get("label", "").lower() == str(goal).lower():
                return pos
        raise KeyError(f"Goal desconocido: {goal}")



    def get_path(
        self,
        start: Tuple[int,int],
        goal:str,
        method: str = "bfs",
        forbid_other_sections: bool = True
        ) -> List[Tuple[int, int]]: 
        
        goal_pos = self._resolve_goal_pos(goal)

        G = self.graph.copy()

        if forbid_other_sections:
            for sec_key, sec_pos in self.sections.items():
                # no eliminar la sección objetivo, ni entry/exit
                if sec_pos == goal_pos:
                    continue
                if sec_pos == self.entry or sec_pos == self.exit:
                    continue
                if sec_pos in G:
                    G.remove_node(sec_pos)

        try:
            if method.lower() in ("astar", "a*"):
                def manhattan(a, b):
                    return abs(a[0] - b[0]) + abs(a[1] - b[1])
                return nx.astar_path(G, start, goal_pos, heuristic=manhattan)
            else:
                return nx.shortest_path(G, source=start, target=goal_pos)
        except nx.NetworkXNoPath:
            return []
        
    def add_customer(self, customer: "Customer", debug:bool = False) -> None:
        self.customers.append(customer)
        if debug:
            print(f"[DEBUG] Customer agregado. Total: {len(self.customers)}")