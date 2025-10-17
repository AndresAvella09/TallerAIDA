from pathlib import Path
import csv
import random
import networkx as nx
import pandas as pd

class Store:
    def __init__(self, rows=8, cols=6, seed: int | None = None):

        self.rows = rows
        self.cols = cols
        self.graph = nx.grid_2d_graph(rows, cols)

        self.entry = (0, 0)
        self.exit = (rows - 1, cols - 1)
        reserved = {self.entry, self.exit}

        csv_path = r"C:\Users\Diego Andres\Documents\Universidad\Semestres\8 Semestre\AIDA_M\TallerAIDA\store_sim\data\retail_sales_dataset.csv"
        df = pd.read_csv(csv_path)
        categories = df['Product Category'].unique().tolist()
        #seen = set()

        available_positions = [(r, c) for r in range(rows) for c in range(cols) if (r, c) not in reserved]

        if seed is not None:
            random.seed(seed)

        n_to_place = min(len(categories), len(available_positions))

        chosen_positions = random.sample(available_positions, k=n_to_place)

        self.sections = {}
        for cat, pos in zip(categories[:n_to_place], chosen_positions):
            key = cat.lower().replace(" ", "_")
            self.sections[key] = pos
            self.graph.nodes[pos]["zone"] = key
            self.graph.nodes[pos]["label"] = cat

        self._init_sections()

    def _init_sections(self):
        for node in self.graph.nodes:
            self.graph.nodes[node]["zone"] = "pasillo"
        for name, pos in self.sections.items():
            self.graph.nodes[pos]["zone"] = name
    


    def _resolve_goal_pos(self, goal):
        if isinstance(goal, tuple):
            return goal
        if goal in self.sections:
            return self.sections[goal]
        for k, pos in self.sections.items():
            if self.graph.nodes[pos].get("label", "").lower() == str(goal).lower():
                return pos
        raise KeyError(f"Goal desconocido: {goal}")



    def get_path(self, start, goal, method="bfs", forbid_other_sections: bool = True):
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