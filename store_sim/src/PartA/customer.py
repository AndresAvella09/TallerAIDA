from typing import List, Tuple

class Customer:
    def __init__(self, store:Store, goal_section, algorithm = "bfs"):
        self.store = store
        self.position = store.entry
        self.goal = store.sections[goal_section]
        self.algorithm = algorithm
        self.finished = False

        try:
            self.path: List[Tuple[int,int]] = store.get_path(self.position, self.goal, method=self.algorithm)
        except Exception as e:
            print(f"[Customer] no tiene como llegar de {self.position} hasta {self.goal}: {e}")
            self.path = [self.position]
            self.finished = True

        self.step_index = 1 if len(self.path) > 1 else 0

    def update(self):
        if self. finished:
            return
        
        if self.step_index < len(self.path):
            self.position = self.path[self.step_index]
            self.step_index += 1
        else:
            self.finished = True