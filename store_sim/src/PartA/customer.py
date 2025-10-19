from typing import List, Tuple, TYPE_CHECKING

if TYPE_CHECKING:
    from .store import Store

class Customer:
    def __init__(self, store: "Store",
                 goal_section:str,
                 algorithm:str = "bfs",
                 start_delay_ticks:int = 1):
        self.store = store
        self.position: Tuple[int,int] = store.entry
        self.goal:Tuple[int,int] = store.sections[goal_section]
        self.goal_str = goal_section
        self.algorithm = algorithm
        self.finished = False
        self.step_index = 0
        self.wait_ticks_remaining = start_delay_ticks
        try:
            self.path: List[Tuple[int,int]] = store.get_path(self.position, self.goal, method=self.algorithm)
        except Exception as e:
            print(f"[Customer] no tiene como llegar de {self.position} hasta {self.goal}: {e}")
            self.path = [self.position]
            self.finished = True

        

    def update(self):
        if self.finished:
            return
        
        if self.step_index < len(self.path):
            self.position = self.path[self.step_index]
            self.step_index += 1
        else:
            self.finished = True
            
    def get_algorithm(self) -> str:
        return self.algorithm