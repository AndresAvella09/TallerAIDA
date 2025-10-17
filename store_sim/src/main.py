import pygame
from store import Store
from simulation import Simulation
from heatmap import create_heatmap

class Game:
    def __init__(self, cell_size = 80):
        pygame.init()
        self.store = Store()
        self.sim = Simulation(self.store, num_customers = 3)
        self.cell_size = cell_size
        self.screen = pygame.display.set_mode((self.store.cols*cell_size, self.store.rows*cell_size))
        self.clock = pygame.time.Clock()

    def draw_grid(self):
        for row in range(self.store.rows):
            for col in range(self.store.cols):
                rect = pygame.Rect(col * self.cell_size, row * self.cell_size, self.cell_size, self.cell_size)
                
                node = self.store.graph.nodes.get((row, col), {})
                zone = node.get("zone", "pasillo")

                if zone != "pasillo":
                    pygame.draw.rect(self.screen, (50, 150, 50), rect)

                pygame.draw.rect(self.screen, (200, 200, 200), rect, 1)

    def draw_customers(self):
        for c in self.sim.customers:
            px = c.position[1]*self.cell_size + self.cell_size//2
            py = c.position[0]*self.cell_size + self.cell_size//2
            pygame.draw.circle(self.screen, (0, 100, 255), (px, py), 10)
    
    def run(self):
        running = True
        while running:
            for e in pygame.event.get():
                if e.type == pygame.QUIT:
                    running = False
            self.screen.fill((30, 30, 30))
            self.sim.update()
            
            self.sim.customers = [c for c in self.sim.customers if not c.finished]
            
            self.draw_grid()
            self.draw_customers()
            pygame.display.flip()
            self.clock.tick(1)
        
        # Generar el mapa de calor al final de la simulación
        create_heatmap(self.sim.traffic, self.store)
        
        pygame.quit()

if __name__ == "__main__":
    Game().run()