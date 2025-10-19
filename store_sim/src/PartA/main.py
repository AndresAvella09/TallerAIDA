# game_interactive.py
import pygame
from typing import Tuple
from .store import Store
from .simulation import Simulation


class Slider:
    def __init__(self, x: int, y: int, width: int, min_val: int, max_val: int, start_val: int, label: str):
        self.rect = pygame.Rect(x, y, width, 8)
        self.knob_radius = 10
        self.min_val = min_val
        self.max_val = max_val
        self.value = max(min(start_val, max_val), min_val)
        self.label = label
        self.dragging = False
        self._update_knob_from_value()

    def _update_knob_from_value(self):
        t = (self.value - self.min_val) / (self.max_val - self.min_val)
        self.knob_x = int(self.rect.x + t * self.rect.w)
        self.knob_y = self.rect.y + self.rect.h // 2

    def _update_value_from_mouse(self, mx: int):
        t = (mx - self.rect.x) / max(1, self.rect.w)
        t = max(0.0, min(1.0, t))
        self.value = int(round(self.min_val + t * (self.max_val - self.min_val)))
        self._update_knob_from_value()

    def handle_event(self, event: pygame.event.EventType):
        if event.type == pygame.MOUSEBUTTONDOWN:
            if (event.pos[0] - self.knob_x) ** 2 + (event.pos[1] - self.knob_y) ** 2 <= self.knob_radius ** 2:
                self.dragging = True
        elif event.type == pygame.MOUSEBUTTONUP:
            self.dragging = False
        elif event.type == pygame.MOUSEMOTION and self.dragging:
            self._update_value_from_mouse(event.pos[0])

    def draw(self, surf: pygame.Surface, font: pygame.font.Font):
        pygame.draw.rect(surf, (180, 180, 180), self.rect, border_radius=4)
        pygame.draw.circle(surf, (30, 144, 255), (self.knob_x, self.knob_y), self.knob_radius)
        label_s = font.render(f"{self.label}: {self.value}", True, (20, 20, 20))
        surf.blit(label_s, (self.rect.x, self.rect.y - 26))


class Button:
    def __init__(self, x: int, y: int, w: int, h: int, text: str):
        self.rect = pygame.Rect(x, y, w, h)
        self.text = text
        self.pressed = False

    def handle_event(self, event: pygame.event.EventType) -> bool:
        if event.type == pygame.MOUSEBUTTONDOWN and self.rect.collidepoint(event.pos):
            self.pressed = True
            return True
        return False

    def draw(self, surf: pygame.Surface, font: pygame.font.Font):
        color = (50, 160, 80) if not self.pressed else (40, 120, 60)
        pygame.draw.rect(surf, color, self.rect, border_radius=6)
        pygame.draw.rect(surf, (20, 80, 40), self.rect, 2, border_radius=6)
        txt = font.render(self.text, True, (255, 255, 255))
        tx = self.rect.centerx - txt.get_width() // 2
        ty = self.rect.centery - txt.get_height() // 2
        surf.blit(txt, (tx, ty))


class GameInteractive:
    def __init__(
        self,
        customers: int = 8,
        categories: int = 10,
        cell_size: int = 80,
        fps: int = 30,
        control_panel_width: int = 320,
        seed_:int = 42
    ):
        pygame.init()
        pygame.display.set_caption("Simulación interactiva — AIDA")

        self.cell_size = cell_size
        self.control_width = control_panel_width

        # Inicial
        self.categories_ = categories
        self.store = Store(categories=self.categories_, seed = seed_)
        self.customers_ = customers
        self.fps = fps

        # Mezcla inicial: 70% BFS, 30% A*
        self.bfs_percent = 70

        # crea sim inicial con esa mezcla
        algo_mix = {'bfs': self.bfs_percent / 100.0, 'astar': 1.0 - self.bfs_percent / 100.0}
        self.sim = Simulation(self.store, num_customers=self.customers_, algorithm_mix=algo_mix)

        sim_w = self.store.cols * self.cell_size
        sim_h = self.store.rows * self.cell_size
        win_w = self.control_width + sim_w
        win_h = max(sim_h, 420)
        self.screen = pygame.display.set_mode((win_w, win_h))
        self.clock = pygame.time.Clock()

        self.font = pygame.font.SysFont("Arial", 18)

        # Sliders
        sx, sy, sw = 24, 60, self.control_width - 48
        self.slider_categories = Slider(sx, sy + 0 * 70, sw, 1, 30, self.categories_, "Categorías")
        self.slider_customers  = Slider(sx, sy + 1 * 70, sw, 1, 200, self.customers_, "Personas")
        self.slider_fps        = Slider(sx, sy + 2 * 70, sw, 1, 20, self.fps, "FPS")
        self.slider_bfs        = Slider(sx, sy + 3 * 70, sw, 0, 100, self.bfs_percent, "% BFS (resto A*)")
        self.slider_seed       = Slider(sx, sy + 4.5 * 70, sw, 0, 100, self.bfs_percent, "Semilla")
        # Botón
        self.btn_start = Button(sx, sy + 5 * 70 + 10, sw, 44, "Iniciar simulación")

        self.sim_origin = (self.control_width, 0)

        print(f"[DEBUG] Ventana {win_w}x{win_h}, panel {self.control_width}px, sim {sim_w}x{sim_h}")

    # ---------- Paletas por algoritmo ----------
    @staticmethod
    def _color_for_bfs(intensity: float) -> Tuple[int, int, int]:
        # 0..1 → blanco a azul
        intensity = max(0.0, min(1.0, intensity))
        r = int(255 * (1 - intensity))
        g = int(255 * (1 - intensity))
        b = 255
        return (r, g, b)

    @staticmethod
    def _color_for_astar(intensity: float) -> Tuple[int, int, int]:
        # 0..1 → blanco a rojo
        intensity = max(0.0, min(1.0, intensity))
        r = 255
        g = int(255 * (1 - intensity))
        b = int(255 * (1 - intensity))
        return (r, g, b)

    @staticmethod
    def _blend_colors_weighted(c1: Tuple[int, int, int], w1: float,
                               c2: Tuple[int, int, int], w2: float) -> Tuple[int, int, int]:
        total = max(1e-9, w1 + w2)
        r = int((c1[0] * w1 + c2[0] * w2) / total)
        g = int((c1[1] * w1 + c2[1] * w2) / total)
        b = int((c1[2] * w1 + c2[2] * w2) / total)
        return (r, g, b)

    def _cell_color_mixed(self, row: int, col: int) -> Tuple[int, int, int]:
        """
        Si Simulation expone traffic_by_algo['bfs'] y ['astar'], mezclamos por proporción.
        Si no existe, caemos al total traffic y coloreamos con paleta BFS por defecto.
        """
        # fallback si tu Simulation NO tiene traffic_by_algo:
        traffic_by_algo = getattr(self.sim, "traffic_by_algo", None)

        if traffic_by_algo is None:
            # Fallback: usa total traffic con paleta BFS
            v = int(self.sim.traffic[row, col])
            cap = max(1, self.customers_)
            t = 0 if v <= 0 else min(1.0, v / cap)
            return self._color_for_bfs(t)

        # Si hay tráfico separado por algoritmo:
        v_bfs = int(traffic_by_algo.get('bfs', self.sim.traffic)[row, col])
        v_ast = int(traffic_by_algo.get('astar', self.sim.traffic)[row, col])

        if v_bfs == 0 and v_ast == 0:
            return (255, 255, 255)

        cap = max(1, self.customers_)
        t_bfs = min(1.0, v_bfs / cap)
        t_ast = min(1.0, v_ast / cap)

        color_bfs = self._color_for_bfs(t_bfs)
        color_ast = self._color_for_astar(t_ast)

        return self._blend_colors_weighted(color_bfs, float(v_bfs), color_ast, float(v_ast))

    # ---- Dibujo sim ----
    def draw_heatmap(self, surf: pygame.Surface) -> None:
        for row in range(self.store.rows):
            for col in range(self.store.cols):
                node = self.store.graph.nodes.get((row, col), {})
                if node.get("zone", "pasillo") != "pasillo":
                    continue
                color = self._cell_color_mixed(row, col)
                rect = pygame.Rect(
                    self.sim_origin[0] + col * self.cell_size,
                    self.sim_origin[1] + row * self.cell_size,
                    self.cell_size, self.cell_size
                )
                pygame.draw.rect(surf, color, rect)

    def draw_grid(self, surf: pygame.Surface):
        for row in range(self.store.rows):
            for col in range(self.store.cols):
                rect = pygame.Rect(
                    self.sim_origin[0] + col * self.cell_size,
                    self.sim_origin[1] + row * self.cell_size,
                    self.cell_size, self.cell_size
                )
                node = self.store.graph.nodes.get((row, col), {})
                zone = node.get("zone", "pasillo")
                if zone != "pasillo":
                    pygame.draw.rect(surf, (50, 150, 50), rect)
                pygame.draw.rect(surf, (180, 180, 180), rect, 1)

    def draw_customers(self, surf: pygame.Surface):
        for c in self.sim.customers:
            px = self.sim_origin[0] + c.position[1] * self.cell_size + self.cell_size // 2
            py = self.sim_origin[1] + c.position[0] * self.cell_size + self.cell_size // 2

            # Normaliza y colorea por algoritmo
            algo = c.get_algorithm().lower()
            if algo in ("bfs",):
                color = (0, 100, 255)      # azul para BFS
            elif algo in ("astar", "a*"):
                color = (255, 80, 80)      # rojo para A*
            else:
                color = (80, 80, 80)       # gris si llega algo raro

            # [DEBUG] descomenta si quieres verificar qué algoritmo trae cada customer
            # print(f"[DEBUG] customer algo={algo} pos={c.position}")

            pygame.draw.circle(surf, color, (px, py), 10)


    # ---- Panel ----
    def draw_panel(self, surf: pygame.Surface):
        pygame.draw.rect(surf, (245, 245, 245), pygame.Rect(0, 0, self.control_width, surf.get_height()))
        pygame.draw.line(surf, (200, 200, 200), (self.control_width, 0), (self.control_width, surf.get_height()), 2)

        title = self.font.render("Controles:", True, (20, 20, 20))
        surf.blit(title, (24, 5))

        self.slider_categories.draw(surf, self.font)
        self.slider_customers.draw(surf, self.font)
        self.slider_fps.draw(surf, self.font)
        self.slider_bfs.draw(surf, self.font)
        self.slider_seed.draw(surf, self.font)

        bfs = self.slider_bfs.value
        astar = 100 - bfs
        info = self.font.render(f"Mezcla: BFS {bfs}%  /  A* {astar}%", True, (60, 60, 60))
        surf.blit(info, (24, self.slider_bfs.rect.y + 18))

        self.btn_start.draw(surf, self.font)

    # ---- Lógica ----
    def _apply_settings_and_restart(self):
        self.categories_ = self.slider_categories.value
        self.customers_  = self.slider_customers.value
        self.fps         = self.slider_fps.value
        self.bfs_percent = self.slider_bfs.value
        self.seed        = self.slider_seed.value

        bfs_w   = self.bfs_percent / 100.0
        astar_w = 1.0 - bfs_w
        algo_mix = {'bfs': bfs_w, 'astar': astar_w}

        print(f"[DEBUG] Reiniciar con categorías={self.categories_}, personas={self.customers_}, "
              f"fps={self.fps}, mix={algo_mix}")

        self.store = Store(categories=self.categories_, seed = self.seed)
        self.sim = Simulation(self.store, num_customers=self.customers_, algorithm_mix=algo_mix)

        sim_w = self.store.cols * self.cell_size
        sim_h = self.store.rows * self.cell_size
        win_w = self.control_width + sim_w
        win_h = max(sim_h, self.screen.get_height())
        self.screen = pygame.display.set_mode((win_w, win_h))
        print(f"[DEBUG] Nueva ventana {win_w}x{win_h}")

    def run(self):
        running = True
        while running:
            for e in pygame.event.get():
                if e.type == pygame.QUIT:
                    running = False
                self.slider_categories.handle_event(e)
                self.slider_customers.handle_event(e)
                self.slider_fps.handle_event(e)
                self.slider_bfs.handle_event(e)
                self.slider_seed.handle_event(e)
                if self.btn_start.handle_event(e):
                    self._apply_settings_and_restart()
                if e.type == pygame.MOUSEBUTTONUP:
                    self.btn_start.pressed = False

            self.screen.fill((255, 255, 255))

            self.sim.update()
            self.sim.customers = [c for c in self.sim.customers if not c.finished]

            self.draw_panel(self.screen)
            self.draw_heatmap(self.screen)
            self.draw_grid(self.screen)
            self.draw_customers(self.screen)

            pygame.display.flip()
            self.clock.tick(self.fps)

        pygame.quit()


if __name__ == "__main__":
    # Minimal ejecutable con UI
    GameInteractive(
        customers=8,
        categories=10,
        cell_size=80,
        fps=30,
        control_panel_width=320,
        seed_=42
    ).run()
