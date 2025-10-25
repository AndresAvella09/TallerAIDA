import pygame
import pandas as pd
import numpy as np
import os
from pathlib import Path
from stable_baselines3 import PPO
from .dynamic_pricing_rl import DynamicPricingEnv  # Importación relativa explícita
import sys

# Inicializar Pygame
pygame.init()

# Configuración de la ventana
WIDTH, HEIGHT = 1200, 700
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("UrbanStyle - Dynamic Pricing RL Visualization")

# Colores
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
BLUE = (52, 152, 219)
GREEN = (46, 204, 113)
RED = (231, 76, 60)
GRAY = (189, 195, 199)
DARK_GRAY = (52, 73, 94)
ORANGE = (230, 126, 34)
PURPLE = (155, 89, 182)

# Fuentes
font_large = pygame.font.Font(None, 48)
font_medium = pygame.font.Font(None, 32)
font_small = pygame.font.Font(None, 24)

class Slider:
    def __init__(self, x, y, w, h, min_val, max_val, initial_val, label):
        self.rect = pygame.Rect(x, y, w, h)
        self.min_val = min_val
        self.max_val = max_val
        self.value = initial_val
        self.label = label
        self.dragging = False
        
    def draw(self, surface):
        # Dibujar línea del slider
        pygame.draw.rect(surface, GRAY, self.rect, 2)
        
        # Calcular posición del handle
        ratio = (self.value - self.min_val) / (self.max_val - self.min_val)
        handle_x = self.rect.x + int(ratio * self.rect.width)
        handle_rect = pygame.Rect(handle_x - 10, self.rect.y - 5, 20, self.rect.height + 10)
        
        # Dibujar handle
        pygame.draw.rect(surface, BLUE, handle_rect)
        
        # Dibujar label
        label_text = font_small.render(f"{self.label}: ${self.value:.2f}", True, BLACK)
        surface.blit(label_text, (self.rect.x, self.rect.y - 30))
        
    def handle_event(self, event):
        if event.type == pygame.MOUSEBUTTONDOWN:
            handle_x = self.rect.x + int((self.value - self.min_val) / (self.max_val - self.min_val) * self.rect.width)
            handle_rect = pygame.Rect(handle_x - 10, self.rect.y - 5, 20, self.rect.height + 10)
            if handle_rect.collidepoint(event.pos):
                self.dragging = True
                
        elif event.type == pygame.MOUSEBUTTONUP:
            self.dragging = False
            
        elif event.type == pygame.MOUSEMOTION and self.dragging:
            # Actualizar valor basado en posición del mouse
            mouse_x = event.pos[0]
            ratio = (mouse_x - self.rect.x) / self.rect.width
            ratio = max(0, min(1, ratio))
            self.value = self.min_val + ratio * (self.max_val - self.min_val)

class Button:
    def __init__(self, x, y, w, h, text, color):
        self.rect = pygame.Rect(x, y, w, h)
        self.text = text
        self.color = color
        self.hover = False
        
    def draw(self, surface):
        color = tuple(min(255, c + 30) for c in self.color) if self.hover else self.color
        pygame.draw.rect(surface, color, self.rect, border_radius=10)
        pygame.draw.rect(surface, BLACK, self.rect, 2, border_radius=10)
        
        text_surface = font_small.render(self.text, True, WHITE)
        text_rect = text_surface.get_rect(center=self.rect.center)
        surface.blit(text_surface, text_rect)
        
    def handle_event(self, event):
        if event.type == pygame.MOUSEMOTION:
            self.hover = self.rect.collidepoint(event.pos)
        elif event.type == pygame.MOUSEBUTTONDOWN:
            if self.rect.collidepoint(event.pos):
                return True
        return False

def draw_revenue_plot(surface, revenues, x, y, w, h):
    """Dibuja un gráfico de revenue en tiempo real"""
    # Fondo del gráfico
    pygame.draw.rect(surface, WHITE, (x, y, w, h))
    pygame.draw.rect(surface, BLACK, (x, y, w, h), 2)
    
    if len(revenues) < 2:
        return
    
    # Encontrar valores max y min para escalar
    max_rev = max(revenues) if revenues else 1
    min_rev = min(revenues) if revenues else 0
    range_rev = max_rev - min_rev if max_rev != min_rev else 1
    
    # Dibujar líneas de referencia
    for i in range(5):
        y_pos = y + h - (i * h / 4)
        pygame.draw.line(surface, GRAY, (x, y_pos), (x + w, y_pos), 1)
        value = min_rev + (range_rev * i / 4)
        label = font_small.render(f"${value:.0f}", True, BLACK)
        surface.blit(label, (x - 60, y_pos - 10))
    
    # Dibujar puntos y líneas
    points = []
    for i, rev in enumerate(revenues):
        px = x + (i / max(1, len(revenues) - 1)) * w
        py = y + h - ((rev - min_rev) / range_rev) * h
        points.append((px, py))
    
    # Dibujar línea conectando puntos
    if len(points) > 1:
        pygame.draw.lines(surface, BLUE, False, points, 3)
    
    # Dibujar puntos
    for point in points[-20:]:  # Solo últimos 20 puntos para no saturar
        pygame.draw.circle(surface, RED, (int(point[0]), int(point[1])), 4)
    
    # Título
    title = font_medium.render("Revenue per Step", True, BLACK)
    surface.blit(title, (x + w // 2 - title.get_width() // 2, y - 40))

def draw_price_history(surface, prices, x, y, w, h, static_price):
    """Dibuja un gráfico del historial de precios"""
    # Fondo del gráfico
    pygame.draw.rect(surface, WHITE, (x, y, w, h))
    pygame.draw.rect(surface, BLACK, (x, y, w, h), 2)
    
    if len(prices) < 2:
        return
    
    # Encontrar valores max y min para escalar
    all_prices = prices + [static_price]
    max_price = max(all_prices)
    min_price = min(all_prices)
    range_price = max_price - min_price if max_price != min_price else 1
    
    # Línea de precio estático
    static_y = y + h - ((static_price - min_price) / range_price) * h
    pygame.draw.line(surface, RED, (x, static_y), (x + w, static_y), 2)
    static_label = font_small.render(f"Static: ${static_price:.2f}", True, RED)
    surface.blit(static_label, (x + w - 120, static_y - 20))
    
    # Dibujar puntos y líneas de precio RL
    points = []
    for i, price in enumerate(prices):
        px = x + (i / max(1, len(prices) - 1)) * w
        py = y + h - ((price - min_price) / range_price) * h
        points.append((px, py))
    
    # Dibujar línea conectando puntos
    if len(points) > 1:
        pygame.draw.lines(surface, GREEN, False, points, 3)
    
    # Título
    title = font_medium.render("Price History", True, BLACK)
    surface.blit(title, (x + w // 2 - title.get_width() // 2, y - 40))

def main():
    # Obtener ruta absoluta al archivo de datos
    current_dir = Path(__file__).parent
    data_path = current_dir.parent.parent.parent / "store_sim" / "data" / "retail_sales_dataset.csv"
    
    # Verificar que el archivo existe
    if not data_path.exists():
        # Intentar ruta alternativa
        data_path = Path("store_sim/data/retail_sales_dataset.csv")
        if not data_path.exists():
            print(f"ERROR: No se encontró el archivo de datos en: {data_path}")
            print("Asegúrate de que existe: store_sim/data/retail_sales_dataset.csv")
            return
    
    # Cargar datos y modelo
    print("Cargando datos y modelo...")
    print(f"Datos desde: {data_path}")
    df = pd.read_csv(data_path)
    
    try:
        model = PPO.load("ppo_dynamic_pricing")
        print("Modelo cargado exitosamente")
    except:
        print("No se encontró el modelo. Ejecuta training.py primero.")
        return
    
    # Precio inicial y estático
    static_price = float(df["Price per Unit"].mean())
    initial_price = static_price
    
    # Crear slider para precio inicial
    price_slider = Slider(50, HEIGHT - 80, 300, 10, 20, 550, initial_price, "Initial Price")
    
    # Crear botones
    reset_button = Button(400, HEIGHT - 95, 150, 40, "Reset", BLUE)
    start_button = Button(570, HEIGHT - 95, 150, 40, "Start/Pause", GREEN)
    step_button = Button(740, HEIGHT - 95, 150, 40, "Step", ORANGE)
    
    # Estado del juego
    env = None
    obs = None
    revenues = []
    prices = []
    current_step = 0
    running = True
    paused = True
    episode_revenue = 0
    static_revenue = 0
    
    clock = pygame.time.Clock()
    
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            
            # Manejar slider
            price_slider.handle_event(event)
            
            # Manejar botones
            if reset_button.handle_event(event):
                # Reset environment
                initial_price = price_slider.value
                env = DynamicPricingEnv(
                    data=df,
                    window_size=10,
                    max_steps=100,
                    price_min=20,
                    price_max=550,
                    seed=None
                )
                obs, _ = env.reset()
                # Ajustar precio inicial
                env.price = initial_price
                obs = env._get_state()
                
                revenues = []
                prices = []
                current_step = 0
                episode_revenue = 0
                static_revenue = 0
                paused = True
                print(f"Environment reset con precio inicial: ${initial_price:.2f}")
            
            if start_button.handle_event(event):
                if env is None:
                    # Inicializar por primera vez
                    initial_price = price_slider.value
                    env = DynamicPricingEnv(
                        data=df,
                        window_size=10,
                        max_steps=100,
                        price_min=20,
                        price_max=550,
                        seed=None
                    )
                    obs, _ = env.reset()
                    env.price = initial_price
                    obs = env._get_state()
                paused = not paused
                print("Paused" if paused else "Running")
            
            if step_button.handle_event(event):
                if env is None:
                    # Inicializar por primera vez
                    initial_price = price_slider.value
                    env = DynamicPricingEnv(
                        data=df,
                        window_size=10,
                        max_steps=100,
                        price_min=20,
                        price_max=550,
                        seed=None
                    )
                    obs, _ = env.reset()
                    env.price = initial_price
                    obs = env._get_state()
                
                # Ejecutar un solo step
                if not paused or True:  # Permitir step incluso si está pausado
                    action, _ = model.predict(obs, deterministic=True)
                    obs, reward, done, truncated, info = env.step(action)
                    
                    revenues.append(info['revenue'])
                    prices.append(info['price'])
                    episode_revenue += reward
                    static_revenue += static_price * df["Quantity"].mean()
                    current_step += 1
                    
                    if done or truncated:
                        print(f"Episode terminado. Revenue RL: ${episode_revenue:.2f}, Static: ${static_revenue:.2f}")
                        paused = True
        
        # Ejecutar step automáticamente si no está pausado
        if not paused and env is not None:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, truncated, info = env.step(action)
            
            revenues.append(info['revenue'])
            prices.append(info['price'])
            episode_revenue += reward
            static_revenue += static_price * df["Quantity"].mean()
            current_step += 1
            
            if done or truncated:
                print(f"Episode terminado. Revenue RL: ${episode_revenue:.2f}, Static: ${static_revenue:.2f}")
                paused = True
        
        # Dibujar todo
        screen.fill(WHITE)
        
        # Título
        title = font_large.render("UrbanStyle - Dynamic Pricing RL", True, DARK_GRAY)
        screen.blit(title, (WIDTH // 2 - title.get_width() // 2, 20))
        
        # Estadísticas principales
        stats_y = 80
        
        # Step actual
        step_text = font_medium.render(f"Step: {current_step} / 100", True, BLACK)
        screen.blit(step_text, (50, stats_y))
        
        # Revenue RL
        rl_revenue_text = font_medium.render(f"RL Revenue: ${episode_revenue:.2f}", True, BLUE)
        screen.blit(rl_revenue_text, (300, stats_y))
        
        # Revenue estático
        static_revenue_text = font_medium.render(f"Static Revenue: ${static_revenue:.2f}", True, RED)
        screen.blit(static_revenue_text, (650, stats_y))
        
        # Precio actual
        if env is not None:
            current_price_text = font_medium.render(f"Current Price: ${env.price:.2f}", True, GREEN)
            screen.blit(current_price_text, (50, stats_y + 40))
            
            # Segmento actual
            segment_text = font_small.render(f"Segment: {env.segment}", True, BLACK)
            screen.blit(segment_text, (50, stats_y + 75))
        
        # Comparación (improvement)
        if static_revenue > 0:
            improvement = ((episode_revenue - static_revenue) / static_revenue) * 100
            color = GREEN if improvement > 0 else RED
            improvement_text = font_medium.render(f"Improvement: {improvement:.2f}%", True, color)
            screen.blit(improvement_text, (650, stats_y + 40))
        
        # Dibujar gráficos
        draw_revenue_plot(screen, revenues, 50, 200, 500, 300)
        draw_price_history(screen, prices, 620, 200, 530, 300, static_price)
        
        # Dibujar controles
        price_slider.draw(screen)
        reset_button.draw(screen)
        start_button.draw(screen)
        step_button.draw(screen)
        
        # Status
        status_text = "PAUSED" if paused else "RUNNING"
        status_color = ORANGE if paused else GREEN
        status = font_medium.render(status_text, True, status_color)
        screen.blit(status, (950, HEIGHT - 85))
        
        pygame.display.flip()
        clock.tick(10 if not paused else 30)  # 10 FPS cuando corre, 30 cuando pausa
    
    pygame.quit()

if __name__ == "__main__":
    main()
