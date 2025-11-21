"""
Interfaz gráfica principal para entrenamiento y visualización de modelos LSTM
"""
import pygame
import torch
import sys
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np

from data_processor import DataProcessor
from lstm_model import LSTMTrainer, set_seed
from ui_components import Button, Slider, Dropdown, ProgressBar, TextBox
from chart_generator import ChartGenerator


class LSTMApp:
    """Aplicación principal con interfaz pygame"""
    
    def __init__(self, width=None, height=None, fullscreen=True, seed=42):
        pygame.init()
        
        # Fijar semilla para reproducibilidad
        self.seed = seed
        set_seed(seed)
        
        # Obtener resolución de pantalla
        if fullscreen or (width is None or height is None):
            display_info = pygame.display.Info()
            self.width = display_info.current_w
            self.height = display_info.current_h
            self.screen = pygame.display.set_mode((self.width, self.height), pygame.FULLSCREEN)
            self.fullscreen = True
        else:
            self.width = width
            self.height = height
            self.screen = pygame.display.set_mode((width, height))
            self.fullscreen = False
            
        pygame.display.set_caption("LSTM Sales Forecasting")
        self.clock = pygame.time.Clock()
        self.running = True
        
        # Colores
        self.bg_color = (30, 30, 30)
        self.panel_color = (45, 45, 45)
        
        # Fuentes
        self.title_font = pygame.font.Font(None, 36)
        self.subtitle_font = pygame.font.Font(None, 24)
        self.text_font = pygame.font.Font(None, 20)
        
        # Cargar datos
        print("Cargando datos...")
        self.data_processor = DataProcessor()
        self.data_processor.load_and_prepare()
        categories = self.data_processor.get_categories()
        print(f"Categorías cargadas: {len(categories)}")
        
        # Device para PyTorch
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Usando device: {self.device}")
        print(f"Semilla fijada: {self.seed}")
        
        # Trainer (con semilla para reproducibilidad)
        self.trainer = LSTMTrainer(device=self.device, seed=self.seed)
        
        # UI Components - Panel Izquierdo (responsive al tamaño de pantalla)
        left_x = 20
        left_width = min(350, int(self.width * 0.25))  # 25% del ancho o máximo 350px
        
        self.category_dropdown = Dropdown(left_x, 70, left_width, 40, categories, label="Categoría de Producto")
        
        # Input para seed (justo después del dropdown con su espacio desplegado)
        self.seed_input_active = False
        self.seed_input_rect = pygame.Rect(left_x, 260, left_width, 32)
        self.seed_input_text = str(self.seed)
        
        # Input para learning rate
        self.lr_input_active = False
        self.lr_input_rect = pygame.Rect(left_x, 322, left_width, 32)
        self.learning_rate = 0.001  # Valor por defecto
        self.lr_input_text = "0.001"
        
        self.epochs_slider = Slider(left_x, 395, left_width, 1000, 5000, 2000, label="Épocas")
        self.train_button = Button(left_x, 445, left_width, 45, "🚀 Entrenar Modelo", color=(76, 175, 80))
        self.progress_bar = ProgressBar(left_x, 505, left_width, 25)
        self.status_box = TextBox(left_x, 545, left_width, int(self.height * 0.27), title="Estado")
        
        # Variables de entrenamiento
        self.is_training = False
        self.training_epoch = 0
        self.training_epochs_total = 0
        self.losses = []
        
        # Resultados
        self.current_results = None
        self.chart_surface = None
        self.metrics_surface = None
        self.loss_surface = None
        
        self.status_lines = ["Listo para entrenar.", "Selecciona categoría y épocas."]
        self.status_box.set_text(self.status_lines)
        
        # Calcular dimensiones dinámicas para gráficos
        self.chart_x = left_x + left_width + 20
        self.chart_width = self.width - self.chart_x - 20
        self.chart_height = int(self.height * 0.55)  # 55% de altura para gráfico principal
        
        self.metrics_x = self.chart_x
        self.metrics_y = self.chart_height + 90
        self.metrics_width = int(self.chart_width * 0.48)
        self.metrics_height = self.height - self.metrics_y - 20
        
        self.loss_x = self.metrics_x + self.metrics_width + 20
        self.loss_y = self.metrics_y
        self.loss_width = self.chart_width - self.metrics_width - 20
        self.loss_height = self.metrics_height
        
    def train_model_incremental(self):
        """Entrena el modelo de forma incremental (época por época) para no bloquear UI"""
        if not self.is_training:
            return
            
        category = self.category_dropdown.get_selected()
        
        # Entrenar una época
        loss = self.trainer.train_epoch(self.X_train_t, self.y_train_t)
        self.losses.append(loss)
        self.training_epoch += 1
        
        # Actualizar progreso
        progress = self.training_epoch / self.training_epochs_total
        self.progress_bar.set_progress(progress)
        
        # Actualizar estado cada 50 épocas o al final
        if self.training_epoch % 50 == 0 or self.training_epoch >= self.training_epochs_total:
            self.status_lines[-1] = f"Época {self.training_epoch}/{self.training_epochs_total} - Loss: {loss:.6f}"
            self.status_box.set_text(self.status_lines)
        
        # Verificar si terminó
        if self.training_epoch >= self.training_epochs_total:
            self.finish_training(category)
            
    def finish_training(self, category):
        """Finaliza el entrenamiento y genera visualizaciones"""
        self.is_training = False
        self.status_lines.append("Evaluando modelo...")
        self.status_box.set_text(self.status_lines)
        
        # Evaluar en test set
        y_pred_scaled = self.trainer.predict(self.X_test_t)
        
        # Desescalar
        y_true = self.data_processor.inverse_transform(category, self.y_test)
        y_pred = self.data_processor.inverse_transform(category, y_pred_scaled)
        
        # Métricas
        mae = mean_absolute_error(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        
        # Predicción próxima semana
        last_window = self.data_processor.get_last_window(category)
        next_scaled = self.trainer.forecast_next(last_window)
        next_value = self.data_processor.inverse_transform(category, np.array([next_scaled]))[0]
        
        # Guardar resultados
        self.current_results = {
            "category": category,
            "y_true": y_true,
            "y_pred": y_pred,
            "mae": mae,
            "rmse": rmse,
            "next_forecast": next_value,
            "dates": self.data_info["dates"],
            "split": self.data_info["split"],
            "train_size": len(self.X_train_t),
            "test_size": len(self.X_test_t),
            "learning_rate": self.learning_rate,
            "seed": self.seed,
            "epochs": self.training_epochs_total
        }
        
        # Generar gráficos
        self.status_lines.append("Generando gráficos...")
        self.status_box.set_text(self.status_lines)
        
        self.generate_charts()
        
        self.status_lines.append(f"✓ Completado! MAE={mae:.2f}, RMSE={rmse:.2f}")
        self.status_box.set_text(self.status_lines[-8:])  # Últimas 8 líneas
        
    def generate_charts(self):
        """Genera las visualizaciones"""
        if not self.current_results:
            return
            
        r = self.current_results
        
        # Generar gráficos en alta resolución basados en el tamaño de pantalla
        self.chart_surface = ChartGenerator.create_time_series_chart(
            r["dates"],
            self.data_info["values"],
            r["y_pred"],
            r["split"],
            r["mae"],
            r["rmse"],
            r["next_forecast"],
            width=int(self.chart_width * 1.2),  # Generar más grande para mejor calidad
            height=int(self.chart_height * 1.2)
        )
        
        # Tabla de métricas (incluye hiperparámetros)
        self.metrics_surface = ChartGenerator.create_metrics_table(
            r["category"],
            r["mae"],
            r["rmse"],
            r["next_forecast"],
            r["train_size"],
            r["test_size"],
            learning_rate=r.get("learning_rate"),
            seed=r.get("seed"),
            epochs=r.get("epochs"),
            width=int(self.metrics_width * 1.2),
            height=int(self.metrics_height * 1.2)
        )
        
        # Gráfico de pérdida
        if self.losses:
            self.loss_surface = ChartGenerator.create_loss_chart(
                self.losses,
                width=int(self.loss_width * 1.2),
                height=int(self.loss_height * 1.2)
            )
        
    def start_training(self):
        """Inicia el proceso de entrenamiento"""
        category = self.category_dropdown.get_selected()
        if not category:
            return
        
        # Actualizar learning rate desde el input antes de entrenar
        try:
            new_lr = float(self.lr_input_text)
            if 0.0 < new_lr <= 1.0:
                self.learning_rate = new_lr
        except ValueError:
            pass  # Mantener el valor actual si hay error
        
        # Restablecer semilla antes de entrenar
        set_seed(self.seed)
            
        self.is_training = True
        self.training_epoch = 0
        self.training_epochs_total = int(self.epochs_slider.value)
        self.losses = []
        self.progress_bar.set_progress(0)
        
        self.status_lines = [
            f"Categoría: {category[:30]}",
            f"Épocas: {self.training_epochs_total}",
            f"Semilla: {self.seed}",
            f"Learning Rate: {self.learning_rate}",
            "Preparando datos...",
        ]
        self.status_box.set_text(self.status_lines)
        
        # Obtener datos
        self.data_info = self.data_processor.get_data_for_category(category)
        X_train = self.data_info["X_train"]
        X_test = self.data_info["X_test"]
        y_train = self.data_info["y_train"]
        y_test = self.data_info["y_test"]
        
        # Convertir a tensores
        self.X_train_t = torch.tensor(X_train, dtype=torch.float32, device=self.device)
        self.y_train_t = torch.tensor(y_train.reshape(-1, 1), dtype=torch.float32, device=self.device)
        self.X_test_t = torch.tensor(X_test, dtype=torch.float32, device=self.device)
        self.y_test = y_test
        
        # Crear modelo con el learning rate configurado
        self.trainer.create_model(input_size=1, hidden_size=64, output_size=1, lr=self.learning_rate)
        
        self.status_lines.append("Entrenando...")
        self.status_lines.append("")
        self.status_box.set_text(self.status_lines)
        
    def handle_events(self):
        """Maneja eventos de pygame"""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
            
            # Tecla ESC para salir de pantalla completa
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE and not self.seed_input_active and not self.lr_input_active:
                    self.running = False
                
                # Manejo del input de seed
                if self.seed_input_active:
                    if event.key == pygame.K_RETURN:
                        # Aplicar nueva semilla
                        try:
                            new_seed = int(self.seed_input_text)
                            self.seed = new_seed
                            set_seed(new_seed)
                            self.trainer = LSTMTrainer(device=self.device, seed=new_seed)
                            self.status_lines = [f"Semilla cambiada a: {new_seed}", "Listo para entrenar."]
                            self.status_box.set_text(self.status_lines)
                        except ValueError:
                            self.seed_input_text = str(self.seed)
                        self.seed_input_active = False
                    elif event.key == pygame.K_BACKSPACE:
                        self.seed_input_text = self.seed_input_text[:-1]
                    elif event.key == pygame.K_ESCAPE:
                        self.seed_input_active = False
                        self.seed_input_text = str(self.seed)
                    elif event.unicode.isdigit() and len(self.seed_input_text) < 10:
                        self.seed_input_text += event.unicode
                
                # Manejo del input de learning rate
                if self.lr_input_active:
                    if event.key == pygame.K_RETURN:
                        # Aplicar nuevo learning rate
                        try:
                            new_lr = float(self.lr_input_text)
                            if 0.0 < new_lr <= 1.0:
                                self.learning_rate = new_lr
                                self.status_lines = [f"Learning rate: {new_lr}", "Listo para entrenar."]
                                self.status_box.set_text(self.status_lines)
                            else:
                                self.lr_input_text = str(self.learning_rate)
                        except ValueError:
                            self.lr_input_text = str(self.learning_rate)
                        self.lr_input_active = False
                    elif event.key == pygame.K_BACKSPACE:
                        self.lr_input_text = self.lr_input_text[:-1]
                    elif event.key == pygame.K_ESCAPE:
                        self.lr_input_active = False
                        self.lr_input_text = str(self.learning_rate)
                    elif (event.unicode.isdigit() or event.unicode == '.') and len(self.lr_input_text) < 10:
                        # Evitar múltiples puntos decimales
                        if event.unicode == '.' and '.' in self.lr_input_text:
                            pass
                        else:
                            self.lr_input_text += event.unicode
            
            # Click en los inputs
            if event.type == pygame.MOUSEBUTTONDOWN:
                if self.seed_input_rect.collidepoint(event.pos) and not self.is_training:
                    self.seed_input_active = True
                    self.lr_input_active = False
                elif self.lr_input_rect.collidepoint(event.pos) and not self.is_training:
                    self.lr_input_active = True
                    self.seed_input_active = False
                else:
                    self.seed_input_active = False
                    self.lr_input_active = False
                    
            # Manejar componentes UI
            if self.category_dropdown.handle_event(event):
                # Cambió la categoría - limpiar resultados
                self.current_results = None
                self.chart_surface = None
                self.metrics_surface = None
                self.loss_surface = None
                
            self.epochs_slider.handle_event(event)
            
            if not self.is_training:
                if self.train_button.handle_event(event):
                    self.start_training()
                    
    def draw(self):
        """Dibuja la interfaz"""
        self.screen.fill(self.bg_color)
        
        # Título (responsive)
        title = self.title_font.render("LSTM Sales Forecasting", True, (255, 255, 255))
        self.screen.blit(title, (20, 20))
        
        # Info: Presiona ESC para salir
        esc_font = pygame.font.Font(None, 18)
        esc_text = esc_font.render("Presiona ESC para salir", True, (150, 150, 150))
        self.screen.blit(esc_text, (self.width - 200, 20))
        
        # Panel izquierdo - Controles (responsive)
        left_width = self.category_dropdown.rect.width
        panel_height = self.height - 80
        pygame.draw.rect(self.screen, self.panel_color, (10, 60, left_width + 20, panel_height), border_radius=10)
        
        self.category_dropdown.draw(self.screen)
        
        # Input de semilla (seed) - después del dropdown
        seed_label = self.text_font.render("Semilla (Seed):", True, (255, 255, 255))
        self.screen.blit(seed_label, (self.seed_input_rect.x, self.seed_input_rect.y - 25))
        
        # Caja de input seed
        input_color = (100, 150, 200) if self.seed_input_active else (80, 80, 80)
        pygame.draw.rect(self.screen, input_color, self.seed_input_rect, border_radius=5)
        pygame.draw.rect(self.screen, (255, 255, 255), self.seed_input_rect, 2, border_radius=5)
        
        # Texto del input seed
        input_text_surf = self.text_font.render(self.seed_input_text, True, (255, 255, 255))
        self.screen.blit(input_text_surf, (self.seed_input_rect.x + 10, self.seed_input_rect.y + 8))
        
        # Hint seed
        if self.seed_input_active:
            hint_font = pygame.font.Font(None, 16)
            hint = hint_font.render("Enter para aplicar, ESC para cancelar", True, (200, 200, 200))
            self.screen.blit(hint, (self.seed_input_rect.x, self.seed_input_rect.y + 40))
        
        # Input de learning rate
        lr_label = self.text_font.render("Learning Rate:", True, (255, 255, 255))
        self.screen.blit(lr_label, (self.lr_input_rect.x, self.lr_input_rect.y - 25))
        
        # Caja de input lr
        lr_input_color = (100, 150, 200) if self.lr_input_active else (80, 80, 80)
        pygame.draw.rect(self.screen, lr_input_color, self.lr_input_rect, border_radius=5)
        pygame.draw.rect(self.screen, (255, 255, 255), self.lr_input_rect, 2, border_radius=5)
        
        # Texto del input lr
        lr_text_surf = self.text_font.render(self.lr_input_text, True, (255, 255, 255))
        self.screen.blit(lr_text_surf, (self.lr_input_rect.x + 10, self.lr_input_rect.y + 8))
        
        # Hint lr
        if self.lr_input_active:
            hint_font = pygame.font.Font(None, 16)
            hint = hint_font.render("Enter para aplicar, ESC para cancelar", True, (200, 200, 200))
            self.screen.blit(hint, (self.lr_input_rect.x, self.lr_input_rect.y + 40))
        
        # Ahora dibujar el slider y el resto
        self.epochs_slider.draw(self.screen)
        
        if not self.is_training:
            self.train_button.draw(self.screen)
        else:
            # Mostrar que está entrenando
            training_text = self.subtitle_font.render("Entrenando...", True, (255, 200, 50))
            self.screen.blit(training_text, (self.train_button.rect.x + 50, self.train_button.rect.y + 10))
            
        self.progress_bar.draw(self.screen)
        self.status_box.draw(self.screen)
        
        # Panel derecho - Visualizaciones (responsive y en alta calidad)
        if self.chart_surface:
            scaled_chart = pygame.transform.smoothscale(self.chart_surface, (self.chart_width, self.chart_height))
            self.screen.blit(scaled_chart, (self.chart_x, 70))
            
        if self.metrics_surface:
            scaled_metrics = pygame.transform.smoothscale(self.metrics_surface, (self.metrics_width, self.metrics_height))
            self.screen.blit(scaled_metrics, (self.metrics_x, self.metrics_y))
            
        if self.loss_surface:
            scaled_loss = pygame.transform.smoothscale(self.loss_surface, (self.loss_width, self.loss_height))
            self.screen.blit(scaled_loss, (self.loss_x, self.loss_y))
            
        # Info del dispositivo (más pequeño)
        small_font = pygame.font.Font(None, 16)
        device_text = small_font.render(f"Device: {self.device.upper()}", True, (150, 150, 150))
        self.screen.blit(device_text, (self.width - 120, self.height - 25))
        
        pygame.display.flip()
        
    def run(self):
        """Loop principal de la aplicación"""
        while self.running:
            self.handle_events()
            
            # Entrenar incrementalmente si está en proceso
            if self.is_training:
                # Entrenar varias épocas por frame para acelerar
                for _ in range(10):  # 10 épocas por frame
                    if self.is_training:
                        self.train_model_incremental()
                    else:
                        break
            
            self.draw()
            self.clock.tick(60)  # 60 FPS
            
        pygame.quit()
        sys.exit()


if __name__ == "__main__":
    app = LSTMApp()
    app.run()
