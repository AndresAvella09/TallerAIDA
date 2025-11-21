"""
Componentes de UI personalizados para pygame
"""
import pygame
import math


class Button:
    """Botón clickeable con hover"""
    
    def __init__(self, x, y, width, height, text, color=(70, 130, 180), hover_color=(100, 160, 210)):
        self.rect = pygame.Rect(x, y, width, height)
        self.text = text
        self.color = color
        self.hover_color = hover_color
        self.is_hovered = False
        self.font = pygame.font.Font(None, 24)
        
    def draw(self, screen):
        color = self.hover_color if self.is_hovered else self.color
        pygame.draw.rect(screen, color, self.rect, border_radius=5)
        pygame.draw.rect(screen, (255, 255, 255), self.rect, 2, border_radius=5)
        
        text_surf = self.font.render(self.text, True, (255, 255, 255))
        text_rect = text_surf.get_rect(center=self.rect.center)
        screen.blit(text_surf, text_rect)
        
    def handle_event(self, event):
        if event.type == pygame.MOUSEMOTION:
            self.is_hovered = self.rect.collidepoint(event.pos)
        elif event.type == pygame.MOUSEBUTTONDOWN:
            if self.rect.collidepoint(event.pos):
                return True
        return False


class Slider:
    """Slider para seleccionar valores numéricos"""
    
    def __init__(self, x, y, width, min_val, max_val, initial_val, label=""):
        self.x = x
        self.y = y
        self.width = width
        self.height = 20
        self.min_val = min_val
        self.max_val = max_val
        self.value = initial_val
        self.label = label
        self.dragging = False
        
        self.track_rect = pygame.Rect(x, y, width, 8)
        self.handle_radius = 12
        
        self.font = pygame.font.Font(None, 20)
        
    def get_handle_x(self):
        """Calcula la posición x del handle basada en el valor actual"""
        ratio = (self.value - self.min_val) / (self.max_val - self.min_val)
        return self.x + int(ratio * self.width)
        
    def draw(self, screen):
        # Etiqueta
        label_surf = self.font.render(f"{self.label}: {int(self.value)}", True, (255, 255, 255))
        screen.blit(label_surf, (self.x, self.y - 25))
        
        # Track
        pygame.draw.rect(screen, (100, 100, 100), self.track_rect, border_radius=4)
        
        # Handle
        handle_x = self.get_handle_x()
        pygame.draw.circle(screen, (70, 130, 180), (handle_x, self.y + 4), self.handle_radius)
        pygame.draw.circle(screen, (255, 255, 255), (handle_x, self.y + 4), self.handle_radius, 2)
        
    def handle_event(self, event):
        if event.type == pygame.MOUSEBUTTONDOWN:
            handle_x = self.get_handle_x()
            mouse_x, mouse_y = event.pos
            dist = math.sqrt((mouse_x - handle_x)**2 + (mouse_y - (self.y + 4))**2)
            if dist <= self.handle_radius:
                self.dragging = True
                
        elif event.type == pygame.MOUSEBUTTONUP:
            self.dragging = False
            
        elif event.type == pygame.MOUSEMOTION:
            if self.dragging:
                mouse_x = event.pos[0]
                mouse_x = max(self.x, min(mouse_x, self.x + self.width))
                ratio = (mouse_x - self.x) / self.width
                self.value = self.min_val + ratio * (self.max_val - self.min_val)
                return True
        return False


class Dropdown:
    """Menú desplegable"""
    
    def __init__(self, x, y, width, height, options, label=""):
        self.rect = pygame.Rect(x, y, width, height)
        self.options = options
        self.selected_index = 0
        self.label = label
        self.expanded = False
        self.font = pygame.font.Font(None, 20)
        self.small_font = pygame.font.Font(None, 18)
        
        self.option_height = 28
        self.max_visible = 7  # Balance entre opciones visibles y espacio
        
    def get_selected(self):
        return self.options[self.selected_index] if self.options else None
        
    def draw(self, screen):
        # Etiqueta
        if self.label:
            label_surf = self.font.render(self.label, True, (255, 255, 255))
            screen.blit(label_surf, (self.rect.x, self.rect.y - 25))
        
        # Caja principal
        pygame.draw.rect(screen, (60, 60, 60), self.rect, border_radius=5)
        pygame.draw.rect(screen, (255, 255, 255), self.rect, 2, border_radius=5)
        
        # Texto seleccionado
        if self.options:
            selected_text = self.options[self.selected_index]
            # Truncar si es muy largo
            if len(selected_text) > 25:
                selected_text = selected_text[:22] + "..."
            text_surf = self.small_font.render(selected_text, True, (255, 255, 255))
            screen.blit(text_surf, (self.rect.x + 10, self.rect.y + 10))
        
        # Flecha
        arrow_x = self.rect.right - 20
        arrow_y = self.rect.centery
        if self.expanded:
            points = [(arrow_x, arrow_y + 5), (arrow_x - 5, arrow_y - 5), (arrow_x + 5, arrow_y - 5)]
        else:
            points = [(arrow_x, arrow_y - 5), (arrow_x - 5, arrow_y + 5), (arrow_x + 5, arrow_y + 5)]
        pygame.draw.polygon(screen, (255, 255, 255), points)
        
        # Opciones expandidas
        if self.expanded:
            visible_options = min(self.max_visible, len(self.options))
            dropdown_height = visible_options * self.option_height
            dropdown_rect = pygame.Rect(
                self.rect.x, 
                self.rect.bottom + 2, 
                self.rect.width, 
                dropdown_height
            )
            pygame.draw.rect(screen, (50, 50, 50), dropdown_rect, border_radius=5)
            pygame.draw.rect(screen, (255, 255, 255), dropdown_rect, 2, border_radius=5)
            
            for i, option in enumerate(self.options[:visible_options]):
                option_rect = pygame.Rect(
                    self.rect.x,
                    self.rect.bottom + 2 + i * self.option_height,
                    self.rect.width,
                    self.option_height
                )
                
                # Hover
                mouse_pos = pygame.mouse.get_pos()
                if option_rect.collidepoint(mouse_pos):
                    pygame.draw.rect(screen, (80, 80, 80), option_rect)
                
                # Texto de opción (truncado)
                option_text = option
                if len(option_text) > 25:
                    option_text = option_text[:22] + "..."
                    
                text_surf = self.small_font.render(option_text, True, (255, 255, 255))
                screen.blit(text_surf, (option_rect.x + 10, option_rect.y + 8))
                
    def handle_event(self, event):
        if event.type == pygame.MOUSEBUTTONDOWN:
            if self.rect.collidepoint(event.pos):
                self.expanded = not self.expanded
                return False
            elif self.expanded:
                # Verificar click en opciones
                visible_options = min(self.max_visible, len(self.options))
                for i in range(visible_options):
                    option_rect = pygame.Rect(
                        self.rect.x,
                        self.rect.bottom + 2 + i * self.option_height,
                        self.rect.width,
                        self.option_height
                    )
                    if option_rect.collidepoint(event.pos):
                        self.selected_index = i
                        self.expanded = False
                        return True
                # Click fuera cierra el dropdown
                self.expanded = False
        return False


class ProgressBar:
    """Barra de progreso"""
    
    def __init__(self, x, y, width, height):
        self.rect = pygame.Rect(x, y, width, height)
        self.progress = 0.0  # 0.0 a 1.0
        self.font = pygame.font.Font(None, 18)
        
    def set_progress(self, progress):
        self.progress = max(0.0, min(1.0, progress))
        
    def draw(self, screen):
        # Fondo
        pygame.draw.rect(screen, (60, 60, 60), self.rect, border_radius=5)
        
        # Barra de progreso
        if self.progress > 0:
            progress_width = int(self.rect.width * self.progress)
            progress_rect = pygame.Rect(self.rect.x, self.rect.y, progress_width, self.rect.height)
            pygame.draw.rect(screen, (50, 180, 50), progress_rect, border_radius=5)
        
        # Borde
        pygame.draw.rect(screen, (255, 255, 255), self.rect, 2, border_radius=5)
        
        # Texto de porcentaje
        percent_text = f"{int(self.progress * 100)}%"
        text_surf = self.font.render(percent_text, True, (255, 255, 255))
        text_rect = text_surf.get_rect(center=self.rect.center)
        screen.blit(text_surf, text_rect)


class TextBox:
    """Caja de texto para mostrar información"""
    
    def __init__(self, x, y, width, height, title=""):
        self.rect = pygame.Rect(x, y, width, height)
        self.title = title
        self.lines = []
        self.font = pygame.font.Font(None, 18)
        self.title_font = pygame.font.Font(None, 22)
        
    def set_text(self, text):
        """Establece texto (puede ser lista de líneas o string)"""
        if isinstance(text, str):
            self.lines = text.split('\n')
        else:
            self.lines = text
            
    def draw(self, screen):
        # Fondo
        pygame.draw.rect(screen, (40, 40, 40), self.rect, border_radius=5)
        pygame.draw.rect(screen, (255, 255, 255), self.rect, 2, border_radius=5)
        
        # Título
        if self.title:
            title_surf = self.title_font.render(self.title, True, (255, 255, 255))
            screen.blit(title_surf, (self.rect.x + 10, self.rect.y + 5))
            
        # Líneas de texto
        y_offset = 30 if self.title else 10
        for i, line in enumerate(self.lines):
            if y_offset + 20 > self.rect.height:
                break
            text_surf = self.font.render(str(line), True, (200, 200, 200))
            screen.blit(text_surf, (self.rect.x + 10, self.rect.y + y_offset))
            y_offset += 20
