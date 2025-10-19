"""
Interfaz gráfica independiente para generar imágenes con la GAN entrenada.
Permite seleccionar el tipo de prenda a generar y visualizar el resultado.
"""

import pygame
import torch
import os
import sys
from pathlib import Path

# Añadir el directorio raíz al path para imports
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from store_sim.src.PartD.models import Generator
from store_sim.src.PartD.utils import denormalize_image

class GANInterface:
    def __init__(self, checkpoint_path="checkpoints/gan_final.pth", latent_dim=100):
        """
        Inicializa la interfaz de la GAN.
        
        Args:
            checkpoint_path: Ruta al checkpoint del modelo entrenado
            latent_dim: Dimensión del vector latente
        """
        pygame.init()
        
        # Configuración de ventana
        self.width = 800
        self.height = 600
        self.screen = pygame.display.set_mode((self.width, self.height))
        pygame.display.set_caption("UrbanStyle - Generador de Imágenes con GAN")
        
        # Colores
        self.WHITE = (255, 255, 255)
        self.BLACK = (0, 0, 0)
        self.GRAY = (200, 200, 200)
        self.DARK_GRAY = (100, 100, 100)
        self.BLUE = (70, 130, 180)
        self.GREEN = (60, 179, 113)
        self.HOVER = (100, 149, 237)
        
        # Fuentes
        self.title_font = pygame.font.Font(None, 48)
        self.button_font = pygame.font.Font(None, 32)
        self.label_font = pygame.font.Font(None, 24)
        
        # Cargar el generador
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.latent_dim = latent_dim
        self.generator = None
        self.checkpoint_path = checkpoint_path
        self.load_generator()
        
        # Estado de la interfaz
        self.current_seed = 0
        self.generated_image = None
        self.image_surface = None
        
        # Botones
        self.buttons = self.create_buttons()
        
        # Sliders para noise seed
        self.sliders = self.create_sliders()
        
        self.clock = pygame.time.Clock()
        
    def load_generator(self):
        """Carga el generador desde el checkpoint."""
        try:
            self.generator = Generator(latent_dim=self.latent_dim)
            
            # Cargar el checkpoint
            checkpoint = torch.load(self.checkpoint_path, map_location=self.device)
            
            # El checkpoint puede tener diferentes formatos
            if isinstance(checkpoint, dict):
                if 'generator_state_dict' in checkpoint:
                    self.generator.load_state_dict(checkpoint['generator_state_dict'])
                elif 'model_state_dict' in checkpoint:
                    self.generator.load_state_dict(checkpoint['model_state_dict'])
                else:
                    self.generator.load_state_dict(checkpoint)
            else:
                self.generator.load_state_dict(checkpoint)
            
            self.generator.to(self.device)
            self.generator.eval()
            print(f" Generador cargado exitosamente desde {self.checkpoint_path}")
        except Exception as e:
            print(f" Error al cargar el generador: {e}")
            self.generator = None
    
    def create_buttons(self):
        """Crea los botones de la interfaz."""
        buttons = {
            'generate': pygame.Rect(50, 500, 200, 60),
            'save': pygame.Rect(300, 500, 200, 60),
            'random': pygame.Rect(550, 500, 200, 60)
        }
        return buttons
    
    def create_sliders(self):
        """Crea el slider para el seed."""
        sliders = {
            'seed': {
                'rect': pygame.Rect(50, 150, 700, 20),
                'handle': pygame.Rect(50, 145, 15, 30),
                'min': 0,
                'max': 10000,
                'value': 0,
                'dragging': False,
                'label': 'Noise Seed'
            }
        }
        return sliders
    
    def update_slider_handle(self, slider_name):
        """Actualiza la posición del handle del slider."""
        slider = self.sliders[slider_name]
        slider_range = slider['rect'].width
        normalized = (slider['value'] - slider['min']) / (slider['max'] - slider['min'])
        handle_x = slider['rect'].x + int(normalized * slider_range)
        slider['handle'].centerx = handle_x
    
    def update_seed_from_slider(self, slider_name, mouse_x):
        """Actualiza el seed según la posición del mouse en el slider."""
        slider = self.sliders[slider_name]
        slider_range = slider['rect'].width
        relative_x = max(0, min(slider_range, mouse_x - slider['rect'].x))
        normalized = relative_x / slider_range
        slider['value'] = int(slider['min'] + normalized * (slider['max'] - slider['min']))
        self.current_seed = slider['value']
        self.update_slider_handle(slider_name)
    
    def generate_image(self):
        """Genera una nueva imagen usando la GAN con el seed actual."""
        if self.generator is None:
            print("✗ Generador no disponible")
            return
        
        try:
            # Fijar seed para reproducibilidad
            torch.manual_seed(self.current_seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed(self.current_seed)
            
            # Generar vector latente usando el seed fijado
            z = torch.randn(1, self.latent_dim, device=self.device)
            
            # Generar imagen
            with torch.no_grad():
                fake_image = self.generator(z)
            
            # Desnormalizar y convertir a numpy
            image_np = denormalize_image(fake_image[0])
            
            # Convertir a surface de pygame (escalar a 280x280)
            image_np = (image_np * 255).astype('uint8')
            # Repetir canal para obtener RGB
            image_rgb = image_np.repeat(3, axis=0).transpose(1, 2, 0)
            
            # Crear surface y escalar
            temp_surface = pygame.surfarray.make_surface(image_rgb.swapaxes(0, 1))
            self.image_surface = pygame.transform.scale(temp_surface, (280, 280))
            
            print(f"✓ Imagen generada con seed: {self.current_seed}")
            
        except Exception as e:
            print(f"✗ Error al generar imagen: {e}")
    
    def generate_random(self):
        """Genera una imagen con seed aleatorio."""
        import random
        self.current_seed = random.randint(0, 10000)
        self.sliders['seed']['value'] = self.current_seed
        self.update_slider_handle('seed')
        self.generate_image()
    
    def save_image(self):
        """Guarda la imagen generada."""
        if self.image_surface is None:
            print("✗ No hay imagen para guardar")
            return
        
        # Crear directorio de salida si no existe
        output_dir = Path("generated_images")
        output_dir.mkdir(exist_ok=True)
        
        # Generar nombre de archivo con el seed
        filename = output_dir / f"gan_seed_{self.current_seed}_{len(list(output_dir.glob('*.png')))}.png"
        
        # Guardar
        pygame.image.save(self.image_surface, str(filename))
        print(f"✓ Imagen guardada en: {filename}")
    
    def draw_button(self, rect, text, color, hover=False):
        """Dibuja un botón."""
        btn_color = self.HOVER if hover else color
        pygame.draw.rect(self.screen, btn_color, rect, border_radius=10)
        pygame.draw.rect(self.screen, self.BLACK, rect, 2, border_radius=10)
        
        text_surface = self.button_font.render(text, True, self.WHITE)
        text_rect = text_surface.get_rect(center=rect.center)
        self.screen.blit(text_surface, text_rect)
    
    def draw(self):
        """Dibuja la interfaz."""
        self.screen.fill(self.WHITE)
        
        # Título
        title = self.title_font.render("Generador GAN - Fashion MNIST", True, self.BLACK)
        title_rect = title.get_rect(center=(self.width // 2, 40))
        self.screen.blit(title, title_rect)
        
        # Subtítulo
        subtitle = self.label_font.render("Ajusta el Noise Seed y genera imágenes", True, self.DARK_GRAY)
        subtitle_rect = subtitle.get_rect(center=(self.width // 2, 80))
        self.screen.blit(subtitle, subtitle_rect)
        
        # Dibujar slider de seed
        slider = self.sliders['seed']
        
        # Etiqueta del slider
        seed_label = self.label_font.render(f"{slider['label']}: {slider['value']}", True, self.BLACK)
        self.screen.blit(seed_label, (slider['rect'].x, slider['rect'].y - 30))
        
        # Barra del slider
        pygame.draw.rect(self.screen, self.GRAY, slider['rect'], border_radius=5)
        
        # Handle del slider
        mouse_pos = pygame.mouse.get_pos()
        handle_color = self.HOVER if slider['dragging'] else self.BLUE
        pygame.draw.rect(self.screen, handle_color, slider['handle'], border_radius=5)
        pygame.draw.rect(self.screen, self.BLACK, slider['handle'], 2, border_radius=5)
        
        # Área de visualización de imagen
        image_area = pygame.Rect(260, 220, 280, 280)
        pygame.draw.rect(self.screen, self.GRAY, image_area)
        pygame.draw.rect(self.screen, self.BLACK, image_area, 2)
        
        if self.image_surface:
            self.screen.blit(self.image_surface, image_area)
        else:
            no_image_text = self.label_font.render(
                "Ajusta el seed y presiona 'Generar'", 
                True, 
                self.DARK_GRAY
            )
            text_rect = no_image_text.get_rect(center=image_area.center)
            self.screen.blit(no_image_text, text_rect)
        
        # Botones de acción
        self.draw_button(
            self.buttons['generate'], 
            "Generar", 
            self.GREEN,
            self.buttons['generate'].collidepoint(mouse_pos)
        )
        self.draw_button(
            self.buttons['save'], 
            "Guardar", 
            self.BLUE,
            self.buttons['save'].collidepoint(mouse_pos) and self.image_surface is not None
        )
        self.draw_button(
            self.buttons['random'], 
            "Aleatorio", 
            self.BLUE,
            self.buttons['random'].collidepoint(mouse_pos)
        )
        
        # Información del modelo
        info_text = self.label_font.render(
            f"Modelo: {os.path.basename(self.checkpoint_path)}", 
            True, 
            self.DARK_GRAY
        )
        self.screen.blit(info_text, (10, self.height - 30))
        
        pygame.display.flip()
    
    def handle_events(self):
        """Maneja los eventos de pygame."""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False
            
            if event.type == pygame.MOUSEBUTTONDOWN:
                mouse_pos = event.pos
                
                # Botón generar
                if self.buttons['generate'].collidepoint(mouse_pos):
                    self.generate_image()
                
                # Botón guardar
                elif self.buttons['save'].collidepoint(mouse_pos) and self.image_surface:
                    self.save_image()
                
                # Botón aleatorio
                elif self.buttons['random'].collidepoint(mouse_pos):
                    self.generate_random()
                
                # Click en el handle del slider
                elif self.sliders['seed']['handle'].collidepoint(mouse_pos):
                    self.sliders['seed']['dragging'] = True
                
                # Click directo en la barra del slider
                elif self.sliders['seed']['rect'].collidepoint(mouse_pos):
                    self.update_seed_from_slider('seed', mouse_pos[0])
                    self.generate_image()
            
            elif event.type == pygame.MOUSEBUTTONUP:
                self.sliders['seed']['dragging'] = False
            
            elif event.type == pygame.MOUSEMOTION:
                if self.sliders['seed']['dragging']:
                    self.update_seed_from_slider('seed', event.pos[0])
            
            # Atajos de teclado
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE or event.key == pygame.K_g:
                    self.generate_image()
                elif event.key == pygame.K_s and self.image_surface:
                    self.save_image()
                elif event.key == pygame.K_r:
                    self.generate_random()
                elif event.key == pygame.K_LEFT:
                    # Decrementar seed
                    self.sliders['seed']['value'] = max(0, self.sliders['seed']['value'] - 100)
                    self.current_seed = self.sliders['seed']['value']
                    self.update_slider_handle('seed')
                elif event.key == pygame.K_RIGHT:
                    # Incrementar seed
                    self.sliders['seed']['value'] = min(10000, self.sliders['seed']['value'] + 100)
                    self.current_seed = self.sliders['seed']['value']
                    self.update_slider_handle('seed')
        
        return True
    
    def run(self):
        """Ejecuta el bucle principal de la interfaz."""
        if self.generator is None:
            print("✗ No se puede ejecutar la interfaz sin un generador válido")
            pygame.quit()
            return
        
        print("\n" + "="*60)
        print("Fashion MNIST - Generador de Imágenes con GAN")
        print("="*60)
        print("\nControles:")
        print("  • ESPACIO o G: Generar imagen con seed actual")
        print("  • S: Guardar imagen actual")
        print("  • R: Generar con seed aleatorio")
        print("  • ← →: Ajustar seed (±100)")
        print("  • Slider: Ajustar seed manualmente (0-10000)")
        print("  • Click en botones para las mismas acciones")
        print("="*60 + "\n")
        
        running = True
        while running:
            running = self.handle_events()
            self.draw()
            self.clock.tick(60)
        
        pygame.quit()
        print("\n✓ Interfaz cerrada")


def main():
    """Función principal."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Interfaz para generar imágenes con GAN")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="checkpoints/gan_final.pth",
        help="Ruta al checkpoint del generador (default: checkpoints/gan_final.pth)"
    )
    parser.add_argument(
        "--latent-dim",
        type=int,
        default=100,
        help="Dimensión del espacio latente (default: 100)"
    )
    
    args = parser.parse_args()
    
    # Verificar que existe el checkpoint
    if not os.path.exists(args.checkpoint):
        print(f" Error: No se encontró el checkpoint en {args.checkpoint}")
        print("  Asegúrate de haber entrenado el modelo primero con:")
        print("  python -m store_sim.src.PartD.train")
        return
    
    # Crear y ejecutar la interfaz
    interface = GANInterface(
        checkpoint_path=args.checkpoint,
        latent_dim=args.latent_dim
    )
    interface.run()


if __name__ == "__main__":
    main()
