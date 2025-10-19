import pygame
from typing import Tuple

class Button:
    def __init__(self, 
                 x:int,
                 y:int,
                 width:int,
                 height:int,
                 text:str):
        self.rect = pygame.Rect(x, y, width, height)
        self.text = text
        self.is_pressed = False
        
        
    def handle_event(self, event:pygame.event.EventType) -> bool:
        if event.type == pygame.MOUSEBUTTONDOWN and self.rect.collidepoint(event.pos):
            self.is_pressed = True
            print(f"[DEBUG] Button '{self.text}' pressed")
            return True
        if event.type == pygame.MOUSEBUTTOMUP:
            self.is_pressed = False
        return False
    
    def draw(self, surface:pygame.Surface, font:pygame.font.Font):
        fill_color = (50, 160, 80) if not self.is_pressed else (40, 120, 60)
        border_color = (20, 80, 40)
        pygame.draw.rect(surface, fill_color)
        