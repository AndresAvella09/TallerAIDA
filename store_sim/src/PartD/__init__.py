"""
PartD - Simple GAN for Promotion Images
========================================

Esta parte del proyecto implementa una GAN (Generative Adversarial Network)
usando PyTorch para generar imágenes promocionales basadas en Fashion-MNIST.

Módulos:
    - dataset: Carga y preprocesa Fashion-MNIST
    - models: Arquitectura DCGAN (Generador y Discriminador)
    - train: Script de entrenamiento con checkpoints
    - generate: Generación de imágenes desde vectores latentes
    - pygame_ui: Interfaz gráfica interactiva
    - utils: Utilidades auxiliares
"""

from .models import Generator, Discriminator
from .generate import generate_from_seed

__all__ = ['Generator', 'Discriminator', 'generate_from_seed']
