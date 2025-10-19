"""
Utility functions
=================

Funciones auxiliares para checkpoints, visualización y conversión de imágenes.
"""

import torch
import logging
from pathlib import Path
from typing import Dict, Optional, Union
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from torchvision.utils import make_grid

logger = logging.getLogger(__name__)


def setup_logging(level: int = logging.INFO):
    """
    Configura el sistema de logging.
    
    Args:
        level: Nivel de logging (default: INFO)
    """
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )


def save_checkpoint(
    generator: torch.nn.Module,
    discriminator: torch.nn.Module,
    optimizer_G: torch.optim.Optimizer,
    optimizer_D: torch.optim.Optimizer,
    epoch: int,
    path: Union[str, Path]
):
    """
    Guarda checkpoint del entrenamiento.
    
    Args:
        generator: Modelo generador
        discriminator: Modelo discriminador
        optimizer_G: Optimizador del generador
        optimizer_D: Optimizador del discriminador
        epoch: Época actual
        path: Ruta donde guardar
    """
    checkpoint = {
        'epoch': epoch,
        'generator_state_dict': generator.state_dict(),
        'discriminator_state_dict': discriminator.state_dict(),
        'optimizer_G_state_dict': optimizer_G.state_dict(),
        'optimizer_D_state_dict': optimizer_D.state_dict(),
        'latent_dim': generator.latent_dim
    }
    
    torch.save(checkpoint, path)
    logger.debug(f"Checkpoint guardado: {path}")


def load_checkpoint(
    path: Union[str, Path],
    generator: torch.nn.Module,
    discriminator: Optional[torch.nn.Module] = None,
    optimizer_G: Optional[torch.optim.Optimizer] = None,
    optimizer_D: Optional[torch.optim.Optimizer] = None,
    device: str = 'cpu'
) -> int:
    """
    Carga checkpoint del entrenamiento.
    
    Args:
        path: Ruta al checkpoint
        generator: Modelo generador
        discriminator: Modelo discriminador (opcional)
        optimizer_G: Optimizador G (opcional)
        optimizer_D: Optimizador D (opcional)
        device: Device donde cargar
        
    Returns:
        Época del checkpoint
    """
    checkpoint = torch.load(path, map_location=device)
    
    generator.load_state_dict(checkpoint['generator_state_dict'])
    
    if discriminator is not None and 'discriminator_state_dict' in checkpoint:
        discriminator.load_state_dict(checkpoint['discriminator_state_dict'])
    
    if optimizer_G is not None and 'optimizer_G_state_dict' in checkpoint:
        optimizer_G.load_state_dict(checkpoint['optimizer_G_state_dict'])
    
    if optimizer_D is not None and 'optimizer_D_state_dict' in checkpoint:
        optimizer_D.load_state_dict(checkpoint['optimizer_D_state_dict'])
    
    epoch = checkpoint.get('epoch', 0)
    logger.info(f"Checkpoint cargado desde {path} (época {epoch})")
    
    return epoch


def load_generator(
    checkpoint_path: Union[str, Path],
    latent_dim: int = 100,
    device: str = 'cpu'
) -> torch.nn.Module:
    """
    Carga solo el generador desde un checkpoint.
    
    Args:
        checkpoint_path: Ruta al checkpoint
        latent_dim: Dimensión latente
        device: Device
        
    Returns:
        Generador cargado
    """
    from .models import Generator
    
    generator = Generator(latent_dim=latent_dim).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    generator.load_state_dict(checkpoint['generator_state_dict'])
    
    logger.info(f"Generador cargado desde {checkpoint_path}")
    return generator


def denormalize_image(tensor: torch.Tensor) -> np.ndarray:
    """
    Desnormaliza una imagen de [-1, 1] a [0, 1] y la convierte a numpy.
    
    Args:
        tensor: Tensor [C, H, W] en rango [-1, 1]
        
    Returns:
        Array numpy [C, H, W] en rango [0, 1]
    """
    # Desnormalizar de [-1, 1] a [0, 1]
    img = (tensor + 1) / 2.0
    img = torch.clamp(img, 0, 1)
    
    # A numpy
    img_np = img.cpu().detach().numpy()
    
    return img_np


def tensor_to_numpy(tensor: torch.Tensor) -> np.ndarray:
    """
    Convierte tensor de imagen a numpy array [0, 255].
    
    Args:
        tensor: Tensor [C, H, W] en rango [-1, 1]
        
    Returns:
        Array numpy [H, W, C] en rango [0, 255]
    """
    # Desnormalizar de [-1, 1] a [0, 1]
    img = (tensor + 1) / 2.0
    img = torch.clamp(img, 0, 1)
    
    # A numpy
    img_np = img.cpu().detach().numpy()
    
    # [C, H, W] -> [H, W, C]
    if img_np.ndim == 3:
        img_np = np.transpose(img_np, (1, 2, 0))
    
    # Escalar a [0, 255]
    img_np = (img_np * 255).astype(np.uint8)
    
    return img_np


def tensor_to_pil(tensor: torch.Tensor) -> Image.Image:
    """
    Convierte tensor a PIL Image.
    
    Args:
        tensor: Tensor [C, H, W] o [H, W]
        
    Returns:
        PIL Image
    """
    img_np = tensor_to_numpy(tensor)
    
    # Si es grayscale, asegurar que sea 2D
    if img_np.shape[-1] == 1:
        img_np = img_np.squeeze(-1)
    
    return Image.fromarray(img_np, mode='L')  # 'L' para grayscale


def save_sample_images(
    images: torch.Tensor,
    path: Union[str, Path],
    nrow: int = 8,
    normalize: bool = True
):
    """
    Guarda un grid de imágenes.
    
    Args:
        images: Tensor [N, C, H, W]
        path: Ruta de salida
        nrow: Número de imágenes por fila
        normalize: Si normalizar o no
    """
    grid = make_grid(images, nrow=nrow, normalize=normalize, value_range=(-1, 1))
    grid_np = tensor_to_numpy(grid)
    
    # Convertir a PIL y guardar
    # Si es grayscale con dimensión extra, eliminarla
    if grid_np.ndim == 3 and grid_np.shape[-1] == 1:
        grid_np = grid_np.squeeze(-1)
    
    # Asegurar que sea 2D para modo 'L'
    if grid_np.ndim == 2:
        img = Image.fromarray(grid_np, mode='L')
    else:
        # Si tiene 3 canales, convertir a RGB
        img = Image.fromarray(grid_np, mode='RGB')
    
    img.save(path)


def plot_losses(
    g_losses: list,
    d_losses: list,
    save_path: Optional[Union[str, Path]] = None
):
    """
    Grafica las pérdidas del entrenamiento.
    
    Args:
        g_losses: Lista de losses del generador
        d_losses: Lista de losses del discriminador
        save_path: Ruta para guardar (opcional)
    """
    plt.figure(figsize=(10, 5))
    plt.plot(g_losses, label='Generator Loss')
    plt.plot(d_losses, label='Discriminator Loss')
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('GAN Training Losses')
    plt.grid(True)
    
    if save_path:
        plt.savefig(save_path)
        logger.info(f"Gráfica guardada en {save_path}")
    else:
        plt.show()
    
    plt.close()


def count_parameters(model: torch.nn.Module) -> int:
    """
    Cuenta el número de parámetros entrenables.
    
    Args:
        model: Modelo PyTorch
        
    Returns:
        Número de parámetros
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
