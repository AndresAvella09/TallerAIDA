"""
Image generation from trained GAN
==================================

Genera imágenes usando un generador entrenado a partir de vectores latentes.
"""

import torch
import argparse
from pathlib import Path
from typing import Optional, Union
import logging
import numpy as np
from PIL import Image

from .models import Generator
from .utils import load_generator, tensor_to_pil, setup_logging

logger = logging.getLogger(__name__)


def generate_from_seed(
    seed: int,
    checkpoint_path: Union[str, Path],
    latent_dim: int = 100,
    device: Optional[str] = None,
    return_tensor: bool = False
) -> Union[Image.Image, torch.Tensor]:
    """
    Genera una imagen a partir de una semilla (seed) específica.
    
    Esta es la función principal para usar en otros scripts.
    
    Args:
        seed: Semilla para el generador de números aleatorios
        checkpoint_path: Ruta al checkpoint del modelo entrenado
        latent_dim: Dimensión del espacio latente
        device: 'cpu' o 'cuda' (auto si None)
        return_tensor: Si True retorna tensor, si False retorna PIL Image
        
    Returns:
        Imagen generada (PIL Image o Tensor según return_tensor)
        
    Example:
        >>> img = generate_from_seed(42, "checkpoints/gan_final.pth")
        >>> img.save("output.png")
    """
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Cargar generador
    generator = load_generator(checkpoint_path, latent_dim=latent_dim, device=device)
    generator.eval()
    
    # Fijar semilla y generar ruido
    torch.manual_seed(seed)
    if device == 'cuda':
        torch.cuda.manual_seed(seed)
    
    noise = torch.randn(1, latent_dim, device=device)
    
    # Generar imagen
    with torch.no_grad():
        fake_image = generator(noise)
    
    if return_tensor:
        return fake_image.squeeze(0)  # [1, 28, 28]
    else:
        return tensor_to_pil(fake_image.squeeze(0))


def generate_batch(
    num_images: int,
    checkpoint_path: Union[str, Path],
    output_dir: str = "./generated",
    latent_dim: int = 100,
    seed: Optional[int] = None,
    device: Optional[str] = None
):
    """
    Genera un batch de imágenes y las guarda.
    
    Args:
        num_images: Número de imágenes a generar
        checkpoint_path: Ruta al checkpoint
        output_dir: Directorio de salida
        latent_dim: Dimensión latente
        seed: Semilla inicial (opcional)
        device: Device a usar
    """
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Cargar modelo
    generator = load_generator(checkpoint_path, latent_dim=latent_dim, device=device)
    generator.eval()
    
    if seed is not None:
        torch.manual_seed(seed)
        if device == 'cuda':
            torch.cuda.manual_seed(seed)
    
    logger.info(f"Generando {num_images} imágenes...")
    
    with torch.no_grad():
        for i in range(num_images):
            noise = torch.randn(1, latent_dim, device=device)
            fake_image = generator(noise)
            
            # Convertir a PIL y guardar
            pil_img = tensor_to_pil(fake_image.squeeze(0))
            output_path = Path(output_dir) / f"generated_{i:04d}.png"
            pil_img.save(output_path)
            
            if (i + 1) % 10 == 0:
                logger.info(f"Generadas {i+1}/{num_images} imágenes")
    
    logger.info(f"Imágenes guardadas en {output_dir}")


def interpolate_latent(
    seed_start: int,
    seed_end: int,
    steps: int,
    checkpoint_path: Union[str, Path],
    output_dir: str = "./interpolation",
    latent_dim: int = 100,
    device: Optional[str] = None
):
    """
    Genera interpolación entre dos vectores latentes.
    
    Args:
        seed_start: Semilla inicial
        seed_end: Semilla final
        steps: Número de pasos de interpolación
        checkpoint_path: Ruta al checkpoint
        output_dir: Directorio de salida
        latent_dim: Dimensión latente
        device: Device
    """
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    generator = load_generator(checkpoint_path, latent_dim=latent_dim, device=device)
    generator.eval()
    
    # Generar vectores latentes inicial y final
    torch.manual_seed(seed_start)
    z_start = torch.randn(1, latent_dim, device=device)
    
    torch.manual_seed(seed_end)
    z_end = torch.randn(1, latent_dim, device=device)
    
    logger.info(f"Interpolando entre seed {seed_start} y {seed_end} en {steps} pasos")
    
    with torch.no_grad():
        for i in range(steps):
            alpha = i / (steps - 1)
            z_interp = (1 - alpha) * z_start + alpha * z_end
            
            fake_image = generator(z_interp)
            pil_img = tensor_to_pil(fake_image.squeeze(0))
            
            output_path = Path(output_dir) / f"interp_{i:03d}.png"
            pil_img.save(output_path)
    
    logger.info(f"Interpolación guardada en {output_dir}")


def main():
    """Entry point para línea de comandos."""
    parser = argparse.ArgumentParser(description="Generar imágenes con GAN entrenada")
    
    subparsers = parser.add_subparsers(dest='command', help='Comando a ejecutar')
    
    # Comando: single
    parser_single = subparsers.add_parser('single', help='Generar una imagen')
    parser_single.add_argument('--seed', type=int, required=True, help='Semilla')
    parser_single.add_argument('--checkpoint', type=str, required=True, help='Checkpoint')
    parser_single.add_argument('--output', type=str, default='generated.png', help='Archivo salida')
    parser_single.add_argument('--latent_dim', type=int, default=100)
    
    # Comando: batch
    parser_batch = subparsers.add_parser('batch', help='Generar múltiples imágenes')
    parser_batch.add_argument('--num', type=int, required=True, help='Número de imágenes')
    parser_batch.add_argument('--checkpoint', type=str, required=True, help='Checkpoint')
    parser_batch.add_argument('--output_dir', type=str, default='./generated')
    parser_batch.add_argument('--seed', type=int, default=None)
    parser_batch.add_argument('--latent_dim', type=int, default=100)
    
    # Comando: interpolate
    parser_interp = subparsers.add_parser('interpolate', help='Interpolación latente')
    parser_interp.add_argument('--seed_start', type=int, required=True)
    parser_interp.add_argument('--seed_end', type=int, required=True)
    parser_interp.add_argument('--steps', type=int, default=10)
    parser_interp.add_argument('--checkpoint', type=str, required=True)
    parser_interp.add_argument('--output_dir', type=str, default='./interpolation')
    parser_interp.add_argument('--latent_dim', type=int, default=100)
    
    args = parser.parse_args()
    
    setup_logging()
    
    if args.command == 'single':
        img = generate_from_seed(
            seed=args.seed,
            checkpoint_path=args.checkpoint,
            latent_dim=args.latent_dim
        )
        img.save(args.output)
        logger.info(f"Imagen generada guardada en {args.output}")
    
    elif args.command == 'batch':
        generate_batch(
            num_images=args.num,
            checkpoint_path=args.checkpoint,
            output_dir=args.output_dir,
            latent_dim=args.latent_dim,
            seed=args.seed
        )
    
    elif args.command == 'interpolate':
        interpolate_latent(
            seed_start=args.seed_start,
            seed_end=args.seed_end,
            steps=args.steps,
            checkpoint_path=args.checkpoint,
            output_dir=args.output_dir,
            latent_dim=args.latent_dim
        )
    
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
