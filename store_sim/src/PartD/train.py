"""
Training script for Fashion-MNIST GAN
======================================

Entrena el Generador y Discriminador con el algoritmo GAN estándar.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
import logging
import argparse
from typing import Optional
import time

from .models import create_models
from .dataset import get_fashion_mnist_loader
from .utils import save_checkpoint, save_sample_images, setup_logging

logger = logging.getLogger(__name__)


def train_gan(
    latent_dim: int = 100,
    epochs: int = 10,
    batch_size: int = 128,
    lr: float = 2e-4,
    beta1: float = 0.5,
    data_dir: str = "./data",
    checkpoint_dir: str = "./checkpoints",
    samples_dir: str = "./samples",
    save_interval: int = 1,
    device: Optional[str] = None
):
    """
    Función principal de entrenamiento GAN.
    
    Args:
        latent_dim: Dimensión del vector latente
        epochs: Número de épocas de entrenamiento
        batch_size: Tamaño del batch
        lr: Learning rate para Adam
        beta1: Beta1 para Adam optimizer
        data_dir: Directorio de datos
        checkpoint_dir: Directorio para guardar checkpoints
        samples_dir: Directorio para guardar muestras generadas
        save_interval: Guardar checkpoint cada N épocas
        device: 'cpu' o 'cuda' (auto-detect si None)
    """
    # Setup
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)
    Path(samples_dir).mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Iniciando entrenamiento en device: {device}")
    logger.info(f"Parámetros: epochs={epochs}, batch_size={batch_size}, lr={lr}")
    
    # Cargar datos
    dataloader = get_fashion_mnist_loader(
        batch_size=batch_size,
        data_dir=data_dir
    )
    
    # Crear modelos
    generator, discriminator = create_models(latent_dim=latent_dim, device=device)
    
    # Loss y optimizadores
    criterion = nn.BCEWithLogitsLoss()
    
    optimizer_G = optim.Adam(
        generator.parameters(),
        lr=lr,
        betas=(beta1, 0.999)
    )
    
    optimizer_D = optim.Adam(
        discriminator.parameters(),
        lr=lr,
        betas=(beta1, 0.999)
    )
    
    # Labels para entrenamiento
    real_label = 1.0
    fake_label = 0.0
    
    # Vector fijo para visualización
    fixed_noise = torch.randn(64, latent_dim, device=device)
    
    logger.info("Comenzando entrenamiento...")
    
    # Loop de entrenamiento
    for epoch in range(epochs):
        epoch_start = time.time()
        d_losses = []
        g_losses = []
        
        for i, (real_images, _) in enumerate(dataloader):
            batch_size_actual = real_images.size(0)
            real_images = real_images.to(device)
            
            # ===========================
            # Entrenar Discriminador
            # ===========================
            optimizer_D.zero_grad()
            
            # Loss en imágenes reales
            label_real = torch.full((batch_size_actual, 1), real_label, device=device)
            output_real = discriminator(real_images)
            loss_d_real = criterion(output_real, label_real)
            
            # Loss en imágenes falsas
            noise = torch.randn(batch_size_actual, latent_dim, device=device)
            fake_images = generator(noise)
            label_fake = torch.full((batch_size_actual, 1), fake_label, device=device)
            output_fake = discriminator(fake_images.detach())
            loss_d_fake = criterion(output_fake, label_fake)
            
            # Total discriminador
            loss_d = loss_d_real + loss_d_fake
            loss_d.backward()
            optimizer_D.step()
            
            # ===========================
            # Entrenar Generador
            # ===========================
            optimizer_G.zero_grad()
            
            # Queremos que el discriminador clasifique las fake como reales
            label_real_for_g = torch.full((batch_size_actual, 1), real_label, device=device)
            output_fake_for_g = discriminator(fake_images)
            loss_g = criterion(output_fake_for_g, label_real_for_g)
            loss_g.backward()
            optimizer_G.step()
            
            # Guardar losses
            d_losses.append(loss_d.item())
            g_losses.append(loss_g.item())
            
            # Log progreso
            if i % 50 == 0:
                logger.info(
                    f"[Epoch {epoch+1}/{epochs}] [Batch {i}/{len(dataloader)}] "
                    f"Loss_D: {loss_d.item():.4f} Loss_G: {loss_g.item():.4f}"
                )
        
        # Estadísticas de época
        avg_d_loss = sum(d_losses) / len(d_losses)
        avg_g_loss = sum(g_losses) / len(g_losses)
        epoch_time = time.time() - epoch_start
        
        logger.info(
            f"Época {epoch+1}/{epochs} completada en {epoch_time:.2f}s | "
            f"Avg Loss_D: {avg_d_loss:.4f} | Avg Loss_G: {avg_g_loss:.4f}"
        )
        
        # Guardar muestras
        with torch.no_grad():
            fake_samples = generator(fixed_noise)
            save_path = Path(samples_dir) / f"epoch_{epoch+1:03d}.png"
            save_sample_images(fake_samples, save_path)
            logger.info(f"Muestras guardadas en {save_path}")
        
        # Guardar checkpoint
        if (epoch + 1) % save_interval == 0 or (epoch + 1) == epochs:
            checkpoint_path = Path(checkpoint_dir) / f"gan_epoch_{epoch+1:03d}.pth"
            save_checkpoint(
                generator=generator,
                discriminator=discriminator,
                optimizer_G=optimizer_G,
                optimizer_D=optimizer_D,
                epoch=epoch + 1,
                path=checkpoint_path
            )
            logger.info(f"Checkpoint guardado en {checkpoint_path}")
    
    logger.info("¡Entrenamiento completado!")
    
    # Guardar modelo final
    final_path = Path(checkpoint_dir) / "gan_final.pth"
    save_checkpoint(
        generator=generator,
        discriminator=discriminator,
        optimizer_G=optimizer_G,
        optimizer_D=optimizer_D,
        epoch=epochs,
        path=final_path
    )
    logger.info(f"Modelo final guardado en {final_path}")


def main():
    """Entry point para ejecutar desde línea de comandos."""
    parser = argparse.ArgumentParser(description="Entrenar GAN para Fashion-MNIST")
    parser.add_argument('--latent_dim', type=int, default=100, help='Dimensión del vector latente')
    parser.add_argument('--epochs', type=int, default=10, help='Número de épocas')
    parser.add_argument('--batch_size', type=int, default=128, help='Tamaño del batch')
    parser.add_argument('--lr', type=float, default=2e-4, help='Learning rate')
    parser.add_argument('--data_dir', type=str, default='./data', help='Directorio de datos')
    parser.add_argument('--checkpoint_dir', type=str, default='./checkpoints', help='Dir checkpoints')
    parser.add_argument('--samples_dir', type=str, default='./samples', help='Dir muestras')
    parser.add_argument('--save_interval', type=int, default=1, help='Guardar cada N épocas')
    parser.add_argument('--device', type=str, default=None, help='cpu o cuda')
    
    args = parser.parse_args()
    
    setup_logging()
    
    train_gan(
        latent_dim=args.latent_dim,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        data_dir=args.data_dir,
        checkpoint_dir=args.checkpoint_dir,
        samples_dir=args.samples_dir,
        save_interval=args.save_interval,
        device=args.device
    )


if __name__ == "__main__":
    main()
