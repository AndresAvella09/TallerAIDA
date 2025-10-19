"""
GAN Models - Generator and Discriminator
=========================================

Implementa arquitecturas DCGAN (Deep Convolutional GAN) simplificadas
para generar imágenes de 28x28 píxeles.
"""

import torch
import torch.nn as nn
from typing import Tuple
import logging

logger = logging.getLogger(__name__)


class Generator(nn.Module):
    """
    Generador DCGAN que transforma ruido latente en imágenes 28x28.
    
    Arquitectura:
        Input: [batch, latent_dim] 
        -> Linear + Reshape: [batch, 128, 7, 7]
        -> ConvTranspose2d blocks
        -> Output: [batch, 1, 28, 28]
    """
    
    def __init__(self, latent_dim: int = 100, feature_maps: int = 64):
        """
        Args:
            latent_dim: Dimensión del vector de ruido latente
            feature_maps: Número base de feature maps
        """
        super(Generator, self).__init__()
        
        self.latent_dim = latent_dim
        self.init_size = 7  # Tamaño inicial 7x7
        
        # Proyección lineal del vector latente
        self.fc = nn.Sequential(
            nn.Linear(latent_dim, 128 * self.init_size * self.init_size),
            nn.BatchNorm1d(128 * self.init_size * self.init_size),
            nn.ReLU(True)
        )
        
        # Capas convolucionales transpuestas para upsampling
        self.conv_blocks = nn.Sequential(
            # 7x7 -> 14x14
            nn.ConvTranspose2d(128, feature_maps * 2, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(feature_maps * 2),
            nn.ReLU(True),
            
            # 14x14 -> 28x28
            nn.ConvTranspose2d(feature_maps * 2, feature_maps, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(feature_maps),
            nn.ReLU(True),
            
            # 28x28 -> 28x28 (capa final)
            nn.Conv2d(feature_maps, 1, kernel_size=3, stride=1, padding=1),
            nn.Tanh()  # Output en rango [-1, 1]
        )
        
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Inicializa pesos según recomendaciones DCGAN."""
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.normal_(m.weight.data, 0.0, 0.02)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.normal_(m.weight.data, 1.0, 0.02)
                nn.init.constant_(m.bias.data, 0)
    
    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Args:
            z: Vector latente [batch, latent_dim]
            
        Returns:
            Imagen generada [batch, 1, 28, 28]
        """
        x = self.fc(z)
        x = x.view(x.size(0), 128, self.init_size, self.init_size)
        img = self.conv_blocks(x)
        return img


class Discriminator(nn.Module):
    """
    Discriminador DCGAN que clasifica imágenes como reales o falsas.
    
    Arquitectura:
        Input: [batch, 1, 28, 28]
        -> Conv2d blocks con downsampling
        -> Output: [batch, 1] (logit, no sigmoid)
    """
    
    def __init__(self, feature_maps: int = 64):
        """
        Args:
            feature_maps: Número base de feature maps
        """
        super(Discriminator, self).__init__()
        
        self.conv_blocks = nn.Sequential(
            # 28x28 -> 14x14
            nn.Conv2d(1, feature_maps, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout2d(0.25),
            
            # 14x14 -> 7x7
            nn.Conv2d(feature_maps, feature_maps * 2, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(feature_maps * 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout2d(0.25),
            
            # 7x7 -> 3x3
            nn.Conv2d(feature_maps * 2, feature_maps * 4, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(feature_maps * 4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout2d(0.25),
        )
        
        # Clasificador final
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(feature_maps * 4 * 3 * 3, 1)
            # No sigmoid aquí - usamos BCEWithLogitsLoss
        )
        
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Inicializa pesos según recomendaciones DCGAN."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight.data, 0.0, 0.02)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.normal_(m.weight.data, 1.0, 0.02)
                nn.init.constant_(m.bias.data, 0)
    
    def forward(self, img: torch.Tensor) -> torch.Tensor:
        """
        Args:
            img: Imagen [batch, 1, 28, 28]
            
        Returns:
            Logit de clasificación [batch, 1]
        """
        x = self.conv_blocks(img)
        validity = self.fc(x)
        return validity


def create_models(latent_dim: int = 100, device: str = 'cpu') -> Tuple[Generator, Discriminator]:
    """
    Factory function para crear y mover modelos al device.
    
    Args:
        latent_dim: Dimensión del espacio latente
        device: 'cpu' o 'cuda'
        
    Returns:
        Tupla (generator, discriminator)
    """
    generator = Generator(latent_dim=latent_dim).to(device)
    discriminator = Discriminator().to(device)
    
    logger.info(f"Modelos creados en device: {device}")
    logger.info(f"Generator params: {sum(p.numel() for p in generator.parameters()):,}")
    logger.info(f"Discriminator params: {sum(p.numel() for p in discriminator.parameters()):,}")
    
    return generator, discriminator
