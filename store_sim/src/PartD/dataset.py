"""
Dataset module for Fashion-MNIST
=================================

Proporciona funciones para cargar y preprocesar el dataset Fashion-MNIST
para entrenamiento de la GAN.
"""

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from typing import Tuple
import logging

logger = logging.getLogger(__name__)


def get_fashion_mnist_loader(
    batch_size: int = 128,
    data_dir: str = "./data",
    num_workers: int = 2,
    shuffle: bool = True
) -> DataLoader:
    """
    Carga el dataset Fashion-MNIST y retorna un DataLoader.
    
    Args:
        batch_size: Tamaño del batch
        data_dir: Directorio donde descargar/cargar los datos
        num_workers: Número de workers para carga paralela
        shuffle: Si se debe barajar el dataset
        
    Returns:
        DataLoader con Fashion-MNIST transformado
    """
    # Transformaciones: normalizar a [-1, 1] para mejor entrenamiento GAN
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])  # Normaliza a [-1, 1]
    ])
    
    logger.info(f"Cargando Fashion-MNIST desde {data_dir}")
    
    dataset = datasets.FashionMNIST(
        root=data_dir,
        train=True,
        download=True,
        transform=transform
    )
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available()
    )
    
    logger.info(f"Dataset cargado: {len(dataset)} imágenes, batch_size={batch_size}")
    
    return dataloader


def get_sample_images(dataloader: DataLoader, num_samples: int = 64) -> torch.Tensor:
    """
    Obtiene un batch de imágenes reales para visualización.
    
    Args:
        dataloader: DataLoader de Fashion-MNIST
        num_samples: Número de muestras a obtener
        
    Returns:
        Tensor con imágenes [num_samples, 1, 28, 28]
    """
    real_batch = next(iter(dataloader))
    images = real_batch[0][:num_samples]
    return images


# Etiquetas de Fashion-MNIST para referencia
FASHION_MNIST_CLASSES = [
    'T-shirt/top',
    'Trouser',
    'Pullover',
    'Dress',
    'Coat',
    'Sandal',
    'Shirt',
    'Sneaker',
    'Bag',
    'Ankle boot'
]
