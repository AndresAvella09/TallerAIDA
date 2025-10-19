# Part D - Simple GAN for Promotion Images

Implementación de una **Generative Adversarial Network (GAN)** usando PyTorch para generar imágenes promocionales basadas en el dataset Fashion-MNIST.

## Características

- **Arquitectura DCGAN** (Deep Convolutional GAN) simplificada
- Generación de imágenes 28x28 píxeles en escala de grises
- Entrenamiento configurable con checkpoints automáticos
- Interfaz gráfica interactiva con Pygame
- Generación controlada mediante semillas (seeds)
- Interpolación en espacio latente
- Logging detallado del proceso

## Arquitectura

### Generador
```
Input: [batch, latent_dim=100]
  ↓
Linear + BatchNorm + ReLU → [batch, 128*7*7]
  ↓
Reshape → [batch, 128, 7, 7]
  ↓
ConvTranspose2d (128→128) → [batch, 128, 14, 14]
  ↓
ConvTranspose2d (128→64) → [batch, 64, 28, 28]
  ↓
Conv2d (64→1) + Tanh → [batch, 1, 28, 28]
```

### Discriminador
```
Input: [batch, 1, 28, 28]
  ↓
Conv2d (1→64) + LeakyReLU → [batch, 64, 14, 14]
  ↓
Conv2d (64→128) + BatchNorm + LeakyReLU → [batch, 128, 7, 7]
  ↓
Conv2d (128→256) + BatchNorm + LeakyReLU → [batch, 256, 3, 3]
  ↓
Flatten + Linear → [batch, 1]
```

### Hiperparámetros
- **Latent Dimension**: 100
- **Learning Rate**: 2e-4
- **Optimizer**: Adam (beta1=0.5, beta2=0.999)
- **Loss Function**: BCEWithLogitsLoss
- **Batch Size**: 128 (default)
- **Epochs**: 10 (default)

## Instalación

```bash
# Instalar dependencias
pip install -r requirements.txt

# O manualmente
pip install torch torchvision pygame matplotlib Pillow numpy
```

## Uso Rápido

### 1. Entrenar la GAN

```bash
# Entrenamiento básico (10 épocas)
python -m store_sim.src.PartD.train

# Entrenamiento personalizado
python -m store_sim.src.PartD.train \
    --epochs 20 \
    --batch_size 256 \
    --lr 0.0002 \
    --checkpoint_dir ./my_checkpoints \
    --samples_dir ./my_samples
```

**Parámetros disponibles:**
- `--latent_dim`: Dimensión del vector latente (default: 100)
- `--epochs`: Número de épocas (default: 10)
- `--batch_size`: Tamaño del batch (default: 128)
- `--lr`: Learning rate (default: 2e-4)
- `--data_dir`: Directorio para Fashion-MNIST (default: ./data)
- `--checkpoint_dir`: Directorio para checkpoints (default: ./checkpoints)
- `--samples_dir`: Directorio para muestras (default: ./samples)
- `--save_interval`: Guardar cada N épocas (default: 1)
- `--device`: cpu o cuda (auto-detect si no se especifica)

### 2. Generar Imágenes

#### Generar una sola imagen
```bash
python -m store_sim.src.PartD.generate single \
    --seed 42 \
    --checkpoint ./checkpoints/gan_final.pth \
    --output my_image.png
```

#### Generar múltiples imágenes
```bash
python -m store_sim.src.PartD.generate batch \
    --num 100 \
    --checkpoint ./checkpoints/gan_final.pth \
    --output_dir ./generated_images \
    --seed 12345
```

#### Interpolación en espacio latente
```bash
python -m store_sim.src.PartD.generate interpolate \
    --seed_start 0 \
    --seed_end 1000 \
    --steps 20 \
    --checkpoint ./checkpoints/gan_final.pth \
    --output_dir ./interpolation
```
