# Arquitectura y Flujo de Datos

## Diagrama de Componentes

```
┌─────────────────────────────────────────────────────────────────┐
│                         Part D - GAN System                      │
└─────────────────────────────────────────────────────────────────┘

┌──────────────┐      ┌──────────────┐      ┌──────────────┐
│   dataset.py │      │   models.py  │      │   train.py   │
│              │      │              │      │              │
│ Fashion-MNIST│─────▶│  Generator   │◀─────│  Training    │
│   Loader     │      │              │      │    Loop      │
│              │      │ Discriminator│      │              │
└──────────────┘      └──────────────┘      └──────────────┘
                             │                      │
                             │                      ▼
                             │              ┌──────────────┐
                             │              │ Checkpoints  │
                             │              │ (.pth files) │
                             │              └──────────────┘
                             │                      │
                             ▼                      │
                      ┌──────────────┐             │
                      │ generate.py  │◀────────────┘
                      │              │
                      │ Image Gen    │
                      └──────────────┘
                             │
                ┌────────────┴────────────┐
                │                         │
                ▼                         ▼
        ┌──────────────┐          ┌──────────────┐
        │ pygame_ui.py │          │  Your App    │
        │              │          │              │
        │ Interactive  │          │ Integration  │
        │     UI       │          │              │
        └──────────────┘          └──────────────┘
```

## Flujo de Entrenamiento

```
1. PREPARACIÓN
   ┌─────────────────────┐
   │ dataset.py          │
   │ - Descarga MNIST    │
   │ - Normaliza [-1,1]  │
   │ - Crea DataLoader   │
   └──────────┬──────────┘
              │
              ▼
2. INICIALIZACIÓN
   ┌─────────────────────┐
   │ models.py           │
   │ - Crea Generator    │
   │ - Crea Discrim.     │
   │ - Init weights      │
   └──────────┬──────────┘
              │
              ▼
3. TRAINING LOOP (train.py)
   ┌─────────────────────────────────────┐
   │ For each epoch:                     │
   │   For each batch:                   │
   │                                     │
   │   1) Train Discriminator:           │
   │      ┌─────────────────────────┐    │
   │      │ Real images → D → Loss  │    │
   │      │ Fake images → D → Loss  │    │
   │      │ Backprop & Update D     │    │
   │      └─────────────────────────┘    │
   │                                     │
   │   2) Train Generator:               │
   │      ┌─────────────────────────┐    │
   │      │ Noise → G → Fake        │    │
   │      │ Fake → D → Loss         │    │
   │      │ Backprop & Update G     │    │
   │      └─────────────────────────┘    │
   │                                     │
   │   Save checkpoint every epoch       │
   │   Generate samples for visual       │
   └─────────────────────────────────────┘
              │
              ▼
4. OUTPUT
   ┌─────────────────────┐
   │ checkpoints/        │
   │ - gan_epoch_*.pth   │
   │ - gan_final.pth     │
   │                     │
   │ samples/            │
   │ - epoch_*.png       │
   └─────────────────────┘
```

## Flujo de Generación

```
USER INPUT
   ├─ UI (pygame_ui.py)
   │   └─ Seed slider: 0-10000
   │
   └─ CLI (generate.py)
       └─ --seed 42

         │
         ▼
┌────────────────────────┐
│ generate_from_seed()   │
│                        │
│ 1) Set seed            │
│ 2) Generate noise z    │
│    z ~ N(0,1)^100      │
└───────────┬────────────┘
            │
            ▼
┌────────────────────────┐
│ Generator Forward      │
│                        │
│ z [100]                │
│  ↓ Linear              │
│ [128*7*7]              │
│  ↓ Reshape             │
│ [128,7,7]              │
│  ↓ ConvT2d             │
│ [128,14,14]            │
│  ↓ ConvT2d             │
│ [64,28,28]             │
│  ↓ Conv2d + Tanh       │
│ [1,28,28] ∈ [-1,1]     │
└───────────┬────────────┘
            │
            ▼
┌────────────────────────┐
│ Post-processing        │
│                        │
│ 1) Denormalize         │
│    [-1,1] → [0,255]    │
│ 2) To PIL/numpy        │
│ 3) Optional scale      │
└───────────┬────────────┘
            │
            ▼
      OUTPUT IMAGE
    28x28 grayscale
```

## Arquitectura Detallada

### Generator Architecture
```
Input: Latent Vector z [batch, 100]
│
├─ FC Layer 1
│  ├─ Linear(100, 128*7*7)
│  ├─ BatchNorm1d(128*7*7)
│  └─ ReLU
│  Output: [batch, 128*7*7]
│
├─ Reshape
│  Output: [batch, 128, 7, 7]
│
├─ ConvBlock 1
│  ├─ ConvTranspose2d(128, 128, k=4, s=2, p=1)
│  ├─ BatchNorm2d(128)
│  └─ ReLU
│  Output: [batch, 128, 14, 14]
│
├─ ConvBlock 2
│  ├─ ConvTranspose2d(128, 64, k=4, s=2, p=1)
│  ├─ BatchNorm2d(64)
│  └─ ReLU
│  Output: [batch, 64, 28, 28]
│
└─ Output Layer
   ├─ Conv2d(64, 1, k=3, s=1, p=1)
   └─ Tanh
   Output: [batch, 1, 28, 28] ∈ [-1, 1]

Total params: ~500K
```

### Discriminator Architecture
```
Input: Image [batch, 1, 28, 28]
│
├─ ConvBlock 1
│  ├─ Conv2d(1, 64, k=4, s=2, p=1)
│  ├─ LeakyReLU(0.2)
│  └─ Dropout2d(0.25)
│  Output: [batch, 64, 14, 14]
│
├─ ConvBlock 2
│  ├─ Conv2d(64, 128, k=4, s=2, p=1)
│  ├─ BatchNorm2d(128)
│  ├─ LeakyReLU(0.2)
│  └─ Dropout2d(0.25)
│  Output: [batch, 128, 7, 7]
│
├─ ConvBlock 3
│  ├─ Conv2d(128, 256, k=4, s=2, p=1)
│  ├─ BatchNorm2d(256)
│  ├─ LeakyReLU(0.2)
│  └─ Dropout2d(0.25)
│  Output: [batch, 256, 3, 3]
│
├─ Flatten
│  Output: [batch, 256*3*3]
│
└─ FC Layer
   └─ Linear(256*3*3, 1)
   Output: [batch, 1] (logit)

Total params: ~400K
```

## Configuration Options

```
Training:
  ├─ latent_dim: 100 (fixed by architecture)
  ├─ epochs: 1-100 (default 10)
  ├─ batch_size: 32-512 (default 128)
  ├─ lr: 1e-5 to 1e-3 (default 2e-4)
  ├─ beta1: 0.5 (Adam momentum)
  └─ device: cpu/cuda

Generation:
  ├─ seed: 0-∞ (for reproducibility)
  ├─ checkpoint: path to .pth
  └─ return_tensor: bool (PIL vs Tensor)

UI:
  ├─ window_size: (width, height)
  └─ checkpoint: path to .pth
```

---