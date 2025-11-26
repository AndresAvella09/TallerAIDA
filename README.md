# Simulador de Tienda - Taller AIDA

Proyecto integral de simulación y análisis de retail usando múltiples técnicas de IA: algoritmos de búsqueda (BFS/A*), redes neuronales LSTM, aprendizaje por refuerzo (RL) y redes generativas adversarias (GAN).

## 📋 Estructura del Proyecto

```
TallerAIDA/
├── store_sim/
│   ├── data/
│   │   └── retail_sales_dataset.csv   # Dataset de productos
│   └── src/
│       ├── PartA/                      # Simulación visual con algoritmos de búsqueda
│       │   ├── main.py                 # Simulación con pygame
│       │   ├── store.py                # Definición de la tienda
│       │   ├── customer.py             # Comportamiento de clientes
│       │   ├── simulation.py           # Motor de simulación
│       │   └── heatmap.py              # Generación de mapas de calor
│       ├── PartB/                      # Predicción de ventas con LSTM
│       │   ├── lstm_gui.py             # Interfaz gráfica interactiva
│       │   ├── lstm_model.py           # Modelo LSTM y trainer
│       │   ├── data_processor.py       # Procesamiento de datos
│       │   ├── ui_components.py        # Widgets pygame
│       │   ├── chart_generator.py      # Generación de gráficos
│       │   ├── experiments_replicable.py  # Experimentos reproducibles
│       │   └── RNN_LSTM.py             # Script original
│       ├── PartC/                      # Dynamic Pricing con RL
│       │   ├── dynamic_pricing_rl.py   # Ambiente de RL
│       │   ├── training.py             # Entrenamiento PPO
│       │   └── pygame_visualization.py # Visualización interactiva
│       └── PartD/                      # Generación de imágenes con GAN
│           ├── train.py                # Entrenamiento GAN
│           ├── generate.py             # Generación de imágenes
│           ├── models.py               # Arquitectura Generator/Discriminator
│           ├── dataset.py              # Carga de Fashion-MNIST
│           ├── utils.py                # Utilidades
│           └── gan_interface.py        # Interfaz interactiva
├── run_simulation.py                   # Script para ejecutar Parte A
├── run_lstm_gui.py                     # Script para ejecutar Parte B GUI
└── requirements.txt                    # Dependencias del proyecto
```

## 🚀 Instalación

### 1. Crear y activar entorno virtual:

**Windows (PowerShell):**
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

**Linux/macOS:**
```bash
python -m venv .venv
source .venv/bin/activate
```

### 2. Instalar dependencias:

```bash
pip install -r requirements.txt
```

Intentar instalar lo anterior, si hay errores ingrese al `requirements.txt` y descomente las lineas comentadas y comente las anteriores y vuelva a intentar.


### 3. Instalar PyTorch (para PartB y PartD):

**CPU:**
```bash
pip install torch torchvision torchaudio
```

**GPU (CUDA 11.8):**
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### 4. Instalar Stable-Baselines3 (para PartC):

```bash
pip install stable-baselines3
```

## 💻 Ejecución

### Parte A - Simulación Visual con BFS/A*

**Script de ejecución (Recomendado):**
```powershell
python run_simulation.py
```

**Como módulo:**
```powershell
python -m store_sim.src.PartA.main
```

Abre una ventana con pygame mostrando clientes navegando por la tienda en tiempo real usando algoritmos de búsqueda.

---

### Parte B - Predicción de Ventas con LSTM

**Interfaz Gráfica Interactiva (Recomendado):**
```powershell
python run_lstm_gui.py
```

**Alternativamente:**
```powershell
python -m store_sim.src.PartB.lstm_gui
```

**Experimentos Reproducibles: (Resultados Consola)**
```powershell
python -m store_sim.src.PartB.experiments_replicable
```

---

### Parte C - Dynamic Pricing con Reinforcement Learning

Ejecutar en orden

**1.Entrenar modelo PPO:**
```powershell
python -m store_sim.src.PartC.training
```

**2.Visualización interactiva:**
```powershell
python -m store_sim.src.PartC.pygame_visualization
```

---

### Parte D - Generación de Imágenes con GAN

Es necesario primero entrenar el modelo GAN antes de generar imágenes.

**Entrenar GAN (Fashion-MNIST):**
```powershell
# Entrenamiento básico (10 épocas)
python -m store_sim.src.PartD.train

# Entrenamiento personalizado
python -m store_sim.src.PartD.train --epochs 20 --batch_size 256 --lr 0.0002
```

**Generar interfaz:**
```powershell
python -m store_sim.src.PartD.gan_interface
```

**Generar imágenes: (Consola)**
```powershell
# Generar una imagen
python -m store_sim.src.PartD.generate single --seed 42 --checkpoint ./checkpoints/gan_final.pth

# Generar múltiples imágenes
python -m store_sim.src.PartD.generate batch --num 100 --checkpoint ./checkpoints/gan_final.pth

# Interpolación en espacio latente
python -m store_sim.src.PartD.generate interpolate --seed_start 0 --seed_end 1000 --steps 20
```

## 📊 ¿Qué hace cada parte?

### Parte A: Simulación Visual con Algoritmos de Búsqueda

- Crea una tienda como grid de 8x6 celdas
- Genera clientes que se mueven desde la entrada hacia secciones específicas
- Compara algoritmos **BFS** (Breadth-First Search) vs **A*** (A-Star)
- Visualiza el movimiento en tiempo real con pygame (clientes = círculos azules)
- Genera mapa de calor del tráfico al finalizar

**Métricas calculadas:**
- Longitud promedio de caminos
- Score de eficiencia
- Identificación de cuellos de botella

**Archivos generados:**
- `bfs_report.txt` / `astar_report.txt` - Reportes detallados
- `heatmap_bfs.png` / `heatmap_astar.png` - Mapas de calor
- `algorithm_comparison.png` - Gráfico comparativo
- `combined_report.txt` - Reporte combinado

---

### Parte B: Predicción de Ventas con LSTM

Interfaz gráfica interactiva para entrenar redes **LSTM** (Long Short-Term Memory) y predecir ventas por categoría de producto.

**Características:**
- ✅ Selección de categoría mediante dropdown
- ✅ Control de épocas con slider (1000-5000)
- ✅ Entrenamiento en tiempo real
- ✅ Visualización de serie temporal con bandas de confianza
- ✅ Métricas: MAE, RMSE
- ✅ Predicción próxima semana
- ✅ Gráfico de pérdida durante entrenamiento

**Arquitectura LSTM:**
- 2 capas LSTM con 64 unidades ocultas
- Dropout: 0.2
- Optimizador: Adam (lr=1e-3)
- Loss: MSE

---

### Parte C: Dynamic Pricing con Reinforcement Learning

Sistema de **fijación dinámica de precios** usando algoritmo **PPO** (Proximal Policy Optimization) de Stable-Baselines3.

**Características:**
- Ambiente de RL personalizado basado en datos históricos de ventas
- Entrenamiento de agente PPO para maximizar ingresos
- Visualización interactiva con pygame mostrando:
  - Precios en tiempo real
  - Demanda estimada
  - Ingresos acumulados
  - Gráficos de evolución
- Control de hiperparámetros mediante sliders

**Métricas:**
- Ingresos totales por episodio
- Precio promedio óptimo
- Demanda satisfecha

---

### Parte D: Generación de Imágenes con GAN

Implementación de **DCGAN** (Deep Convolutional GAN) para generar imágenes promocionales basadas en Fashion-MNIST.

**Arquitectura:**
- **Generador**: ConvTranspose2d (latent_dim=100 → 28x28 grayscale)
- **Discriminador**: Conv2d con BatchNorm + LeakyReLU
- Optimizador: Adam (lr=2e-4, beta1=0.5)
- Loss: BCEWithLogitsLoss

**Funcionalidades:**
- Entrenamiento con checkpoints automáticos
- Generación de imágenes individuales o en batch
- Interpolación en espacio latente
- Control mediante semillas (seeds) para reproducibilidad
- Visualización de progreso durante entrenamiento

**Salidas:**
- Checkpoints del modelo en `./checkpoints/`
- Imágenes de muestra en `./samples/`
- Imágenes generadas según parámetros

## 🔧 Componentes Principales

### Parte A:
- **Store**: Representa la tienda con un grafo de NetworkX
- **Customer**: Modela el comportamiento de un cliente navegando por la tienda
- **Simulation**: Motor que maneja múltiples clientes y registra tráfico
- **StoreAnalytics**: Calcula métricas y genera análisis

### Parte B:
- **DataProcessor**: Carga y prepara datos de ventas
- **LSTMTrainer**: Entrenamiento del modelo LSTM
- **LSTMApp**: Interfaz gráfica con pygame
- **ChartGenerator**: Generación de gráficos matplotlib

### Parte C:
- **DynamicPricingEnv**: Ambiente de RL personalizado (Gym)
- **PPO**: Agente de aprendizaje por refuerzo
- **PygameVisualization**: Interfaz interactiva de simulación

### Parte D:
- **Generator**: Red neuronal generadora (DCGAN)
- **Discriminator**: Red neuronal discriminadora
- **GANTrainer**: Proceso de entrenamiento adversario
- **GANInterface**: Interfaz de generación interactiva

## 🐛 Solución de Problemas

### Error de imports relativos:
Asegúrate de ejecutar los scripts desde la raíz del proyecto (`TallerAIDA`), no desde subdirectorios.

### Error de pygame:
Si pygame no se inicia correctamente, verifica que el entorno virtual esté activado.

### Error: "No module named 'torch'":
```bash
pip install torch torchvision torchaudio
```

### Error: "No se encuentra el archivo CSV":
Verifica que existe `store_sim/data/retail_sales_dataset.csv`

### Error: "No module named 'stable_baselines3'":
```bash
pip install stable-baselines3
```

### Entrenamiento LSTM muy lento:
Reduce el número de épocas a 1000-1500 o usa GPU con PyTorch CUDA.

### GAN no converge:
Ajusta learning rate (`--lr 0.0001`) o aumenta épocas (`--epochs 50`).

## 📦 Dependencias

### Básicas (Todas las partes):
- **pygame==2.6.1** - Visualización e interfaces gráficas
- **networkx==3.5** - Grafos y algoritmos de búsqueda
- **pandas==2.3.3** - Manejo de datos
- **matplotlib==3.10.6** - Gráficos
- **seaborn==0.13.2** - Visualizaciones estadísticas
- **numpy==2.3.3** - Cálculos numéricos

### Adicionales para Parte B y D (LSTM/GAN):
- **torch** - Framework de deep learning
- **torchvision** - Datasets y transformaciones
- **scikit-learn** - Preprocesamiento y métricas

### Adicionales para Parte C (RL):
- **stable-baselines3** - Algoritmos de RL (PPO)
- **gymnasium** - Ambientes de RL

---

## 📚 Documentación Adicional

- **Parte B**: Ver `store_sim/src/PartB/README_LSTM_GUI.md`
- **Parte D**: Ver `store_sim/src/PartD/README.md` y `ARCHITECTURE.md`

---

## 🎯 Resumen de Tecnologías

| Parte | Técnica | Framework Principal | Objetivo |
|-------|---------|-------------------|----------|
| A | Algoritmos de Búsqueda | NetworkX + Pygame | Optimización de rutas en tienda |
| B | LSTM (Deep Learning) | PyTorch + Pygame | Predicción de ventas |
| C | Reinforcement Learning | Stable-Baselines3 (PPO) | Fijación dinámica de precios |
| D | GAN (Deep Learning) | PyTorch | Generación de imágenes promocionales |
