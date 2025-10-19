# Simulador de Tienda - Taller AIDA

Proyecto de simulación de comportamiento de clientes en una tienda usando algoritmos de búsqueda (BFS y A\*).

## 📋 Estructura del Proyecto

```
TallerAIDA/
├── store_sim/
│   ├── data/
│   │   └── retail_sales_dataset.csv   # Dataset de productos
│   └── src/
│       ├── PartA/                      # Simulación visual
│       │   ├── main.py                 # Simulación con pygame
│       │   ├── store.py                # Definición de la tienda
│       │   ├── customer.py             # Comportamiento de clientes
│       │   ├── simulation.py           # Motor de simulación
│       │   └── heatmap.py              # Generación de mapas de calor
│       └── PartB/                      # Análisis y experimentos
│           ├── analytics.py            # Métricas y análisis
│           └── task_b_experiments.py   # Comparación de algoritmos
├── run_simulation.py                   # Script para ejecutar Parte A
├── run_experiments.py                  # Script para ejecutar Parte B
└── requirements.txt                    # Dependencias del proyecto
```

## 🚀 Instalación

### 1. Activar el entorno virtual:

```bash
source .venv/bin/activate
```

### 2. Verificar dependencias (ya instaladas):

```bash
pip list
```

## 💻 Ejecución

### Opción 1: Scripts de ejecución (Recomendado)

#### Parte A - Simulación Visual:

```bash
python run_simulation.py
```

Abre una ventana con pygame mostrando clientes navegando por la tienda en tiempo real.

#### Parte B - Experimentos Comparativos:

```bash
python run_experiments.py
```

Ejecuta análisis comparativo entre BFS y A\*, genera reportes y gráficos.

### Opción 2: Como módulos de Python

#### Parte A:

```bash
python -m store_sim.src.PartA.main
```

#### Parte B:

```bash
python -m store_sim.src.PartB.task_b_experiments
```

## 📊 ¿Qué hace cada parte?

### Parte A: Simulación Visual

- Crea una tienda como grid de 8x6 celdas
- Genera clientes que se mueven desde la entrada hacia secciones específicas
- Visualiza el movimiento en tiempo real con pygame (clientes = círculos azules)
- Genera un mapa de calor del tráfico al finalizar

### Parte B: Análisis Comparativo

- Compara algoritmos BFS vs A\*
- Calcula métricas:
  - Longitud promedio de caminos
  - Score de eficiencia
  - Identificación de cuellos de botella
- Genera:
  - `bfs_report.txt` - Reporte detallado BFS
  - `astar_report.txt` - Reporte detallado A\*
  - `heatmap_bfs.png` - Mapa de calor BFS
  - `heatmap_astar.png` - Mapa de calor A\*
  - `algorithm_comparison.png` - Gráfico comparativo
  - `combined_report.txt` - Reporte combinado

## 🔧 Componentes Principales

- **Store**: Representa la tienda con un grafo de NetworkX
- **Customer**: Modela el comportamiento de un cliente navegando por la tienda
- **Simulation**: Motor que maneja múltiples clientes y registra tráfico
- **StoreAnalytics**: Calcula métricas y genera análisis

## 🐛 Solución de Problemas

### Error de imports relativos:

Asegúrate de ejecutar los scripts desde la raíz del proyecto (`TallerAIDA`), no desde subdirectorios.

### Error de pygame:

Si pygame no se inicia correctamente, verifica que el entorno virtual esté activado.

## 📦 Dependencias

- pygame==2.6.1 - Visualización
- networkx==3.5 - Grafos y algoritmos de búsqueda
- pandas==2.3.3 - Manejo de datos
- matplotlib==3.10.6 - Gráficos
- seaborn==0.13.2 - Visualizaciones
- numpy==2.3.3 - Cálculos numéricos
