# 🧠 LSTM Sales Forecasting - Interfaz Gráfica

Interfaz gráfica interactiva con **pygame** para entrenar y visualizar modelos LSTM de predicción de ventas por categoría de producto.

## 📸 Características

- ✅ **Selección de categoría** mediante dropdown
- ✅ **Control de épocas** con slider (1000-5000)
- ✅ **Entrenamiento en tiempo real** sin bloquear la UI
- ✅ **Gráfico de serie temporal** con datos de entrenamiento y predicciones
- ✅ **Bandas de confianza** basadas en MAE y RMSE
- ✅ **Predicción próxima semana** con visualización
- ✅ **Tabla de métricas** detallada
- ✅ **Gráfico de pérdida** durante entrenamiento
- ✅ **Interfaz oscura** profesional

## 🚀 Instalación Rápida

### 1. Instalar PyTorch

**macOS / Linux:**

```bash
pip install torch torchvision torchaudio
```

**Para GPU CUDA (opcional):**

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### 2. Instalar dependencias adicionales

```bash
cd store_sim/src/PartB
pip install -r requirements_lstm_gui.txt
```

## 💻 Ejecución

Desde la raíz del proyecto:

```bash
python run_lstm_gui.py
```

O directamente:

```bash
cd store_sim/src/PartB
python lstm_gui.py
```

## 🎮 Cómo Usar la Interfaz

### Panel Izquierdo - Controles

1. **Seleccionar Categoría**

   - Click en el dropdown "Categoría de Producto"
   - Selecciona la categoría que deseas analizar

2. **Ajustar Épocas**

   - Arrastra el slider "Épocas"
   - Rango: 1000 - 5000 épocas
   - Recomendado: 2000-3000 para balance velocidad/precisión

3. **Entrenar Modelo**
   - Click en "🚀 Entrenar Modelo"
   - La barra de progreso muestra el avance
   - El cuadro de estado muestra la pérdida en tiempo real

### Panel Derecho - Visualizaciones

#### 1. **Gráfico Principal (Superior)**

- **Verde**: Datos de entrenamiento
- **Azul**: Datos reales de prueba
- **Rojo**: Predicciones del modelo
- **Naranja (banda)**: Intervalo de confianza ±MAE
- **Rojo claro (banda)**: Intervalo de confianza ±RMSE
- **Morado**: Predicción próxima semana

#### 2. **Tabla de Métricas (Inferior Izquierdo)**

- Categoría seleccionada
- MAE (Mean Absolute Error)
- RMSE (Root Mean Squared Error)
- Predicción próxima semana
- Tamaño conjuntos train/test

#### 3. **Gráfico de Pérdida (Inferior Derecho)**

- Evolución de la pérdida (MSE) durante entrenamiento
- Ayuda a identificar convergencia del modelo

## 📁 Estructura de Archivos

```
store_sim/src/PartB/
├── lstm_gui.py              # Aplicación principal
├── lstm_model.py            # Modelo LSTM y trainer
├── data_processor.py        # Carga y preparación de datos
├── ui_components.py         # Widgets pygame (Button, Slider, Dropdown, etc.)
├── chart_generator.py       # Generación de gráficos matplotlib
├── requirements_lstm_gui.txt  # Dependencias
└── RNN_LSTM.py              # Script original (referencia)
```

## 🔧 Arquitectura del Sistema

### Flujo de Datos

```
CSV Data → DataProcessor → Train/Test Split
                              ↓
                         LSTM Model
                              ↓
                    Predictions + Metrics
                              ↓
                      ChartGenerator
                              ↓
                    Pygame Surfaces
                              ↓
                        UI Display
```

### Componentes Principales

1. **DataProcessor**

   - Carga `retail_sales_dataset.csv`
   - Agrupa ventas por categoría y semana
   - Normaliza con MinMaxScaler
   - Crea secuencias temporales (ventanas de 10 semanas)

2. **LSTMTrainer**

   - Red LSTM: 2 capas, 64 unidades ocultas
   - Dropout: 0.2
   - Optimizador: Adam (lr=1e-3)
   - Loss: MSE

3. **UI Components**

   - Widgets personalizados con pygame
   - Interacción mouse y eventos
   - Diseño responsive

4. **ChartGenerator**
   - matplotlib con backend Agg
   - Convierte plots a superficies pygame
   - Tema oscuro consistente

## 📊 Interpretación de Resultados

### Métricas

- **MAE (Mean Absolute Error)**: Error promedio absoluto en unidades de ventas ($)

  - Más bajo = mejor
  - Interpretación: "En promedio, el modelo se equivoca por $X"

- **RMSE (Root Mean Squared Error)**: Penaliza más los errores grandes
  - Más bajo = mejor
  - Más sensible a outliers que MAE

### Bandas de Confianza

- **±MAE**: Rango donde caen ~68% de las predicciones
- **±RMSE**: Rango más amplio, captura errores mayores

### Predicción Próxima Semana

- Usa los últimos 10 datos reales
- Predice la venta de la semana siguiente
- Útil para planificación de inventario

## 🐛 Solución de Problemas

### Error: "No module named 'torch'"

```bash
pip install torch
```

### Error: "No se encuentra el archivo CSV"

Verifica que existe `store_sim/data/retail_sales_dataset.csv`

### Entrenamiento muy lento

- Reduce épocas a 1000-1500
- El código entrena 10 épocas por frame (600 épocas/segundo aprox)

### La ventana no responde durante entrenamiento

- Es normal, el entrenamiento es intensivo
- La UI se actualiza cada 10 épocas
- Espera a que termine o reduce épocas

### Gráficos no se generan

Verifica que matplotlib esté instalado:

```bash
pip install matplotlib
```

## 🎯 Tips de Uso

1. **Primera vez**: Empieza con 1000 épocas para probar
2. **Categorías con pocos datos**: Pueden tener métricas altas
3. **Comparar categorías**: Entrena varias y compara MAE/RMSE
4. **Épocas óptimas**: 2000-3000 es buen balance
5. **CPU vs GPU**: Con GPU es mucho más rápido (pytorch con CUDA)

## 🔮 Mejoras Futuras

- [ ] Guardar modelos entrenados
- [ ] Exportar predicciones a CSV
- [ ] Comparación entre múltiples categorías
- [ ] Ajuste de hiperparámetros en UI
- [ ] Histórico de entrenamientos
- [ ] Modo batch (todas las categorías)

## 📝 Créditos

Basado en el notebook original `RNN_LSTM.py` con mejoras de UX y visualización interactiva.
