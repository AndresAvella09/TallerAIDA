import pandas as pd
import numpy as np
import os
from pathlib import Path
from stable_baselines3 import PPO
from .dynamic_pricing_rl import DynamicPricingEnv  # Importación relativa explícita

# Obtener ruta absoluta al archivo de datos
current_dir = Path(__file__).parent
data_path = current_dir.parent.parent.parent / "store_sim" / "data" / "retail_sales_dataset.csv"

# Verificar que el archivo existe
if not data_path.exists():
    # Intentar ruta alternativa
    data_path = Path("store_sim/data/retail_sales_dataset.csv")
    if not data_path.exists():
        raise FileNotFoundError(
            f"No se encontró el archivo de datos en: {data_path}\n"
            "Asegúrate de que existe: store_sim/data/retail_sales_dataset.csv"
        )

# Cargar datos
print(f"Cargando datos desde: {data_path}")
df = pd.read_csv(data_path)

# Crear ambiente
env = DynamicPricingEnv(
    data=df, 
    window_size=10, 
    max_steps=100,
    price_min=20,  # Precio mínimo basado en el dataset
    price_max=550,  # Precio máximo basado en el dataset
    seed=42
)

# Entrenar el modelo PPO
print("Entrenando modelo PPO...")
model = PPO("MlpPolicy", env, verbose=1, learning_rate=0.0003, n_steps=2048)
model.learn(total_timesteps=50000)

# Guardar el modelo
model.save("ppo_dynamic_pricing")
print("Modelo guardado como 'ppo_dynamic_pricing'")

# Evaluar el modelo
print("\n=== Evaluación del Modelo ===")
obs, _ = env.reset()
total_revenue = 0
episode_revenues = []

for step in range(env.max_steps):
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, done, truncated, info = env.step(action)
    total_revenue += reward
    episode_revenues.append(info['revenue'])
    
    if step % 10 == 0:  # Mostrar cada 10 pasos
        env.render()
    
    if done or truncated:
        break

print(f"\n=== Resultados ===")
print(f"Total revenue (RL): ${total_revenue:.2f}")
print(f"Average revenue per step: ${np.mean(episode_revenues):.2f}")

# Comparar con estrategia estática
static_price = df["Price per Unit"].mean()
static_quantity = df["Quantity"].mean()
static_revenue = static_price * static_quantity * env.max_steps

print(f"\nStatic revenue (baseline): ${static_revenue:.2f}")
print(f"Improvement: {((total_revenue - static_revenue) / static_revenue * 100):.2f}%")