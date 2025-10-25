import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd
import random

class DynamicPricingEnv(gym.Env):

    def __init__(self, data, window_size=10, max_steps=100, price_min=10, price_max=600, seed=None):
        super().__init__()
        self.data = data.reset_index(drop=True)
        self.window_size = window_size
        self.max_steps = max_steps
        self.price_min = price_min
        self.price_max = price_max

        # Calcular estadísticas históricas para referencia
        self.reference_price = float(self.data["Price per Unit"].mean())
        self.reference_sales = float(self.data["Quantity"].mean())

        # derive segments mapping from data
        self.segments = list(self.data["Product Category"].unique())
        self.n_segments = len(self.segments)
        self.segment2idx = {s: i for i, s in enumerate(self.segments)}

        self.action_space = spaces.Discrete(3)  # bajar, mantener o subir precio

        # observation: price_norm (1) + last_n_sales (window_size) + segment one-hot (n_segments)
        obs_dim = 1 + self.window_size + self.n_segments
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32)

        # Estado inicial
        if seed is not None:
            self.seed(seed)
        self.reset()

    def seed(self, seed=None):
        np.random.seed(seed)
        random.seed(seed)
        return [seed]

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)
        
        self.current_step = 0
        self.sales_history = [0.0] * self.window_size
        # Precio inicial = promedio histórico
        self.price = float(np.clip(self.reference_price, self.price_min, self.price_max))
        self._prev_price = self.price  # Para calcular cambios de precio
        # Elije un segmento random para el episodio
        self.segment = random.choice(self.segments) if self.n_segments > 0 else None
        self.cumulative_revenue = 0.0
        return self._get_state(), {}
    
    def _get_state(self):
        # Precio normalizado por el precio historico promedio
        max_price = max(1.0, float(self.data["Price per Unit"].max()))
        price_norm = self.price / max_price
        sales_vector = np.array(self.sales_history[-self.window_size:], dtype=np.float32)
        # one-hot segment encoding
        seg_onehot = np.zeros(self.n_segments, dtype=np.float32)
        if self.segment is not None:
            seg_onehot[self.segment2idx[self.segment]] = 1.0
        obs = np.concatenate(([price_norm], sales_vector, seg_onehot)).astype(np.float32)
        return obs

    def _simulate_sales(self, price):
        # Base sales depende del segmento si hay datos
        df_seg = self.data[self.data["Product Category"] == self.segment] if self.segment is not None else self.data
        base_sales = float(df_seg["Quantity"].mean()) if len(df_seg) > 0 else float(self.data["Quantity"].mean())
        
        # Elasticidad de precio: -1.5 (balanceada para evitar extremos)
        elasticity = -1.5
        price_ratio = price / self.reference_price
        demand_multiplier = np.power(price_ratio, elasticity)
        
        # Demanda esperada
        expected_sales = base_sales * demand_multiplier
        
        # Añade ruido estocástico (+-20%)
        noise = np.random.normal(1.0, 0.20)
        actual_sales = max(0, expected_sales * noise)
        
        return actual_sales  

    def step(self, action):
        # interpret action (ensure scalar int)
        if isinstance(action, np.ndarray):
            if action.ndim == 0:  # Scalar array
                action = int(action.item())
            else:  # Array with elements
                action = int(action[0])
        elif isinstance(action, list):
            action = int(action[0])
        else:
            action = int(action)

        prev_price = self.price
        if action == 0:  # Bajar precio
            self.price = float(max(self.price * 0.9, self.price_min))
        elif action == 2:  # Subir precio
            self.price = float(min(self.price * 1.1, self.price_max))
        # action == 1 -> mantener precio

        sales = float(self._simulate_sales(self.price))
        revenue = self.price * sales

        # Componente base: revenue es el objetivo principal
        base_reward = revenue
        
        # 1) Penalización por precios extremos (AMBOS LADOS)
        price_ratio = self.price / self.reference_price
        extreme_price_penalty = 0.0
        if price_ratio > 2.0:  # Si precio > 200% del referencia
            extreme_price_penalty = -revenue * 0.5  # Penaliza con 50% del revenue
        elif price_ratio < 0.5:  # Si precio < 50% del referencia
            extreme_price_penalty = -revenue * 0.8  # Penaliza FUERTE con 80% del revenue
        
        # 2) Penalización ligera por cambios bruscos
        price_change = abs(self.price - prev_price) / prev_price if prev_price > 0 else 0
        price_change_penalty = 0.0
        if price_change > 0.2:  # Solo si cambia más del 20%
            price_change_penalty = -revenue * 0.15  # Penaliza con 15% del revenue
        
        # 3) Bonus moderado por estar en rango razonable (80%-150% del precio referencia)
        optimal_bonus = 0.0
        if 0.8 <= price_ratio <= 1.5:  # Rango óptimo
            optimal_bonus = revenue * 0.2  # Bonus de 20% del revenue
        
        # Reward total
        reward = (base_reward + 
                  extreme_price_penalty + 
                  price_change_penalty + 
                  optimal_bonus)

        # update history and counters
        self.sales_history.append(sales)
        if len(self.sales_history) > self.window_size:
            self.sales_history = self.sales_history[-self.window_size:]
        self.current_step += 1
        self.cumulative_revenue += revenue

        done = (self.current_step >= self.max_steps)
        truncated = False 
        info = {"price": self.price, "sales": sales, "revenue": revenue, "cum_revenue": self.cumulative_revenue}

        return self._get_state(), float(reward), done, truncated, info

    def render(self, mode='human'):
        last_sales = self.sales_history[-1] if len(self.sales_history) > 0 else 0.0
        print(f"Step {self.current_step}: Price={self.price:.2f}, Sales={last_sales:.2f}, CumRevenue={self.cumulative_revenue:.2f}, Segment={self.segment}")

    def close(self):
        pass