import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import mean_absolute_error, mean_squared_error

df = pd.read_csv('retail_sales_dataset.csv')
df['Date'] = pd.to_datetime(df['Date'], errors = "coerce")
# Ordernar df por Date
df = df.sort_values('Date')

# -- Agregar las ventas por categoría y semanal 
weekly = (
    df.set_index("Date")
    .groupby("Product Category")["Total Amount"]
    .resample('W')
    .sum()
    .reset_index()
)

# -- Crea una serie semanal por categoría
series_dict = {}
for cat in weekly["Product Category"].unique():
    cat_df = (
        weekly[weekly["Product Category"] == cat]
        .set_index("Date")
        .sort_index()
        .fillna(0.0)
    )
    series_dict[cat] = cat_df[["Total Amount"]]
    
# -- Escala las series (MinMaxScaler)
scalers = {}
scaled_series = {}

for cat, s in series_dict.items():
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled = scaler.fit_transform(s)
    scalers[cat] = scaler
    scaled_series[cat] = scaled
    
# -- Funcion conjunto de entrenamiento
def make_sequences(data, timesteps=10):
    X, y = [], []
    for i in range(timesteps, len(data)):
        X.append(data[i - timesteps:i, 0])  # ventana de 10 pasos
        y.append(data[i, 0])                # siguiente valor
    X = np.array(X)
    y = np.array(y)
    X = X.reshape(X.shape[0], X.shape[1], 1)  # (samples, timesteps, features)
    return X, y
        
# -- Conjuntos re-formateados cada 10 pasos
timesteps = 10  

Xy_dict = {}
for cat, serie in scaled_series.items():
    X, y = make_sequences(serie, timesteps)
    Xy_dict[cat] = {"X": X, "y": y}
    print(f"[DEBUG] {cat} → X: {X.shape}, y: {y.shape}")
    

# =====================================================
# Definición del modelo LSTM (del .ipynb original)
# =====================================================
class StockPriceLSTM(nn.Module):
    def __init__(self, input_size=1, hidden_size=64, output_size=1):
        super(StockPriceLSTM, self).__init__()
        self.lstm = nn.LSTM(
            input_size,
            hidden_size,
            num_layers=2,
            batch_first=True,
            dropout=0.2
        )
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])  # tomamos el último paso temporal
        return out


# =====================================================
# Entrenamiento por categoría
# =====================================================
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"[DEBUG] Device: {device}")

resultados = []

for cat in scaled_series.keys():
    print(f"\n[DEBUG] === {cat} ===")

    X = Xy_dict[cat]["X"]
    y = Xy_dict[cat]["y"]

    split = int(len(X) * 0.8)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    # Tensores
    X_train_t = torch.tensor(X_train, dtype=torch.float32, device=device)
    y_train_t = torch.tensor(y_train.reshape(-1, 1), dtype=torch.float32, device=device)
    X_test_t  = torch.tensor(X_test, dtype=torch.float32, device=device)

    # =====================================================
    # Modelo LSTM (usa la clase del .ipynb)
    # =====================================================
    model = StockPriceLSTM(input_size=1, hidden_size=64, output_size=1).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    # =====================================================
    # Entrenamiento
    # =====================================================
    epochs = 4000
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        outputs = model(X_train_t)
        loss = criterion(outputs, y_train_t)
        loss.backward()
        optimizer.step()
        if (epoch + 1) % 50 == 0:
            print(f"[DEBUG] {cat} | Epoch {epoch+1}/{epochs} - Loss: {loss.item():.6f}")

    # =====================================================
    # Evaluación
    # =====================================================
    model.eval()
    with torch.no_grad():
        y_pred_scaled = model(X_test_t).cpu().numpy().ravel()

    # Desescalado a valores reales
    scaler = scalers[cat]
    y_true = scaler.inverse_transform(y_test.reshape(-1, 1)).ravel()
    y_pred = scaler.inverse_transform(y_pred_scaled.reshape(-1, 1)).ravel()

    mae  = mean_absolute_error(y_true, y_pred)
    rmse = mean_squared_error(y_true, y_pred)

    # =====================================================
    # Predicción próxima semana (forecast t+1)
    # =====================================================
    last_window = scaled_series[cat][-10:, 0].reshape(1, 10, 1)
    with torch.no_grad():
        next_scaled = model(
            torch.tensor(last_window, dtype=torch.float32, device=device)
        ).cpu().numpy().ravel()[0]

    next_value = scaler.inverse_transform(np.array([[next_scaled]])).ravel()[0]

    print(f"[DEBUG] {cat} | MAE={mae:.2f}  RMSE={rmse:.2f}  NextWeek={next_value:.2f}")

    resultados.append({
        "Product Category": cat,
        "MAE": mae,
        "RMSE": rmse,
        "NextWeekForecast": next_value
    })

# =====================================================
# Resumen de resultados
# =====================================================
if resultados:
    resultados_df = pd.DataFrame(resultados)
    display(resultados_df)
else:
    print("[DEBUG] No hubo categorías suficientes para entrenar.")
    
