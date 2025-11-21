"""
Modelo LSTM para predicción de ventas por categoría
"""
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random


def set_seed(seed=42):
    """
    Fija todas las semillas para reproducibilidad
    
    Args:
        seed: Semilla para generadores aleatorios
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    # Para que cuDNN sea determinístico
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class StockPriceLSTM(nn.Module):
    """Modelo LSTM para predicción de series temporales de ventas"""
    
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


class LSTMTrainer:
    """Clase para entrenar y evaluar el modelo LSTM"""
    
    def __init__(self, device="cpu", seed=None):
        self.device = device
        self.model = None
        self.optimizer = None
        self.criterion = nn.MSELoss()
        self.seed = seed
        
        # Fijar semilla si se proporciona
        if seed is not None:
            set_seed(seed)
        
    def create_model(self, input_size=1, hidden_size=64, output_size=1, lr=1e-3):
        """Crea un nuevo modelo LSTM"""
        self.model = StockPriceLSTM(input_size, hidden_size, output_size).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        return self.model
    
    def train_epoch(self, X_train_t, y_train_t):
        """Entrena una época y retorna la pérdida"""
        self.model.train()
        self.optimizer.zero_grad()
        outputs = self.model(X_train_t)
        loss = self.criterion(outputs, y_train_t)
        loss.backward()
        self.optimizer.step()
        return loss.item()
    
    def train(self, X_train_t, y_train_t, epochs=1000, callback=None):
        """
        Entrena el modelo por un número de épocas
        
        Args:
            X_train_t: Datos de entrenamiento (tensor)
            y_train_t: Etiquetas de entrenamiento (tensor)
            epochs: Número de épocas
            callback: Función a llamar cada época con (epoch, loss)
        
        Returns:
            Lista de pérdidas por época
        """
        losses = []
        for epoch in range(epochs):
            loss = self.train_epoch(X_train_t, y_train_t)
            losses.append(loss)
            
            if callback:
                callback(epoch, loss)
                
        return losses
    
    def predict(self, X_test_t):
        """Realiza predicciones"""
        self.model.eval()
        with torch.no_grad():
            y_pred_scaled = self.model(X_test_t).cpu().numpy().ravel()
        return y_pred_scaled
    
    def forecast_next(self, last_window):
        """Predice el siguiente valor dada una ventana"""
        self.model.eval()
        with torch.no_grad():
            next_scaled = self.model(
                torch.tensor(last_window, dtype=torch.float32, device=self.device)
            ).cpu().numpy().ravel()[0]
        return next_scaled
