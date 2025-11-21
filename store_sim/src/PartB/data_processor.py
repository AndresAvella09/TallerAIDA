"""
Procesamiento de datos para el modelo LSTM
"""
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from pathlib import Path


class DataProcessor:
    """Clase para cargar y procesar datos de ventas"""
    
    def __init__(self, csv_path=None):
        if csv_path is None:
            # Ruta relativa al archivo
            csv_path = Path(__file__).parent.parent.parent / "data" / "retail_sales_dataset.csv"
        
        self.csv_path = csv_path
        self.df = None
        self.weekly = None
        self.series_dict = {}
        self.scalers = {}
        self.scaled_series = {}
        self.Xy_dict = {}
        self.timesteps = 10
        
    def load_and_prepare(self):
        """Carga y prepara los datos"""
        # Cargar CSV
        self.df = pd.read_csv(self.csv_path)
        self.df['Date'] = pd.to_datetime(self.df['Date'], errors="coerce")
        self.df = self.df.sort_values('Date')
        
        # Agregar ventas por categoría y semanal
        self.weekly = (
            self.df.set_index("Date")
            .groupby("Product Category")["Total Amount"]
            .resample('W')
            .sum()
            .reset_index()
        )
        
        # Crear series por categoría
        for cat in self.weekly["Product Category"].unique():
            cat_df = (
                self.weekly[self.weekly["Product Category"] == cat]
                .set_index("Date")
                .sort_index()
                .fillna(0.0)
            )
            self.series_dict[cat] = cat_df[["Total Amount"]]
        
        # Escalar series
        for cat, s in self.series_dict.items():
            scaler = MinMaxScaler(feature_range=(0, 1))
            scaled = scaler.fit_transform(s)
            self.scalers[cat] = scaler
            self.scaled_series[cat] = scaled
        
        # Crear secuencias
        for cat, serie in self.scaled_series.items():
            X, y = self._make_sequences(serie, self.timesteps)
            self.Xy_dict[cat] = {"X": X, "y": y}
            
        return self
    
    def _make_sequences(self, data, timesteps=10):
        """Crea secuencias para LSTM"""
        X, y = [], []
        for i in range(timesteps, len(data)):
            X.append(data[i - timesteps:i, 0])
            y.append(data[i, 0])
        X = np.array(X)
        y = np.array(y)
        X = X.reshape(X.shape[0], X.shape[1], 1)
        return X, y
    
    def get_categories(self):
        """Retorna lista de categorías disponibles"""
        return list(self.series_dict.keys())
    
    def get_data_for_category(self, category, split_ratio=0.8):
        """
        Obtiene datos de entrenamiento y test para una categoría
        
        Returns:
            dict con X_train, X_test, y_train, y_test, dates, values
        """
        X = self.Xy_dict[category]["X"]
        y = self.Xy_dict[category]["y"]
        
        split = int(len(X) * split_ratio)
        X_train, X_test = X[:split], X[split:]
        y_train, y_test = y[:split], y[split:]
        
        # Obtener fechas y valores originales
        serie_df = self.series_dict[category]
        dates = serie_df.index[self.timesteps:].tolist()
        values = serie_df["Total Amount"].values[self.timesteps:]
        
        return {
            "X_train": X_train,
            "X_test": X_test,
            "y_train": y_train,
            "y_test": y_test,
            "dates": dates,
            "values": values,
            "split": split
        }
    
    def get_last_window(self, category):
        """Obtiene la última ventana para predicción futura"""
        return self.scaled_series[category][-self.timesteps:, 0].reshape(1, self.timesteps, 1)
    
    def inverse_transform(self, category, scaled_values):
        """Desescala valores a escala original"""
        scaler = self.scalers[category]
        return scaler.inverse_transform(scaled_values.reshape(-1, 1)).ravel()
