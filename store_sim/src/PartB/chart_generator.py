"""
Generación de gráficos con matplotlib para mostrar en pygame
"""
import matplotlib
matplotlib.use('Agg')  # Backend sin GUI
import matplotlib.pyplot as plt
import numpy as np
import pygame
from io import BytesIO


class ChartGenerator:
    """Genera gráficos matplotlib como superficies pygame"""
    
    @staticmethod
    def create_time_series_chart(dates, y_true, y_pred, split_idx, mae, rmse, next_forecast, width=600, height=400):
        """
        Crea gráfico de serie temporal con predicciones y bandas de confianza
        
        Args:
            dates: Lista de fechas
            y_true: Valores reales
            y_pred: Valores predichos (solo para test set)
            split_idx: Índice donde termina training y empieza test
            mae: Mean Absolute Error
            rmse: Root Mean Squared Error
            next_forecast: Predicción próxima semana
            width, height: Dimensiones en píxeles
        """
        fig, ax = plt.subplots(figsize=(width/100, height/100), dpi=100)
        
        # Convertir fechas a índices numéricos para graficar
        x_indices = np.arange(len(dates))
        
        # Training data (todo hasta split)
        ax.plot(x_indices[:split_idx], y_true[:split_idx], 
                'o-', color='#4CAF50', linewidth=2, markersize=3, label='Entrenamiento', alpha=0.8)
        
        # Test data - valores reales
        test_indices = x_indices[split_idx:]
        ax.plot(test_indices, y_true[split_idx:], 
                'o-', color='#2196F3', linewidth=2, markersize=3, label='Real (Test)', alpha=0.8)
        
        # Predicciones
        ax.plot(test_indices, y_pred, 
                's-', color='#FF5722', linewidth=2, markersize=4, label='Predicción', alpha=0.8)
        
        # Bandas de confianza
        # MAE band
        ax.fill_between(test_indices, 
                       y_pred - mae, 
                       y_pred + mae,
                       color='#FF9800', alpha=0.2, label=f'±MAE ({mae:.2f})')
        
        # RMSE band
        ax.fill_between(test_indices, 
                       y_pred - rmse, 
                       y_pred + rmse,
                       color='#F44336', alpha=0.15, label=f'±RMSE ({rmse:.2f})')
        
        # Predicción futura (próxima semana)
        if next_forecast is not None:
            next_x = len(dates)
            ax.plot([test_indices[-1], next_x], [y_pred[-1], next_forecast],
                   '--', color='#9C27B0', linewidth=2, label=f'Próx. semana: {next_forecast:.2f}')
            ax.plot(next_x, next_forecast, 'D', color='#9C27B0', markersize=8)
        
        # Línea vertical en el split
        ax.axvline(x=split_idx, color='gray', linestyle='--', alpha=0.5, linewidth=1)
        ax.text(split_idx, ax.get_ylim()[1]*0.95, 'Train/Test Split', 
               rotation=90, va='top', ha='right', fontsize=8, color='gray')
        
        ax.set_xlabel('Semanas', fontsize=10)
        ax.set_ylabel('Ventas Totales ($)', fontsize=10)
        ax.set_title('Predicción de Ventas - Serie Temporal', fontsize=12, fontweight='bold')
        ax.legend(loc='upper left', fontsize=8)
        ax.grid(True, alpha=0.3)
        
        # Estilo
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        fig.patch.set_facecolor('#2C2C2C')
        ax.set_facecolor('#1E1E1E')
        ax.tick_params(colors='white', labelsize=8)
        ax.xaxis.label.set_color('white')
        ax.yaxis.label.set_color('white')
        ax.title.set_color('white')
        for spine in ax.spines.values():
            spine.set_color('white')
        legend = ax.legend(loc='upper left', fontsize=8, facecolor='#2C2C2C')
        for text in legend.get_texts():
            text.set_color('white')
        
        plt.tight_layout()
        
        # Convertir a superficie pygame
        buf = BytesIO()
        plt.savefig(buf, format='png', facecolor=fig.get_facecolor())
        buf.seek(0)
        surface = pygame.image.load(buf)
        plt.close(fig)
        
        return surface
    
    @staticmethod
    def create_metrics_table(category, mae, rmse, next_forecast, train_size, test_size, 
                           learning_rate=None, seed=None, epochs=None, width=400, height=300):
        """
        Crea tabla de métricas como imagen
        """
        fig, ax = plt.subplots(figsize=(width/100, height/100), dpi=100)
        ax.axis('off')
        
        # Datos de la tabla
        metrics_data = [
            ['Categoría', category[:30]],
            ['', ''],
            ['MAE', f'{mae:.2f}'],
            ['RMSE', f'{rmse:.2f}'],
            ['', ''],
            ['Próxima semana', f'${next_forecast:.2f}'],
            ['', ''],
            ['Datos entrenamiento', f'{train_size}'],
            ['Datos prueba', f'{test_size}'],
        ]
        
        # Crear tabla
        table = ax.table(cellText=metrics_data,
                        cellLoc='left',
                        loc='center',
                        colWidths=[0.45, 0.55])
        
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 2.5)
        
        # Estilo de celdas
        for i, row in enumerate(metrics_data):
            cell = table[(i, 0)]
            cell.set_facecolor('#424242')
            cell.set_text_props(weight='bold', color='white')
            
            cell = table[(i, 1)]
            cell.set_facecolor('#2C2C2C')
            cell.set_text_props(color='white')
            
            # Destacar métricas principales
            if i in [2, 3, 5]:
                table[(i, 0)].set_facecolor('#1565C0')
                table[(i, 1)].set_facecolor('#1976D2')
        
        fig.patch.set_facecolor('#2C2C2C')
        plt.tight_layout()
        
        # Convertir a superficie pygame
        buf = BytesIO()
        plt.savefig(buf, format='png', facecolor=fig.get_facecolor())
        buf.seek(0)
        surface = pygame.image.load(buf)
        plt.close(fig)
        
        return surface
    
    @staticmethod
    def create_loss_chart(losses, width=400, height=250):
        """Crea gráfico de pérdida durante entrenamiento"""
        fig, ax = plt.subplots(figsize=(width/100, height/100), dpi=100)
        
        epochs = np.arange(1, len(losses) + 1)
        ax.plot(epochs, losses, color='#00BCD4', linewidth=2, alpha=0.8)
        
        ax.set_xlabel('Época', fontsize=10)
        ax.set_ylabel('Loss (MSE)', fontsize=10)
        ax.set_title('Pérdida durante Entrenamiento', fontsize=11, fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        # Estilo oscuro
        fig.patch.set_facecolor('#2C2C2C')
        ax.set_facecolor('#1E1E1E')
        ax.tick_params(colors='white', labelsize=8)
        ax.xaxis.label.set_color('white')
        ax.yaxis.label.set_color('white')
        ax.title.set_color('white')
        for spine in ax.spines.values():
            spine.set_color('white')
        
        plt.tight_layout()
        
        # Convertir a superficie pygame
        buf = BytesIO()
        plt.savefig(buf, format='png', facecolor=fig.get_facecolor())
        buf.seek(0)
        surface = pygame.image.load(buf)
        plt.close(fig)
        
        return surface
