import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

def create_heatmap(traffic_matrix, store, output_path="heatmap.png"):

    plt.figure(figsize=(8, 10))
    
    # seaborn para un mapa de calor estético
    ax = sns.heatmap(
        traffic_matrix, 
        cmap="viridis", 
        annot=True,  # Muestra los números de tráfico en cada celda
        fmt="d",     # Formatea los números como enteros
        linewidths=.5
    )

    ax.set_title("Mapa de Calor del Tráfico de Clientes")
    ax.set_xlabel("Columnas de la Tienda")
    ax.set_ylabel("Filas de la Tienda")

    for name, pos in store.sections.items():
        row, col = pos
        label = store.graph.nodes[pos].get("label", name)
        # Las coordenadas para el texto en matplotlib son (col, row)
        plt.text(col + 0.5, row + 0.5, label,
                 ha='center', va='center', color='white', fontsize=7, weight='bold')

    ax.invert_yaxis()

    plt.savefig(output_path)
    plt.close()
    print(f"Mapa de calor guardado en: {output_path}")
