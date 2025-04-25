import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


def plot_sales_data(df, title="Análisis de Ventas Material A"):
    """
    Visualize sales data with multiple plots showing different aspects of the time series.

    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame containing the sales data
    title : str
        Main title for the plots
    """
    # Configurar el estilo de los gráficos
    plt.style.use("classic")  # Cambiado de 'seaborn' a 'ggplot'
    sns.set_palette("husl")

    # Crear figura con subplots
    fig = plt.figure(figsize=(15, 12))
    gs = fig.add_gridspec(3, 2)

    # 1. Serie temporal completa
    ax1 = fig.add_subplot(gs[0, :])
    ax1.plot(df.index, df["ventas"], linewidth=1)
    ax1.set_title("Serie Temporal de Ventas")
    ax1.set_xlabel("Fecha")
    ax1.set_ylabel("Unidades Vendidas")
    ax1.grid(True, alpha=0.3)

    # 2. Promedio mensual
    ax2 = fig.add_subplot(gs[1, 0])
    monthly_sales = df.groupby("month")["ventas"].mean()
    monthly_sales.plot(kind="bar", ax=ax2)
    ax2.set_title("Promedio de Ventas por Mes")
    ax2.set_xlabel("Mes")
    ax2.set_ylabel("Promedio de Ventas")

    # 3. Promedio por día de la semana
    ax3 = fig.add_subplot(gs[1, 1])
    weekday_sales = df.groupby("day_of_week")["ventas"].mean()
    weekday_sales.plot(kind="bar", ax=ax3)
    ax3.set_title("Promedio de Ventas por Día de la Semana")
    ax3.set_xlabel("Día de la Semana")
    ax3.set_ylabel("Promedio de Ventas")
    ax3.set_xticklabels(["Lun", "Mar", "Mié", "Jue", "Vie", "Sáb", "Dom"])

    # 4. Boxplot por temporada
    ax4 = fig.add_subplot(gs[2, 0])
    sns.boxplot(x="season", y="ventas", data=df, ax=ax4)
    ax4.set_title("Distribución de Ventas por Temporada")
    ax4.set_xlabel("Temporada")
    ax4.set_ylabel("Ventas")

    # 5. Heatmap de ventas por mes y día de la semana
    ax5 = fig.add_subplot(gs[2, 1])
    pivot_table = df.pivot_table(
        values="ventas", index="day_of_week", columns="month", aggfunc="mean"
    )
    sns.heatmap(pivot_table, cmap="YlOrRd", ax=ax5)
    ax5.set_title("Heatmap de Ventas: Día de la Semana vs Mes")
    ax5.set_xlabel("Mes")
    ax5.set_ylabel("Día de la Semana")
    ax5.set_yticklabels(["Lun", "Mar", "Mié", "Jue", "Vie", "Sáb", "Dom"])

    plt.tight_layout()
    plt.suptitle(title, y=1.02, fontsize=16)
    plt.show()


# Ejemplo de uso:
# plot_sales_data(df, title="Análisis de Ventas Material A - 2022-2023")
