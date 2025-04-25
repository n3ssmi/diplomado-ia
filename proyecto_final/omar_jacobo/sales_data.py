# type: ignore[reportAttributeAccessIssue]
import numpy as np
import pandas as pd
from datetime import datetime


def generate_senoidal_sales_data() -> pd.DataFrame:
    """Genera datos sintéticos de ventas con patrones estacionales y características adicionales.

    Returns:
        pandas.DataFrame: DataFrame con los datos sintéticos de ventas.
    """
    # Crear fechas para 2 años de datos
    start_date = datetime(2022, 1, 1)
    end_date = datetime(2023, 12, 31)
    dates = pd.date_range(start=start_date, end=end_date, freq="D")

    # Crear el DataFrame base
    df = pd.DataFrame(index=dates)

    # Agregar componentes de tiempo
    df["year"] = df.index.year  # type: ignore
    df["month"] = df.index.month  # type: ignore
    df["day_of_week"] = df.index.dayofweek  # 0 = Lunes, 6 = Domingo # type: ignore
    df["day_of_year"] = df.index.dayofyear  # type: ignore

    # Crear el patrón base senoidal con 2 picos por año (verano e invierno)
    days = np.arange(len(df))

    # Crear dos patrones senoidales con diferentes fases
    # Pico de verano (junio-julio)
    summer_pattern = np.sin(
        2 * np.pi * (2 / 365) * (days - 180)
    )  # 180 días = mitad del año
    # Pico de invierno (diciembre-enero)
    winter_pattern = np.sin(2 * np.pi * (1 / 365) * days)

    # Combinar los patrones
    seasonal_pattern = (summer_pattern + winter_pattern) / 2
    seasonal_pattern = summer_pattern

    # Agregar tendencia y ruido
    trend = days * 0.1  # Tendencia lineal creciente
    noise = np.random.normal(0, 0.5, len(df))

    # Crear ventas base
    base_sales = 1000  # Ventas base
    amplitude = 500  # Amplitud de la variación estacional

    df["ventas"] = (
        base_sales
        + amplitude * seasonal_pattern  # Patrón estacional combinado
        + trend  # Tendencia
        + noise  # Ruido aleatorio
    )

    # Ajustar por día de la semana (menos ventas en fines de semana)
    weekend_effect = df["day_of_week"].apply(lambda x: 0.7 if x >= 5 else 1.0)
    df["ventas"] = df["ventas"] * weekend_effect

    # Agregar efectos especiales
    # 1. Efecto de promociones (aumentos aleatorios en algunos días)
    promociones = np.random.choice([1.0, 1.3, 1.5], size=len(df), p=[0.85, 0.1, 0.05])
    df["ventas"] = df["ventas"] * promociones

    # 2. Asegurar que las ventas sean positivas y redondear
    df["ventas"] = np.round(df["ventas"].clip(lower=0))

    # Agregar columnas categóricas
    df["is_weekend"] = df["day_of_week"] >= 5
    df["season"] = pd.cut(
        df["month"],
        bins=[0, 3, 6, 9, 12],
        labels=["Invierno", "Primavera", "Verano", "Otoño"],
    )

    return df


def generate_weekend_sales_data(
    start_date: str = "2023-01-01",
    end_date: str = "2023-12-31",
    base_sales: float = 100,
    weekend_boost: float = 1.5,
    monthly_variation: float = 0.2,
    noise_level: float = 0.1,
    trend_factor: float = 0.001,
) -> pd.DataFrame:
    """Genera datos sintéticos de ventas con patrones semanales y picos en fines de semana.

    Args:
        start_date (str, optional): Fecha de inicio en formato 'YYYY-MM-DD'. Defaults to "2023-01-01".
        end_date (str, optional): Fecha de fin en formato 'YYYY-MM-DD'. Defaults to "2023-12-31".
        base_sales (float, optional): Nivel base de ventas para días entre semana. Defaults to 100.
        weekend_boost (float, optional): Multiplicador para ventas en fines de semana.
            Ej: 1.5 significa 50% más ventas. Defaults to 1.5.
        monthly_variation (float, optional): Variación mensual máxima como proporción de ventas base.
            Defaults to 0.2.
        noise_level (float, optional): Nivel de ruido aleatorio como proporción de ventas base.
            Defaults to 0.1.
        trend_factor (float, optional): Factor de tendencia diaria (positivo para tendencia creciente).
            Defaults to 0.001.

    Returns:
        pandas.DataFrame: DataFrame con datos de ventas y características temporales.
    """
    # Convertir fechas a datetime
    start = datetime.strptime(start_date, "%Y-%m-%d")
    end = datetime.strptime(end_date, "%Y-%m-%d")

    # Generar rango de fechas
    date_range = pd.date_range(start=start, end=end, freq="D")

    # Crear DataFrame con fechas
    df = pd.DataFrame(index=date_range)

    # Agregar características temporales
    df["year"] = df.index.year
    df["month"] = df.index.month
    df["day_of_week"] = df.index.dayofweek  # 0=Lunes, 6=Domingo
    df["day_of_year"] = df.index.dayofyear
    df["is_weekend"] = df["day_of_week"] >= 5

    # Generar ventas con patrones
    days = np.arange(len(df))

    # Ventas base con tendencia
    sales = base_sales * (1 + trend_factor * days)

    # Agregar impulso en fines de semana
    weekend_mask = df["is_weekend"].values
    sales[weekend_mask] *= weekend_boost

    # Agregar variación mensual (patrón sinusoidal)
    monthly_pattern = np.sin(2 * np.pi * df["month"] / 12)
    sales *= 1 + monthly_variation * monthly_pattern

    # Agregar ruido aleatorio
    noise = np.random.normal(0, noise_level * base_sales, size=len(df))
    sales += noise

    # Asegurar que no haya ventas negativas
    sales = np.maximum(sales, 0)

    # Agregar ventas al DataFrame
    df["ventas"] = sales

    # Agregar temporada basada en el mes
    season_map = {
        1: "Invierno",
        2: "Invierno",
        3: "Invierno",
        4: "Primavera",
        5: "Primavera",
        6: "Primavera",
        7: "Verano",
        8: "Verano",
        9: "Verano",
        10: "Otoño",
        11: "Otoño",
        12: "Otoño",
    }
    df["season"] = df["month"].map(season_map)

    return df
