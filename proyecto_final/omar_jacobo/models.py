# type: ignore[reportAttributeAccessIssue]
import numpy as np
import pandas as pd
import keras
from keras import layers
from sklearn.preprocessing import MinMaxScaler
from typing import List, Optional, Tuple


def prepare_data_for_lstm(
    df: pd.DataFrame,
    sequence_length: int = 30,
    features: Optional[List[str]] = None,
) -> Tuple[np.ndarray, np.ndarray, MinMaxScaler, Optional[MinMaxScaler]]:
    """Prepara los datos para el modelo LSTM con múltiples características.

    Args:
        df (pandas.DataFrame): DataFrame con los datos de ventas.
        sequence_length (int, optional): Número de días a usar para la predicción. Defaults to 30.
        features (list, optional): Lista de características a usar. Si es None, usa todas las disponibles.
            Características disponibles: ["day_of_week", "month", "is_weekend", "season_ordinal"].
            Defaults to None.

    Returns:
        tuple: Contiene:
            - X_train (numpy.ndarray): Secuencias de entrenamiento.
            - y_train (numpy.ndarray): Valores objetivo.
            - sales_scaler (MinMaxScaler): Escalador para las ventas.
            - features_scaler (MinMaxScaler): Escalador para las características.
    """
    # Crear variable ordinal para las temporadas
    df["season_ordinal"] = df["season"].map(
        {"Primavera": 1, "Verano": 2, "Otoño": 3, "Invierno": 4}
    )

    # Eliminar la columna original de season
    df = df.drop("season", axis=1)

    # Separar ventas y características
    sales = df[["ventas"]]

    # Si solo se usan ventas, no procesar características adicionales
    if not features:
        # Escalar solo ventas
        sales_scaler = MinMaxScaler(feature_range=(0, 1))
        sales_scaled = sales_scaler.fit_transform(sales)

        # Crear secuencias solo con ventas
        X, y = [], []
        for i in range(len(sales_scaled) - sequence_length):
            X.append(sales_scaled[i : (i + sequence_length)])
            y.append(sales_scaled[i + sequence_length, 0])

        return np.array(X), np.array(y), sales_scaler, None

    # Definir características disponibles
    available_features = ["day_of_week", "month", "is_weekend", "season_ordinal"]

    # Si no se especifican características, usar todas las disponibles
    if features is None:
        features = available_features
    else:
        # Verificar que todas las características solicitadas estén disponibles
        for feature in features:
            if feature not in available_features:
                raise ValueError(
                    f"Característica '{feature}' no disponible. Características disponibles: {available_features}"
                )

    # Verificar valores NaN
    print("Valores NaN en features:", df[features].isna().sum())

    # Combinar características
    feature_df = df[features]

    # Escalar ventas y características por separado
    sales_scaler = MinMaxScaler(feature_range=(0, 10))
    features_scaler = MinMaxScaler(feature_range=(0, 1))

    # Verificar datos antes de escalar
    print("Datos antes de escalar: \n", feature_df.head())

    sales_scaled = sales_scaler.fit_transform(sales)
    features_scaled = features_scaler.fit_transform(feature_df)

    # Verificar datos después de escalar
    print("Datos después de escalar: \n", features_scaled[0:5])

    # Combinar datos escalados
    scaled_data = np.concatenate([sales_scaled, features_scaled], axis=1)
    print("Datos combinados: \n", scaled_data[0:5])

    # Crear secuencias
    X, y = [], []
    for i in range(len(scaled_data) - sequence_length):
        X.append(scaled_data[i : (i + sequence_length)])
        y.append(
            scaled_data[i + sequence_length, 0]
        )  # Solo la primera columna (ventas)

    return np.array(X), np.array(y), sales_scaler, features_scaler


def create_lstm_model(
    sequence_length: int,
    n_features: int,
    type_model: str = "simple",
) -> keras.Model:
    """Crea un modelo LSTM para pronóstico de ventas.

    Args:
        sequence_length (int): Longitud de la secuencia de entrada.
        n_features (int): Número de características en los datos de entrada.
        type_model (str, optional): Tipo de modelo a crear. Opciones:
            - "simple": Modelo básico con una capa LSTM.
            - "deep_1": Modelo profundo con tres capas LSTM.
            - "deep_2": Modelo profundo con regularización L2.
            Defaults to "simple".

    Returns:
        keras.Model: Modelo LSTM compilado.
    """

    if type_model == "simple":
        model = keras.Sequential(
            [
                layers.LSTM(64, input_shape=(sequence_length, n_features)),
                # Donde:
                # - n_samples: número total de secuencias de entrenamiento
                # - sequence_length: número de días históricos (60)
                # - 1: una característica (ventas)
                layers.Dense(32, activation="relu"),
                layers.Dense(1),
            ]
        )
    elif type_model == "deep_1":
        model = keras.Sequential(
            [
                # Capa de entrada con más unidades
                layers.LSTM(
                    128,
                    return_sequences=True,
                    input_shape=(sequence_length, n_features),
                ),
                layers.Dropout(0.2),
                # Segunda capa LSTM para capturar patrones más complejos
                layers.LSTM(64, return_sequences=True),
                layers.Dropout(0.2),
                # Tercera capa LSTM
                layers.LSTM(32),
                layers.Dropout(0.2),
                # Capas densas con más unidades
                layers.Dense(64, activation="relu"),
                layers.Dense(32, activation="relu"),
                layers.Dense(1),
            ]
        )
    elif type_model == "deep_2":
        model = keras.Sequential(
            [
                # Capa de entrada con más unidades y regularización
                layers.LSTM(
                    256,
                    return_sequences=True,
                    input_shape=(sequence_length, n_features),
                    kernel_regularizer=keras.regularizers.l2(0.01),
                ),
                layers.Dropout(0.3),
                # Segunda capa LSTM con más unidades
                layers.LSTM(128, return_sequences=True),
                layers.Dropout(0.3),
                # Tercera capa LSTM
                layers.LSTM(64),
                layers.Dropout(0.3),
                # Capas densas con más unidades y regularización
                layers.Dense(
                    128,
                    activation="relu",
                    kernel_regularizer=keras.regularizers.l2(0.01),
                ),
                layers.Dense(64, activation="relu"),
                layers.Dense(32, activation="relu"),
                layers.Dense(1),
            ]
        )

    optimizer = keras.optimizers.Adam(learning_rate=0.001)
    model.compile(loss="huber", optimizer=optimizer, metrics=["mae"])
    return model


def train_model(
    model: keras.Model,
    X_train: np.ndarray,
    y_train: np.ndarray,
    epochs: int = 50,
    batch_size: int = 32,
    validation_split: float = 0.2,
) -> keras.callbacks.History:
    """Entrena el modelo LSTM.

    Args:
        model (keras.Model): Modelo LSTM a entrenar.
        X_train (numpy.ndarray): Características de entrenamiento.
        y_train (numpy.ndarray): Valores objetivo de entrenamiento.
        epochs (int, optional): Número de épocas de entrenamiento. Defaults to 50.
        batch_size (int, optional): Tamaño del batch. Defaults to 32.
        validation_split (float, optional): Proporción de datos para validación. Defaults to 0.2.

    Returns:
        keras.callbacks.History: Historial de entrenamiento.
    """
    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=10,  # restore_best_weights=True
        )
    ]

    history = model.fit(
        X_train,
        y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=validation_split,
        callbacks=callbacks,
        # verbose=1,
    )

    return history


def predict_sales(
    model: keras.Model,
    last_sequence: np.ndarray,
    sales_scaler: MinMaxScaler,
    features_scaler: Optional[MinMaxScaler] = None,
    feature_df: Optional[pd.DataFrame] = None,
    days_to_predict: int = 30,
    features: Optional[List[str]] = None,
) -> np.ndarray:
    """Predice ventas futuras usando el modelo LSTM.

    Args:
        model (keras.Model): Modelo LSTM entrenado.
        last_sequence (numpy.ndarray): Última secuencia de datos de ventas.
        sales_scaler (MinMaxScaler): Escalador usado para normalizar las ventas.
        features_scaler (MinMaxScaler, optional): Escalador para características. Si es None, solo usa ventas.
            Defaults to None.
        feature_df (pandas.DataFrame, optional): DataFrame con características. Si es None, solo usa ventas.
            Defaults to None.
        days_to_predict (int, optional): Número de días a predecir. Defaults to 30.
        features (list, optional): Lista de características a usar. Si es None, solo usa ventas.
            Defaults to None.

    Returns:
        numpy.ndarray: Valores de ventas predichos.
    """
    predictions = []
    current_sequence = last_sequence.copy()

    # Si no hay características o no se especifican, usar solo ventas
    if not features or features_scaler is None or feature_df is None:
        for _ in range(days_to_predict):
            # Predecir el siguiente valor
            next_pred = model.predict(current_sequence.reshape(1, -1, 1), verbose=0)
            predictions.append(next_pred[0, 0])

            # Actualizar la secuencia
            current_sequence = np.roll(current_sequence, -1, axis=0)
            current_sequence[-1] = next_pred

        # Invertir la normalización solo para las ventas
        predictions = sales_scaler.inverse_transform(
            np.array(predictions).reshape(-1, 1)
        )
        print("se uso solo ventas")
        return predictions.flatten()

    # Crear fechas futuras
    last_date = feature_df.index[-1]
    future_dates = pd.date_range(
        start=last_date + pd.Timedelta(days=1), periods=days_to_predict
    )

    # Crear DataFrame con características futuras
    future_df = pd.DataFrame(index=future_dates)
    future_df["year"] = future_df.index.year
    future_df["month"] = future_df.index.month
    future_df["day_of_week"] = future_df.index.dayofweek
    future_df["day_of_year"] = future_df.index.dayofyear
    future_df["is_weekend"] = future_df["day_of_week"] >= 5

    # Mapear meses a temporadas (1-4)
    month_to_season = {
        1: 4,
        2: 4,
        3: 4,  # Invierno
        4: 1,
        5: 1,
        6: 1,  # Primavera
        7: 2,
        8: 2,
        9: 2,  # Verano
        10: 3,
        11: 3,
        12: 3,  # Otoño
    }
    future_df["season_ordinal"] = future_df["month"].map(month_to_season)

    # Usar solo las características seleccionadas
    future_features = future_df[features]

    # Escalar las características futuras
    future_features_scaled = features_scaler.transform(future_features)

    for i in range(days_to_predict):
        # Predecir el siguiente valor
        next_pred = model.predict(
            current_sequence.reshape(1, -1, current_sequence.shape[-1]), verbose=0
        )
        predictions.append(next_pred[0, 0])

        # Actualizar la secuencia
        current_sequence = np.roll(current_sequence, -1, axis=0)

        # Actualizar con las características futuras escaladas
        current_sequence[-1, 0] = next_pred[0, 0]
        current_sequence[-1, 1:] = future_features_scaled[i]

    # Invertir la normalización solo para las ventas
    predictions = sales_scaler.inverse_transform(np.array(predictions).reshape(-1, 1))
    print("se uso todas las caracteristicas")
    return predictions.flatten()
