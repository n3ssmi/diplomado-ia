# type: ignore[reportAttributeAccessIssue]
# type: ignore[F841]

from visualization import plot_sales_data
from models import (
    prepare_data_for_lstm,
    create_lstm_model,
    predict_sales,
    train_model,
)
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


def main(
    df,
    epochs=50,
    batch_size=32,
    sequence_length=60,
    days_to_predict=180,
    type_model="simple",
    features=["day_of_week", "month", "is_weekend", "season_ordinal"],
):
    """
    Ejecuta el modelo LSTM..
    """
    # Mostrar información básica del dataset
    print("\nInformación del dataset:")
    print(df.info())
    print("\nEstadísticas descriptivas:")
    print(df["ventas"].describe())

    # Separar datos en entrenamiento y prueba (90% train, 10% test)
    train_size = int(len(df) * 0.9)
    train_df = df.iloc[:train_size]
    test_df = df.iloc[train_size:]

    print(f"\nTamaño del conjunto de entrenamiento: {len(train_df)}")
    print(f"Tamaño del conjunto de prueba: {len(test_df)}")

    # Preparar datos para LSTM
    print("\nPreparando datos para el modelo LSTM...")
    X_train, y_train, sales_scaler, features_scaler = prepare_data_for_lstm(
        train_df, sequence_length, features
    )
    n_features = X_train.shape[2]  # Número de características
    print(f"\nNúmero de características: {n_features}")

    # Crear y entrenar el modelo
    print("\nCreando y entrenando el modelo LSTM...")
    model = create_lstm_model(sequence_length, n_features, type_model)
    model.summary()
    history = train_model(model, X_train, y_train, epochs=epochs, batch_size=batch_size)

    # Mostrar progreso del entrenamiento
    plt.figure(figsize=(10, 4))
    plt.plot(history.history["loss"], label="Training Loss")
    plt.plot(history.history["val_loss"], label="Validation Loss")
    plt.title("Model Loss During Training")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.show()

    # Predecir ventas futuras
    print("\nPrediciendo ventas futuras...")
    last_sequence = X_train[-1]
    predictions = predict_sales(
        model,
        last_sequence,
        sales_scaler,
        features_scaler,
        train_df,
        days_to_predict,
        features,
    )

    first_predictions = predict_sales(
        model,
        last_sequence,
        sales_scaler,
        features_scaler,
        train_df,
        10,
        features,
    )

    # Comparar las primeras 10 predicciones
    print("\nComparación de predicciones:")
    print(last_sequence)
    print("\nPrimeras 10 predicciones de la predicción larga:")
    print(predictions[:10])
    print("\nPredicciones de 10 días:")
    print(first_predictions)

    # Calcular la diferencia
    diferencia = predictions[:10] - first_predictions
    print("\nDiferencia absoluta máxima:", np.abs(diferencia).max())
    print("¿Son idénticas?:", np.allclose(predictions[:10], first_predictions))

    # Si hay diferencias, mostrar las diferencias día por día
    if not np.allclose(predictions[:10], first_predictions):
        print("\nDiferencias día por día:")
        for i in range(10):
            print(
                f"Día {i + 1}: {predictions[i]:.2f} vs {first_predictions[i]:.2f} (diferencia: {diferencia[i]:.2f})"
            )

    # Crear fechas futuras para las predicciones
    last_date = train_df.index[-1]
    future_dates = pd.date_range(
        start=last_date + pd.Timedelta(days=1), periods=days_to_predict
    )

    # Crear DataFrame con las predicciones
    predictions_df = pd.DataFrame(index=future_dates)
    predictions_df["ventas"] = predictions

    # Agregar componentes de tiempo al DataFrame de predicciones
    predictions_df["year"] = predictions_df.index.year
    predictions_df["month"] = predictions_df.index.month
    predictions_df["day_of_week"] = predictions_df.index.dayofweek
    predictions_df["day_of_year"] = predictions_df.index.dayofyear
    predictions_df["is_weekend"] = predictions_df["day_of_week"] >= 5
    predictions_df["season"] = pd.cut(
        predictions_df["month"],
        bins=[0, 3, 6, 9, 12],
        labels=["Invierno", "Primavera", "Verano", "Otoño"],
    )

    # Visualizar datos históricos y predicciones
    print("\nVisualizando datos históricos...")
    plot_sales_data(train_df, title="Datos Históricos de Entrenamiento - Producto X")

    print("\nVisualizando predicciones...")
    plot_sales_data(predictions_df, title="Predicciones de Ventas - Producto X")

    # Visualizar datos históricos, de prueba y predicciones juntos
    plt.figure(figsize=(15, 6))
    plt.plot(
        train_df.index, train_df["ventas"], label="Datos de entrenamiento", alpha=0.7
    )
    plt.plot(
        test_df.index,
        test_df["ventas"],
        label="Datos de prueba",
        color="green",
        alpha=0.7,
    )
    plt.plot(
        predictions_df.index,
        predictions_df["ventas"],
        label="Predicciones",
        color="red",
    )
    plt.title("Predicción de Ventas - Producto X")
    plt.xlabel("Fecha")
    plt.ylabel("Unidades Vendidas")
    plt.legend()
    plt.grid(True)
    plt.show()

    # Calcular métricas de error para el conjunto de prueba
    test_predictions = predict_sales(
        model,
        last_sequence,
        sales_scaler,
        features_scaler,
        train_df,
        len(test_df),
        features,
    )

    # Calcular métricas
    mae = np.mean(np.abs(test_df["ventas"].values - test_predictions))
    mse = np.mean((test_df["ventas"].values - test_predictions) ** 2)
    rmse = np.sqrt(mse)
    mape = (
        np.mean(
            np.abs(
                (test_df["ventas"].values - test_predictions) / test_df["ventas"].values
            )
        )
        * 100
    )

    print("\nMétricas de error en el conjunto de prueba:")
    print(f"MAE: {mae:.2f}")
    print(f"MSE: {mse:.2f}")
    print(f"RMSE: {rmse:.2f}")
    print(f"MAPE: {mape:.2f}%")

    print("\nProceso completado ╰(*°▽°*)╯")
