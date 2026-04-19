"""
=============================================================================
PRÁCTICA FINAL — EJERCICIO 2
Inferencia con Scikit-Learn
=============================================================================

DESCRIPCIÓN
-----------
En este ejercicio se construye un modelo de regresión lineal utilizando Scikit-Learn con el objetivo de predecir la variable objetivo del dataset.

Se realiza un proceso completo que incluye:
- preprocesamiento de los datos,
- entrenamiento del modelo,
- evaluación mediante métricas (MAE, RMSE y R²),
- y análisis de los residuos para evaluar el comportamiento del modelo.

El objetivo es interpretar tanto el rendimiento del modelo como sus posibles limitaciones.

LIBRERÍAS USADAS
--------------------
  - pandas
  - numpy
  - matplotlib
  - sklearn

SALIDAS ESPERADAS (carpeta output/)
------------------------------------
  - output/ej2_metricas_regresion.txt   → MAE, RMSE y R² de la regresión lineal
  - output/ej2_residuos.png             → Gráfico de residuos del modelo de regresión lineal

=============================================================================
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Crear carpeta de salida si no existe
os.makedirs("output", exist_ok=True)

# =============================================================================
# FUNCIÓN PREPROCESAMIENTO
# =============================================================================

def preprocesar_datos(df):
  """
  Realiza el preprocesamiento del dataset para su uso en un modelo de regresión.

    Este proceso incluye:
    - Eliminación de variables no relevantes o no numéricas (como 'iso_code').
    - Codificación de variables categóricas ('country' y 'decade') mediante One-Hot Encoding.
    - Separación de la variable objetivo ('primary_energy_consumption') y las variables predictoras.
    - División del dataset en conjunto de entrenamiento (80%) y test (20%) usando una semilla fija para reproducibilidad.
    - Escalado de las variables numéricas mediante StandardScaler, ajustado únicamente sobre el conjunto de entrenamiento.

    Parámetros
    ----------
    df : pd.DataFrame
        Dataset original con variables numéricas y categóricas.

    Retorna
    -------
    X_train : Variables predictoras del conjunto de entrenamiento escaladas.
    X_test : Variables predictoras del conjunto de test escaladas.
    y_train : Variable objetivo del conjunto de entrenamiento.
    y_test : Variable objetivo del conjunto de test.
  """

  target = "primary_energy_consumption"

  # eliminar columnas no numéricas útiles
  df = df.drop(columns=["iso_code"])

  df = pd.get_dummies(df, columns=["country", "decade"], drop_first=True)

  X = df.drop(columns=[target])
  y = df[target]

  X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
  )

  scaler = StandardScaler()

  X_train = scaler.fit_transform(X_train)
  X_test = scaler.transform(X_test)

  return X_train, X_test, y_train, y_test

# =============================================================================
# ENTRENAMIENTO MODELO
# =============================================================================

def entrenar_modelo(X_train, y_train, X_test):
  """
  Entrena un modelo de regresión lineal y genera predicciones sobre el conjunto de test.

  Este proceso incluye:
  - Inicialización del modelo LinearRegression de Scikit-Learn.
  - Entrenamiento del modelo utilizando los datos de entrenamiento.
  - Generación de predicciones sobre el conjunto de test.

  Parámetros
  ----------
  X_train : Variables predictoras del conjunto de entrenamiento.
  y_train : Variable objetivo del conjunto de entrenamiento.
  X_test : Variables predictoras del conjunto de test.

  Retorna
  -------
  modelo : odelo de regresión lineal entrenado.
  y_pred : Predicciones generadas por el modelo sobre el conjunto de test.
  """

  modelo = LinearRegression()
  modelo.fit(X_train, y_train)

  y_pred = modelo.predict(X_test)

  return modelo, y_pred

# =============================================================================
# FUNCIONES DE MÉTRICAS
# =============================================================================

def calcular_mae(y_real, y_pred):
  """
  Calcula el Mean Absolute Error (MAE).

      MAE = (1/n) * Σ |y_real - y_pred|

  Parámetros
  ----------
  y_real : np.ndarray — Valores reales
  y_pred : np.ndarray — Valores predichos

  Retorna
  -------
  float — Valor del MAE
  """
  return mean_absolute_error(y_real, y_pred)


def calcular_rmse(y_real, y_pred):
  """
  Calcula el Root Mean Squared Error (RMSE).

      RMSE = sqrt((1/n) * Σ (y_real - y_pred)²)

  Parámetros
  ----------
  y_real : np.ndarray — Valores reales
  y_pred : np.ndarray — Valores predichos

  Retorna
  -------
  float — Valor del RMSE
  """
  return np.sqrt(mean_squared_error(y_real, y_pred))


def calcular_r2(y_real, y_pred):
  """
  Calcula el coeficiente de determinación R².

      R² = 1 - SS_res / SS_tot
      SS_res = Σ (y_real - y_pred)²
      SS_tot = Σ (y_real - ȳ)²

  Parámetros
  ----------
  y_real : np.ndarray — Valores reales
  y_pred : np.ndarray — Valores predichos

  Retorna
  -------
  float — Valor del R² (entre -∞ y 1; cuanto más cercano a 1, mejor)
  """
  return r2_score(y_real, y_pred)


# =============================================================================
# FUNCIÓN DE VISUALIZACIÓN
# =============================================================================

def graficar_residuos(y_real, y_pred, ruta_salida="output/ej2_residuos.png"):
  """
  Genera un scatter plot de Residuos vs. Valores Predichos.

  Un modelo perfecto produciría todos los puntos sobre la diagonal y=x.
  La dispersión alrededor de esa línea representa el error del modelo.

  Parámetros
  ----------
  y_real      : np.ndarray — Valores reales del test set
  y_pred      : np.ndarray — Predicciones del modelo
  ruta_salida : str        — Ruta donde guardar la imagen
  """
  residuos = y_real - y_pred
  plt.scatter(y_pred, residuos, alpha=0.6)
  plt.axhline(0, color="red", linestyle="--")
  plt.xlabel("Predicciones")
  plt.ylabel("Residuos")
  plt.title("Residuos vs Valores Predichos")
  plt.savefig(ruta_salida, dpi=150, bbox_inches='tight')
  plt.close()

# =============================================================================
# FUNCIÓN IMPORTANCIA VARIABLES
# =============================================================================

def importancia_variables(modelo, X_train):
  importancias = pd.Series(modelo.coef_, index=X_train.columns)
  return importancias.sort_values(ascending=False)


# =============================================================================
# MAIN — Ejecuta el pipeline completo
# =============================================================================

if __name__ == "__main__":

  print("=" * 55)
  print("EJERCICIO 2 — Inferencia con Scikit-Learn")
  print("=" * 55)

  # ---------------------------------------------------------
  # CARGA DE DATOS
  # ---------------------------------------------------------

  df = pd.read_csv("data/World-Energy-Consumption-clean.csv")

  # ---------------------------------------------------------
  # PREPROCESAMIENTO
  # ---------------------------------------------------------

  X_train, X_test, y_train, y_test = preprocesar_datos(df)

  print("Dataset preparado:")
  print("  Train:", X_train.shape)
  print("  Test :", X_test.shape)

  # ---------------------------------------------------------
  # ENTRENAMIENTO
  # ---------------------------------------------------------

  modelo, y_pred = entrenar_modelo(X_train, y_train, X_test)

  # ---------------------------------------------------------
  # MÉTRICAS
  # ---------------------------------------------------------

  mae = calcular_mae(y_test, y_pred)
  rmse = calcular_rmse(y_test, y_pred)
  r2 = calcular_r2(y_test, y_pred)

  print("\nRESULTADOS")
  print("-" * 40)
  print(f"MAE  : {mae:.4f}")
  print(f"RMSE : {rmse:.4f}")
  print(f"R²   : {r2:.4f}")

  # ---------------------------------------------------------
  # GUARDAR MÉTRICAS
  # ---------------------------------------------------------

  with open("output/ej2_metricas_regresion.txt", "w", encoding="utf-8") as f:
    f.write("REGRESIÓN LINEAL — MÉTRICAS\n")
    f.write("=" * 50 + "\n")
    f.write(f"MAE  : {mae:.4f}\n")
    f.write(f"RMSE : {rmse:.4f}\n")
    f.write(f"R²   : {r2:.4f}\n")

  # ---------------------------------------------------------
  # GRÁFICO DE RESIDUOS
  # ---------------------------------------------------------

  graficar_residuos(y_test, y_pred)

  print("\nSalidas guardadas en la carpeta output/")
  print("  → output/ej2_metricas_regresion.txt")
  print("  → output/ej2_residuos.png")

  importancias = importancia_variables(modelo, pd.DataFrame(X_train))

  print("\nTOP VARIABLES MÁS INFLUYENTES")
  print(importancias.head(10))