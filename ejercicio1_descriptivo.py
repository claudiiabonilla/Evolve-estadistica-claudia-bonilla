"""
=============================================================================
PRÁCTICA FINAL — EJERCICIO 1
Análisis Estadístico Descriptivo — World Energy Consumption
=============================================================================

DESCRIPCIÓN
-----------
En este ejercicio se realiza un análisis estadístico descriptivo completo del
dataset "World Energy Consumption", con el objetivo de comprender la estructura,
distribución y relaciones entre las variables.

El análisis incluye:
  - Exploración estructural del dataset (tipos de datos y valores nulos).
  - Limpieza de datos mediante eliminación de valores faltantes.
  - Cálculo de estadísticos descriptivos (media, mediana, varianza, etc.).
  - Análisis de distribuciones mediante histogramas y KDE.
  - Identificación de outliers utilizando el método IQR.
  - Análisis de variables categóricas (frecuencias y desbalance).
  - Estudio de correlaciones y detección de multicolinealidad.

OBJETIVO
--------
Obtener una comprensión profunda del dataset antes de aplicar modelos de
Machine Learning, identificando patrones, anomalías y posibles problemas
como sesgo, outliers o correlaciones fuertes entre variables.

LIBRERÍAS UTILIZADAS
--------------------
  - pandas: manipulación de datos
  - numpy: operaciones numéricas
  - matplotlib / seaborn: visualización de datos

SALIDAS GENERADAS (carpeta output/)
-----------------------------------
  - ej1_descriptivo.csv             → Estadísticos descriptivos
  - ej1_histogramas.png             → Distribuciones de variables numéricas
  - ej1_boxplots.png                → Boxplots de la variable objetivo
  - ej1_outliers.txt                → Detección de outliers (IQR)
  - ej1_categoricas.png             → Distribución de variables categóricas
  - ej1_heatmap_correlacion.png     → Matriz de correlación

NOTAS
-----
  - La eliminación de valores nulos se realiza mediante dropna(), lo que puede
    reducir el tamaño del dataset pero simplifica el análisis.
  - La presencia de outliers y distribuciones asimétricas es común en datos
    económicos y energéticos, por lo que se interpretan como parte del fenómeno.
  - El análisis de correlaciones permite detectar multicolinealidad, lo cual
    es relevante para modelos posteriores de regresión.

=============================================================================
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Crear carpeta de salida si no existe
os.makedirs("output", exist_ok=True)

# Variable objetivo global
TARGET = "primary_energy_consumption"

# =============================================================================
# FUNCIÓN RESUMEN ESTRUCTURAL
# =============================================================================

def resumen_estructural(df):
    """
    Realiza un análisis estructural inicial del dataset.

    Este análisis incluye:
    - Información general del dataset (número de filas, columnas y tipos de datos).
    - Identificación del porcentaje de valores nulos por columna.
    - Eliminación de registros con valores faltantes mediante dropna().
    - Guardado del dataset limpio para su uso posterior.

    Parámetros
    ----------
    df : Dataset original.

    Retorna
    -------
    df_clean : Dataset tras la eliminación de valores nulos.
    """

    print("=== RESUMEN ESTRUCTURAL ===")

    df.info()
    print(df.dtypes)

    nulos = df.isnull().mean() * 100
    nulos = nulos.sort_values(ascending=False)

    print("\nPorcentaje de nulos:")
    print(nulos)

    # Tratamiento de nulos (decisión simple)
    df_clean = df.dropna()

    print("\nShape tras limpieza:", df_clean.shape)

    # Guardar dataset limpio
    df_clean.to_csv("data/World-Energy-Consumption-clean.csv", index=False)
    return df_clean


# =============================================================================
# FUNCIÓN ESTADÍSTICOS DESCRIPTIVOS
# =============================================================================

def estadisticos_descriptivos(df):
    """
    Calcula y guarda estadísticos descriptivos de las variables numéricas.

    Incluye:
    - Media, desviación estándar, mínimo, máximo y cuartiles.
    - Mediana, varianza y moda.
    - Cálculo del rango intercuartílico (IQR) de la variable objetivo.
    - Cálculo de asimetría (skewness) y curtosis.

    Parámetros
    ----------
    df : Dataset limpio.

    Salidas
    -------
    - output/ej1_descriptivo.csv : tabla con los estadísticos calculados.
    """

    print("\n=== ESTADÍSTICOS DESCRIPTIVOS ===")

    num_cols = df.select_dtypes(include="number")

    desc = num_cols.describe().T
    desc["median"] = num_cols.median()
    desc["variance"] = num_cols.var()
    desc["mode"] = num_cols.mode().iloc[0]

    desc.to_csv("output/ej1_descriptivo.csv")

    # IQR

    Q1 = df[TARGET].quantile(0.25)
    Q3 = df[TARGET].quantile(0.75)
    IQR = Q3 - Q1

    print(f"\nIQR {TARGET}: {IQR}")

    # Asimetría y curtosis
    skewness = df[TARGET].skew()
    kurtosis = df[TARGET].kurt()

    print(f"Skewness: {skewness}")
    print(f"Kurtosis: {kurtosis}")


# =============================================================================
# FUNCIÓN DISTRIBUCIONES
# =============================================================================

def distribuciones(df):
    """
    Analiza la distribución de las variables numéricas y detecta outliers.

    Incluye:
    - Histogramas con curva KDE para todas las variables numéricas.
    - Boxplot de la variable objetivo segmentado por los países con más mediana.
    - Detección de outliers mediante el método del rango intercuartílico (IQR).

    Parámetros
    ----------
    df : Dataset limpio.

    Salidas
    -------
    - output/ej1_histogramas.png
    - output/ej1_boxplots.png
    - output/ej1_outliers.txt
    """

    num_cols = df.select_dtypes(include="number")
    num_cols_names = num_cols.columns

    # Histogramas

    n = len(num_cols_names)
    cols_plot = 3
    rows = (n // cols_plot) + 1

    plt.figure(figsize=(15, rows * 4))

    for i, col in enumerate(num_cols_names):
        plt.subplot(rows, cols_plot, i + 1)
        sns.histplot(df[col], kde=True)
        plt.title(col)

    plt.tight_layout()
    plt.savefig("output/ej1_histogramas.png", dpi=150)
    plt.close()

    # Boxplot
    cat_cols = ["country", "decade"]

    plt.figure(figsize=(18, 6))

    for i, col in enumerate(cat_cols):
        plt.subplot(1, len(cat_cols), i + 1)

        if col == "country":
            order = df.groupby(col)[TARGET].median().sort_values(ascending=False).head(10).index
            data_plot = df[df[col].isin(order)]
        else:
            order = None
            data_plot = df

        sns.boxplot(x=col, y=TARGET, data=data_plot, order=order)

        plt.xticks(rotation=45)
        plt.title(f"{TARGET} por {col}")

    plt.tight_layout()
    plt.savefig("output/ej1_boxplots.png", dpi=150)
    plt.close()

    # =========================================================
    # OUTLIERS
    # =========================================================

    print("\n=== OUTLIERS ===")

    Q1 = df[TARGET].quantile(0.25)
    Q3 = df[TARGET].quantile(0.75)
    IQR = Q3 - Q1

    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR

    outliers = df[(df[TARGET] < lower) | (df[TARGET] > upper)]
    n_outliers = len(outliers)

    print(f"Outliers en {TARGET}: {n_outliers}")

    with open("output/ej1_outliers.txt", "w", encoding="utf-8") as f:
        f.write("DETECCIÓN DE OUTLIERS\n")
        f.write("======================\n\n")
        f.write("Método: IQR\n\n")
        f.write(f"Límite inferior: {lower:.2f}\n")
        f.write(f"Límite superior: {upper:.2f}\n\n")
        f.write(f"{TARGET}: {n_outliers} outliers\n")

# =========================================================
# FUNCIÓN VARIABLES CATEGÓRICAS
# =========================================================

def variables_categoricas(df):
    """
    Analiza la distribución de las variables categóricas.

    Incluye:
    - Frecuencia absoluta y relativa de las categorías.
    - Visualización mediante gráficos de barras.
    - Evaluación del desbalance de clases.

    Parámetros
    ----------
    df : Dataset limpio.

    Salidas
    -------
    - output/ej1_categoricas.png
    """

    cat_cols = ["country", "decade"]
    plt.figure(figsize=(18,6))

    for i, col in enumerate(cat_cols):
        print(f"\n--- {col.upper()} ---")

        # Frecuencia absoluta
        freq_abs = df[col].value_counts()
        print("\nFrecuencia absoluta (top 10):")
        print(freq_abs.head(10))

        # Frecuencia relativa
        freq_rel = df[col].value_counts(normalize=True) * 100
        print("\nFrecuencia relativa (%) (top 10):")
        print(freq_rel.head(10))

        # Gráfico
        plt.subplot(1, len(cat_cols), i + 1)
        df[col].value_counts().head(10).plot(kind="bar")

        plt.title(f"Distribución de {col}")
        plt.ylabel("Frecuencia")
        plt.xticks(rotation=45)

    plt.tight_layout()
    plt.savefig(f"output/ej1_categoricas.png", dpi=150)
    plt.close()

    # --- Análisis de desbalance global ---
    print("\n=== ANÁLISIS DE DESBALANCE ===")

    for col in cat_cols:
        imbalance = df[col].value_counts(normalize=True) * 100
        print(f"\n{col}:")
        print("Top categoría:", imbalance.index[0])
        print("Porcentaje:", imbalance.iloc[0])
        print("Top 5 acumulado:", imbalance.head(5).sum(), "%")

# =========================================================
# FUNCIÓN CORRELACIONES
# =========================================================

def correlaciones(df):
    """
    Analiza la relación entre variables numéricas mediante correlación.

    Incluye:
    - Cálculo de la matriz de correlación de Pearson.
    - Visualización mediante heatmap.
    - Identificación de las variables más correlacionadas con la variable objetivo.
    - Detección de multicolinealidad (|r| > 0.9).

    Parámetros
    ----------
    df : Dataset limpio.

    Salidas
    -------
    - output/ej1_heatmap_correlacion.png
    """

    # 1. Matriz de correlación (solo numéricas)
    corr = df.select_dtypes(include="number").corr(method="pearson")

    # -----------------------------
    # HEATMAP
    # -----------------------------
    plt.figure(figsize=(12,8))
    sns.heatmap(corr, cmap="coolwarm", annot=False)

    plt.title("Matriz de correlación (Pearson)")
    plt.tight_layout()
    plt.savefig("output/ej1_heatmap_correlacion.png", dpi=150)
    plt.close()

    # -----------------------------
    # TOP 3 CORRELACIONES CON TARGET
    # -----------------------------
    corr_target = corr[TARGET].drop(TARGET).abs().sort_values(ascending=False)

    print("\nTOP 3 variables más correlacionadas con", TARGET)
    print(corr_target.head(3))

    # -----------------------------
    # MULTICOLINEALIDAD
    # -----------------------------
    print("\n=== MULTICOLINEALIDAD (|r| > 0.9) ===")

    cols = corr.columns

    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            if abs(corr.iloc[i, j]) > 0.9:
                print(f"{cols[i]} - {cols[j]}: {corr.iloc[i, j]:.2f}")


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":

    print("=" * 60)
    print("EJERCICIO 1 — ANÁLISIS ESTADÍSTICO DESCRIPTIVO")
    print("=" * 60)

    # Cargar dataset limpio
    df = pd.read_csv("data/World-Energy-Consumption.csv")

    # Pipelina
    df = resumen_estructural(df)
    estadisticos_descriptivos(df)
    distribuciones(df)
    variables_categoricas(df)
    correlaciones(df)
