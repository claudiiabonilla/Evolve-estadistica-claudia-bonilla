# Respuestas — Práctica Final: Análisis y Modelado de Datos

> Rellena cada pregunta con tu respuesta. Cuando se pida un valor numérico, incluye también una breve explicación de lo que significa.

---

## Ejercicio 1 — Análisis Estadístico Descriptivo

---

El dataset que he seleccionado es World Energy Consumption. Contiene información sobre consumo energético a nivel mundial, con un total de 1278 observaciones tras la limpieza y 14 variables. La mayoría de las variables son numéricas, junto con algunas categóricas como país y década. Su tamaño (2.4 MB) lo hace adecuado para análisis exploratorio.

Enlace: https://www.kaggle.com/datasets/pralabhpoudel/world-energy-consumption

El análisis revela un alto porcentaje de valores nulos, especialmente en variables energéticas, superando en muchos casos el 70%. Incluso variables relevantes como el PIB o el consumo energético presentan niveles importantes de datos faltantes. Por este motivo, se optó por eliminar las filas con valores nulos, lo que simplifica el análisis aunque puede introducir cierto sesgo.

La variable objetivo, `primary_energy_consumption´, presenta una distribución fuertemente asimétrica positiva, con una alta concentración de valores bajos y una larga cola hacia valores altos. Esto se refleja en su elevado skewness (5.10) y curtosis (27.78), indicando la presencia de valores extremos. De hecho, se detectaron 143 outliers mediante el método IQR, que no fueron eliminados al representar casos reales, como países con alto consumo energético.

En cuanto a las variables categóricas, la distribución por país es bastante uniforme, mientras que la variable década está equilibrada entre los años 2000 y 2010. Por otro lado, el análisis de correlaciones muestra relaciones muy fuertes entre el consumo energético y variables como el consumo de combustibles fósiles (0.9979) y las emisiones de gases de efecto invernadero (0.9781).

Finalmente, se observa una fuerte multicolinealidad entre varias variables, con correlaciones superiores a 0.9, lo que indica una alta redundancia de información. Esto puede afectar negativamente a modelos predictivos, especialmente a la regresión lineal, al dificultar la interpretación de los coeficientes y generar problemas de estabilidad.

---

**Pregunta 1.1** — ¿De qué fuente proviene el dataset y cuál es la variable objetivo (target)? ¿Por qué tiene sentido hacer regresión sobre ella?

> El dataset World Energy Consumption está disponible en Kaggle y se basa en datos recopilados por Our World in Data, una plataforma que integra información internacional sobre consumo energético, emisiones, población y variables económicas desagregadas por país y año.
>
> La variable objetivo seleccionada es `primary_energy_consumption`, que representa el consumo total de energía primaria de un país.
>
> Tiene sentido aplicar un modelo de regresión sobre esta variable, ya que es continua y está influida por múltiples factores explicativos del dataset, como el PIB (gdp), la población (population) o el consumo de combustibles fósiles (fossil_fuel_consumption). Esto permite analizar la relación e impacto de estas variables en el consumo energético.

**Pregunta 1.2** — ¿Qué distribución tienen las principales variables numéricas y has encontrado outliers? Indica en qué variables y qué has decidido hacer con ellos.

> Las variables numéricas presentan, en su mayoría, distribuciones asimétricas positivas. Este comportamiento es notable en variables como:
>
> - gdp
> - population
> - primary_energy_consumption
> - electricity_generation
> - fossil_fuel_consumption
> - greenhouse_gas_emissions
>
> Esto es coherente con la naturaleza del dataset, donde pocos países concentran valores extremadamente altos.
>
> Por otro lado, variables como energy_per_gdp y renewables_share_energy presentan distribuciones algo más equilibradas, aunque siguen mostrando cierta asimetría.
>
> Se han detectado 143 outliers en la variable objetivo (primary_energy_consumption) utilizando el método del rango intercuartílico (IQR).
> Estos valores extremos corresponden principalmente a países con consumos energéticos muy elevados, lo que genera una larga cola en la distribución.
> Se ha decidido no eliminar los outliers, ya que representan observaciones reales y relevantes, forman parte de la estructura natural del problema, su eliminación podría sesgar el análisis.
>
> No obstante, estos outliers pueden afectar a métricas como la media y la varianza, así como influir negativamente en el rendimiento de modelos de regresión lineal.
>
> Como mejora futura, se podría aplicar transformaciones como el logaritmo a las variables más sesgadas, con el objetivo de reducir la asimetría y mejorar la estabilidad y capacidad predictiva de los modelos.

**Pregunta 1.3** — ¿Qué tres variables numéricas tienen mayor correlación (en valor absoluto) con la variable objetivo? Indica los coeficientes.

> Las tres variables con mayor correlación (en valor absoluto) con la variable objetivo ´primary_energy_consumption´ son:
>
> - fossil_fuel_consumption → 0.997874
> - electricity_generation → 0.992765
> - greenhouse_gas_emissions → 0.978122
>
> Estas correlaciones son muy altas y positivas, lo que indica una relación lineal muy fuerte entre el consumo de energía primaria y estas variables.
> En particular, el consumo de combustibles fósiles está directamente asociado al consumo energético total, mientras que la generación eléctrica refleja la demanda energética global. Por su parte, las emisiones de gases de efecto invernadero reflejan el impacto del uso intensivo de energía, especialmente de fuentes no renovables.
>
> Estos coeficientes tan elevados también sugieren la posible existencia de multicolinealidad entre las variables explicativas, lo que puede introducir redundancia de información.

**Pregunta 1.4** — ¿Hay valores nulos en el dataset? ¿Qué porcentaje representan y cómo los has tratado?

> Sí, el dataset presenta una cantidad significativa de valores nulos, especialmente en variables relacionadas con energía y emisiones. Los porcentajes más altos son:
>
> - renewables_share_energy → 78.25%
> - fossil_fuel_consumption → 77.46%
> - greenhouse_gas_emissions → 75.89%
> - renewables_consumption → 75.11%
> - electricity_generation → 67.33%
> - energy_per_gdp → 67.24%
>
> Otras variables relevantes como gdp (49.51%) y primary_energy_consumption (42.81%) también presentan un volumen considerable de datos faltantes.
>
> En cambio, variables como year y country no contienen valores nulos.
>
> Para el tratamiento, se ha optado por eliminar las filas con valores nulos (dropna), con el objetivo de trabajar con un conjunto de datos completo y evitar inconsistencias en el análisis estadístico.

---

## Ejercicio 2 — Inferencia con Scikit-Learn

---

Se ha entrenado un modelo de regresión lineal utilizando un conjunto de entrenamiento de 1022 observaciones y un conjunto de test de 256 observaciones, tras aplicar preprocesamiento y codificación de variables categóricas.

El gráfico de residuos muestra la diferencia entre los valores reales y las predicciones del modelo en función de estas últimas.
Se observa que la mayoría de los residuos se concentran alrededor de cero para valores bajos de predicción, lo que indica un buen ajuste en ese rango. Sin embargo, para valores más altos aparecen residuos más dispersos y algunos outliers significativos.
La existencia de valores extremos en los residuos indica que el modelo no captura correctamente algunos casos, especialmente en niveles altos de consumo energético.

En conjunto, aunque el modelo presenta un R² muy alto, el análisis de residuos revela que el modelo no es totalmente fiable.

---

**Pregunta 2.1** — Indica los valores de MAE, RMSE y R² de la regresión lineal sobre el test set. ¿El modelo funciona bien? ¿Por qué?

> Los resultados obtenidos en el conjunto de test son:
>
> - MAE: 15.8314
> - RMSE: 30.4313
> - R²: 1.0000
>
> En este caso, el MAE significa que el modelo se desvía en esa cantidad respecto al valor real. Mientras que el RMSE indica que existen algunas desviaciones más elevadas que aumentan el error total.
> A primera vista, el modelo parece funcionar demasiado bien, ya que el valor de R² es igual a 1, lo que indica que el modelo explica casi el 100% de la variabilidad de la variable objetivo.
>
> Sin embargo, este resultado es poco realista, ya que en problemas reales es muy raro obtener un ajuste prácticamente perfecto en datos de test.
>
> Como se ha observado en el mapa de correlación, existen variables como fossil_fuel_consumption, electricity_generation y greenhouse_gas_emissions que presentan correlaciones muy altas con la variable objetivo, cercanas a 1. Esto permite al modelo reconstruir casi exactamente el valor real en lugar de aprender una relación generalizable.
>
> Además, este comportamiento también puede verse reforzado por el uso de dropna(), ya que al eliminar filas con valores nulos se reduce el dataset a observaciones más completas y homogéneas, lo que puede incrementar artificialmente las correlaciones entre variables.
>
> Por otro lado, el análisis previo mostró la existencia de multicolinealidad, lo que implica redundancia entre variables predictoras y contribuye a que el modelo dependa excesivamente de ciertas variables.

---

## Ejercicio 3 — Regresión Lineal Múltiple en NumPy

---

En este ejercicio se ha implementado un modelo de regresión lineal múltiple utilizando NumPy, con el objetivo de predecir la variable.

A diferencia del ejercicio anterior, donde se utilizaba Scikit-Learn, en este caso el modelo se construye manualmente mediante álgebra matricial, utilizando la ecuación normal: β = (XᵀX)⁻¹ Xᵀy

En conjunto, el modelo de regresión lineal múltiple construido mediante la ecuación normal presenta un rendimiento adecuado, con errores de predicción dentro de un rango esperado para este tipo de aproximación. No obstante, su capacidad explicativa resulta ligeramente inferior en comparación con el modelo de referencia, lo que puede deberse a la sensibilidad de la inversión matricial y a la ausencia de técnicas de regularización o validación empleadas en implementaciones como Scikit-Learn.

---

**Pregunta 3.1** — Explica en tus propias palabras qué hace la fórmula β = (XᵀX)⁻¹ Xᵀy y por qué es necesario añadir una columna de unos a la matriz X.

> Representa la solución analítica de los Mínimos Cuadrados Ordinarios (OLS), que permite encontrar los coeficientes que minimizan la diferencia entre los valores reales y los predichos.
>
> La columna de unos se añade para permitir que el modelo de regresión tenga una intersección distinta de cero.

**Pregunta 3.2** — Copia aquí los cuatro coeficientes ajustados por tu función y compáralos con los valores de referencia del enunciado.

| Parametro | Valor real | Valor ajustado |
| --------- | ---------- | -------------- |
| β₀        | 5.0        | 4.864995       |
| β₁        | 2.0        | 2.063618       |
| β₂        | -1.0       | -1.117038      |
| β₃        | 0.5        | 0.438517       |

> Los coeficientes estimados por el modelo son muy cercanos a los valores reales, lo que indica que la implementación de la regresión lineal es correcta.
>
> Las diferencias observadas son pequeñas y se deben principalmente al ruido aleatorio presente en los datos. En particular:
>
> - β₀ presenta una ligera subestimación.
> - β₁ se aproxima bastante bien al valor real.
> - β₂ muestra una desviación algo mayor, pero mantiene el signo y orden de magnitud.
> - β₃ también es cercano al valor esperado, con una ligera subestimación.
>
> En conjunto, el modelo ha sido capaz de recuperar adecuadamente la relación lineal subyacente, lo que valida el uso de la solución analítica de mínimos cuadrados.

**Pregunta 3.3** — ¿Qué valores de MAE, RMSE y R² has obtenido? ¿Se aproximan a los de referencia?

> Los valores obtenidos son:

> - MAE: 1.166462
> - RMSE: 1.461243
> - R²: 0.689672
>
> Los valores de MAE (~1.17) y RMSE (~1.46) se encuentran dentro del rango esperado según el enunciado (MAE ≈ 1.20 ± 0.20 y RMSE ≈ 1.50 ± 0.20), lo que indica que el error de predicción es coherente con lo esperado.
>
> Sin embargo, el valor de R² (~0.69) es inferior al valor de referencia (~0.80 ± 0.05), lo que sugiere que el modelo está explicando una menor proporción de la variabilidad de los datos de la esperada.
>
> En conjunto, el modelo presenta un rendimiento aceptable, con errores dentro de lo esperado, aunque con una capacidad explicativa algo inferior a la de referencia.

---

## Ejercicio 4 — Series Temporales

---

Se analiza una serie temporal sintética generada con semilla fija, descomponiéndola en tres componentes: tendencia, estacionalidad y residuo.

La tendencia muestra la evolución general a largo plazo, la estacionalidad recoge patrones periódicos repetitivos y el residuo representa la variabilidad no explicada por los otros dos componentes.

Para evaluar la calidad de la descomposición, se analiza el residuo, comprobando si se comporta como ruido gaussiano (media cercana a cero, varianza constante y ausencia de patrones).

---

**Pregunta 4.1** — ¿La serie presenta tendencia? Descríbela brevemente (tipo, dirección, magnitud aproximada).

> La serie presenta una tendencia creciente y aproximadamente lineal, con una pendiente positiva constante. El crecimiento es progresivo y estable a lo largo del tiempo (2018–2023), con una magnitud moderada.

**Pregunta 4.2** — ¿Hay estacionalidad? Indica el periodo aproximado en días y la amplitud del patrón estacional.

> Se observa una estacionalidad clara con un periodo aproximado de 365 días. El patrón se repite de forma anual con oscilaciones regulares y una amplitud moderada.

**Pregunta 4.3** — ¿Se aprecian ciclos de largo plazo en la serie? ¿Cómo los diferencias de la tendencia?

> Sí, se identifican ciclos de largo plazo de aproximadamente 4 años. Estos se diferencian de la tendencia porque introducen oscilaciones alrededor de ella, mientras que la tendencia es monotónica y creciente. Además, los ciclos tienen una frecuencia mucho menor que la estacionalidad.

**Pregunta 4.4** — ¿El residuo se ajusta a un ruido ideal? Indica la media, la desviación típica y el resultado del test de normalidad (p-value) para justificar tu respuesta.

> El residuo se comporta aproximadamente como ruido blanco.. Presenta media cercana a 0, desviación típica alrededor de 3.5 y un p-value del test de normalidad generalmente superior a 0.05, por lo que no se rechaza la hipótesis de normalidad. Esto sugiere que la descomposición captura adecuadamente la estructura de la serie, aunque pueden existir pequeñas desviaciones.

_Fin del documento de respuestas_
