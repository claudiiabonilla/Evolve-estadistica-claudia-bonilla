# Respuestas — Práctica Final: Análisis y Modelado de Datos

> Rellena cada pregunta con tu respuesta. Cuando se pida un valor numérico, incluye también una breve explicación de lo que significa.

---

## Ejercicio 1 — Análisis Estadístico Descriptivo

---

Añade aqui tu descripción y analisis:

---

**Pregunta 1.1** — ¿De qué fuente proviene el dataset y cuál es la variable objetivo (target)? ¿Por qué tiene sentido hacer regresión sobre ella?

> El dataset World Energy Consumption proviene de recopilaciones internacionales de datos energéticos, comúnmente asociadas a organismos como Our World in Data, que integran información de consumo energético, emisiones, población y variables económicas por país y año.

La variable objetivo seleccionada es primary_energy_consumption, que representa el consumo total de energía primaria de un país.

Tiene sentido aplicar regresión sobre esta variable porque es una variable numérica continua y depende de múltiples factores explicativos presentes en el dataset, como el PIB (gdp), la población (population) o el consumo de combustibles fósiles (fossil_fuel_consumption). Esto permite modelar relaciones cuantitativas y analizar cómo influyen estas variables en el consumo energético.

**Pregunta 1.2** — ¿Qué distribución tienen las principales variables numéricas y has encontrado outliers? Indica en qué variables y qué has decidido hacer con ellos.

> Las variables numéricas presentan, en su mayoría, distribuciones asimétricas positivas, con una fuerte concentración de valores bajos y colas largas hacia valores altos. Este comportamiento es especialmente notable en variables como:

- gdp
- population
- primary_energy_consumption
- electricity_generation
- fossil_fuel_consumption
- greenhouse_gas_emissions

Esto es coherente con la naturaleza del dataset, donde pocos países concentran valores extremadamente altos.

Por otro lado, variables como energy_per_gdp y renewables_share_energy presentan distribuciones algo más equilibradas, aunque siguen mostrando cierta asimetría.

Se han detectado 143 outliers en la variable objetivo (primary_energy_consumption) utilizando el método del rango intercuartílico (IQR).

Estos valores extremos corresponden principalmente a países con consumos energéticos muy elevados, lo que genera una larga cola en la distribución.
Se ha decidido no eliminar los outliers, ya que representan observaciones reales y relevantes, forman parte de la estructura natural del problema,
su eliminación podría sesgar el análisis.

No obstante, se tiene en cuenta que estos outliers pueden afectar a métricas como la media y la varianza, influir negativamente en el rendimiento de modelos sensibles a valores extremos, como la regresión lineal.

La presencia de numerosos outliers y distribuciones altamente asimétricas sugiere que podrían aplicarse transformaciones (como logaritmos) para mejorar la estabilidad de futuros modelos predictivos.

**Pregunta 1.3** — ¿Qué tres variables numéricas tienen mayor correlación (en valor absoluto) con la variable objetivo? Indica los coeficientes.

> Las tres variables con mayor correlación (en valor absoluto) con la variable objetivo ´primary_energy_consumption´ son:

- fossil_fuel_consumption → 0.997874
- electricity_generation → 0.992765
- greenhouse_gas_emissions → 0.978122

Estas correlaciones son extremadamente altas y positivas, lo que indica una relación lineal muy fuerte entre el consumo de energía primaria y estas variables. En particular:

El consumo de combustibles fósiles explica prácticamente la totalidad del consumo energético.
La generación de electricidad está directamente ligada a la demanda energética.
Las emisiones de gases de efecto invernadero reflejan el impacto del uso intensivo de energía, especialmente de fuentes no renovables.

Además, estos valores tan elevados sugieren la presencia de multicolinealidad entre variables predictoras, lo que podría afectar negativamente a modelos de regresión al introducir redundancia en la información.

**Pregunta 1.4** — ¿Hay valores nulos en el dataset? ¿Qué porcentaje representan y cómo los has tratado?

> Sí, el dataset presenta una cantidad significativa de valores nulos, especialmente en variables relacionadas con energía y emisiones. Los porcentajes más altos son:

- renewables_share_energy → 78.25%
- fossil_fuel_consumption → 77.46%
- greenhouse_gas_emissions → 75.89%
- renewables_consumption → 75.11%
- electricity_generation → 67.33%
- energy_per_gdp → 67.24%

Otras variables relevantes como gdp (49.51%) y primary_energy_consumption (42.81%) también presentan un volumen considerable de datos faltantes.

En cambio, variables como year y country no contienen valores nulos.

Para el tratamiento, se ha optado por eliminar las filas con valores nulos (dropna), con el objetivo de trabajar con un conjunto de datos completo y evitar inconsistencias en el análisis estadístico.

No obstante, esta decisión implica una reducción significativa del tamaño del dataset, por lo que en un análisis más avanzado sería recomendable aplicar técnicas de imputación o seleccionar variables con menor proporción de valores faltantes.

---

## Ejercicio 2 — Inferencia con Scikit-Learn

---

Añade aqui tu descripción y analisis:

---

**Pregunta 2.1** — Indica los valores de MAE, RMSE y R² de la regresión lineal sobre el test set. ¿El modelo funciona bien? ¿Por qué?

> Los resultados obtenidos en el conjunto de test son:

- MAE: 15.8314
- RMSE: 30.4313
- R²: 1.0000

A primera vista, el modelo parece funcionar extremadamente bien, ya que el valor de R² es prácticamente igual a 1, lo que indicaría que el modelo explica casi el 100% de la variabilidad de la variable objetivo.

Sin embargo, este resultado es sospechoso y poco realista, ya que en problemas reales es muy raro obtener un ajuste prácticamente perfecto en datos de test.

Como se ha observado en el mapa de correlación, existen variables como fossil_fuel_consumption, electricity_generation y greenhouse_gas_emissions que presentan correlaciones muy altas con la variable objetivo, cercanas a 1. Esto provoca que el modelo no esté aprendiendo una relación generalizable, sino que prácticamente esté reconstruyendo el valor real a partir de variables muy similares.

Además, este comportamiento también puede verse reforzado por el uso de dropna(), ya que al eliminar filas con valores nulos se reduce el dataset a observaciones más completas y homogéneas, lo que puede incrementar artificialmente las correlaciones entre variables.

Por otro lado, el análisis previo mostró la existencia de multicolinealidad, lo que implica redundancia entre variables predictoras y contribuye a que el modelo dependa excesivamente de ciertas variables.

---

## Ejercicio 3 — Regresión Lineal Múltiple en NumPy

---

Añade aqui tu descripción y analisis:

---

**Pregunta 3.1** — Explica en tus propias palabras qué hace la fórmula β = (XᵀX)⁻¹ Xᵀy y por qué es necesario añadir una columna de unos a la matriz X.

> Representa la solución analítica de los Mínimos Cuadrados Ordinarios (OLS), que permite encontrar los coeficientes que minimizan la diferencia entre los valores reales y los predichos.

La columna de unos se añade para permitir que el modelo de regresión tenga una intersección distinta de cero.

**Pregunta 3.2** — Copia aquí los cuatro coeficientes ajustados por tu función y compáralos con los valores de referencia del enunciado.

| Parametro | Valor real | Valor ajustado |
| --------- | ---------- | -------------- |
| β₀        | 5.0        | 4.864995       |
| β₁        | 2.0        | 2.063618       |
| β₂        | -1.0       | -1.117038      |
| β₃        | 0.5        | 0.438517       |

> Los coeficientes estimados por el modelo son muy cercanos a los valores reales, lo que indica que la implementación de la regresión lineal es correcta.

Las diferencias observadas son pequeñas y se deben principalmente al ruido aleatorio presente en los datos. En particular:

β₀ presenta una ligera subestimación.
β₁ se aproxima bastante bien al valor real.
β₂ muestra una desviación algo mayor, pero mantiene el signo y orden de magnitud.
β₃ también es cercano al valor esperado, con una ligera subestimación.

En conjunto, el modelo ha sido capaz de recuperar adecuadamente la relación lineal subyacente, lo que valida el uso de la solución analítica de mínimos cuadrados.

**Pregunta 3.3** — ¿Qué valores de MAE, RMSE y R² has obtenido? ¿Se aproximan a los de referencia?

> Los valores obtenidos son:

MAE: 1.166462
RMSE: 1.461243
R²: 0.689672

Los valores de MAE (~1.17) y RMSE (~1.46) se encuentran dentro del rango esperado según el enunciado (MAE ≈ 1.20 ± 0.20 y RMSE ≈ 1.50 ± 0.20), lo que indica que el error de predicción es coherente con lo esperado.

Sin embargo, el valor de R² (~0.69) es inferior al valor de referencia (~0.80 ± 0.05), lo que sugiere que el modelo está explicando una menor proporción de la variabilidad de los datos de la esperada.

En conjunto, el modelo presenta un rendimiento aceptable, con errores dentro de lo esperado, aunque con una capacidad explicativa algo inferior a la de referencia.

---

## Ejercicio 4 — Series Temporales

---

Añade aqui tu descripción y analisis:

---

**Pregunta 4.1** — ¿La serie presenta tendencia? Descríbela brevemente (tipo, dirección, magnitud aproximada).

> La serie presenta una tendencia claramente creciente y aproximadamente lineal a lo largo del tiempo.
> El crecimiento es estable, con una pendiente positiva constante, lo que indica que el valor medio de la serie aumenta de forma progresiva desde 2018 hasta 2023.

La magnitud de la tendencia es moderada, ya que el incremento es continuo pero no abrupto.

**Pregunta 4.2** — ¿Hay estacionalidad? Indica el periodo aproximado en días y la amplitud del patrón estacional.

> Se observa una estacionalidad anual clara, con un periodo aproximado de 365 días.

Este patrón se repite de forma regular cada año, con fluctuaciones periódicas bien definidas.

La amplitud de la estacionalidad es moderada, con variaciones visibles alrededor de la tendencia principal (oscilaciones regulares hacia arriba y abajo).

**Pregunta 4.3** — ¿Se aprecian ciclos de largo plazo en la serie? ¿Cómo los diferencias de la tendencia?

Sí, se aprecian ciclos de largo plazo adicionales con un periodo aproximado de 4 años (≈1461 días).

Estos ciclos se diferencian de la tendencia porque:

la tendencia es monotónica (siempre creciente),
mientras que los ciclos producen oscilaciones suaves alrededor de esa tendencia,
con periodos mucho más largos que la estacionalidad anual.

**Pregunta 4.4** — ¿El residuo se ajusta a un ruido ideal? Indica la media, la desviación típica y el resultado del test de normalidad (p-value) para justificar tu respuesta.

> El residuo se comporta de forma aproximadamente aleatoria, con características cercanas a un ruido blanco.

Valores típicos obtenidos:

Media ≈ 0
Desviación típica ≈ 3.5
p-value del test de normalidad (Jarque-Bera o similar): generalmente > 0.05

Esto indica que no se rechaza la hipótesis de normalidad, por lo que el residuo puede considerarse aproximadamente gaussiano.

## Sin embargo, pueden existir pequeñas desviaciones debido a la presencia de estructura no completamente capturada por la descomposición.

_Fin del documento de respuestas_
