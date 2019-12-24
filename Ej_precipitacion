#!/usr/bin/env python
# coding: utf-8

# # Ejercicios de precipitación
# 
# __Índice__<br>
# 
# __[Ejercicio 1 - Interpolación de datos faltantes](#Ejercicio-1---Interpolación-de-datos-faltantes)__<br>
# [Método de la media](#Método-de-la-media)<br>
# [Método de la razón normal](#Método-de-la-razón-normal)<br>
# [Método de la distancia inversa](#Método-de-la-distancia-inversa)<br>
# 
# __[Ejercicio 2 - Curvas de doble masa](#Ejercicio-2---Curvas-de-doble-masa)__<br>
# 
# __[Ejercicio 3 - Curvas de doble masa](#Ejercicio-2---Curvas-de-doble-masa)__<br>
# 
# __[Ejercicio 4 - Método hipsométrico](#Ejercicio-4---Método-hipsométrico)__<br>
# 
# __[Exercise 5 - Curva intensidad-duración-frecuencia](#Exercise-5---Curva-intensidad-duración-frecuencia)__<br>

# ### Aspectos generales
# El primer paso en todo código en Python es cargar aquellos paquetes que vamos a necesitar en nuestros cálculos posteriores. Un paquete es un conjunto de herramientas que no vienen en la instalación por defecto de Python.
# 
# Hay tres paquetes de uso habitual en Python que importaremos habitualmente: 
# *  __[NumPy](http://www.numpy.org/)__ es un paquete básico en programación científica que incluye el tratamiento de vectores n-dimensionales, álgebra lineal, transformadas de Fourier, generación de números aleatorios según diversas funciones de distribución... _Numpy_ utiliza una estructura vectorial de datos llamada _array_; los datos de un _array_ deben tener siempre la misma naturaleza (enteros, decimales...).
# 
# *  __[pandas](https://pandas.pydata.org/)__ es un paquete que permite la organización de los datos en una estructura llamada _data frame_. Un _data frame_ asemeja el uso habitual de una tabla Excel, es decir, cada columna representa una variable con un mismo tipo de datos (numérico, texto, booleano...) y cada fila una observación. Un _data frame_ tiene un índice que identifica cada una de las filas (observaciones) y un encabezado que identifica a cada una de las columnas (variables).
# 
# * __[matplotlib](https://matplotlib.org/)__ es un paquete que permite la generación de gráficos.
# 
# * __[SciPy](https://www.scipy.org/)__ contiene numerosas herramientas numéricas eficientes y de fácil manejo, por ejemplo integración numérica y optimización.

# In[1]:


import numpy as np

import pandas as pd

from matplotlib import pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
plt.style.use('seaborn-whitegrid')

from scipy.stats import genextreme
from scipy.optimize import curve_fit


# Es necesario instalar una serie de paquetes.<br>
# *  Abrir Anaconda prompt<br>
# *  `conda install scipy` + `Enter`<br>
# 
# Instalaremos un visor con el que saber qué variables hemos declarado.<br>
# *  Abrir Anaconda prompt<br>
# *  `pip install jupyter_contrib_nbextensions` + `Enter`<br>
# *  `jupyter contrib nbextension install --user` + `Enter`<br>
# *  `jupyter nbextension enable varInspector/main` + `Enter`<br>

# ### Estructuras de datos básicas en Python
# **Listas**<br>
# Las listas son conjuntos de datos de cualquier tipo (enteros, reales, texto...). Son mutables, es decir, permiten modificar sus valores una vez creada la lista.

# In[2]:


# crear una lista
a = [1, 'hola', 1.5]


# In[3]:


# ver uno de los valores de la lista
a[1]


# In[4]:


# modificar uno de los valores de la lista
a[1] = 'adiós'


# **Tuplas**<br>
# Otra estructura que agrupa valores de diversa naturaleza. Las tuplas, a diferencia de las listas, no permiten modificar sus valores.

# In[5]:


# crear una lista
b = (2, 'rojo', np.nan)


# In[6]:


# ver uno de los valores de la tupla
b[2]


# In[7]:


# modificar uno de los valores de la lista
b[2] = 0


# **Arrays**<br>
# Es una estructura propia del paquete *NumPy* que permite trabajar con vectores ymatrices, y hacer cálculos de forma sencilla. Para ello, todos los datos de un *array* han de ser de la misma naturaleza.

# In[8]:


# crear un array a partir de 'a'
c = np.array([1.5, 2.1, 4.5])


# In[9]:


# extraer varios valores del array
c[1:]


# In[10]:


# invertir la posición de un array
c[::-1]


# In[11]:


# modificar algún valor del array
c[2] = 10.3


# In[12]:


# hacer algún cálculo con el array (la media)
c.mean()


# **Pandas: series y _data frames_**<br>
# El paquete pandas permite trabajar con tablas de 2 dimensiones (data frames) o de una dimensión (series). Hace uso de las herramientas de *NumPy* para permitir cálculos de forma sencilla a partir de los datos contenidos en la tabla. Dentro de una columna de la tabla, todos los datos deben ser de la misma naturaleza; en columnas distintas puede haber distintos tipos de datos.

# In[13]:


# crear un 'data frame' con el nombre, edad y peso
d = [['Pedro', 36, 71],
     ['Laura', 40, 58],
     ['Juan', 25, 65]]
d = pd.DataFrame(data=d, columns=['nombre', 'edad', 'peso'])
d


# In[14]:


# una de las columnas del 'data frame' es una serie
d2 = d.nombre
d2


# In[15]:


# sobre los datos de un 'data frame' se pueden aplicar funciones al igual que en NumPy
d.mean()


# **Diccionarios**<br>
# Los diccionarios permiten guardar datos en diversas de las estructuras mencionadas anteriormente en un objeto único, asignando un nombre (clave) a cada una de ellas.

# In[16]:


# crear un diccionario que contenga todos los datos anteriormente creados
# siendo la clave el tipo de estructura
e = {'list': a,
     'tuple': b,
     'array': c,
     'dataframe': d}


# In[17]:


# extraer uno de los datos del diccionario
e['list']


# ## <font color=steelblue>Ejercicio 1 - Interpolación de datos faltantes
# 
# <font color=steelblue>La figura muestra una cuenca hidrográfica experimental y la ubicación de 11 pluviómetros en su zona de influencia. La estación F dejó de funcionar durante una tormenta, con lo que no se dispone de datos en ella. Utiliza los datos que sí registraron el resto de estaciones de la cuenca para estimar la precipitación en F.
# 
# Los datos de precipitación se encuentran en el archivo *RainfallData_Exercise_001.csv*.

# <img src="RainGages.png" alt="Mountain View" style="width:300px">

# | Gage | X      | Y       | Average Annual Precip. (mm) | Measured Storm Precip. (mm) |
# |------|--------|---------|-----------------------------|-----------------------------|
# | C    | 385014 | 4778553 | 1404                        | 11.6                        |
# | D    | 389634 | 4779045 | 1433                        | 14.8                        |
# | E    | 380729 | 4775518 | 1665                        | 13.3                        |
# | F    | 387259 | 4776670 | 1137                        | -                           |
# | G    | 389380 | 4776484 | 1235                        | 12.3                        |
# | H    | 382945 | 4772356 | 1114                        | 11.5                        |
# | I    | 386399 | 4771795 | 1101                        | 11.6                        |
# | J    | 388397 | 4772419 | 1086                        | 11.2                        |
# | K    | 389287 | 4771097 | 1010                        | 9.7                         |

# Los métodos de completado de datos siguen, al igual que muchos métodos de interpolación, la siguiente fórmula general:
# 
# $$\hat{p_o} = \sum_{i=1}^{n} w_i·p_i$$
# 
# Donde $\hat{p_o}$ es el dato de precipitación en la estación que queremos interpolar, $n$ es el número de estaciones, $w_i$ y $p_i$ son la ponderación y la precipitación recogida en cada una de las estaciones a partir de las cuales se completa/interpola.

# Se importan los datos a través de _pandas_ y se guardan en un objeto (_data frame_ en terminología _pandas_).

# In[19]:


# Importar 'RainfallData_Exercise_001.csv'
data1 = pd.read_csv('RainfallData_Exercise_001.csv', index_col='Gage')
data1


# Un _data frame_ tiene asociados una serie de atributos. Por ejemplo, se puede extraer la dimensión de la tabla (`shape`), el número total de elemenos (`size`), el nombre de las variables (`columns`) o los datos en forma de _numpy array_ (`values`).

# In[20]:


#dimensión
data1.shape


# In[21]:


# nº de datos
data1.size


# In[22]:


# índice
data1.index


# In[23]:


# columnas
data1.columns


# Además, se pueden aplicar sobre él funciones de cualquier tipo. Por ejemplo, la función `describe` genera un resumen estadístico de cada una de las variables (columnas) del _data frame_.

# In[24]:


# resumen
data1.describe()


# In[25]:


# media
round(data1.mean(), 0)


# In[26]:


# primeras líneas
data1.head()


# Para seleccionar un dato o un conjunto de datos de un _data frame_ debe indicarse la fila y columna bien a través de su nombre o de su posición. 
# 
# * La función `.loc` permite extraer datos a través del nombre de las filas y columnas.
# 
# * La función `.iloc` permite extraer datos a través de la posición de las filas y columnas. IMPORTANTE: la posición en Python se inicia desde el 0, es decir, la primera fila es la fila 0, no la 1.

# In[27]:


# Extraer mediante .loc
data1.loc['A', 'Measured Storm Precip. (mm)']
data1.loc['A', :]
data1.loc[['A', 'C'], 'Average Annual Precip. (mm)']


# In[28]:


# extraer mediante .iloc
data1.iloc[0, 0]
data1.iloc[0, :]
data1.iloc[[0, 2], 1]


# In[29]:


# extraer una columna por nombre
data1['Average Annual Precip. (mm)']


# In[30]:


# Simpliciar el nombre de las columnas
# d: distancia a F en km
# P: precipitación media anual en mm
# p: precipitación en la tormenta de interés en mm
data1.columns = ['X', 'Y', 'Pan', 'p']

data1.head(2)


# ### Método de la media
# 
# En el método de la media se asimila la precipitación en el punto de estudio como la media de la precipitación en las estaciones en su alrededor.
# 
# Siguiendo la ecuación del inicio, se da el mismo  peso a todas las estaciones ($\frac{1}{n}$), siendo *n* el número de estaciones consideradas.
# 
# $$w_i=\frac{1}{n}$$
# 
# $$\hat{p_o} = \frac{1}{n}\sum_{i=1}^{n} p_i$$

# In[31]:


po_mm = data1['p'].mean()


# In[32]:


print('La precipitación en F es:')
print('pf =', round(po_mm, 1), 'mm')


# Al hacer la media de todas las estaciones de las que disponemos de datos obtenemos un valor suavizado de la precipitación en F. Es decir, puede ser que estemos teniendo en cuenta datos de estaciones lo suficientemente lejanas como para que su dato de precipitación no sea representativo de la precipitación en F.
# 
# Para evitar en este problema, se suele aplicar el método de la media utilizando únicamente la estación más cercana del punto de interés en cada uno de los cuatro cuadrantes.

# In[33]:


closest = ['C', 'D', 'G', 'I']
po_mmc = data1.loc[closest, 'p'].mean()

print('La precipitación en F es:')
print('pf =', round(po_mmc, 1), 'mm')


# ### Método de la razón normal
# 
# Otra forma de tener en cuenta la conexión entre la precipitación en una estación cualquiera y la precipitación objetivo es incluir el cociente entre la precipitación anual media en ambas estaciones, es decir, la razón normal.
# 
# Aplicando esta correción sobre el método de la media se obtiene el método de la razón normal.
# 
# $$w_i = \frac{1}{n}\frac{P_o}{P_i} = \frac{1}{n}RN$$
# 
# $$\hat{p_o} = \frac{1}{n}\sum_{i=1}^{n} \frac{P_o}{P_i} p_i$$
# 
# Donde $P_o$ y $P_i$ son la precipitación media anual en la estación objetivo y las estaciones utilizadas en la interpolación, repectivamete, y $NR$ es la razón normal.

# In[34]:


# Calcular el cociente entre la precipitación anual en 'F' y en cada una de las otras estaciones
data1['RN'] = data1.loc['F', 'Pan'] / data1['Pan']

data1


# In[35]:


# Extraer las estaciones con registro
data1_ = data1.drop('F')
data1_


# Then we just multiply the **normal ratio** by the precipitation measured at the remaining stations

# In[36]:


# Producto de la razón normal por la precipitación observada en cada estación
NR_p = data1_['RN'] * data1_['p']
NR_p


# In[37]:


# La media de ese producto es la precipitación interpolada por el método de la razón normal
po_rn = NR_p.mean()

print('La precipitación en F es:')
print('pf =', round(po_rn, 1), 'mm')


# In[38]:


# Todo de una vez
po_rn = np.mean(data1.loc[:, 'RN'] * data1.loc[:, 'p'])

print('La precipitación en F es:')
print('pf =', round(po_rn, 1), 'mm')


# El método de la razón normal se puede aplicar también sólo sobre la estación más cercana en cada cuadrante.

# In[39]:


po_rnc = np.mean(data1.loc[closest, 'RN'] * data1.loc[closest, 'p'])

print('La precipitación en F es:')
print('pf =', round(po_rnc, 1), 'mm')


# ###  Método de la distancia inversa
# En el método de la distancia inversa se basa en que la precipitación en las estaciones más cercanas es más representativo de la precipitación en la estación objetivo. Para ello, la peso de cada estación se calcula con la inversa de la distancia a la estación objetivo elevada a un exponente; para que la suma de los pesos sea 1, se divide la distancia inversa por la suma de las distancias inversas de todas las estaciones.
# 
# $$w_i = \frac{d_{i}^{-b}}{\sum_{i=1}^{n}d_{i}^{-b}}$$
# 
# $$\hat{p_o} = \sum_{i=1}^{n}\frac{d_{i}^{-b}}{\sum_{i=1}^{n}d_{i}^{-b}}·p_i = \frac{1}{\sum_{i=1}^{n}d_{i}^{-b}}\sum_{i=1}^{n}d_{i}^{-b}·p_i$$
# 
# Donde $d_i$ es la distancia entre la estación *i* y la estación objetivo, y $b$ es un exponente a elegir por el modelador. Habitualmente se utiliza como exponente el cuadrado, dando lugar al método de la distancia inversa al cuadrado.

# In[40]:


# calcular distancia a la estación F
distX = data1.loc['F', 'X'] - data1.loc[:, 'X'] # distancia en el eje X
distY = data1.loc['F', 'Y'] - data1.loc[:, 'Y'] # distancia en el eje X
data1['d'] = np.sqrt(distX**2 + distY**2)       # distancia total


# In[41]:


# Extraer las estaciones con registro
data1_ = data1.drop('F')
data1_


# __$b=-1$ paso a paso__

# In[42]:


# Definir el exponente
b = -1


# In[43]:


# Calcular el inverso de la distancia para cada estación
data1_['di'] = data1_['d']**b
data1_


# In[44]:


# Calcular la suma de los inversos de la distancia
Sd = data1_['di'].sum()
Sd


# In[45]:


# Calcular el peso de cada estación
w = data1_.di / Sd
w


# In[46]:


# Calcular la precipitación en F
po_di1 = np.sum(w * data1_.p)

print('La precipitación en F es:')
print('pf =', round(po_di1, 1), 'mm')


# __$b=-2$ abreviadamente__

# In[47]:


b = -2


# In[48]:


# Calcula el inverso de la distancia al cuadrado
data1_['di2'] = data1_.d**b
data1_


# In[49]:


# Calcular la precipitación en F
po_di2 = np.sum(data1_.di2 / np.sum(data1_.di2) * data1_.p) 

print('La precipitación en F es:')
print('pf =', round(po_di2, 1), 'mm')


# Al igual que ocurría en los otros métodos, la distancia inversa puede aplicarse sólo a la estación más cercana en cada cuadrante.

# In[68]:


# Calcular la precipitación en F
po_di2c = np.sum(data1_.loc[closest, 'di2'] * data1_.loc[closest, 'p']) /           np.sum(data1_.loc[closest, 'di2'])

print('La precipitación en F es:')
print('pf =', round(po_di2c, 1), 'mm')


# __Comparativa de métodos__

# In[70]:


resultados = [po_mm, po_mmc, po_rn, po_rnc, po_di1, po_di2, po_di2c]

plt.bar(range(len(resultados)), resultados, width=0.4, alpha=.75)
plt.title('Comparativa de métodos', fontsize=16, weight='bold')
plt.xlabel('metodo', fontsize=13)
plt.xticks(range(len(resultados)), ['med', 'med_c', 'RN', 'RN_c', 'DI',
                                    'DI2', 'DI2_c'])
plt.ylim((0, 16))
plt.ylabel('precipitación en f (mm)', fontsize=13);


# ### Interpolación
# Los métodos anteriormente mostrados se aplican también para la interpolación de mapas de precipitación. Seguidamente, generaremos un mapa de precipitación en la tormenta mediante el método de la distancia inversa.
# 
# El primer paso es crear una función de Python que calcule la distancia inversa.

# In[71]:


def IDW(x, y, estX, estY, estP, b=-2):
    """Interpolar mediante el método de la distancia inversa (inverse distance
    weighted)
    
    Entradas:
    ---------
    x:       float. Coordenada X del punto objetivo
    y:       float. Coordenada Y del punto objetivo
    estX:    Series. Serie de coordenadas X de las estaciones con dato
    estY:    Series. Serie de coordenadas Y de las estaciones con dato
    estP:    Series. Serie con el dato observado en las estaciones
    b:       int. Exponente de la distancia para calcular su inverso
    
    Salida:
    -------
    p:       float. Precipitación interpolada en el punto (x, y)
    """
    
    # distancia al punto de cálculo
    distX = x - estX                    # distancia en el eje X
    distY = y - estY                    # distancia en el eje X
    dist = np.sqrt(distX**2 + distY**2) # distancia total
    # inverso de la distancia
    idw = dist**b
    # interpolar
    p = np.sum(idw / np.sum(idw) * estP)
    
    return round(p, 1)


# In[72]:


# Prueba de la función para repetir la interpolación en el punto F
IDW(data1.loc['F', 'X'], data1.loc['F', 'Y'], data1_.X, data1_.Y,
    data1_.p, b=-2)


# Ahora aplicaremos la función de interpolación iterativamente sobre las celdas del mapa que queremos interpolar. Para ello, hay que crear el mapa antes que nada.

# In[73]:


# Coordenadas X e Y de un raster cuadrados
xo, xf = 382200, 390200
X = np.arange(xo, xf, 100)
yo, yf = 4771400, 4779400
Y = np.arange(yo, yf, 100)


# In[74]:


# crear un mapa vacío (NaN) con las dimensiones de las coordenadas
pcp = np.zeros((len(X), len(Y)))


# In[75]:


# interpolar la precipitación en cada una de las celdas del mapa
for i, y in enumerate(Y[::-1]): # importante invertir la posición de 'Y'
    for j, x in enumerate(X):
        pcp[i, j] = IDW(x, y, data1_.X, data1_.Y, data1_.p, b=-2)


# In[76]:


# gráfico con las estaciones y el mapa de precipitación interpolada
# -----------------------------------------------------------------
# configuración
plt.figure(figsize=(6, 6))
plt.axis('equal')

# mapa interpolado
pmap = plt.imshow(pcp, extent=[xo, xf, yo, yf], cmap='Blues')
cb = plt.colorbar(pmap)
cb.set_label('precipitación (mm)', rotation=90, fontsize=12)

# puntos con las estaciones
plt.scatter(data1_.X, data1_.Y, c='k', s=data1_.p**3/30);


# ## <font color=steelblue>Ejercicio 2 - Curvas de doble masa<br>
# 
# <font color=steelblue>La tabla *MassCurve* en el archivo *RainfallData.xlsx* proporciona la precipitación anual medida durante 17 años en cinco estaciones pluviométricas de una región. La ubicación de la estación C cambió en el año 1974. Realiza un análisis mediante una curva de doble masa para verificar la consistencia en la información del pluviómetro y realiza los ajustes pertinentes para corregir las inconsistencias descubiertas.</font>

# Una **curva de doble masa** es un gráfico de datos acumulado de una serie de datos de una variable frente a la serie de datos acumulados de otra variable en el mismo periodo de medición. Habitualmente, la variable de comparación es la serie acumulada de la media de las observaciones en otras estaciones.
#     
# <img src="Double mass curve.JPG" alt="Mountain View" style="width:450px">
# > <font color=grey>Curva de doble masa de datos de precipitación. *(Double-Mass Curves. USGS, 1960)*
#     
# La serie de una estación es correcta si la curva de doble masa es una línea recta; la pendiente de dicha recta es la constante de proporcionalidad entre las series. Un cambio de pendiente en la recta significa un cambio en la constante de proporcionalidad y que la serie antes o después de ese punto debe ser corregida.
# 
# La curva de doble masa, cuando se aplica a precipitación, toma la forma $Y=bX$, donde $b$ es la pendiente. No hay ordenada en el origen.

# In[77]:


# Importar los datos de la hoja 'Data' en '2MassCurve.xls'
data2 = pd.read_excel('RainfallData.xlsx', sheet_name='2MassCurve',
                      skiprows=4,
             index_col=0, usecols=range(6))
data2.head()


# In[78]:


# Calcular la media anual entre todas las estaciones
data2['AVG'] = data2.mean(axis=1)
data2.head()


# Primeramente, creamos un gráfico de dispersión que compare la serie de precipitación anual en el pluviómetro C frente a la media de todas las estaciones. La gráfica muestra también la regresión lineal entre las dos series según la fórmula $Y=bX$.

# In[79]:


def linear_reg(x, b):
    """Linear regression with no intecept
    
    y = b * x   
    
    Input:
    ------
    x:         float. Independet value
    b:         float. Slope of the linear regression
    
    Output:
    -------
    y:         float. Regressed value"""
    
    y = b * x
    return y


# In[81]:


# Ajustar la regresión lineal
b = curve_fit(linear_reg, data2.AVG, data2.C)[0][0]
b


# In[82]:


fig, ax = plt.subplots(figsize=(5,5))

# configuración
ax.set_title('Serie de precipitación media anual (mm)', fontsize=14,
             fontweight='bold')
ax.set_xlabel('media de todas las estaciones', fontsize=13)
ax.set_ylabel('estación C', fontsize=13)
ax.set(xlim=(600, 1600), ylim=(600, 1600))

# diagrama de dispersión
ax.scatter(data2.AVG, data2.C)

# recta de regresión
ax.plot([0, 3000], [0, b * 3000], 'k--', linewidth=1)

# label one every five years
years = data2.index[::5]
xyear = [data2.loc[year, 'AVG'] + 10 for year in years]
yyear = [data2.loc[year, 'C'] - 20 for year in years] 
for i, year in enumerate(years):
    ax.text(xyear[i], yyear[i], year, verticalalignment='center')

plt.tight_layout()


# Este tipo de gráfico tiene mucha dispersión causada por la variabilidad anual del clima, por lo que no es conveniente para encontrar anomalías. 
# 
# Por eso se utiliza la **curva de doble masa**. Esta gráfica se crea a partir de las series de **precipitación acumulada**. De esta manera, la gráfica ha de tener siempre una pendiente positiva y continua; cualquier cambio de pendiente representa una anomalía en la serie de precipitación.

# In[83]:


# Serie anual de precipitación acumulada
accData2 = data2.cumsum()
accData2.head()


# In[85]:


# ajustar la regresión lineal
b = curve_fit(linear_reg, accData2.AVG, accData2.C)[0][0]
b


# In[86]:


fig, ax = plt.subplots(figsize=(5,5))
lim = 20000
thr = 0.1

# configuración
ax.set_title('Curva de doble masa (mm)', fontsize=14, fontweight='bold')
ax.set_xlabel('media de todas las estaciones', fontsize=13)
ax.set_ylabel('estación C', fontsize=13)
ax.set(xlim=(0, lim), ylim=(0, lim))

# diagrama de dispersión
ax.scatter(accData2.AVG, accData2.C, label='original')

# regresión lineal
ax.plot([0, lim], [0, b * lim], 'k--', linewidth=1)

# etiquetar uno de cada cinco años
years = accData2.index[::5]
xyear = [accData2.loc[year, 'AVG'] + 200 for year in years]
yyear = [accData2.loc[year, 'C'] for year in years] 
for i, year in enumerate(years):
    ax.text(xyear[i], yyear[i], year, verticalalignment='center')
          
plt.tight_layout()


# In[87]:


# identificar años con anomalía
for j, year in enumerate(accData2.index[4:-4]):
    # pendiente de la recta de regresión hasta 'year'
    p1 = np.polyfit(accData2.loc[:year, 'AVG'],
                    accData2.loc[:year, 'C'], 1)
    # pendiente de la recta de regresión a partir de 'year'
    p2 = np.polyfit(accData2.loc[year + 1:, 'AVG'],
                    accData2.loc[year + 1:, 'C'], 1)
    # identificar como anomalía si el cociente de las pendientes se aleja de 1
    if (p1[0] / p2[0] < 1 - thr) | (p1[0] / p2[0] > 1 + thr):
        print("Potential anomaly: year {0}".format(year))


# El análisis muestra un cambio en la pendiente a partir de 1976, lo que concuerda con el hecho de que la estación cambió de ubicación en 1974. Puesto que no disponemos de información para decidir si la serie correcta es la anterior o posterior a 1976, corregiremos los datos previos a dicha fecha.

# In[88]:


# año de la anomalía
year = 1976


# In[89]:


# pendiente antes de la anomalía
# ------------------------------
b_wrong = curve_fit(linear_reg, accData2.loc[:year, 'AVG'],
                    accData2.loc[:year, 'C'])[0][0]
b_wrong


# In[90]:


# pendiente después de la anomalía
# --------------------------------
# extraer datos posteriores a la anomalía
temp = data2.loc[year + 1:, :].copy()
# calcular serie acumulada
accTemp = temp.cumsum(axis=0)    
# ajustar la regresión
b_right = curve_fit(linear_reg, accTemp.AVG, accTemp.C)[0][0]
b_right


# In[91]:


# corregir datos originales
# -------------------------
# crear columna para los datos corregidos
data2['C_c'] = data2.C
# corregir datos
data2.loc[:year, 'C_c'] = data2.loc[:year, 'C'] * b_right / b_wrong
# recalcular la serie acumulada
accData2 = data2.cumsum(axis=0)


# In[92]:


# GRÁFICO CON LA SERIE ORIGINAL Y LA CORREGIDA
# --------------------------------------------
fig, ax = plt.subplots(figsize=(5,5))
lim = 20000

# configuración
ax.set_title('Curva de doble masa (mm)', fontsize=14, fontweight='bold')
ax.set_xlabel('media de las estaciones', fontsize=13)
ax.set_ylabel('estación C', fontsize=13)
ax.set(xlim=(0, lim), ylim=(0, lim))

# diagramas de dispersión
ax.scatter(accData2.AVG, accData2.C, label='original')
ax.scatter(accData2.AVG, accData2.C_c, marker='x', label='corregido')

# regresión linal
b = curve_fit(linear_reg, accData2.AVG, accData2.C_c)[0][0]
ax.plot([0, lim], [0, b * lim], 'k--', linewidth=1)

# etiquetas uno de cada cinco años
years = accData2.index[::5]
xyear = [accData2.loc[year, 'AVG'] + 200 for year in years]
yyear = [accData2.loc[year, 'C'] for year in years] 
for i, year in enumerate(years):
    ax.text(xyear[i], yyear[i], year, verticalalignment='center')

ax.legend(loc=4, ncol=1, fontsize=13)
plt.tight_layout()


# ## <font color=steelblue>Ejercicio 3 - Curvas de doble masa<br>
# 
# <font color=steelblue>Realiza un análisis mediante la curva de doble masa con los datos proporcionados en la tabla *Exercise_003* del archivo *RainfallData.xlsx*.</font>

# In[93]:


# Importar los datos
data3 = pd.read_excel('RainfallData.xlsx', sheet_name='Exercise_003',
                      skiprows=0, index_col=0)

# Calcular la media anual entre todas las estaciones
data3['AVG'] = data3.mean(axis=1)

# Serie de precipitación acumulada
accData3 = data3.cumsum()

data3.head()


# In[98]:


fig, ax = plt.subplots(2, 3, figsize=(12,8))
fig.text(0.5, 1.02, 'Gráficos de doble masa de precipitación anual', 
         horizontalalignment='center', fontsize=16, weight='bold')
ax[1, 2].axis("off")
lim = 800
thr = 0.1 # umbral para definir anomalías

for idx, gage in enumerate(["A", "B", "C", "D", "E"]):
    print('Estación', gage)
    # Definir la posición del gráfico
    (ii, jj) = np.unravel_index(idx, (2, 3))
    # Configurar
    ax[ii, jj].set(xlim=(0, lim), ylim=(0, lim))
    ax[ii, jj].set_xlabel('estación ' + gage, fontsize=13)
    ax[ii, jj].set_ylabel('media de las estaciones', fontsize=13)
    
    # Recta de pendiente 1
    b = curve_fit(linear_reg, accData3.AVG, accData3[gage])[0][0]
    ax[ii, jj].plot([0, lim], [0, b * lim], 'k--', linewidth=1)
    
    # Gráfico de dispersión
    ax[ii, jj].plot(accData3.AVG, accData3[gage], 'o')
    
    # label one every five years
    years = accData3.index[::5]
    xyear = [accData3.loc[year, 'AVG'] + 20 for year in years]
    yyear = [accData3.loc[year, gage] for year in years] 
    for i, year in enumerate(years):
        ax[ii, jj].text(xyear[i], yyear[i], year,
                        verticalalignment='center')
                        
    # identificar estaciones y años con anomalía
    for j, year in enumerate(accData3.index[4:-4]):
        # pendiente de la regresión lineal hasta j
        p1 = np.polyfit(accData3.loc[:year, 'AVG'],
                        accData3.loc[:year, gage], 1)
        # pendiente de la regresión linean desde j+1
        p2 = np.polyfit(accData3.loc[year + 1:, 'AVG'],
                        accData3.loc[year + 1:, gage], 1)
        # hay anomalía si el cambio de la pendiente es notable
        if (p1[0] / p2[0] < 1 - thr) | (p1[0] / p2[0] > 1 + thr):
            print("Potencial anomalía: año {0}".format(year))
    print()
    
plt.tight_layout()


# Dos pluviómetros muestran inconsistencias: el pluviómetro B dos cambios de pendiente en 1930 y 1935,  y el pluviómetro E en el año 1930.
# 
# **Corregir pluviómetro B**

# In[99]:


# pendiente antes de 1930
# -----------------------
b1 = curve_fit(linear_reg, accData3.loc[:1930, 'AVG'],
               accData3.loc[:1930, 'B'])[0][0]
b1


# In[100]:


# pendiente de 1931 a 1935
# ------------------------
temp = data3.loc[1931:1935, :]
accTemp = temp.cumsum(axis=0)
b2 = curve_fit(linear_reg, accTemp.loc[:, 'AVG'],
               accTemp.loc[:, 'B'])[0][0]
del temp, accTemp
b2


# In[102]:


# pendiente a partir de 1936
# --------------------------
temp = data3.loc[1936:, :]
accTemp = temp.cumsum(axis=0)
b3 = curve_fit(linear_reg, accTemp.loc[:, 'AVG'],
               accTemp.loc[1:, 'B'])[0][0]
del temp, accTemp
b3


# Puesto que la pendiente en el segundo y tercer periodo es similar, asumimos que el periodo incorrecto es hasta 1930.

# In[104]:


# pendiente desde 1931
# --------------------
temp = data3.loc[1931:, :]
accTemp = temp.cumsum(axis=0)
b_ok = curve_fit(linear_reg, accTemp.loc[:, 'AVG'],
                 accTemp.loc[:, 'B'])[0][0]
del temp, accTemp
b_ok


# In[105]:


# corregir la serie hasta 1930
# ----------------------------
data3['B_c'] = data3.B.copy()
data3.loc[:1930, 'B_c'] = data3.loc[:1930, 'B'] * b_ok / b1
# accumulate corrected data
accData3 = data3.cumsum(axis=0)


# In[106]:


# Gráfico serie corregida vs original
# -----------------------------------
fig, ax = plt.subplots(figsize=(5,5))
# setup
ax.set(xlim=(0, lim), ylim=(0, lim))
ax.set_xlabel('media estaciones', fontsize=13)
ax.set_ylabel('estación B', fontsize=13)

b = curve_fit(linear_reg, accData3.AVG, accData3.B_c)[0][0]
ax.plot([0, lim], [0, b * lim], '--k', linewidth=1)

ax.scatter(accData3.AVG, accData3.B, label='original')
ax.scatter(accData3.AVG, accData3.B_c, marker='x', label='corregido')

ax.legend(loc=4, fontsize=13);


# **Corregir estación E**<br>
# Asumimos que la serie correcta es a partir de 1931 en adelante.

# In[107]:


# pendiente hasta 1930
# --------------------
b_wrong = curve_fit(linear_reg, accData3.loc[:1930, 'AVG'],
                    accData3.loc[:1930, 'E'])[0][0]
b_wrong


# In[109]:


# pendiente desde 1931
# --------------------
temp = data3.loc[1931:, :]     # extract raw data
accTemp = temp.cumsum(axis=0) # accumulate series
b_ok = curve_fit(linear_reg, accTemp.AVG, accTemp.E)[0][0]
del temp, accTemp
b_ok


# In[110]:


# corregir series hasta 1930
# --------------------------
data3['E_c'] = data3.E
data3.loc[:1930, 'E_c'] = data3.loc[:1930, 'E'] * b_ok / b_wrong
# accumulate corrected data
accData3 = data3.cumsum(axis=0)


# In[111]:


# Plot corrected vs original values
# ---------------------------------
fig, ax = plt.subplots(figsize=(5,5))
# setup
ax.set(xlim=(0, lim), ylim=(0, lim))
ax.set_xlabel('media estaciones', fontsize=13)
ax.set_ylabel('estación E', fontsize=13)

b = curve_fit(linear_reg, accData3.AVG, accData3.E_c)[0][0]
ax.plot([0, lim], [0, b * lim], '--k', linewidth=1)

ax.scatter(accData3.AVG, accData3.E, label='original')
ax.scatter(accData3.AVG, accData3.E_c, marker='x', label='corregido')

ax.legend(loc=4, fontsize=13);


# ## <font color=steelblue>Ejercicio 4 - Método hipsométrico
# 
# <font color=steelblue>Dada la curva hipsométrica de una cuenca (relación área-elevación) y la información de varias estaciones pluviométricas en dicha cuenca (tabla *Exercise_004* del archivo *RainfallData.xlsx*), calcula la precipitación media anual para la cuenca usando el método hipsométrico.<tfont>
# 
# | **Rango de altitud (m)** | **Fracción del área de la cuenca** |
# |-------------------------|-----------------------------------|
# | 311-400                 | 0.028                             |
# | 400-600                 | 0.159                             |
# | 600-800                 | 0.341                             |
# | 800-1000                | 0.271                             |
# | 1000-1200               | 0.151                             |
# | 1200-1400               | 0.042                             |
# | 1400-1600               | 0.008                             |

# __Curva hipsométrica__<br>
# La curva hipsométrica define el porcentaje de área de la cuenca que está por debajo de una altitud dada.
# 
# En este ejercicio utilizaremos la curva hipsométrica para asignar la proporción de la cuenca (en tanto por uno) correspondiente a cada franja de altitud.

# In[112]:


# Rangos de altitud
Zs = np.array([311, 400, 600, 800, 1000, 1200, 1400, 1600])
Zs = np.mean([Zs[:-1], Zs[1:]], axis=0)
# Área asociada
As = np.array([0.028, 0.159, 0.341, 0.271, 0.151, 0.042, 0.008])


# In[113]:


# crear data frame
hipso = pd.DataFrame(data=[Zs, As]).transpose()
hipso.columns = ['Z', 'A']
hipso['Aac'] = hipso.A.cumsum()
hipso


# In[118]:


# Gráfico de la curva hipsométrica
plt.plot(hipso.Z, hipso.Aac * 100)
plt.title('Curva hipsométrica', fontsize=16, weight='bold')
plt.xlabel('altitud (msnm)', fontsize=13)
plt.xlim(Zs[0], Zs[-1])
plt.ylabel('área (%)', fontsize=13)
plt.ylim((0, 100));


# __Regresión precipitación-altitud__
# 
# Utilizaremos los datos de precipitación anual en las estaciones de la cuenca para establecer la regresión lineal de la precipitación con la altitud.

# In[120]:


# Importar datos de precipitación
data4 = pd.read_excel('RainfallData.xlsx', sheet_name='Exercise_004',
                      index_col='Gage')
# Simplificar nombres de las variables
data4.columns = ['Z', 'P']
data4


# Se calcula la regresión lineal de la precipitación media anual con respecto a la altura.
# 
# $$P=a·Z+b$$
# 
# Donde $P$ es la precipitación media anual (mm) de un punto a cota $Z$ (msnm).

# In[121]:


# ajustar la recta de regresión
(a, b) = np.polyfit(data4.Z, data4.P, deg=1)
print('P = {0:.3f} Z + {1:.3f}'.format(a,b))


# In[125]:


# Gráfico altitud vs precipitación anual
plt.scatter(data4.Z, data4.P)
# recta de regresión
xlim = np.array([0, Zs[-1]])
plt.plot(xlim, a * xlim + b, 'k--')
# configuración
plt.title('', fontsize=16, weight='bold')
plt.xlabel('altitud (msnm)', fontsize=13)
plt.xlim(xlim)
plt.ylabel('Panual (mm)', fontsize=13)
plt.ylim(0, 2200);


# __Precipitación areal__
# 
# Conocida la regresión, se calcula la precipitación media anual en los puntos intermedios de cada una de los rangos de altitud que define la curva hipsométrica.

# In[126]:


hipso['P'] = a * hipso.Z + b
hipso


# La precipitación areal es el sumatorio del producto del área y precipitación en cada uno de los rangos de altitud.

# In[127]:


Pareal = np.sum(hipso.A * hipso.P)

print('La precipitación media anual sobre la cuenca es {0:.1f} mm'.format(Pareal))


# Hacer lo mismo de forma simplificada:

# In[128]:


p = np.polyfit(Data3.Z,  Data3.P, deg=1) # ajustar la regresión
Ps = np.polyval(p, Zs)                   # interpolar precipitación
Pareal = np.sum(Ps * As)                 # precipitación areal

print('La precipitación media anual sobre la cuenca es {0:.1f} mm'.format(Pareal))


# Si se hubiera calculado la precipitación areal por el **método de la media de las estacions**, habríamos subestimado la precipitación areal de la cuenca.

# In[129]:


Pareal2 = Data3.P.mean()

print('La precipitación media anual sobre la cuenca {0:.1f} mm'.format(Pareal2))


# ## <font color=steelblue>Exercise 5 - Curva intensidad-duración-frecuencia
# 
# <font color=steelblue>Construye la curva IDF (intensidad-duración-frecuencia) a partir de la información en la tabla *ChiAnnMax* del archivo *RainfallData.xlsx*.<tfont>

# Las **curvas de intensidad-duración-frecuencia (IDF)** son una aproximación habitual en los proyectos de hidrología para definir las tormentas de diseño. Las curvas IDF relacionan la intensidad de la precipitación, con su duración y su frecuencia de ocurrencia (expresada como periodo de retorno).
#  
# <img src="IDF curves.JPG" alt="Mountain View" style="width:500px">
# > <font color=grey>Curva de intensidad-duración-frecuenca para la ciudad de Oklahoma. *(Applied Hydrology. Chow, 1988)*
# 
# Cuando se va a diseñar una estructura hidráulica (puente, drenaje, presa...), es necesario conocer la intensidad máxima de precipitación que puede ocurrir para un periodo de retorno y una duración de la tormenta. El periodo de retorno suele estar definido por la normativa para cada tipo de estructura; el peor escenario de duración de la tormenta es el tiempo de concentración de la cuenca de drenaje de la estructura.
# 
# **Curvas IDF empíricas**<br>
# Para construir las curvas IDF a partir de datos locales, se lleva a cabo un análisis de frecuencia de extremos. Los valores de entrada son la serie anual de máxima intensidad de precipitación para diversas duraciones de tormenta. La serie correspondiente a cada duración se ajusta a una función de distribución de valores extremos para estimar el periodo de retorno. 
# 
# **Curvas IDF analíticas**
# Para generar las curvas IDF analíticas no es necesario el análisis de frecuencia de extremos anterior. En su lugar, se ajusta una ecuación representativa de la curva IDF a las observaciones.
# 

# ### Importación y  análisis de datos
# Para generar las curvas de intensidad-duración-frecuencia se necesitan los máximos anuales de precipitación acumulada a distintas duraciones. En nuestro caso estudiaremos eventos de duración 1, 6 y 24 horas.

# In[130]:


# Cargar los datos de intensidad
intensity = pd.read_excel('RainfallData.xlsx', sheet_name='ChiAnnMax', skiprows=7,
                          usecols=[0, 5, 6, 7], index_col='Year')
# Convertir datos de pulgadas a mm
intensity = intensity * 25.4
# Corregir columnas
D = np.array([1, 6 , 24]) # duración de la tormenta
intensity.columns = D
intensity.head()


# Vamos a generar un gráfico que muestre ordenadas de menor a mayor las series de máxima intensidad de precipitación para las tres duraciones que estamos analizando.
# 
# En este gráfico se observa que a menor duración, la intensidad es siempre mayor. Además, se intuye una mayor variabilidad (mayor pendiente) de la intensidad a menor duracion.

# In[131]:


# Configurar el gráfico
fig = plt.figure(figsize=(10, 6))
plt.title('Series ordenadas de máxima intensidad anual', fontsize=16, weight='bold')
plt.xlabel('', fontsize=13)
plt.xlim((0, 25))
plt.ylabel('intensidad (mm/h)', fontsize=13)
plt.ylim((0, 60))

# Tres gráficos de dispersión para cada duración de tormenta
plt.scatter(range(intensity.shape[0]), intensity.sort_values(1)[1], label='1 h')
plt.scatter(range(intensity.shape[0]), intensity.sort_values(6)[6], label='6 h')
plt.scatter(range(intensity.shape[0]), intensity.sort_values(24)[24], label='24 h')

# Leyenda
fig.legend(loc=8, ncol= 3, fontsize=13);


# ### Ajuste de la función GEV a los datos
# 
# Hemos de ajustar una distribución estadística de extremos a los datos. A partir de este ajuste podremos calcular los periodos de retorno. Utilizaremos la función de distribución **GEV (generalized extreme values)**. La función de distribución GEV sigue, para el caso de variables siempre positivas como la precipitación, la siguiente ecuación:
# 
# $$F(s,\xi)=e^{-(1+\xi s)^{-1/\xi}}  \quad \forall \xi>0$$
# $$ s = \frac{x-\mu}{\sigma} \quad \sigma>0$$
# 
# Donde $s$ es la variable de estudio estandarizada a través del parámetro de posición $\mu$ y el parámetro de escala $\sigma$, y $\xi$ es el parámetro de forma. Por tanto, la distribución GEV tiene 3 parámetros. En la siguiente figura se muestra la función de densidad y la función de distribución de extremos del tipo II, la distribución de Frechet, para diversos valores de los parámetros de escala y forma.
# 
# <img src="Frechet.png" alt="Mountain View" style="width:600px">
# 
# Para ajustar la función GEV utilizaremos la función `genextreme.fit` del paquete `SciPy.stats` de Python. Esta función devuelve los valores de los 3 parámetros de la GEV (forma, localización y escala) que mejor se ajustan a nuestros datos.

# In[132]:


# Ejemplo
# Ajustar la GEV para duración 1 h
par_int1h = genextreme.fit(intensity[1])


# In[133]:


print('Parámetros ajustados para la intensidad en 1 h:')
print('xi =', round(par_int1h[0], 4))
print('mu =', round(par_int1h[1], 4))
print('sigma =', round(par_int1h[2], 4))


# Lo haremos con un bucle para las tres duraciones (1, 6 y 24 h). Los parámetros se guardarán en el data frame *parametros*.

# In[139]:


# Ajustar los parámetros de las 3 duraciones
parametros = pd.DataFrame(index=['xi', 'mu', 'sigma'], columns=D)
for duracion in D:
    # Ajustar la GEV y guardar los parámetros
    parametros[duracion] = genextreme.fit(intensity[duracion])
parametros


# ### Curva IDF empírica
# 
# La **probabilidad de no excedencia** (el valor de la función de distribución) y el **periodo de retorno** de una variable estan relacionados mediante la siguiente ecuación:
# 
# \\[R = \frac{1}{1-CDF(x)}\\]
# 
# Donde $R$ es el periodo de retorno en años, y $CDF(x)$ (del inglés, cumulative density function) es el valor de la función de distribución (o probabilidad de no excendencia)  del valor de precipitación $x$.
# 
# A partir de esta expresión se pueden calcular los **cuantiles** de un **periodo de retorno** dado:
# 
# \\[CDF(x) = \frac{R-1}{R} = 1 - \frac{1}{R}\\]
# 
# Analizaremos los periodos de retorno de 10, 25, 50 y 100 años. Calculamos los cuantiles ($Q$) correspondientes a estos periodos de retorno de acuerdo a las distribuciones anteriormente ajustadas.

# In[140]:


# Periodos de retorno
R = np.array([10, 25, 50, 100], dtype="float64")


# In[141]:


# Probabilidad de no excedencia
Q = 1. - 1. / R


# Como ejemplo, generamos los valores extremos de la intensidad de una tormenta de 1 h de duración para las probabilidades de no excedencia (Q). Para ello utilizamos la función `genextrem.ppf` (*percent point function*) del paquete `SciPy.stats`.

# In[142]:


# intensidad de 1 h para los periodos de retorno
P1 = genextreme.ppf(Q, *par[1]) # ppf: percent point function

print('Intensidad de precipitación en 1 h según periodo de retorno:')
for i, Tr in enumerate(R):
     print('I(Tr=', int(Tr), ') = ', round(P1[i], 1), ' mm/h', sep='')


# Podemos iterar el cálculo de extremos para cada una de las duraciones y cuantiles, guardando los datos en un *data frame* al que llamaremos *IDF*, el cual podemos graficar.

# In[143]:


# data frame con los valores de la curva IDF
IDF = pd.DataFrame(index=R, columns=D)
for duracion in D:
    IDF[duracion] = genextreme(*parametros[duracion]).ppf(Q)
IDF


# Gráfico de líneas que muestra, para cada periodo de retorno, la intensidad de precipitación en función de la duración de la tormenta. 
# 
# Sólo tenemos los datos para tres duraciones de tormenta, motivo por el que la curva es tan quebrada. Para solventar este problema habría que repetir el cálculo para más duraciones de tormenta, o aplicar las **curvas IDF analíticas**.

# In[144]:


# configuración del gráfico
fig = plt.figure(figsize=(12, 6))
plt.title('Curva IDF', fontsize=16, weight='bold')
plt.xlabel('duración (h)', fontsize=13)
plt.xlim(0, IDF.columns.max() + 1)
plt.ylabel('intensidad (mm/h)', fontsize=13)
plt.ylim((0, 80))
color = ['tan', 'darkkhaki', 'olive', 'darkolivegreen']

for i, Tr in enumerate(IDF.index):
    plt.plot(IDF.loc[Tr,:], color=color[i], label='Tr = ' + str(int(Tr)) + ' años')

fig.legend(loc=8, ncol=4, fontsize=12);


# ### Curva IDF analítica
# Hasta ahora hemos calculado una serie de puntos de la **curva IDF**, los correspondientes a las tormentas de 1, 6 y 24 h para los periodos de retorno de 10, 25, 50 y 100 años. Aplicando las ecuaciones analíticas de la curva IDF, podemos generar la curva completa.
# 
# Dos de las formas analíticas de la curva IDF son:
# 
# \\[I = \frac{a}{(D + c)^b}\\]
# 
# \\[I = \frac{a}{D ^b + c}\\]
# 
# donde \\(I\\) es la intensidad de preciptiación, \\(D\\) es la duración de la tormenta, \\(a\\) es una constante dependiente del periodo de retorno y \\(b\\) y \\(c\\) son constantes que dependen de la localización del estudio.
# 
# Asumiremos que la relación entre $a$ y el periodo de retorno sigue la siguiente función lineal:
# 
# \\[a = d \cdot R + e\\]
# 
# Crearemos funciones de Python para estas curvas analíticas.

# In[145]:


def IDF_type_I(x, b, c, d, e):
    """Calcula la intensidad de la precipitación para un periodo de retorno y duración de la tormenta dadas a
    partir de la fórmula I = d * R + e / (D + c)**b.    
    
    Parámetros:
    -----------
    x:         list [2x1]. Par de valores de periodo de retorno(años) y duración (h)
    b:         float. Parámetro de la curva IDF
    c:         float. Parámetro de la curva IDF
    d:         float. Parámetro de la curva IDF
    e:         float. Parámetro de la curva IDF
    
    Salida:
    -------
    I:         float. Intensidad de precipitación (mm/h)"""
    
    a = d * x[0] + e
    I = a / (x[1] + c)**b
    return I

def IDF_type_II(x, b, c, d, e):
    """Calcula la intensidad de la precipitación para un periodo de retorno y duración de la tormenta dadas a
    partir de la fórmula I = d * R + e / (D**b + c).    
    
    Parámetros:
    -----------
    x:         list [2x1]. Par de valores de periodo de retorno(años) y duración (h)
    b:         float. Parámetro de la curva IDF
    c:         float. Parámetro de la curva IDF
    d:         float. Parámetro de la curva IDF
    e:         float. Parámetro de la curva IDF
    
    Salida:
    -------
    I:         float. Intensidad de precipitación (mm/h)"""
    
    a = d * x[0] + e
    I = a / (x[1]**b + c)
    return I

def IDF_type_III(x, b, c, d, e):
    """Calcula la intensidad de la precipitación para un periodo de retorno y duración de la tormenta dadas a
    partir de la fórmula I = d * R**e / (D + c)**b.    
    
    Parámetros:
    -----------
    x:         list [2x1]. Par de valores de periodo de retorno(años) y duración (h)
    b:         float. Parámetro de la curva IDF
    c:         float. Parámetro de la curva IDF
    d:         float. Parámetro de la curva IDF
    e:         float. Parámetro de la curva IDF
    
    Salida:
    -------
    I:         float. Intensidad de precipitación (mm/h)"""
    
    a = d * x[0]**e 
    I = a / (x[1] + c)**b
    return I

def IDF_type_IV(x, b, c, d, e):
    """Calcula la intensidad de la precipitación para un periodo de retorno y duración de la tormenta dadas a
    partir de la fórmula I = d * R**e / (D**b + c).    
    
    Parámetros:
    -----------
    x:         list [2x1]. Par de valores de periodo de retorno(años) y duración (h)
    b:         float. Parámetro de la curva IDF
    c:         float. Parámetro de la curva IDF
    d:         float. Parámetro de la curva IDF
    e:         float. Parámetro de la curva IDF
    
    Salida:
    -------
    I:         float. Intensidad de precipitación (mm/h)"""
    
    a = d * x[0]**e
    I = a / (x[1]**b + c)
    return I 


# Para ajustar la curva hemos de crear primero una malla de pares de valores de periodo de retorno y duración. Utilizaremos las tres duraciones ('D') y los cuatro periodos de retorno ('R') ya empleados hasta ahora, para los cuales hemos calculado la intensidad de precipitación asociada (data frame 'IDF').

# In[146]:


# malla con todas las posibles combinaciones de periodo de retorno 'R' y duración 'D'
(RR, DD) = np.meshgrid(R, D)
RR.shape, DD.shape


# In[147]:


# convertir 'RR' y 'DD' en un vector unidimensional
RR = RR.reshape(-1)
DD = DD.reshape(-1)
RR.shape, DD.shape


# In[148]:


# unir los vectores 'RR' y 'DD'
RD = np.vstack([RR, DD])

RD.shape


# In[149]:


# vector unidimensional a partir de 'IDF'
I = np.hstack([IDF[1], IDF[6], IDF[24]])

I.shape


# Para ajustar la curva utilizaremos la función `curve_fit` de `SciPy.optimize`. A esta función hemos de asignarle la función de la curva a ajustar, los valores de entrada (pares retorno-duración) y el valor de la función en esos pares (intensidad). La función devuelve un vector con los parámetros de la curva optimizados y otro vector con las covarianza entre dichos parámetros

# In[150]:


# ajustar la curva
curva = IDF_type_IV
popt, pcov = curve_fit(curva, RD, I)

print('Parámetros optimizados de la curva IDF analítica')
for i, par in enumerate(['b', 'c', 'd', 'e']):
    print(par, '=', round(popt[i], 4))


# In[151]:


fig = plt.figure(figsize=(12, 6))
plt.xlim(0, D.max()+1)
plt.xlabel('duración (h)', fontsize=13)
plt.ylabel('intensidad (mm/h)', fontsize=13)
color = ['tan', 'darkkhaki', 'olive', 'darkolivegreen']

xx = np.linspace(.25, D.max(), 1000) # valores de duración
y = np.zeros((xx.size,)) # vector vacío de valores de intensidad

for i, Tr in enumerate(R): # para cada periodo de retorno
    for j, d in enumerate(xx): # para cada duración
        y[j] = curva((Tr, d), *popt)
    # gráfico de línea
    plt.plot(xx, y, color=color[i], label='Tr = ' + str(int(Tr)) + ' años')
    # gráfico de dispersión
    plt.scatter(D, IDF.loc[Tr], s=8, marker='o', c=color[i], label=None)

fig.legend(loc=8, ncol=4, fontsize=12);
