#!/usr/bin/env python
# coding: utf-8

# # Ejercicios de agua subterránea
# 
# __Índice__<br>
# 
# __[Ejercicio 1 - Infiltración. Método de Green-Ampt](#Ejercicio-1---Infiltración.-Método-de-Green-Ampt)__<br>
# 
# __[Ejercicio 2 - Textura de los suelos](#Ejercicio-2---Textura-de-los-suelos)__<br>
# 
# __[Ejercicio 3 - Propiedades de los suelos](#Ejercicio-3---Propiedades-de-los-suelos)__<br>
# 
# __[Ejercicio 4 - Tensiómetros](#Ejercicio-4---Tensiómetros)__<br>

# In[1]:


import numpy as np

import pandas as pd

from matplotlib import pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
plt.style.use('seaborn-whitegrid')


# ## <font color=steelblue>Ejercicio 1 - Infiltración. Método de Green-Ampt
# 
# <font color=steelblue>Usando el modelo de Green-Ampt, calcula la __infiltración acumulada__, la __tasa de infiltración__ y la __profundidad del frente de mojado__ durante una precipitación constante de 5 cm/h que dure 2 h en un _loam_ limoso típico con un contenido de agua inicial de 0,45.
#     
# Las propiedades típicas del _loam_ limoso son: <br>
# $\phi=0.485$ <br>
# $K_{s}=2.59 cm/h$ <br>
# $|\Psi_{ae}|=78.6 cm$ <br>
# $b=5.3$ <br>

# In[2]:


# datos del enunciado
phi = 0.485     # -
theta_o = 0.45  # -
Ks = 2.59       # cm/h
psi_ae = 78.6   # cm
b = 5.3         # -

ho = 0          # cm
i = 5           # cm/h
tc = 2          # h

epsilon = 0.001 # cm


# ### Modelo de infiltración de Green-Ampt
# 
# Hipótesis:
# *  Suelo encharcado con una lámina de altura $h_o$ desde el inicio.
# *  Frente de avance de la humedad plano (frente pistón).
# *  Suelo profundo y homogéneo ($\theta_o$, $\theta_s$, $K_s$ constantes).
# 
# Tasa de infiltración, $f \left[ \frac{L}{T} \right]$:
# 
# $$f = K_s \left( 1 + \frac{\Psi_f · \Delta\theta}{F} \right) \qquad \textrm{(1)}$$ 
# 
# Infiltración acumulada, $f \left[ L \right]$:
# $$F = K_s · t + \Psi_f · \Delta\theta · \ln \left(1 + \frac{F}{\Psi_f · \Delta\theta} \right) \qquad \textrm{(2)}$$
# 
# Es una ecuación implícita. Para resolverla, se puede utilizar, por ejemplo, el método de Picard. Se establece un valor inicial de ($F_o=K_s·t$) y se itera el siguiente cálculo hasta converger ($F_{m+1}-F_m<\varepsilon$):
# $$F_{m+1} = K_s · t + \Psi_f · \Delta\theta · \ln \left(1 + \frac{F_m}{\Psi_f · \Delta\theta} \right)  \qquad \textrm{(3)}$$
# 
# 
# ##### Suelo no encharcado al inicio
# Si no se cumple la hipótesis de encharcamiento desde el inicio, se debe calcular el tiempo de encharcamiento ($t_p$) y la cantidad de agua infiltrada hata ese momento ($F_p$):
# $$t_p = \frac{K_s · \Psi_f · \Delta\theta}{i \left( i - K_s \right)} \qquad \textrm{(4)}$$
# $$F_p = i · t_p = \frac{K_s · \Psi_f · \Delta\theta}{i - K_s} \qquad \textrm{(5)}$$
# 
# Conocidos $t_p$ y $F_p$, se ha de resolver la ecuación (1) sobre una nueva variable tiempo $t_p'=t_p-t_o$, con lo que se llega a la siguiente ecuación emplícita:
# $$F_{m+1} = K_s · (t - t_o) + \Psi_f · \Delta\theta · \ln \left(1 + \frac{F_m}{\Psi_f · \Delta\theta} \right)  \qquad \textrm{(6)}$$
# donde $t_o$ es:<br>
# $$t_o = t_p - \frac{F_p - \Psi_f · \Delta\theta · \ln \left(1 + \frac{F_p}{\Psi_f · \Delta\theta} \right)}{K_s} \qquad \textrm{(7)}$$

# In[3]:


# calcular variables auxiliares
Atheta = phi - theta_o                     # incremento de la humedad del suelo
psi_f = (2 * b + 3) / (2 * b + 6) * psi_ae # tensión en el frente húmedo


# In[4]:


# tiempo hasta el encharcamiento
tp = psi_f * Atheta * Ks / (i * (i - Ks))


# In[5]:


# infiltración acumulada cuando ocurre el encharcamiento
Fp = tp * i


# In[6]:


# tiempo de inicio de la curva de infiltración
to = tp - (Fp - psi_f * Atheta * np.log(1 + Fp / (psi_f * Atheta))) / Ks


# In[7]:


# infiltración acumulada en el tiempo de cálculo
Fo = Ks * (tc - to)
Fi = Ks * (tc - to) + psi_f * Atheta * np.log(1 + Fo / (psi_f * Atheta))
while (Fi - Fo) > epsilon:
    Fo = Fi
    Fi = Ks * (tc - to) + psi_f * Atheta * np.log(1 + Fo / (psi_f * Atheta))
    print(Fo, Fi)
Fc = Fi

print()
print('Fc = {0:.3f} cm'.format(Fc))


# In[8]:


# tasa de infiltración en el tiempo de cálculo
fc = Ks * (1 + psi_f * Atheta / Fc)

print('fc = {0:.3f} cm/h'.format(fc))


# In[9]:


# profundidad del frente de húmedo
L = Fc / Atheta

print('L = {0:.3f} cm'.format(L))


# In[10]:


def GreenAmpt(i, tc, ho, phi, theta_o, Ks, psi_ae, b=5.3, epsilon=0.001):
    """Se calcula la infiltración en un suelo para una precipitación constante mediante el método de Green-Ampt.
    
    Entradas:
    ---------
    i:       float. Intensidad de precipitación (cm/h)
    tc:      float. Tiempo de cálculo (h)
    ho:      float. Altura de la lámina de agua del encharcamiento en el inicio (cm)
    phi:     float. Porosidad (-)
    theta_o: float. Humedad del suelo en el inicio (-)
    Ks:      float. Conductividad saturada (cm/h)
    psi_ae:  float. Tensión del suelo para el punto de entrada de aire (cm)
    b:       float. Coeficiente para el cálculo de la tensión en el frente húmedo (cm)
    epsilo:  float. Error tolerable en el cálculo (cm)
    
    Salidas:
    --------
    Fc:      float. Infiltración acumulada en el tiempo de cálculo (cm)
    fc:      float. Tasa de infiltración en el tiempo de cálculo (cm/h)
    L:       float. Profundidad del frente húmedo en el tiempo de cálculo (cm)"""
    
    # calcular variables auxiliares
    Atheta = phi - theta_o                     # incremento de la humedad del suelo
    psi_f = (2 * b + 3) / (2 * b + 6) * psi_ae # tensión en el frente húmedo
    
    if ho > 0: # encharcamiento inicial
        tp = 0
        to = 0
    elif ho == 0: # NO hay encharcamiento inicial
        # tiempo hasta el encharcamiento
        tp = psi_f * Atheta * Ks / (i * (i - Ks))
        # infiltración acumulada cuando ocurre el encharcamiento
        Fp = tp * i
        # tiempo de inicio de la curva de infiltración
        to = tp - (Fp - psi_f * Atheta * np.log(1 + Fp / (psi_f * Atheta))) / Ks
    
    # infiltración acumulada en el tiempo de cálculo
    if tc <= tp:
        Fc = i * tc
    elif tc > tp:
        Fo = Ks * (tc - to)
        Fi = Ks * (tc - to) + psi_f * Atheta * np.log(1 + Fo / (psi_f * Atheta))
        while (Fi - Fo) > epsilon:
            Fo = Fi
            Fi = Ks * (tc - to) + psi_f * Atheta * np.log(1 + Fo / (psi_f * Atheta))
        Fc = Fi
    
    # tasa de infiltración en el tiempo de cálculo
    fc = Ks * (1 + psi_f * Atheta / Fc)
    
    # profundidad del frente de húmedo
    L = Fc / Atheta
    
    return Fc, fc, L


# In[11]:


Fc, fc, L = GreenAmpt(i, tc, ho, phi, theta_o, Ks, psi_ae, b, epsilon)

print('Fc = {0:.3f} cm'.format(Fc))
print('fc = {0:.3f} cm/h'.format(fc))
print('L = {0:.3f} cm'.format(L))


# ## <font color=steelblue>Ejercicio 2 - Textura de los suelos
# 
# <font color=steelblue>Determinar la textura de los siguientes suelos.</font>
#     
# <img src="SoilTextureTriangle.jpg" alt="Mountain View" style="width:500px">
#     
# | Ø (mm)| 50  | 19  | 9.5 | 4.76 | 2   | 0.42 | 0.074 | 0.02 | 0.005 | 0.002 |
# |-------|-----|-----|-----|------|-----|------|-------|------|-------|-------|
# | A1    | 100 | 100 | 100 | 100  | 100 | 100  | 97    | 79   | 45    | 16    |
# | B1    | 100 | 100 | 98  | 94   | 70  | 19   | 15    | 8    | 3     | 2     |
# | C1    | 93  | 91  | 88  | 85   | 69  | 44   | 40    | 27   | 13    | 6     |
# | D1    | 100 | 100 | 100 | 100  | 100 | 97   | 92    | 75   | 47    | 31    |

# In[12]:


suelos = ['A1', 'B1', 'C1', 'D1']
phi = [50, 19, 9.5, 4.76, 2.0, 0.420, 0.074, 0.02, 0.005, 0.002]
A1 = [100, 100, 100, 100, 100, 100, 97, 79, 45, 16]
B1 = [100, 100, 98, 94, 70, 19, 15, 8, 3, 2]
C1 = [93, 91, 88, 85, 69, 44, 40, 27, 13, 6]
D1 = [100, 100, 100, 100, 100, 97, 92, 75, 47, 31]

datos = pd.DataFrame(data=[A1, B1, C1, D1], index=suelos, columns=phi)
datos


# In[13]:


# extraer datos de los límites de cada una de las texturas
limites = [2, 0.074, 0.02, 0.002]
datos_f = datos.loc[:, limites]
datos_f


# In[14]:


# interpolar linealmente el % que pasa un diámetro de 0.05 mm, límite entre arena y limo
datos_f[0.05] = (0.05 - 0.074) * (datos_f[0.02] - datos_f[0.074]) / (0.02 - 0.074) + datos_f[0.074]
datos_f.sort_index(axis=1, ascending=False, inplace=True)
datos_f.drop([0.074, 0.02], axis=1, inplace=True)
datos_f


# In[15]:


# tabla donde guardar el % de cada textura
textura = pd.DataFrame(index=datos.index, columns=['grava', 'arena', 'limo', 'arcilla'])
textura


# In[16]:


# completar la tabla de texturas
textura.grava = 100 - datos_f[2]
textura.arena = round(datos_f[2] - datos_f[0.05], 1)
textura.limo = round(datos_f[0.05] - datos_f[0.002], 1)
textura.arcilla = datos_f[0.002]
textura


# In[17]:


# extraer los porcentajes sólo para los finos
finos = ['arena', 'limo', 'arcilla']
textura_finos = textura[finos].copy()
textura_finos


# In[18]:


# corregir de manera que los finos sumen 100
for soil in textura_finos.index:
    perc_finos = textura_finos.loc[soil, :].sum()
    textura_finos.loc[soil,:] = round(textura_finos.loc[soil,:] * 100 / perc_finos, 1)

textura_finos


# ## <font color=steelblue>Ejercicio 3 - Propiedades de los suelos
# <font color=steelblue>Seproporcionan los pesos en campo y tras secado para cuatro muestras cilíndricas de suelo de 10 cm de longitud y 5 cm de diámetro. Asumiento que $\rho_{m}=2650 kg/m³$, calcular el __contenido en agua__, la __saturación__, la __densidad del suelo__ y la __porosidad__ de cada muestra.
#     
# |    | m (g) | ms (g) |
# |----|-------|--------|
# | A2 | 302.5 | 264.8  |
# | B2 | 376.3 | 308.0  |
# | C2 | 422.6 | 388.6  |
# | D2 | 468.3 | 441.7  |

# In[ ]:


rho_m = 2.65 # g/cm³
rho_w = 1    # g/cm³
D = 5        # cm
L = 10       # cm


# #### Muestra A2

# In[19]:


m = 302.5    # g
ms = 264.8   # g


# In[20]:


Vt = np.pi / 4 * D**2 * L # volumen total


# In[21]:


Vw = (m - ms) / rho_w   # volumen de agua


# In[22]:


rho_b = ms / Vt         # densidad del suelo


# In[25]:


phi = 1 - rho_b / rho_m # porosidad


# In[26]:


theta = Vw / Vt         # contenido de humedad del suelo


# In[27]:


Se = theta / phi        # grado de saturación


# #### Todas las muestras

# In[28]:


def prop_suelo(m, ms, rho_m, D, L):
    """Se calculan las propiedades del suelo
    
    Entradas:
    ---------
    m:       float. Masa de la muestra de suelo (g)
    ms:      float. Masa seca de la muestra (g)
    rho_m:   float. Densidad del mineral (g/cm³)
    D:       float. Diámetro del cilindro de la muestra (cm)
    L:       float. Longitud del cilindro de la muestra (cm)
    
    Salidas:
    --------
    rho_b:   float. Densidad bruta (g/cm³)
    phi:     float. Porosidad (-)
    theta:   float. Humedad del suelo (-)
    Se:      float. Grado de saturación (-)"""
    
    rho_w = 1    # g/cm³
    
    # Calcular volúmenes necesarios
    Vt = np.pi / 4 * D**2 * L # volumen total
    Vw = (m - ms) / rho_w   # volumen de agua
    
    # Calcular propiedades del suelo
    rho_b = ms / Vt         # densidad del suelo
    phi = 1 - rho_b / rho_m # porosidad
    theta = Vw / Vt         # contenido de humedad del suelo
    Se = theta / phi        # grado de saturación
    
    return rho_b, phi, theta, Se


# In[29]:


m = [302.5, 376.3, 422.6, 468.3]
ms = [264.8, 308.0, 388.6, 441.7]
soilnames = ['A2', 'B2', 'C2', 'D2']

# dataframe con los datos
soils = pd.DataFrame(data=[m, ms], index=['m', 'ms'], columns=soilnames).T
soils


# In[30]:


# dataframe con las propiedades
props = pd.DataFrame(index=soils.index, columns=['rho_b', 'phi', 'theta', 'Se'])
props


# In[31]:


# calcular propiedades
for soil in soils.index:
    m, ms = soils.loc[soil, 'm'], soils.loc[soil, 'ms']
    props.loc[soil,:] = prop_suelo(m, ms, rho_m, D, L)
props


# ## <font color=steelblue>Ejercicio 4 - Tensiómetros
# 
# <font color=steelblue>Considérense dos tensiómetros adyacentes insertados en un _loam_ arenoso no saturado. Ambos tensiómetros tienen una longitud de 20 cm. El tensiómetro A se sitúa a una profundidad de 20 cm y el B a 60 cm. De acuerdo a las lecturas de la siguiente tabla, ¿en qué dirección fluye el agua en cada caso?

# |Medición            | Caso 1 | Caso 2 | Caso 3 | Caso 4 | Caso 5 |
# |------------------- |--------|--------|--------|--------|--------|
# | $\Psi_{m,A} (cm)$  | 123    | 106    |  39    | 211    |  20    |    
# | $\Psi_{m,B} (cm)$  |  22    |  51    |  65    | 185    |  36    |

# In[ ]:


Lten = 20 # cm
profA = 20   # cm
profB = 60   # cm


# #### Caso 1

# In[44]:


psi_mA = -123 # cm
psi_mB = -22  # cm


# In[55]:


# array de los datos
psi_m = np.array([psi_mA, psi_mB])
prof = np.array([profA, profB])


# In[56]:


# tensión en el bulbo del tensiómetro
psi = psi_m + Lten
psi


# In[57]:


# altura potencial relativa a la superficie
z = - prof
z


# In[58]:


# altura de energía
H = z + psi
print('Ha = {0} cm\tHb = {1} cm'.format(H[0], H[1]))


# In[51]:


# definir la dirección del flujo
if H[0] > H[1]:
    print('flujo en dirección a B')
elif H[0] == H[1]:
    print('no hay flujo')
elif H[0] < H[1]:
    print('flujo en dirección a A')


# #### Todos los casos

# In[59]:


def direc_flujo(psi_m, prof, Lten):
    """Se calcula la altura hidráulica en dos puntos del suelo y en función de ella la dirección del flujo subterráneo.
    
    Entradas:
    ---------
    psi_m:   list (2,). Valores medidos de la tensión (negativos) en los dos puntos del suelo (cm).
    prof:    list (2,). Profundidad (positivos) a la que se ubican los dos tensiómetros (cm)
    Lten:    float. Longitud del tensiómetro (cm)
    
    Salida:
    -------
    H:       list (2,). Altura de energía en los dos puntos (cm)"""
    
    # tensión en el bulbo de los tensiómetros
    psi = psi_m + Lten
    
    # altura potencial relativa a la superficie
    z = - prof
    
    # altura de energía
    H = z + psi
    print('Ha = {0} cm\tHb = {1} cm'.format(H[0], H[1]))
    
    # definir dirección del flujo
    if H[0] > H[1]:
        print('flujo en dirección a B')
    elif H[0] == H[1]:
        print('no hay flujo')
    elif H[0] < H[1]:
        print('flujo en dirección a A')
        
    return H


# In[60]:


# comprobación para el caso 1
H = direc_flujo(psi_m, prof, Lten)


# In[61]:


# tabla con los datos de las mediciones
psi_mA = [123, 106, 39, 211, 20]
psi_mB = [22, 51, 65, 185, 36]

# dataframe con los datos
Psi_m = pd.DataFrame(data=[psi_mA, psi_mB], index=['A', 'B'], columns=range(1,6))
Psi_m


# In[62]:


# tabla con los resultados
H = pd.DataFrame(index=Psi_m.index, columns=Psi_m.columns)
H


# In[64]:


# calcular potenciales y dirección de flujo
for i in Psi_m.columns:
    print('Caso', i)
    psi_m = - Psi_m.loc[:, i].values
    H.loc[:, i] = direc_flujo(psi_m, prof, Lten)
    print()


# In[65]:


H


# In[ ]:




