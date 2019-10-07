#!/usr/bin/env python
# coding: utf-8

# # clusterizacion
# 

# In[1]:


import sys
sys.path.append("../scripts/")
from funciones_paralectura import carga_de_datos
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


# In[2]:


# carga de caminata aleatoria
documento = '../../Battery_Uniform_Distribution_Discharge_Room_Temp_DataSet_2Post/data/Matlab/RW3.mat'
comentarios = ['discharge (random walk)',
               'rest (random walk)',
               'rest post random walk discharge']
for s in comentarios: print(s)
comentario_final = ['charge (after random walk discharge)']
for s in comentario_final: print(s)
t, v, c, temp, T = carga_de_datos(documento, comentarios, comentario_final, nofinal=False)


# In[3]:


# print(t.keys())
plt.plot(t[400], c[400])


# In[4]:


def guardar_serie_entre(serie, ti, deltaT, tamanno=0, corriente=False):
    """
    Funcion para guardar generar las series de tiempo a guardar en el dataframe
    En especifico: si no se da un tamanno, se asume que es una curva de voltaje y se guarda hasta
    el punto de mayor voltaje anterior
    
    Input:
        serie: serie de tiempo (arreglo de numpy)
        ti: indice inicial de referencia para guardar los datos
        deltaT: cantidad de puntos para guardar mas alla de ti
        tamanno: (opcional) si no se entrega, se asume que es voltaje
            si se entrega, se asume que se debe guardar desde ti + deltaT, la
            cantidad de datos dada por tamanno, hacia atras.
        corriente: si es True, se asume que se quiere guardar por corriente, del nivel anterior al siguiente 0
    
    Return:
        entrega un arreglo de numpy con los datos pedidos
    """
    arr = []
    
    if corriente:
        consulta = False
        inicio = ti
        fin = ti + 2
        for i in range(10):
            if ti - i == 0:
                inicio = 0
                break
            elif serie[ti - i] == 0:
                consulta = True
            elif consulta and serie[ti - i] > 0:
                inicio = ti - i
                break
        i = 0
        while serie[ti + i] != 0:
            i += 1
        fin = ti + i
        return np.array(serie[inicio:fin])
        
    
    fin = ti + deltaT
    if fin > len(serie): fin = len(serie) - 1
    ej = serie[ti]
    i = 0
    epsilon = 0.1

    
    if tamanno != 0: i = fin - tamanno
    else:
        for idx in range(40):
            if serie[ti - idx] > ej:
                ej = serie[ti - idx]
                i = idx
            elif serie[ti - idx] < ej - epsilon: break
        i = ti - i
    
    if i < 0: return np.array(serie[:fin])
    
    return np.array(serie[i : fin])


# In[5]:


# creacion de vectores de diferencia de consumo de corriente

# creacion de diccionarios para guardar datos
centro_c = {}
t_salto = {}
pot = {}
pot_salto = {}
dif_salto_c = {}
centro_salto_c = {}
tiempo = {}
ciclo_salto = {}

# saltos = {}
# j = 0

arr_dif_salto_c = []
arr_pot_salto  = []
arr_centro_salto_c  = []
arr_ciclo_salto = []

arr_voltaje_salto = []
arr_corriente_salto = []
arr_tiempo_salto = []
arr_temperatura_salto = []

dx = 90

for i in c.keys():
#     if i == 0: continue
    if i == len(c.keys()) - 1: break
#     print(i)
    # creacion de vectores para guardar datos
    dif_salto_c[i] = []
    centro_c[i] = []
    centro_salto_c[i] = []
    t_salto[i] = []
    pot[i] = []
    pot_salto[i] = []
    tiempo[i] = []
    ciclo_salto[i] = []
    
    # condiciones iniciales
    anterior = 0
    POT = 0
    t_anterior = t[i][0]
    aux_t = 0
    ti = 0
    
    for idx, elem in enumerate(c[i]):
        # lectura de serie de tiempo
        
        dt = t[i][idx] - t_anterior # dt como variable auxiliar
        t_anterior = t[i][idx]
        POT += c[i][idx] * v[i][idx] * dt # se calcula la potencia integrando en el tiempo
        aux_t += dt # su acumulacion de tiempo
        
        centro_c[i].append((elem + anterior) / 2) # centro del salto
        pot[i].append(POT)
        tiempo[i].append(aux_t)
        
        if (elem - anterior >= 0.1):# or (elem - anterior <= -0.1):
            
            # solo entra si ocurre un salto
            
            dif_salto_c[i].append(elem - anterior) # magnitud del salto
            pot_salto[i].append(POT) # energia entregada al momento del salto
            centro_salto_c[i].append((elem + anterior) / 2) # media del salto
            t_salto[i].append(aux_t) # tiempo en que ocurre el salto
            ciclo_salto[i].append(i)
            
#             saltos[j] = {'ciclo': i,
#                         'energia': POT,
#                         'media': (elem + anterior) / 2,
#                         'magnitud': elem - anterior}
            
            arr_dif_salto_c.append(elem - anterior)
            arr_pot_salto.append(POT)
            arr_centro_salto_c.append((elem + anterior) / 2)
            arr_ciclo_salto.append(i)
            
            # como guardar la serie de tiempo de voltaje, corriente y temperatura 
            # del salto en un intervalo de tiempo
            dx = 90
            di = 10
            
            aux = guardar_serie_entre(v[i], idx, 2*dx)
            arr_voltaje_salto.append(aux)
            arr_corriente_salto.append(guardar_serie_entre(c[i], idx, 2*dx, tamanno=len(aux)))
            arr_tiempo_salto.append(guardar_serie_entre(t[i], idx, 2*dx, tamanno=len(aux)))
            arr_temperatura_salto.append(guardar_serie_entre(temp[i], idx, 2*dx, tamanno=len(aux)))
            
            ti = idx

        anterior = elem


# In[6]:



plt.plot(tiempo[2], c[2], color='r')
plt.plot(t_salto[2], dif_salto_c[2], '+g')
plt.plot(t_salto[2], centro_salto_c[2], '*b'), plt.show()

plt.plot(pot_salto[2], dif_salto_c[2], '+g')
plt.plot(pot_salto[2], centro_salto_c[2], '*b'), plt.show()

plt.plot(tiempo[2], pot[2]), plt.ylabel('energía [J]'), plt.xlabel('tiempo'), plt.show()
plt.plot(tiempo[2], v[2]), plt.ylabel('voltaje [V]'), plt.xlabel('tiempo'), plt.show()
plt.plot(tiempo[2], c[2]), plt.ylabel('corriente [A]'), plt.xlabel('tiempo'), plt.show()

plt.plot(pot[2], pot[2]), plt.show()
plt.plot(pot[2], v[2]), plt.ylabel('voltaje [V]'), plt.xlabel('energía [J]'), plt.show()
plt.plot(pot[2], c[2]), plt.ylabel('corriente [A]'), plt.xlabel('energía [J]'), plt.show()


# In[7]:


plt.plot(t[1], c[1]), plt.show()
plt.plot(t[1], c[1], '+')#, plt.show()
# plt.plot(t[2], c[2], '-r')
# plt.plot(t[1], v[1]), plt.show()
# plt.plot(t[2], c[2]), plt.show()
# plt.plot(t[2], v[2]), plt.show()
for i in range(9):
#     print(len(arr_voltaje_salto[i]))
    print('V: ', arr_voltaje_salto[i][0], arr_voltaje_salto[i][-1])
    print('C: ', arr_corriente_salto[i][0], arr_corriente_salto[i][-1])
#     plt.plot(arr_tiempo_salto[i], arr_voltaje_salto[i]), plt.show()
    plt.plot(arr_tiempo_salto[i], arr_corriente_salto[i])#, plt.show()
plt.show()

print(arr_tiempo_salto[13][0], arr_tiempo_salto[13][-1])
print(arr_tiempo_salto[14][0], arr_tiempo_salto[14][-1])

print(arr_corriente_salto[13][0], arr_corriente_salto[13][-1])
print(arr_corriente_salto[14][0], arr_corriente_salto[14][-1])

print(t[2][0], t[2][-1])
print(c[2][0], c[2][-1])


# In[8]:


import pandas as pd
saltos = {'ciclo': arr_ciclo_salto,
          'energia': arr_pot_salto,
          'magnitud': arr_dif_salto_c,
          'media': arr_centro_salto_c}
df = pd.DataFrame(saltos, columns=['ciclo', 'energia', 'magnitud', 'media'])
df['voltaje'] = [x for x in arr_voltaje_salto]
df['corriente'] = [x for x in arr_corriente_salto]
df['temperatura'] = [x for x in arr_temperatura_salto]
df['tiempo'] = [x for x in arr_tiempo_salto]
df.head()


# In[9]:


df['temp_ini'] = [x[0] for x in df['temperatura']]
df.head()


# In[10]:


idx = 600
plt.plot(t[idx], c[idx]), plt.show()
plt.plot(t[idx], c[idx], '+')

tiemposs = df.loc[(df['ciclo'] == idx), 'tiempo'].values
corrientos = df.loc[(df['ciclo'] == idx), 'corriente'].values

for i, tempp in enumerate(tiemposs):
    plt.plot(tempp, corrientos[i])#, plt.show()#, plt.show()
plt.show()

plt.plot(t[idx], v[idx]), plt.show()
plt.plot(t[idx], v[idx], '+')

tiemposs = df.loc[(df['ciclo'] == idx), 'tiempo'].values
voltajos = df.loc[(df['ciclo'] == idx), 'voltaje'].values

for i, tempp in enumerate(tiemposs):
    plt.plot(tempp, voltajos[i])#, plt.show()#, plt.show()
plt.show()


# In[11]:


# analisis estadistico de los ciclos: ploteo de histogramas
plt.hist(df['ciclo'], bins=70), plt.title('Histograma de ciclos'), plt.show()
plt.hist(df['magnitud'], bins=30), plt.title('Histograma de magnitudes'), plt.show()
plt.hist(df['media'], bins=30), plt.title('Histograma de medias'), plt.show()
plt.hist(df['energia'], bins=60), plt.title('Histograma de energia'), plt.show()


# In[12]:


# definiendo features
def primera_derivada(arr_serie):
    aux = []
    for serie in arr_serie:
        aux2 = list(serie[:-1] - serie[1:])
        if len(aux2) == 0: aux2 = [0]
        aux.append(aux2)
    return aux


# In[13]:


# obtener features
df["F1"] = [np.mean(x[:1]) for x in primera_derivada(df["voltaje"])]
df


# In[14]:


WIN = 1
alpha = 0.3
df_aux = df[(df['magnitud'] >= 0.3) & (df['magnitud'] <= 0.7)]
df_aux = df_aux[df_aux['F1'] > 0.01]
df_aux['F1'] = df_aux['F1'].rolling(window=WIN).mean()
plt.plot(df_aux['ciclo'], df_aux['F1'], label='mag: 0.5 - #' + str(len(df_aux)))

df_aux = df[(df['magnitud'] >= 0.8) & (df['magnitud'] <= 1.2)]
df_aux = df_aux[df_aux['F1'] > 0.01]
df_aux['F1'] = df_aux['F1'].rolling(window=WIN).mean()
plt.plot(df_aux['ciclo'], df_aux['F1'], label='mag: 1.0 - #' + str(len(df_aux)))

df_aux = df[(df['magnitud'] >= 1.4) & (df['magnitud'] <= 1.6)]
df_aux = df_aux[df_aux['F1'] > 0.01]
df_aux['F1'] = df_aux['F1'].rolling(window=WIN).mean()
plt.plot(df_aux['ciclo'], df_aux['F1'], label='mag: 1.5 - #' + str(len(df_aux)))

df_aux = df[(df['magnitud'] >= 1.8) & (df['magnitud'] <= 2.2)]
df_aux = df_aux[df_aux['F1'] > 0.01]
df_aux['F1'] = df_aux['F1'].rolling(window=WIN).mean()
plt.plot(df_aux['ciclo'], df_aux['F1'], label='mag: 2.0 - #' + str(len(df_aux)))

df_aux = df[(df['magnitud'] >= 2.3) & (df['magnitud'] <= 2.7)]
df_aux = df_aux[df_aux['F1'] > 0.01]
df_aux['F1'] = df_aux['F1'].rolling(window=WIN).mean()
plt.plot(df_aux['ciclo'], df_aux['F1'], label='mag: 2.5 - #' + str(len(df_aux)))

df_aux = df[(df['magnitud'] >= 2.8) & (df['magnitud'] <= 3.2)]
df_aux = df_aux[df_aux['F1'] > 0.01]
df_aux['F1'] = df_aux['F1'].rolling(window=WIN).mean()
plt.plot(df_aux['ciclo'], df_aux['F1'], label='mag: 3.0 - #' + str(len(df_aux)))

df_aux = df[(df['magnitud'] >= 3.3) & (df['magnitud'] <= 3.7)]
df_aux = df_aux[df_aux['F1'] > 0.01]
df_aux['F1'] = df_aux['F1'].rolling(window=WIN).mean()
plt.plot(df_aux['ciclo'], df_aux['F1'], label='mag: 3.5 - #' + str(len(df_aux)))

df_aux = df[(df['magnitud'] >= 3.8) & (df['magnitud'] <= 4.2)]
df_aux = df_aux[df_aux['F1'] > 0.01]
df_aux['F1'] = df_aux['F1'].rolling(window=WIN).mean()
plt.plot(df_aux['ciclo'], df_aux['F1'], label='mag: 4.0 - #' + str(len(df_aux)))
plt.xlabel('ciclos')
plt.ylabel(r'$V_i - V_{i-1}$')
plt.title('Diferencia de voltaje salida del reposo\nRespuesta resistiva')
plt.legend(bbox_to_anchor=(1, 1)), plt.grid('on'), plt.show()

WIN = 1
alpha = 0.3
df_aux = df[(df['magnitud'] >= 0.3) & (df['magnitud'] <= 0.7)]
df_aux = df_aux[df_aux['F1'] > 0.01]
df_aux['F1'] = df_aux['F1'].rolling(window=WIN).mean()
plt.scatter(df_aux['energia'], df_aux['F1'], label='emag: 0.5 - #' + str(len(df_aux)), alpha=alpha)

df_aux = df[(df['magnitud'] >= 0.8) & (df['magnitud'] <= 1.2)]
df_aux = df_aux[df_aux['F1'] > 0.01]
df_aux['F1'] = df_aux['F1'].rolling(window=WIN).mean()
plt.scatter(df_aux['energia'], df_aux['F1'], label='emag: 1.0 - #' + str(len(df_aux)), alpha=alpha)

df_aux = df[(df['magnitud'] >= 1.4) & (df['magnitud'] <= 1.6)]
df_aux = df_aux[df_aux['F1'] > 0.01]
df_aux['F1'] = df_aux['F1'].rolling(window=WIN).mean()
plt.scatter(df_aux['energia'], df_aux['F1'], label='emag: 1.5 - #' + str(len(df_aux)), alpha=alpha)

df_aux = df[(df['magnitud'] >= 1.8) & (df['magnitud'] <= 2.2)]
df_aux = df_aux[df_aux['F1'] > 0.01]
df_aux['F1'] = df_aux['F1'].rolling(window=WIN).mean()
plt.scatter(df_aux['energia'], df_aux['F1'], label='emag: 2.0 - #' + str(len(df_aux)), alpha=alpha)

df_aux = df[(df['magnitud'] >= 2.3) & (df['magnitud'] <= 2.7)]
df_aux = df_aux[df_aux['F1'] > 0.01]
df_aux['F1'] = df_aux['F1'].rolling(window=WIN).mean()
plt.scatter(df_aux['energia'], df_aux['F1'], label='emag: 2.5 - #' + str(len(df_aux)), alpha=alpha)

df_aux = df[(df['magnitud'] >= 2.8) & (df['magnitud'] <= 3.2)]
df_aux = df_aux[df_aux['F1'] > 0.01]
df_aux['F1'] = df_aux['F1'].rolling(window=WIN).mean()
plt.scatter(df_aux['energia'], df_aux['F1'], label='emag: 3.0 - #' + str(len(df_aux)), alpha=alpha)

df_aux = df[(df['magnitud'] >= 3.3) & (df['magnitud'] <= 3.7)]
df_aux = df_aux[df_aux['F1'] > 0.01]
df_aux['F1'] = df_aux['F1'].rolling(window=WIN).mean()
plt.scatter(df_aux['energia'], df_aux['F1'], label='emag: 3.5 - #' + str(len(df_aux)), alpha=alpha)

df_aux = df[(df['magnitud'] >= 3.8) & (df['magnitud'] <= 4.2)]
df_aux = df_aux[df_aux['F1'] > 0.01]
df_aux['F1'] = df_aux['F1'].rolling(window=WIN).mean()
plt.scatter(df_aux['energia'], df_aux['F1'], label='emag: 4.0 - #' + str(len(df_aux)), alpha=alpha)
plt.xlabel('energia')
plt.ylabel(r'$V_i - V_{i-1}$')
plt.title('Diferencia de voltaje salida del reposo\nRespuesta resistiva')
plt.legend(bbox_to_anchor=(1, 1)), plt.grid('on'), plt.show()


# In[15]:


def plot_energia_magnitud(df, m, ei, ef, WIN, alpha):
    df_aux = df[(df['magnitud'] >= m - 0.2) & (df['magnitud'] <= m + 0.2)]
    df_aux = df_aux[(df['energia'] >= ei) & (df['energia'] <= ef)]
    df_aux = df_aux[df_aux['F1'] > 0.01]
    df_aux['F1'] = df_aux['F1'].rolling(window=WIN).mean()
    plt.plot(df_aux['ciclo'], df_aux['F1'],
             label='mag: ' + str(m) + ' - #' + str(len(df_aux)) + ' i: ' + str(ei) + '-' + str(ef))
    


# In[16]:


WIN = 1
plot_energia_magnitud(df, 0.5, 0, 5000, WIN, alpha)
# plot_energia_magnitud(df, 0.5, 10000, 15000, WIN, alpha)
# plot_energia_magnitud(df, 0.5, 15000, 20000, WIN, alpha)
# plot_energia_magnitud(df, 0.5, 20000, 25000, WIN, alpha)
# plt.legend(bbox_to_anchor=(1, 1)), plt.grid('on'), plt.show()

plot_energia_magnitud(df, 1.0, 0, 5000, WIN, alpha)
# plot_energia_magnitud(df, 1.0, 10000, 15000, WIN, alpha)
# plot_energia_magnitud(df, 1.0, 15000, 20000, WIN, alpha)
# plot_energia_magnitud(df, 1.0, 20000, 25000, WIN, alpha)
# plt.legend(bbox_to_anchor=(1, 1)), plt.grid('on'), plt.show()

plot_energia_magnitud(df, 1.5, 0, 5000, WIN, alpha)
# plot_energia_magnitud(df, 1.5, 10000, 15000, WIN, alpha)
# plot_energia_magnitud(df, 1.5, 15000, 20000, WIN, alpha)
# plot_energia_magnitud(df, 1.5, 20000, 25000, WIN, alpha)
# plt.legend(bbox_to_anchor=(1, 1)), plt.grid('on'), plt.show()

plot_energia_magnitud(df, 2.0, 0, 5000, WIN, alpha)
# plot_energia_magnitud(df, 2.0, 10000, 15000, WIN, alpha)
# plot_energia_magnitud(df, 2.0, 15000, 20000, WIN, alpha)
# plot_energia_magnitud(df, 2.0, 20000, 25000, WIN, alpha)
# plt.legend(bbox_to_anchor=(1, 1)), plt.grid('on'), plt.show()

plot_energia_magnitud(df, 2.5, 0, 5000, WIN, alpha)
# plot_energia_magnitud(df, 2.5, 10000, 15000, WIN, alpha)
# plot_energia_magnitud(df, 2.5, 15000, 20000, WIN, alpha)
# plot_energia_magnitud(df, 2.5, 20000, 25000, WIN, alpha)
# plt.legend(bbox_to_anchor=(1, 1)), plt.grid('on'), plt.show()

plot_energia_magnitud(df, 3.0, 0, 5000, WIN, alpha)
# plot_energia_magnitud(df, 3.0, 10000, 15000, WIN, alpha)
# plot_energia_magnitud(df, 3.0, 15000, 20000, WIN, alpha)
# plot_energia_magnitud(df, 3.0, 20000, 25000, WIN, alpha)
# plt.legend(bbox_to_anchor=(1, 1)), plt.grid('on'), plt.show()

plot_energia_magnitud(df, 3.5, 0, 5000, WIN, alpha)
# plot_energia_magnitud(df, 3.5, 10000, 15000, WIN, alpha)
# plot_energia_magnitud(df, 3.5, 15000, 20000, WIN, alpha)
# plot_energia_magnitud(df, 3.5, 20000, 25000, WIN, alpha)
# plt.legend(bbox_to_anchor=(1, 1)), plt.grid('on'), plt.show()

plot_energia_magnitud(df, 4.0, 0, 5000, WIN, alpha)
# plot_energia_magnitud(df, 4.0, 10000, 15000, WIN, alpha)
# plot_energia_magnitud(df, 4.0, 15000, 20000, WIN, alpha)
# plot_energia_magnitud(df, 4.0, 20000, 25000, WIN, alpha)
plt.title('Diferencia de voltaje, salida del reposo\nRespuesta resistiva')
plt.xlabel('ciclos')
plt.ylabel(r'$V_i - V_{i-1}$')
plt.legend(bbox_to_anchor=(1, 1)), plt.grid('on'), plt.show()


# In[17]:


# comportamiento de potencia instantanea entregada
def potencia_instantanea_salto(arr_voltaje, arr_corriente, limite=10):
    aux = np.zeros(len(arr_voltaje))
    for i, serie in enumerate(arr_corriente):
        contador = 0
        for j, c in enumerate(serie):
            if c != 0 and aux[i] == 0:
                aux[i] = c * arr_voltaje[i][j]
                contador = 1
            elif contador < limite:
                aux[i] += c * arr_voltaje[i][j]
                contador += 1
            elif contador == limite: break
            
    return aux


# In[18]:


df['F1'] = potencia_instantanea_salto(df['voltaje'], df['corriente'], limite=1)
df


# In[19]:


WIN = 1
df_aux = df[(df['magnitud'] >= 0.3) & (df['magnitud'] <= 0.7)]
df_aux = df_aux[df_aux['F1'] > 0.01]
df_aux['F1'] = df_aux['F1'].rolling(window=WIN).mean()
plt.plot(df_aux['ciclo'], df_aux['F1'], label='mag: 0.5 - #' + str(len(df_aux)))

df_aux = df[(df['magnitud'] >= 0.8) & (df['magnitud'] <= 1.2)]
df_aux = df_aux[df_aux['F1'] > 0.01]
df_aux['F1'] = df_aux['F1'].rolling(window=WIN).mean()
plt.plot(df_aux['ciclo'], df_aux['F1'], label='mag: 1.0 - #' + str(len(df_aux)))

df_aux = df[(df['magnitud'] >= 1.4) & (df['magnitud'] <= 1.6)]
df_aux = df_aux[df_aux['F1'] > 0.01]
df_aux['F1'] = df_aux['F1'].rolling(window=WIN).mean()
plt.plot(df_aux['ciclo'], df_aux['F1'], label='mag: 1.5 - #' + str(len(df_aux)))

df_aux = df[(df['magnitud'] >= 1.8) & (df['magnitud'] <= 2.2)]
df_aux = df_aux[df_aux['F1'] > 0.01]
df_aux['F1'] = df_aux['F1'].rolling(window=WIN).mean()
plt.plot(df_aux['ciclo'], df_aux['F1'], label='mag: 2.0 - #' + str(len(df_aux)))

df_aux = df[(df['magnitud'] >= 2.3) & (df['magnitud'] <= 2.7)]
df_aux = df_aux[df_aux['F1'] > 0.01]
df_aux['F1'] = df_aux['F1'].rolling(window=WIN).mean()
plt.plot(df_aux['ciclo'], df_aux['F1'], label='mag: 2.5 - #' + str(len(df_aux)))

df_aux = df[(df['magnitud'] >= 2.8) & (df['magnitud'] <= 3.2)]
df_aux = df_aux[df_aux['F1'] > 0.01]
df_aux['F1'] = df_aux['F1'].rolling(window=WIN).mean()
plt.plot(df_aux['ciclo'], df_aux['F1'], label='mag: 3.0 - #' + str(len(df_aux)))

# df_aux = df[(df['magnitud'] >= 3.3) & (df['magnitud'] <= 3.7)]
# df_aux = df_aux[df_aux['F1'] > 0.01]
# df_aux['F1'] = df_aux['F1'].rolling(window=WIN).mean()
# plt.plot(df_aux['ciclo'], df_aux['F1'], label='mag: 3.5 - #' + str(len(df_aux)))

# df_aux = df[(df['magnitud'] >= 3.8) & (df['magnitud'] <= 4.2)]
# df_aux = df_aux[df_aux['F1'] > 0.01]
# df_aux['F1'] = df_aux['F1'].rolling(window=WIN).mean()
# plt.plot(df_aux['ciclo'], df_aux['F1'], label='mag: 4.0 - #' + str(len(df_aux)))
plt.xlabel('ciclos')
plt.ylabel(r'$\min_i \ V_i \times I_i \ \ne \ 0$')
plt.title('Potencia entregada al instante despues del salto')
plt.legend(bbox_to_anchor=(1, 1)), plt.grid('on'), plt.show()

WIN = 1
alpha = 0.3
df_aux = df[(df['magnitud'] >= 0.3) & (df['magnitud'] <= 0.7)]
df_aux = df_aux[df_aux['F1'] > 0.01]
df_aux['F1'] = df_aux['F1'].rolling(window=WIN).mean()
plt.scatter(df_aux['energia'], df_aux['F1'], label='emag: 0.5 - #' + str(len(df_aux)), alpha=alpha)

df_aux = df[(df['magnitud'] >= 0.8) & (df['magnitud'] <= 1.2)]
df_aux = df_aux[df_aux['F1'] > 0.01]
df_aux['F1'] = df_aux['F1'].rolling(window=WIN).mean()
plt.scatter(df_aux['energia'], df_aux['F1'], label='emag: 1.0 - #' + str(len(df_aux)), alpha=alpha)

df_aux = df[(df['magnitud'] >= 1.4) & (df['magnitud'] <= 1.6)]
df_aux = df_aux[df_aux['F1'] > 0.01]
df_aux['F1'] = df_aux['F1'].rolling(window=WIN).mean()
plt.scatter(df_aux['energia'], df_aux['F1'], label='emag: 1.5 - #' + str(len(df_aux)), alpha=alpha)

df_aux = df[(df['magnitud'] >= 1.8) & (df['magnitud'] <= 2.2)]
df_aux = df_aux[df_aux['F1'] > 0.01]
df_aux['F1'] = df_aux['F1'].rolling(window=WIN).mean()
plt.scatter(df_aux['energia'], df_aux['F1'], label='emag: 2.0 - #' + str(len(df_aux)), alpha=alpha)

df_aux = df[(df['magnitud'] >= 2.3) & (df['magnitud'] <= 2.7)]
df_aux = df_aux[df_aux['F1'] > 0.01]
df_aux['F1'] = df_aux['F1'].rolling(window=WIN).mean()
plt.scatter(df_aux['energia'], df_aux['F1'], label='emag: 2.5 - #' + str(len(df_aux)), alpha=alpha)

df_aux = df[(df['magnitud'] >= 2.8) & (df['magnitud'] <= 3.2)]
df_aux = df_aux[df_aux['F1'] > 0.01]
df_aux['F1'] = df_aux['F1'].rolling(window=WIN).mean()
plt.scatter(df_aux['energia'], df_aux['F1'], label='emag: 3.0 - #' + str(len(df_aux)), alpha=alpha)

# df_aux = df[(df['magnitud'] >= 3.3) & (df['magnitud'] <= 3.7)]
# df_aux = df_aux[df_aux['F1'] > 0.01]
# df_aux['F1'] = df_aux['F1'].rolling(window=WIN).mean()
# plt.scatter(df_aux['energia'], df_aux['F1'], label='emag: 3.5 - #' + str(len(df_aux)), alpha=alpha)

# df_aux = df[(df['magnitud'] >= 3.8) & (df['magnitud'] <= 4.2)]
# df_aux = df_aux[df_aux['F1'] > 0.01]
# df_aux['F1'] = df_aux['F1'].rolling(window=WIN).mean()
# plt.scatter(df_aux['energia'], df_aux['F1'], label='emag: 4.0 - #' + str(len(df_aux)), alpha=alpha)
plt.xlabel('energia')
plt.ylabel(r'$\min_i \ V_i \times I_i \ \ne \ 0$')
plt.title('Potencia entregada al instante despues del salto')
plt.legend(bbox_to_anchor=(1, 1)), plt.grid('on'), plt.show()


# In[20]:


WIN = 5
plot_energia_magnitud(df, 0.5, 0, 5000, WIN, alpha)
# plot_energia_magnitud(df, 0.5, 10000, 15000, WIN, alpha)
# plot_energia_magnitud(df, 0.5, 15000, 20000, WIN, alpha)
# plot_energia_magnitud(df, 0.5, 20000, 25000, WIN, alpha)
# plt.legend(bbox_to_anchor=(1, 1)), plt.grid('on'), plt.show()

plot_energia_magnitud(df, 1.0, 0, 5000, WIN, alpha)
# plot_energia_magnitud(df, 1.0, 10000, 15000, WIN, alpha)
# plot_energia_magnitud(df, 1.0, 15000, 20000, WIN, alpha)
# plot_energia_magnitud(df, 1.0, 20000, 25000, WIN, alpha)
# plt.legend(bbox_to_anchor=(1, 1)), plt.grid('on'), plt.show()

plot_energia_magnitud(df, 1.5, 0, 5000, WIN, alpha)
# plot_energia_magnitud(df, 1.5, 10000, 15000, WIN, alpha)
# plot_energia_magnitud(df, 1.5, 15000, 20000, WIN, alpha)
# plot_energia_magnitud(df, 1.5, 20000, 25000, WIN, alpha)
# plt.legend(bbox_to_anchor=(1, 1)), plt.grid('on'), plt.show()

plot_energia_magnitud(df, 2.0, 0, 5000, WIN, alpha)
# plot_energia_magnitud(df, 2.0, 10000, 15000, WIN, alpha)
# plot_energia_magnitud(df, 2.0, 15000, 20000, WIN, alpha)
# plot_energia_magnitud(df, 2.0, 20000, 25000, WIN, alpha)
# plt.legend(bbox_to_anchor=(1, 1)), plt.grid('on'), plt.show()

plot_energia_magnitud(df, 2.5, 0, 5000, WIN, alpha)
# plot_energia_magnitud(df, 2.5, 10000, 15000, WIN, alpha)
# plot_energia_magnitud(df, 2.5, 15000, 20000, WIN, alpha)
# plot_energia_magnitud(df, 2.5, 20000, 25000, WIN, alpha)
# plt.legend(bbox_to_anchor=(1, 1)), plt.grid('on'), plt.show()

plot_energia_magnitud(df, 3.0, 0, 5000, WIN, alpha)
# plot_energia_magnitud(df, 3.0, 10000, 15000, WIN, alpha)
# plot_energia_magnitud(df, 3.0, 15000, 20000, WIN, alpha)
# plot_energia_magnitud(df, 3.0, 20000, 25000, WIN, alpha)
# plt.legend(bbox_to_anchor=(1, 1)), plt.grid('on'), plt.show()

plot_energia_magnitud(df, 3.5, 0, 5000, WIN, alpha)
# plot_energia_magnitud(df, 3.5, 10000, 15000, WIN, alpha)
# plot_energia_magnitud(df, 3.5, 15000, 20000, WIN, alpha)
# plot_energia_magnitud(df, 3.5, 20000, 25000, WIN, alpha)
# plt.legend(bbox_to_anchor=(1, 1)), plt.grid('on'), plt.show()

plot_energia_magnitud(df, 4.0, 0, 5000, WIN, alpha)
# plot_energia_magnitud(df, 4.0, 10000, 15000, WIN, alpha)
# plot_energia_magnitud(df, 4.0, 15000, 20000, WIN, alpha)
# plot_energia_magnitud(df, 4.0, 20000, 25000, WIN, alpha)
plt.ylabel(r'$\min_i \ V_i \times I_i \ \ne \ 0$')
plt.title('Potencia entregada al instante despues del salto')
plt.xlabel('ciclos')
plt.legend(bbox_to_anchor=(1, 1)), plt.grid('on'), plt.show()


# In[21]:


def tiempo_a_inflexion(arr_voltaje):
    """
    funcion que entrega el tiempo necesario a la inflexion
    """
    
    
    
    return


# Trabajo próximo:
# - guardar curvas hasta el salto anterior (instante antes de 0)
# - guardar voltaje anterior como etiqueta
# - guardar temperatura inicial como etiqueta
# - Para feature de degradación usar delta T del salto completo (inicio - fin)
# - Ver fenomenolgía del circuito RC, ajustar función de transferencia de modelo RC
#     - para encontrar la constante de proporcionalidad, tomar curvas de voltaje 'estables' donde $V_{i+1} - V_{i} < \epsilon$
#     - Encontrar constante de tiempo entre esas curvas

# # Grafico de curva de salud

# In[22]:


lista = [str('reference charge')]
final = [str('rest post reference charge')]
t_carga, v_carga, c_carga, temp_carga, T_carga = carga_de_datos(documento, lista, final, nofinal=True)

lista = [str('reference discharge')]
final = [str('rest post reference discharge')]
t_descarga, v_descarga, c_descarga, temp_descarga, T_descarga = carga_de_datos(documento, lista, final, nofinal=True)
print(len(t_descarga))


# In[23]:


# integrar la descarga
energia_entregada = []
T = []
for k in t_descarga.keys():
    energia_entregada.append(sum(c_descarga[k]) / 360)
    T.append(np.min(t_descarga[k]) / 3600)
plt.scatter(T, energia_entregada, label='descarga', facecolor='none', s=88, edgecolors='b', alpha=0.8)
plt.ylabel('Capacidad [Ah]')
plt.xlabel('Tiempo [horas]')
plt.legend()
# plt.title('Capacidad de energía entregada por batería'), plt.show()

# integrar la carga
energia_entregada = []
T = []
for k in t_carga.keys():
    energia_entregada.append(sum(c_carga[k]) / 360)
    T.append(np.mean(t_carga[k]) / 3600)
# plt.scatter(T, np.abs(energia_entregada), label='carga', s=38, edgecolors='b', alpha=0.4)
plt.ylabel('Capacidad [Ah]')
plt.xlabel('Tiempo [horas]')
plt.legend()
plt.title('Capacidad de energía'), plt.show()


# In[ ]:





# [ciclo, potencia, media y magnitud del salto] como identidad de un salto, a partir de ello obtener una respuesta dinamica con features
