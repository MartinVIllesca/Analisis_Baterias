import numpy as np
from preparacion_entrenamiento import *

def primera_derivada(data_1,data_2):
    aux = []
    for i in range(len(data_1)):
        der_mean = np.mean(np.diff(data_1[i][2:])/10)
        aux.append(der_mean)
    return aux    

def segunda_derivada(data_1,data_2):
    aux = []
    for i in range(len(data_1)):
        der_mean = np.mean(np.diff(np.diff(data_1[i][2:]))/100)
        aux.append(der_mean)
    return aux

#Delta temperatura
def deltaT(arr_serie, magnitud):
    data = arr_serie.values
    mag = magnitud.values
    aux = data[1:]-data[:-1]
    aux = np.insert(aux,0,data[0])
    aux = abs(aux/mag)
    return aux

#Efecto capacitivo
def primera_derivada_RC(data):
    aux = []
    for i in data:
        valid = Valid_data(i)
        if valid[0] == True:
            inf = valid[1]
            der_mean = np.mean(np.diff(i[:inf])/10)
            aux.append(der_mean)
        else:
            aux.append(0)
    return aux

def segunda_derivada_RC(data):
    aux = []
    for i in data:
        valid = Valid_data(i)
        if valid[0] == True:
            inf = valid[1]
            der_mean = np.mean(np.diff(np.diff(i[:inf]))/100)
            aux.append(der_mean)
        else:
            aux.append(0)
    return aux
    
def RC(data,time,ind):
    aux = []
    aux_2 = []
    for i in range(len(data)):
        valid = Valid_data(data[i])
        if valid[0] == True:
            inf = valid[1]
            Rc = (time[i][inf]-time[i][0])*0.632
            aux.append(Rc)
        else:
            aux.append(0)
    n = max(ind)
    k = 0
    for i in range(n-1):
        ind_aux = np.where(ind.values ==i+1)
        k += len(ind_aux[0])
        if len(ind_aux[0]) > 0:
            ini_aux = ind_aux[0][0]
            final_aux = ind_aux[0][-1]
            #print(ini_aux,final_aux)
            mean = np.mean(aux[ini_aux:final_aux])    
            kk = [mean]*len(ind_aux[0])
            aux_2 = np.append(aux_2, kk)
        else:
            aux_2 = np.append(aux_2, 0)
    return [aux,aux_2]

# comportamiento de potencia instantanea entregada
def potencia_instantanea_salto(arr_voltaje, arr_corriente, limite=10):
    aux = np.zeros(len(arr_voltaje))
    for i, serie in enumerate(arr_corriente):
        serie = serie[1:]
        contador = 0
        for j, c in enumerate(serie):
            if c != 0 and aux[i] == 0:
                aux[i] = c * arr_voltaje[i][j]
                contador = 1
            elif c != 0 and contador < limite:
                aux[i] += c * arr_voltaje[i][j]
                contador += 1
            elif contador == limite: break
            
    return aux

#Energia total consumida 
def energia_total(data_ene,data_ciclo):
    ene = np.zeros(len(data_ene))
    data_ene = data_ene.values
    data_ciclo = data_ciclo.values
    aux = 1
    ind_aux = 0
    for i in range(len(data_ene)):
        if data_ciclo[i] == 1:
            ene[i] = data_ene[i]
        elif data_ciclo[i] != aux:
            aux = data_ciclo[i]
            ind_aux = i-1
            ene[i] = ene[ind_aux]+data_ene[i]
        else:
             ene[i] = ene[ind_aux] + data_ene[i]    
    return ene

# definiendo features
def diferencia_voltaje(arr_voltajes, arr_corrientes):
    '''
    Funcion que entrega el valor de diferencia de voltaje del salto con el ultimo voltaje
    del salto anterior
    '''
    DIFERENCIAS = []
    epsilon = 0.3

    for j, r in enumerate(arr_voltajes):

        arr_voltaje = r
        arr_corriente = arr_corrientes[j]


        # guargar voltaje inicial
        voltaje_inicial = arr_voltaje[0]

        # identificar donde cambia la corriente de 0 al nuevo salto
        index = 0

        # buscamos cuando cambia
        anterior = arr_corriente[0]
        for i, x in enumerate(arr_corriente):
            if x - anterior > epsilon:
                index = i
                break
            anterior = x

        voltaje_final = arr_voltaje[index]
        
        # if j == 3:
        #     print(arr_corriente)
        #     print(arr_voltaje)
        #     print(index)
        #     break

        DIFERENCIAS.append(voltaje_final - voltaje_inicial)

    return DIFERENCIAS
    # return np.ones((len(arr_voltajes), ))

# definiendo features
def diferencia_voltaje2(arr_voltajes, arr_corrientes):
    '''
    Funcion que entrega el valor de diferencia de voltaje del salto con el ultimo voltaje
    del salto anterior
    '''
    DIFERENCIAS = []
    epsilon = 0.3

    for j, r in enumerate(arr_voltajes):

        arr_voltaje = r
        arr_corriente = arr_corrientes[j]

        # identificar donde cambia la corriente de 0 al nuevo salto
        index = 0

        # buscamos cuando cambia
        anterior = arr_corriente[0]
        for i, x in enumerate(arr_corriente):
            if x - anterior > epsilon:
                index = i
                break
            anterior = x

        # guargar voltaje inicial
        voltaje_inicial = arr_voltaje[index - 1]

        voltaje_final = arr_voltaje[index]

        # if j == 3:
        #     print(arr_corriente)
        #     print(arr_voltaje)
        #     print(index)
        #     break

        DIFERENCIAS.append(voltaje_final - voltaje_inicial)

    return DIFERENCIAS
    # return np.ones((len(arr_voltajes), ))
#Magnitud relativa al consumo anterior
def Mg_rel (data):
    data = data.values
    aux  = data[1:]-data[:-1]
    aux = np.insert(aux,0,data[0])
    aux_2 = aux/2 
    return [aux,aux_2]
