import scipy.io as sio
import numpy as np


def carga_de_datos(datafile, lista, final, nofinal=False, add_final=False, pr=False):
    """
    recibe:
        datafile:   (string) nombre del archivo a cargar en .mat
        lista:      (lista de string) conjunto de comentarios a cargar
        final:      (lista de string) comentario de la ultima ventana a cargar
        nofinal:    (boolean) cargar solamente los elementos de la lista

    retorna:
        [tiempo, voltage, current, temperatura, T]
    """
    
    data = sio.loadmat(datafile)['data']
    voltage = {}
    current = {}
    temperatura = {}
    tiempo = {}

    T = {}
    j = 0
    if nofinal:
        tiempo[j] = []
        voltage[j] = []
        current[j] = []
        temperatura[j] = []
        T[j] = []
    

#     print(lista)
#     print('final: ', len(data[0][0][0][0]))
    for idx, elem in enumerate(data[0][0][0][0]):
#         if idx == 10000: break
        # print(idx, elem[0] , 'ciclo')
#         print(list(elem[2])[0][0], elem[0][0])
        if idx % 1000 == 0 : print(idx)

    if pr:
        print(lista)
        print('final: ', len(data[0][0][0][0]))
    for idx, elem in enumerate(data[0][0][0][0]):
#         if idx == 10000: break
        # print(idx, elem[0] , 'ciclo')
        if pr: print(list(elem[2])[0][0], elem[0][0])
        if idx % 1000 == 0 :
            if pr: print(idx)

        if any(str(elem[0][0]) in s for s in lista):
#             print(elem[0])
            for v in list(elem[2])[0]: tiempo[j].append(v)
            for v in list(elem[4])[0]: voltage[j].append(v)
            for v in list(elem[5])[0]: current[j].append(v)
            for v in list(elem[6])[0]: temperatura[j].append(v)
            for v in list(elem[7])[0]: T[j].append(v)
            if nofinal:
                # print(idx, j, elem[0])
                j += 1
                tiempo[j] = []
                voltage[j] = []
                current[j] = []
                temperatura[j] = []
                T[j] = []
                continue
        elif any(str(elem[0][0]) in s for s in final) and not nofinal:
            j += 1

            #print(idx, j, str(elem[0][0]), 'input', list(elem[2])[0][0])

            if pr: print(idx, j, str(elem[0][0]), 'input', list(elem[2])[0][0])

            tiempo[j] = []
            voltage[j] = []
            current[j] = []
            temperatura[j] = []
            T[j] = []
            if add_final:
                for v in list(elem[2])[0]: tiempo[j].append(v)
                for v in list(elem[4])[0]: voltage[j].append(v)
                for v in list(elem[5])[0]: current[j].append(v)
                for v in list(elem[6])[0]: temperatura[j].append(v)
                for v in list(elem[7])[0]: T[j].append(v)
    
    tiempo.pop(j)
    voltage.pop(j)
    current.pop(j)
    temperatura.pop(j)
    T.pop(j)
    return [tiempo, voltage, current, temperatura, T]



def guardar_serie_entre2(serie, ti, corriente=False, inicio=0, fin=0):
    """
    Primero entregar la corriente en el instante ti, para encontrar el inicio y el fin. El inicio de la serie
    sera el final del salto anterior, justo antes de cambiar a 0 A. Termina la serie justo en el instante anterior
    al cambiar a 0 A. Setear corriente en True
    
    Luego, pasar el inicio entregado anteriormente y el fin a las nuevas cargas de datos. Setear corriente en False
    
    """
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
        while (serie[ti + i] != 0):
            if (ti + i == len(serie) - 1): break
            i += 1
        fin = ti + i
        return [inicio, fin, np.array(serie[inicio:fin])]
    else:
        return [inicio, fin, np.array(serie[inicio:fin])]


def transformar_datos(t, v, c, temp, T):
    
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
                
    #             aux = guardar_serie_entre(v[i], idx, 2*dx)
    #             arr_voltaje_salto.append(aux)
    #             arr_corriente_salto.append(guardar_serie_entre(c[i], idx, 2*dx, tamanno=len(aux)))
    #             arr_tiempo_salto.append(guardar_serie_entre(t[i], idx, 2*dx, tamanno=len(aux)))
    #             arr_temperatura_salto.append(guardar_serie_entre(temp[i], idx, 2*dx, tamanno=len(aux)))
                
                inicio, fin, aux = guardar_serie_entre2(c[i], idx, corriente=True)
                arr_corriente_salto.append(aux)
                _, _, aux = guardar_serie_entre2(v[i], idx, inicio=inicio, fin=fin)
                arr_voltaje_salto.append(aux)
                _, _, aux = guardar_serie_entre2(t[i], idx, inicio=inicio, fin=fin)
                arr_tiempo_salto.append(aux)
                _, _, aux = guardar_serie_entre2(temp[i], idx, inicio=inicio, fin=fin)
                arr_temperatura_salto.append(aux)
                
                ti = idx

            anterior = elem

    return [arr_dif_salto_c,
            arr_pot_salto,
            arr_centro_salto_c,
            arr_ciclo_salto,
            arr_voltaje_salto,
            arr_corriente_salto,
            arr_tiempo_salto,
            arr_temperatura_salto]

# def guardar_serie_entre(serie, ti, deltaT, tamanno=0):
#     """
#     Funcion para guardar generar las series de tiempo a guardar en el dataframe
#     En especifico: si no se da un tamanno, se asume que es una curva de voltaje y se guarda hasta
#     el punto de mayor voltaje anterior
    
#     Input:
#         serie: serie de tiempo (arreglo de numpy)
#         ti: indice inicial de referencia para guardar los datos
#         deltaT: cantidad de puntos para guardar mas alla de ti
#         tamanno: (opcional) si no se entrega, se asume que es voltaje
#             si se entrega, se asume que se debe guardar desde ti + deltaT, la
#             cantidad de datos dada por tamanno, hacia atras.
    
#     Return:
#         entrega un arreglo de numpy con los datos pedidos
#     """
#     arr = []
        
    
#     fin = ti + deltaT
#     if fin > len(serie): fin = len(serie) - 1
#     ej = serie[ti]
#     i = 0
#     epsilon = 0.1

    
#     if tamanno != 0: i = fin - tamanno
#     else:
#         for idx in range(40):
#             if serie[ti - idx] > ej:
#                 ej = serie[ti - idx]
#                 i = idx
#             elif serie[ti - idx] < ej - epsilon: break
#         i = ti - i
    
#     if i < 0: return np.array(serie[:fin])
    
#     return np.array(serie[i : fin])
