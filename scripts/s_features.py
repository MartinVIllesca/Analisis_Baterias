# features
import numpy as np
import matplotlib.pyplot as plt


def primera_derivada(arr_serie):
    aux = []
    for serie in arr_serie:
        serie = serie[2:]
        aux2 = list(serie[:-1] - serie[1:])
        if len(aux2) == 0: aux2 = [0]
        aux.append(aux2)
    return aux

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

def plot_energia_magnitud(df, m, ei, ef, WIN, alpha):
    df_aux = df[(df['magnitud'] >= m - 0.2) & (df['magnitud'] <= m + 0.2)]
    df_aux = df_aux[(df['energia'] >= ei) & (df['energia'] <= ef)]
    df_aux = df_aux[df_aux['F1'] > 0.01]
    df_aux['F1'] = df_aux['F1'].rolling(window=WIN).mean()
    plt.plot(df_aux['ciclo'], df_aux['F1'],
             label='mag: ' + str(m) + ' - #' + str(len(df_aux)) + ' i: ' + str(ei) + '-' + str(ef))
    