import scipy.io as sio

def carga_de_datos(datafile, lista, final, nofinal=False):
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
    tiempo[j] = []
    voltage[j] = []
    current[j] = []
    temperatura[j] = []
    T[j] = []
    
    print('final: ', len(data[0][0][0][0]))
    for idx, elem in enumerate(data[0][0][0][0]):
#         if idx == 10000: break
#         print(idx, elem[0])
        if idx % 1000 == 0 : print(idx)
        if elem[0] in lista:
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
        elif elem[0] in final and not nofinal:
            for v in list(elem[2])[0]: tiempo[j].append(v)
            for v in list(elem[4])[0]: voltage[j].append(v)
            for v in list(elem[5])[0]: current[j].append(v)
            for v in list(elem[6])[0]: temperatura[j].append(v)
            for v in list(elem[7])[0]: T[j].append(v)
            # print(idx, j, elem[0])
            j += 1
            tiempo[j] = []
            voltage[j] = []
            current[j] = []
            temperatura[j] = []
            T[j] = []
    
    tiempo.pop(j)
    voltage.pop(j)
    current.pop(j)
    temperatura.pop(j)
    T.pop(j)
    return [tiempo, voltage, current, temperatura, T]
