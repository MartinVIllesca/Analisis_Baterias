import numpy as np

#validacion
def Valid_data(data):
    if len(data)>100:
        ini = (data[15]-data[7]) > 0 
        final = [False,0]
        aux = 0
        index = 0
        if ini == True:
            for i in range(len(data)-1):
                if (data[i+1]-data[i]) <=0 :
                    if aux == 0:
                        index = i
                    aux += 1
                    if aux == 25:
                        final = [True,index]
                        break
                else:
                    aux = 0
        if ini == True and final[0] == True:
            return [True,final[1]]
        else:
            return [False,0]
    else:
        return [False,0]

def ChangeLabel(Data, n):
    maxi = max(Data)
    mini = min(Data)
    size = maxi/n
    labels = [int((i-mini)/size)+1 for i in Data]
    return labels

#Magnitud relativa al consumo anterior
def Mg_rel (data):
    data = data.values
    aux  = data[1:]-data[:-1]
    aux = np.insert(aux,0,data[0])
    aux_2 = aux/2 
    return [aux,aux_2]

def poprow(my_array,pr):
    """ row popping in numpy arrays
    Input: my_array - NumPy array, pr: row index to pop out
    Output: [new_array,popped_row] """
    i = pr
    pop = my_array[i,:]
    new_array = np.vstack((my_array[:i,:],my_array[i+1:,:]))
    return [new_array,pop]

def popcol(my_array,pc):
    """ column popping in numpy arrays
    Input: my_array: NumPy array, pc: column index to pop out
    Output: [new_array,popped_col] """
    i = pc
    pop = my_array[:,i]
    new_array = np.hstack((my_array[:,:i],my_array[:,i+1:]))
    return [new_array,pop]


def Mean_n(data, n):
    aux = np.zeros(len(data))
    rg = int(len(data)/n)
    for i in range(rg):
        aux[i*n:i*n+n] =  np.round(np.median(data[i*n:i*n+n])+0.1)
    if(i*n+n<len(data)):
        aux[i*n+n:] = np.round(np.median(data[i*n+n:])+0.1)
    return aux

def data_valid(data_x,data_y,porcentaje,n,rs=None):
    "esta funcion entrega el conjunto de validacion"
    "porcentaje = cantidad de datos minimos del conjunto total que se quieren como conjunto de validación"
    "n = numero de elementos consecutivos que se quieren tomar en el conjunto de validación"
    data_valid_x = np.empty((0,data_x.shape[1]))
    data_valid_y = []
    "Se reordenan los datos para poder trabajarlo apropiadamente"
    new_data_x = data_x
    new_data_y = np.reshape(data_y,(1,-1))
    "porcentaje de parada"
    por_stop = porcentaje/(100-porcentaje)
    por_act = 0
    l = len(new_data_x)-1
    while(por_act<por_stop):
        "Si el conjunto de validación tiene menos % de lo pedido"
        "Se genera un indice random"
        idx = np.random.randint(0,l) 
        idx_fn = idx+n-1
        if(idx_fn<l):
            for i in range(n):
                idx_act = idx
                "Se genera el valor extraido y el conjunto sin ese valor n veces"
                pop_x = poprow(new_data_x,idx_act)
                pop_y = popcol(new_data_y,idx_act)
                "El valor popeado esta en [1] y el conjunto sin el valor en [0]"
                "Por lo que en el conjunto de validación hay que ir adjuntando"
                data_valid_x = np.append(data_valid_x,np.reshape(pop_x[1],(1,-1)),axis=0)
                data_valid_y = np.append(data_valid_y,pop_y[1])
                "Por lo que en el conjunto de entrenamiento no se necesitan adjuntar los datos"
                new_data_x = pop_x[0]
                new_data_y = pop_y[0]
        l = new_data_x.shape[0]-1
        l_valid = data_valid_x.shape[0]-1
        "se toma el pocentaje acutal para ver si se cumple la condición"
        por_act = l_valid/l
    np.random.seed(rs)
    "Se juntan los label en el conjunto de entrenamiento para luego mezclar"
    data_train = np.hstack((new_data_x,np.reshape(new_data_y,(-1,1))))
    np.random.shuffle(data_train)
    "Se obtiene los conjuntos de entrenamiento y validacion separados en x e y"
    data_train_x = data_train[:,:-1]
    data_train_y = data_train[:,-1]
    return [data_train_x,data_train_y,data_valid_x,data_valid_y]