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
