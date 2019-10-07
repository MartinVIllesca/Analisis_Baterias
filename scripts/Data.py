import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio
import scipy.interpolate as inter
from sklearn.linear_model import LinearRegression
from mpl_toolkits.mplot3d import Axes3D
dataMat = sio.loadmat(r'../../Battery_Uniform_Distribution_Discharge_Room_Temp_DataSet_2Post/data/Matlab/RW3.mat')
data = dataMat['data']


def Sep_Array(Data,type,key_ini,n_cycle):
    "Find and take the firts n_cycle index with key_ini"
    data= []
    size = Data.shape[1]
    cycle = 1
    for i in range(size):
        if n_cycle < cycle:
            break
        elif key_ini in str(Data[0][i][0][0]):
            data.append([Data[0][i][type][0], [cycle, Data[0][i][0][0], i]])
            cycle +=1
    return data


def toArray2(Data,type,n_cycle,key_ini,key_final):
    "Take the data and check the index with key_ini (cycle start)"
    "Then take the next one index and add to the list while the key dont be key_final (final cycle)"
    "Repeat this until n_cycle"
    data= []
    size = Data.shape[1]
    cycle = 1
    aux = 0
    for i in range(size-1):
        if key_ini in str(Data[0][i][0][0]) or aux == 1:
            aux = 1
            if key_final[0] not in str(Data[0][i+1][0][0]) and key_final[1] not in str(Data[0][i+1][0][0]):
                data.append([Data[0][i+1][type][0],[cycle,Data[0][i+1][0][0],i+1]])
            else:
                cycle += 1
                aux = 0
                if n_cycle < cycle:
                    break
    return data




def Energy(vol, curr,time):
    data = []
    aux = []
    for i in range(len(vol)):
        for j in range(len(vol[i][0])):
            if j == 0:
                e = vol[i][0][j] * curr[i][0][j]*time
                aux.append(e)
            else:
                aux.append(aux[j-1]+vol[i][0][j] * curr[i][0][j]*time)
        data.append(aux)
        aux = []
    return data


def Energy_label(vol, energy):
    data = []
    aux = []
    ini = 0
    for i in range(len(vol)):
        for j in range(len(vol[i][0])):
            final = ini + len(vol[i][0][j][0])
            if j != len(vol[i][0])-1:
                aux.append(energy[i][ini:final])
                ini = final
            else:
                aux.append(energy[i][ini:final])
                data.append(aux)
                ini = 0
                aux = []
    return data


def Group (Data):
    "This function groups the data corresponding to the same cycle"
    data = []
    data_aux = []
    index_min = Data[0][1][2]
    cycle = Data[0][1][0]
    for i in range(len(Data)):
        if Data[i][1][0] == cycle:
            data_aux = np.concatenate((data_aux,Data[i][0]),axis=0)
        if i+1 <= (len(Data)-1) and Data[i+1][1][0] != cycle:
            data.append([data_aux,[cycle,index_min]])
            data_aux = []
            index_min = Data[i+1][1][2]
            cycle = Data[i+1][1][0]
        if i == (len(Data)-1):
            data.append([data_aux, [cycle, index_min]])
    return data




def toAh (data_current , data_vol, data_time):
    size = int(len(data_current))
    com = 60*6
    Ah_cap = []
    Energy_cap = []
    Time_cap = []
    Ah = []
    Energy = []
    Time = []

    for i in range(size):
        sum = np.sum(data_current[i][0][:])
        if sum != 0:
            Ah_aux = np.sum(data_current[i][0][:])/com
            Ah_cap = np.append(Ah_cap,Ah_aux)
            Energy_aux = np.sum([data_current[i][0][j]*data_vol[i][0][j]*10 for j in range(len(data_vol[i][0][:]))])
            Energy_cap = np.append(Energy_cap, Energy_aux)
            Time_aux = data_time[i][0][0]/3600
            Time_cap = np.append(Time_cap,Time_aux)
    for i in range(int(len(Ah_cap)/2)):
        a = 2*i
        b = 2*i+1

        Ah = np.append(Ah,np.mean((Ah_cap[a],Ah_cap[b])))
        Energy = np.append(Energy, np.mean((Energy_cap[a], Energy_cap[b])))
        Time = np.append(Time, np.mean((Time_cap[a], Time_cap[b])))

    return [Ah_cap,Energy_cap, Time_cap, Ah,Energy,Time]



def PLOT (data_x,data_y,ini,fin,fig_ini):
    if fin > len(data_x):
        print('Excede el numero de gráficos')
    else:
        for i in range((fin-ini)+1):
            plt.figure(fig_ini + i)
            i = ini+i
            plt.plot(data_x[i],data_y[i][0])



def Group_label (Data):
    "This function groups the data corresponding to the same cycle"
    "The output should be call data[0][0][0][0]"
    "the first index walk across the cycles"
    "The second index change between data or the min index in the original data for this cycle"
    "the third wal across the own cycle and get data"
    "the last index is for change between data and tipe (charge, discharge, etc)"
    data = []
    data_aux = []
    index_min = Data[0][1][2]
    cycle = Data[0][1][0]
    for i in range(len(Data)):
        if Data[i][1][0] == cycle:
            data_aux.append(Data[i])
        if i + 1 <= (len(Data) - 1) and Data[i + 1][1][0] != cycle:
            data.append([data_aux, [cycle, index_min]])
            data_aux = []
            index_min = Data[i + 1][1][2]
            cycle = Data[i + 1][1][0]
        if i == (len(Data) - 1):
            data.append([data_aux, [cycle, index_min]])
    return data



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







def Get_parameters1(Data ,energy,time,model, key):
    "Get the first feature of the data "
    featu = []
    label = []
    for i in range(len(Data)):
        for j in range(len(Data[i][0])):
            if key in str(Data[i][0][j][1][1]):
                valid = Valid_data(Data[i][0][j][0])
                if valid[0] == True:
                    der_mean = np.mean(np.diff(Data[i][0][j][0][:valid[1]])/np.diff(energy[i][j][:valid[1]]))  #derivada promedio
                    energy_ini = energy[i][j][0]  #Energia en el punto inicial
                    dur_norm = (time[i][0][j][0][valid[1]]-time[i][0][j][0][0])/(time[i][0][j][0][-1]-time[i][0][j][0][0])  #Duracion normalizada
                    Ah = model.predict(np.reshape(time[i][0][j][0][0]/3600,(-1,1)))[0][0]   #Ah de la caracteristica
                    featu.append([der_mean,energy_ini,dur_norm])
                    label = np.append(label,Ah)
    featu = np.reshape(featu,(len(featu),3))
    return [featu,label]


def ChangeLabel(Data_label, n):
    maxi = max(Data_label)
    mini = min(Data_label)
    range = maxi - mini
    size = range/n
    labels = [int((i-mini)/size) for i in Data_label]
    labels[0] = labels[0]-1
    return labels













# t2 = toArray2(data[0][0][0],2,2,'(after random walk discharge)',['charge (after random walk discharge)','reference charge'])
# voltage2 = toArray2(data[0][0][0],4,2,'(after random walk discharge)',['(after random walk discharge)','reference charge'])
# current2 = toArray2(data[0][0][0],5,2,'(after random walk discharge)',['(after random walk discharge)','reference charge'])
# temp2 = toArray2(data[0][0][0],6,2,'(after random walk discharge)',['(after random walk discharge)','reference charge'])







