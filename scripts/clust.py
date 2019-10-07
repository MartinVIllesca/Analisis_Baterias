from Data import *

t = Sep_Array(data[0][0][0],2,'reference discharge',10)
voltage = Sep_Array(data[0][0][0],4,'reference discharge',10)
current = Sep_Array(data[0][0][0],5,'reference discharge',10)
temp = Sep_Array(data[0][0][0],6,'reference discharge',10)


t2 = toArray2(data[0][0][0],2,1000,'(after random walk discharge)',['charge (after random walk discharge)','reference charge'])
voltage2 = toArray2(data[0][0][0],4,1000,'(after random walk discharge)',['(after random walk discharge)','reference charge'])
current2 = toArray2(data[0][0][0],5,1000,'(after random walk discharge)',['(after random walk discharge)','reference charge'])
temp2 = toArray2(data[0][0][0],6,1000,'(after random walk discharge)',['(after random walk discharge)','reference charge'])

t3 = Sep_Array(data[0][0][0],2,'reference discharge',1000)
voltage3 = Sep_Array(data[0][0][0],4,'reference discharge',1000)
current3 = Sep_Array(data[0][0][0],5,'reference discharge',1000)
temp3 = Sep_Array(data[0][0][0],6,'reference discharge',1000)
date3 = Sep_Array(data[0][0][0],7,'reference discharge',1000)

Vol = Group(voltage3)
Curr = Group(current3)
Time = Group(t3)

#Linear regression
Ah = toAh(Curr,Vol, Time)
model = LinearRegression().fit(np.reshape(Ah[5],(22,1)), np.reshape(Ah[3],(22,1)))
tt2 = np.arange(0,4000)
val2 = model.predict(np.reshape(tt2,(len(tt2),1)))

# plt.figure(1)
# plt.plot(Ah[2],Ah[0],'o')
# plt.figure(2)
# plt.plot(Ah[2],Ah[1],'o')
plt.figure(3)
plt.plot(Ah[5],Ah[3],'o',tt2,val2)
plt.savefig('fig1.png')
# plt.figure(4)
# plt.plot(Ah[5],Ah[4],'o')

V = Group(voltage2)
I = Group(current2)
E = Energy(V,I,10)

# #plot in the  beginning
# plt.figure(5)
# plt.subplot(121)
# plt.plot(E[30],V[30][0])
# plt.subplot(122)
# plt.plot(E[30],I[30][0])
#
# # plot in medium
# plt.figure(6)
# plt.subplot(121)
# plt.plot(E[230],V[230][0])
# plt.subplot(122)
# plt.plot(E[230],I[230][0])
#
# #plot final

# plt.figure(7)
# plt.subplot(121)
# plt.plot(E[430],V[430][0])
# plt.subplot(122)
# plt.plot(E[430],I[430][0])
#
# plt.show()

# plt.figure(5)
# plt.plot(E[822],V[822][0])

V_test = Group_label(voltage2)
Time_test = Group_label(t2)
energy_lab = Energy_label(V_test,E)
data_features = Get_parameters1(V_test,energy_lab,Time_test,model,'discharge (random walk)')
features = data_features[0]
labels = data_features[1]

fig = plt.figure()
ax = fig.add_subplot(111,projection ='3d')
ax.scatter(features[:,0],features[:,1],features[:,2],c='r',marker='o')
ax.set_xlabel('Promedio derivada')
ax.set_ylabel('energia_inicial')
ax.set_zlabel('duracion')
plt.savefig('fig2.png')

labels = ChangeLabel(labels,3)

print(labels)
