#! /usr/bin/python3
from numpy import *
import matplotlib.pyplot as plt # from pylab import plot,show
import warnings
warnings.filterwarnings("ignore")

#################################### Fin del encabezado

import scipy.constants as constants #para la constante g
from mpl_toolkits.mplot3d import Axes3D
from math import *
from MACHADO_modulo import *
################################################################################

def Funcion_1(x):
    return -53.4523*exp(-0.1*x) + 73.4523*exp(0.1*x) + 20


T_1 , x_1, h_1 = Dif_finitas_1(0,10,20,40,200,20)   #6 numero de puntos, 5 intervalos


fig, axes = plt.subplots()
x = linspace(0,10,4000)
f_x = []
cero = []
for x_i in x:
    y = Funcion_1(x_i)
    f_x.append(y)

axes.plot(x,f_x , "-",markersize = 1, label = "Solucion: f(t)" )
axes.plot(x_1, T_1, '--o',markersize = 2 , label='Solucion1')

axes.minorticks_on()
axes.grid(which='minor', linestyle=':', linewidth='0.5', color='black')
axes.grid(True)
axes.legend(loc = 'best')
axes.set_ylabel("T(x)", fontsize=10)
axes.set_xlabel("x ", fontsize=10)
axes.set_title("Diferencias Finitas 1", fontsize=15)
plt.yscale("log")
plt.show()

E_r = []
h_r = []

for i in range(4,10):
    T_1 , x_1, h_1 = Dif_finitas_1(0,10,i,40,200,20)
    #print(x_1,T_1)
    err = []
    for i in range(len(x_1)):
        err.append(T_1[i] - Funcion_1(x_1[i]))
    E_r.append(abs(max(err)))
    h_r.append(h_1)
#print(h_r,E_r)

lE_r = []
lh_r = []

for i in range(len(E_r)):
    lE_r.append(log(E_r[i]))
    lh_r.append(log(h_r[i]))

lE_r = array(lE_r)
lh_r = array(lh_r)

a,b,syx,r2 = Regresion(lh_r,lE_r)
x = linspace(0,1.5,100)
f_l = []
for x_i in x:
    f_l.append(a + b*x_i)
print("La pendiente es: ",b)

fig, axes = plt.subplots()
axes.plot(lh_r,lE_r, '--o',markersize = 3 , label='Error global, refinamiento')
axes.plot(x,f_l, "-",markersize = 3, label = "Ajuste" )
axes.minorticks_on()
axes.grid(which='minor', linestyle=':', linewidth='0.5', color='black')
axes.grid(True)
axes.legend(loc = 'best')
axes.set_ylabel("Error, norma infinita", fontsize=10)
axes.set_xlabel("h ", fontsize=10)
axes.set_title("Error global, refinamiento", fontsize=15)
#plt.xscale("log")
#plt.yscale("log")
plt.show()




################################################################################
def Funcion_2(x):
    return -36.2354*exp(-0.1*x) + 63.7646*exp(0.1*x) + 40


T_2 , x_2, h_2 = Dif_finitas_2(0,10,20,10,200,40)

fig, axes = plt.subplots()
x = linspace(0,10,4000)
f_x = []
cero = []
for x_i in x:
    y = Funcion_2(x_i)
    f_x.append(y)

axes.plot(x,f_x , "-",markersize = 1, label = "Solucion: f(t)" )
axes.plot(x_2, T_2, '--o',markersize = 2 , label='Solucion1')

axes.minorticks_on()
axes.grid(which='minor', linestyle=':', linewidth='0.5', color='black')
axes.grid(True)
axes.legend(loc = 'best')
axes.set_ylabel("T(x)", fontsize=10)
axes.set_xlabel("x ", fontsize=10)
axes.set_title("Diferencias Finitas 1", fontsize=15)
plt.yscale("log")
plt.show()

E_r = []
h_r = []

for i in range(4,10):
    T_1 , x_1, h_1 = Dif_finitas_2(0,10,i,10,200,40)
    #print(x_1,T_1)
    err = []
    for i in range(len(x_1)):
        err.append(T_1[i] - Funcion_2(x_1[i]))
    E_r.append(abs(max(err)))
    h_r.append(h_1)
#print(h_r,E_r)

lE_r = []
lh_r = []

for i in range(len(E_r)):
    lE_r.append(log(E_r[i]))
    lh_r.append(log(h_r[i]))

lE_r = array(lE_r)
lh_r = array(lh_r)

a,b,syx,r2 = Regresion(lh_r,lE_r)
x = linspace(0,1.5,100)
f_l = []
for x_i in x:
    f_l.append(a + b*x_i)
print("La pendiente es: ",b)


fig, axes = plt.subplots()
axes.plot(lh_r,lE_r, '--o',markersize = 3 , label='Error global, refinamiento')
axes.plot(x,f_l, "-",markersize = 3, label = "Ajuste" )
axes.minorticks_on()
axes.grid(which='minor', linestyle=':', linewidth='0.5', color='black')
axes.grid(True)
axes.legend(loc = 'best')
axes.set_ylabel("Error, norma infinita ", fontsize=10)
axes.set_xlabel("h", fontsize=10)
axes.set_title("Error global, refinamiento", fontsize=15)
#plt.xscale("log")
#plt.yscale("log")
plt.show()



################################################################################

def f1(x1,x2,t):
    return x2

def f2(x1,x2,t):
    return -0.01*(20 - x1)

x1_0 = 40
x2_0 = 200
guess1 = 1
guess2 = 100
t_0 = 0
t_final = 10
h = 1

lista_t, lista_x1, lista_x1a, lista_x1b = Metodo_disparo(x1_0, x2_0,guess1,guess2, t_0, t_final, h, f1, f2)

fig, axes = plt.subplots()
x = linspace(0,10,4000)
f_x = []
cero = []
for x_i in x:
    y = Funcion_1(x_i)
    f_x.append(y)

axes.plot(x,f_x , "-",markersize = 1, label = "Solucion: f(t)" )
axes.plot(lista_t, lista_x1, '--o',markersize = 2 , label='Solucion1')
axes.plot(lista_t, lista_x1a, '--o',markersize = 2, label = 'guess1')
axes.plot(lista_t, lista_x1b, '--o',markersize = 2, label = 'guess2')
axes.minorticks_on()
axes.grid(which='minor', linestyle=':', linewidth='0.5', color='black')
axes.grid(True)
axes.legend(loc = 'best')
axes.set_ylabel("T(x)", fontsize=10)
axes.set_xlabel("x ", fontsize=10)
axes.set_title("Comparacion y(t)", fontsize=15)

plt.show()


################################################################################

A = array([[-4,10],[7,5]])
X = array([1,1])
iter_max = 200
result = Metodo_potencia(A, X, iter_max)
Compara(A, result)

A = array([[1,2,-2],[-2,5,-2],[-6,6,-3]])
X = array([1,1,1])
iter_max = 200
result = Metodo_potencia(A, X, iter_max)
Compara(A, result)
