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


#Ejercicio 1
#y' = y*t^3 - 1.5*y
#analiticamente y = exp(0.25*x^4 - 1.5*x)

def funcion(y,t):
	return y*t**3 - 1.5*y


################################################

def solucion_real(t,i):
	return exp(0.25*t**4 - 1.5*t)

fig, axes = plt.subplots()
x = linspace(0,2,4000)
f_x = []
cero = []
for x_i in x:
    y = solucion_real(x_i,1)
    f_x.append(y)

axes.plot(x,f_x , "-",markersize = 1, label = "funcion: y(t)" )

h = 0.1            #En la primera parte usamos como paso a h = 0.1
t_final = 2.0
y0 = 1.0
t0 = 0.0
t, y = Metodo_euler(y0,t0, t_final, h, funcion)
#print(t)
axes.plot(t, y, '--o',markersize = 2 , label='Euler-0.1')

##################################################
h = 0.1            #En la primera parte usamos como paso a h = 0.01
t_final = 2.0
y0 = 1.0
t0 = 0.0
t, y = Metodo_Heund(y0,t0, t_final, h, funcion)   #nos da la lista de t y y, con la que graficamos la solucion aproximada
#print(t)
axes.plot(t, y, '--o',markersize = 2 , label='Heund-0.1')

##################################################
##################################################
h = 0.1            #En la primera parte usamos como paso a h = 0.01
t_final = 2.0
y0 = 1.0
t0 = 0.0
t, y = Metodo_Heund_correccion(y0,t0, t_final, h, funcion)   #nos da la lista de t y y, con la que graficamos la solucion aproximada
#print(t)
axes.plot(t, y, '-',markersize = 2 , label='Heund_correccion-0.1')

##################################################
##################################################
h = 0.1            #En la primera parte usamos como paso a h = 0.01
t_final = 2.0
y0 = 1.0
t0 = 0.0
t, y = Metodo_Ralston(y0,t0, t_final, h, funcion)   #nos da la lista de t y y, con la que graficamos la solucion aproximada
#print(t)
axes.plot(t, y, '--o',markersize = 2 , label='Ralston-0.1')

##################################################
##################################################
h = 0.1            #En la primera parte usamos como paso a h = 0.01
t_final = 2.0
y0 = 1.0
t0 = 0.0
t, y = Metodo_rk4(y0,t0, t_final, h, funcion)   #nos da la lista de t y y, con la que graficamos la solucion aproximada
#print(t)
axes.plot(t, y, '--o',markersize = 2 , label='Rk4-0.1')

##################################################
#Esta parte es usada para graficar la solucion real y ver la comparacion
axes.minorticks_on()
axes.grid(which='minor', linestyle=':', linewidth='0.5', color='black')
axes.grid(True)
axes.legend(loc = 'best')
axes.set_ylabel("y(t)", fontsize=10)
axes.set_xlabel("t ", fontsize=10)
axes.set_title("Comparacion y(t)", fontsize=15)
#plt.yscale("log")
plt.show()

#Ejercicio 2

def f1(y1,y2):
	return 999*y1 + 1999*y2

def f2(y1,y2):
	return -1000*y1 - 2000*y2

def solx1(x):
	return (exp(-1000*x)*(3998*exp(999*x)-2999))/999

def solx2(x):
	return (exp(-1000*x)*(-2000*exp(999*x)+2999))/999

te1,ye1,ye2 = Metodo_euler_sistema_ex(1.0,1.0,0,0.2,0.0005,f1,f2)
#print(te1,ye1,ye2)

ti1,yi1,yi2 = Metodo_euler_sistema_im(1.0,1.0,0.0,0.2,0.0005)
#print(ti1,yi1,yi2)

fig, axes = plt.subplots()
x = linspace(0,0.2,500)
f_x = []
cero = []
for x_i in x:
    y = solx1(x_i)
    f_x.append(y)

axes.plot(x,f_x , "-",markersize = 1, label = "funcion: y(t)" )
axes.plot(te1, ye1, '--o',markersize = 2 , label='explicito')
axes.plot(ti1, yi1, '--o',markersize = 2 , label='implicito')

axes.minorticks_on()
axes.grid(which='minor', linestyle=':', linewidth='0.5', color='black')
axes.grid(True)
axes.legend(loc = 'best')
axes.set_ylabel("y(t)", fontsize=10)
axes.set_xlabel("t ", fontsize=10)
axes.set_title("Comparacion y(t)", fontsize=15)
#plt.yscale("log")
#plt.ylim(0,10)
plt.show()

#Ejercicio 3

def funcion_3(y,t):
	return -0.5*y + exp(-t)

def solucion_real_3(t,i):
	return 12*exp(-0.5*t) - 2*exp(-t)

fig, axes = plt.subplots()
x = linspace(2,3,500)
f_x = []
cero = []
for x_i in x:
    y = solucion_real_3(x_i,1)
    f_x.append(y)

axes.plot(x,f_x , "-",markersize = 1, label = "funcion: y(t)" )

t,y = Metodo_Heund_modificado(5.222138,4.143883,2.0, 3.0, 0.5, funcion_3)
t1,y1 = Metodo_Heund_correccion(4.143883,2.0, 2.5, 0.5, funcion_3)

axes.plot(t, y, '--o',markersize = 6 , label='Heund modificado')
axes.plot(t1, y1, '--o',markersize = 4 , label='Heund correccion')

axes.minorticks_on()
axes.grid(which='minor', linestyle=':', linewidth='0.5', color='black')
axes.grid(True)
axes.legend(loc = 'best')
axes.set_ylabel("y(t)", fontsize=10)
axes.set_xlabel("t ", fontsize=10)
axes.set_title("Comparacion y(t)", fontsize=15)
#plt.yscale("log")
#plt.ylim(0,10)
plt.show()
