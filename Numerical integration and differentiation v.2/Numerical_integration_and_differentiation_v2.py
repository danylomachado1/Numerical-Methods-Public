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
#Tarea 7
"""Ejercicio 1"""
"""La antiderivada obtenida es x^6/3 - x^4 - x^2/2 + x + C, al ser evaluada se obtiene 1104"""

def fun1(x):
    return 1-x-4*x**3+2*x**5

def fun2(x):
    return 0.2 + 25*x -200*x**2 + 675*x**3 - 900*x**4 + 400*x**5

print("Ejercicio 1")
print("La solucion analitica es: 1104")

print("Trapecio simple")
X = linspace(-2,4,2)
integral = M_trapecio(fun1,X)
print("La solucion en base a este metodo es:", integral)

print("Trapecio n = 2")
X = linspace(-2,4,3)
integral = M_trapecio(fun1,X)
print("La solucion en base a este metodo es:", integral)

print("Trapecio n = 4")
X = linspace(-2,4,5)
integral = M_trapecio(fun1,X)
print("La solucion en base a este metodo es:", integral)

print("Simpson 1/3")
X = array([-2,4])
integral = M_simpson_simple(fun1,X,"1/3")
print("La solucion en base a este metodo es:", integral)

print("Simpson 1/3 compuesto")
X = array([-2,4])
integral = M_simpson13(-2,4,333,fun1)
print("La solucion en base a este metodo es:", integral)

print("Simpson 3/8")
X = array([-2,4])
integral = M_simpson_simple(fun1,X,"3/8")
print("La solucion en base a este metodo es:", integral)

print("Simpson 3/8 compuesto")
X = array([-2,4])
integral = M_simpson38(-2,4,333,fun1)
print("La solucion en base a este metodo es:", integral)
print("\n")

print("Ejercicio 2")

print("Trapecio datos")
t = array([1,2,3.25,4.5,6,7,8,8.5,9,10])
v = array([5,6,5.5,7,8.5,8,6,7,7,5])
integral = M_trapecio_datos(t,v)
print("La solucion en base a este metodo es:", integral)
print("\n")

print("Ejercicio 3")
"""La antiderivada obtenida es (200*x^6)/3 - 180*x^5 + (675*x^4)/4 - (200*x^3)/3 + (25*x^2)/2 + x/5 + C, al ser evaluada se obtiene 1.640533333333"""
print("La solucion analitica es: 1.6405333333333")

print("Trapecio 2 puntos")
X = linspace(0,0.8,2)
integral = M_trapecio(fun2,X)
print("La solucion en base a este metodo es:", integral)

print("Trapecio 3 puntos")
X = linspace(0,0.8,3)
integral = M_trapecio(fun2,X)
print("La solucion en base a este metodo es:", integral)

print("Trapecio 5 puntos")
X = linspace(0,0.8,5)
integral = M_trapecio(fun2,X)
print("La solucion en base a este metodo es:", integral)


print("Interpolacion de Richardson")
def I_Richardson(func,i1,i2):
    X1 = linspace(0,0.8,i1+1)
    X2 = linspace(0,0.8,i2+1)
    integral = M_trapecio(func,X2) + (1/((i2/i1)**2 - 1))*(M_trapecio(func,X2)-M_trapecio(func,X1))
    return integral
print("La solucion en base a este metodo es:", I_Richardson(fun2,2,16) )


#Tarea 8
print("Ejercicio 4")
print("El valor de pi hallado con G-L en 3 puntos es:", GL(fun_A_B,0,1,"3"))
print("El valor de pi hallado con G-L en 4 puntos es:", GL(fun_A_B,0,1,"4"))
print("El valor de pi hallado con G-L en 5 puntos es:", GL(fun_A_B,0,1,"5"))

print("El valor de pi hallado con G-R-L en 3 puntos es:", GRL(fun_A_B,0,1,"3"))
print("El valor de pi hallado con G-R-L en 4 puntos es:", GRL(fun_A_B,0,1,"4"))
print("El valor de pi hallado con G-R-L en 5 puntos es:", GRL(fun_A_B,0,1,"5"))

print("El valor de pi hallado con G-L-L en 3 puntos es:", GLL(fun_A_B,0,1,"3"))
print("El valor de pi hallado con G-L-L en 4 puntos es:", GLL(fun_A_B,0,1,"4"))
print("El valor de pi hallado con G-L-L en 5 puntos es:", GLL(fun_A_B,0,1,"5"))


print("Ejercicio 5")


t = array([200,202,204,206,208,210])
theta = array([0.75,0.72,0.70,0.68,0.67,0.66])
r = array([5120,5370,5560,5800,6030,6240])

#theta = theta*180/pi
#v_r = r derivada
#v_theta = r * theta derivada

#a_r = r segunda - r * theta primera al cuadrado
#a_theta = r * theta segunda + 2*r deirvada * theta derivadas
# se debe hallar r derivada primera y segunda al ugual para theta.

Pd_r = zeros(len(t), dtype = float)
Pd_theta = zeros(len(t), dtype = float)

Sd_r = zeros(len(t), dtype = float)
Sd_theta = zeros(len(t), dtype = float)

for i in range(len(t)):
    if i < 2:
        Pd_r[i] = P_derivada_forward(r,t,i)
        Pd_theta[i] =  P_derivada_forward(theta,t,i)
        Sd_r[i] = S_derivada_forward(r,t,i)
        Sd_theta[i] =  S_derivada_forward(theta,t,i)
    if i<4 and i>1:
        Pd_r[i] = P_derivada_central(r,t,i)
        Pd_theta[i] =  P_derivada_central(theta,t,i)
        Sd_r[i] = S_derivada_central(r,t,i)
        Sd_theta[i] =  S_derivada_central(theta,t,i)
    if i > 3:
        Pd_r[i] = P_derivada_backward(r,t,i)
        Pd_theta[i] =  P_derivada_backward(theta,t,i)
        Sd_r[i] = S_derivada_backward(r,t,i)
        Sd_theta[i] =  S_derivada_backward(theta,t,i)

#print(Pd_r)
#print(Pd_theta)

v_r = zeros(len(t), dtype = float)
v_theta = zeros(len(t), dtype = float)

a_r = zeros(len(t), dtype = float)
a_theta = zeros(len(t), dtype = float)

for i in range(len(t)):
    v_r[i] = Pd_r[i]
    v_theta[i] = r[i] * Pd_theta[i]
    a_r[i] = Sd_r[i] - r[i]*((Pd_theta[i])**2)
    a_theta[i] = r[i]*Sd_theta[i] + 2*Pd_r[i]*Pd_theta[i]

print("v_r :", v_r)
print("v_theta :", v_theta)
print("a_r :", a_r)
print("a_theta :", a_theta)

fig = plt.figure(figsize=(6,6))
ax = fig.add_subplot(111,projection = "polar")
ax.plot(theta,r,"o",label = "posicion-polar")
ax.quiver(theta[0],r[0],v_r[0],v_theta[0],color = "red", label= "velocidad")
ax.quiver(theta[1],r[1],v_r[1],v_theta[1],color = "red")
ax.quiver(theta[2],r[2],v_r[2],v_theta[2],color = "red")
ax.quiver(theta[3],r[3],v_r[3],v_theta[3],color = "red")
ax.quiver(theta[4],r[4],v_r[4],v_theta[4],color = "red")
ax.quiver(theta[5],r[5],v_r[5],v_theta[5],color = "red")

ax.quiver(theta[0],r[0],a_r[0],a_theta[0],color = "blue", label= "aceleracion")
ax.quiver(theta[1],r[1],a_r[1],a_theta[1],color = "blue")
ax.quiver(theta[2],r[2],a_r[2],a_theta[2],color = "blue")
ax.quiver(theta[3],r[3],a_r[3],a_theta[3],color = "blue")
ax.quiver(theta[4],r[4],a_r[4],a_theta[4],color = "blue")
ax.quiver(theta[5],r[5],a_r[5],a_theta[5],color = "blue")
ax.set_rmin(5000)
ax.set_rmax(6500)

plt.legend()
plt.show()


#plt.polar(theta,r,"o",label = "posicion-polar")
#plt.polar(v_theta,v_r,"*",label = "velocidad-polar")
#plt.polar(a_theta,a_r,"+",label = "aceleracion-polar")
#plt.quiver()
#plt.legend()
#plt.ylim(5000,6200)
#plt.show()
