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


#Pregunta 1
def V_norm(x):
    return x**(-6) - exp(-x)

def V_norm_D(x):
    return exp(-x) - 6*x**(-7)

def V_norm_mod(x):
    return x**(-6) - exp(-x) + 0.1

"""Primer item"""
print("################################################################################")
print("Pregunta 1")
print("Item 1")
graficador_a(V_norm,1,5,"Potencial normalizado\nV(x)","V(x)","x",-0.15,0.15)

"""Para poder obtener el valor minimo de la función, una opcion es igualar su
primera derivada a cero"""

solucion_1, error_1 = metodo_biseccion(V_norm_D,1,2,1000,1e-06)
print("El valor que minimiza al potencial de Buckingham es:",solucion_1)
K_1 = range(1,len(error_1)+1)
ajuste_error(K_1,error_1, "Error: M. Biseccion\nV(x) = minimo valor", "Linearizacion")
print("\n")

"""existen dos puntos para los cuales es potencial toma el valor de -0.1, estos
puntos estan en la proximidad de 1 y 2 respectivamente."""
print("Item 2")
solucion_2, error_2 = metodo_newtonraphson(V_norm_mod,V_norm_D,1,1000,1e-06)
print("El potencial de Buckingham es -0.1 en x =",solucion_2)
K_2 = range(1,len(error_2)+1)
ajuste_error(K_2,error_2, "Error: M. Newton-R\nV(x) = -0.1", "Linearizacion")
print("\n")
solucion_3, error_3 = metodo_newtonraphson(V_norm_mod,V_norm_D,2,1000,1e-06)
print("El potencial de Buckingham es -0.1 en x =",solucion_3)
K_3 = range(1,len(error_3)+1)
ajuste_error(K_3,error_3, "Error: M. Newton-R\nV(x) = -0.1", "Linearizacion")
print("\n")



#Pregunta 2
print("################################################################################")
print("Pregunta 2")
print("Item 2")
G = 6.674*1e-11
M = 5.974*1e24
m = 7.248*1e22
R = 3.844*1e08
w = 2.662*1e-06

def ecuacion(x):
    return G*( M*((R-x)**2) - m*(x**2) )-( ( (R-x)**2 )*(x**3) )*(w**2)
def D_ecuacion(x):
    return -2*G*( M*(R-x) + m*x )-( 2*(R-x)*(x**3) + 3*(x**2)*((R-x)**2) )*(w**2)
#graficador_a(ecuacion,0,4*1e08,"Ecuacion a resolver","V(x)","x",-1,1000)
solucion, error = metodo_newtonraphson(ecuacion,D_ecuacion,10000,1000,1e-06)
print("El punto L1 se encuentra a x metros de la tierra, donde x = ",solucion)
print("\n")



#Pregunta 3
print("################################################################################")
print("Pregunta 3")
print("Item 2")
print("El sistema de ecuacion a solucionar es el siguiente:")
print("\n")
print("T_1 - 5*a = 5*g")
print("-T_1 + T_2 - 8*a = 8*g")
print("-T_2 + T_3 -10*a = 10*g*cos(45)*0.2-10*g*sin(45)")
print("-T_3 -15*a = 15*g*cos(45)*0.8-15*g*sin(45)")
print("\n")

A = zeros((4,4), dtype = float)
C = zeros(4, dtype = float)

g = constants.g

A[0] = [1,0,0,-5]
A[1] = [-1,1,0,-8]
A[2] = [0,-1,1,-10]
A[3] = [0,0,-1,-15]
C = [5*g,8*g,(10*g*sin(pi/4)*0.2-10*g*cos(pi/4)),(15*g*sin(pi/4)*0.8-15*g*cos(pi/4))]
print("Entonces la matriz A:")
print(A)
print("Y la matriz C:")
print(C)
print("\n")
print("Metodo de Eliminacion de Gauss")
x = E_Gauss(A, C)
print("los valores hallados para [T_1,T_2,T_3,a] :",x)
print("Donde T_1,T_2,T_3 son las tensiones en Newtons y a es la aceleracion del sistema en m/s^2")
print("\n")


#Pregunta 4
print("################################################################################")
print("Pregunta 4")
print("Item 2")
print("\n")
Q = array([0.04,0.24,0.69,0.13,0.82,2.38,0.31,1.95,5.66])
D = array([0.3,0.6,0.9,0.3,0.6,0.9,0.3,0.6,0.9])
S = array([0.001,0.001,0.001,0.01,0.01,0.01,0.05,0.05,0.05])

Y = copy(Q)
X_1 = copy(D)
X_2 = copy(S)

for i in range(len(Q)):
    Y[i] = log(Q[i])
    X_1[i] = log(D[i])
    X_2[i] = log(S[i])

alpha_0,alpha_1,alpha_2 = ajuste_multivariable(Y,X_1,X_2)

print("Los valores de alpha_0, alpha_1, alpha_2 son:", alpha_0,alpha_1,alpha_2)

def fun(x, y):
    return alpha_0*(x**(alpha_1))*(y**(alpha_2))

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
x = linspace(0.0, 1.0, 20)
y = linspace(0,0.08,20)
X, Y = meshgrid(x, y)
zs = array([fun(x,y) for x,y in zip(ravel(X), ravel(Y))])
Z = zs.reshape(X.shape)

ax.plot_wireframe(X, Y, Z, label = "Ajuste")
ax.scatter(D, S, Q, color= "r", marker = "o", label = "Datos Ex.")
ax.set_xlabel('Diametro [m]')
ax.set_ylabel('inclinacion')
ax.set_zlabel('Flujo [m^2/s]')
ax.set_title("Ajuste multivariable")
ax.legend(loc = "best")
plt.show()

print("################################################################################")
