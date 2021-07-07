#! /usr/bin/python3
from numpy import *
import matplotlib.pyplot as plt # from pylab import plot,show
import warnings
warnings.filterwarnings("ignore")

#################################### Fin del encabezado
from math import *
from MACHADO_modulo import *
################################################################################

################################################################################
print("Problema 1")

print("Item b)")
m = 1
a = 2
def V_potencial(x):
    return x**4

def fun_final(x):
    return sqrt(8)*(1/sqrt(V_potencial(a) - V_potencial(x)))

def fun_A_B(x,A,B):
    return fun_final(((B-A)/2)*x+(A+B)/2)


print("El valor del periodo en segundos hallado con Gauss-Legendre considerando 5 puntos para a = 2m es:", GL(fun_A_B,0,a,"5"))

print("Item c)")
print("Se construyó la gráfica de T vs a para a ∈ [0;2].")

a_array = linspace(0.001,2,200)
T_array = []

for a in a_array:
    T_array.append(GL(fun_A_B,0,a,"5"))

fig, axes = plt.subplots()

axes.plot(a_array,T_array, 'r-',markersize = 1 , label='Metodo: Gauss-Legendre')
axes.minorticks_on()
#axes.grid(which='minor', linestyle=':', linewidth='0.5', color='black')
axes.grid(True)
axes.legend(loc = 'best')
axes.set_ylabel("T [s]", fontsize=10)
axes.set_xlabel("a [m] ", fontsize=10)
axes.set_title("Periodo Vs Amplitud", fontsize=15)
plt.yscale("log")
plt.show()

print("\n")

################################################################################
print("Problema 2")

print("Item b)")
print("Se resolvio el sistema con RK4 para t ∈ [0; 10] para las condiciones iniciales x0 = 1, y0 = 0, g0 = 1 y w0 = 0. Se tomó G = 1, M = 10 y L = 2 y se grafico x vs y (la orbita).")
#g = 9.80665 #m/s^2
g = 1 #
M = 10 #m
L = 2

def r_square(x1,x2):
    return x1**2 + x2**2

def f1(x1, x2, x3, x4, t):
	return x3

def f2(x1, x2, x3, x4, t):
	return x4

def f3(x1, x2, x3, x4, t):
	return -1*g*M*(x1/(r_square(x1,x2)*sqrt(r_square(x1,x2) + (L**2)/4)))

def f4(x1, x2, x3, x4, t):
	return -1*g*M*(x2/(r_square(x1,x2)*sqrt(r_square(x1,x2) + (L**2)/4)))



t_0 = 0
t_final = 10
x1_0 = 1.0
x2_0 = 0.0
x3_0 = 0.0
x4_0 = 1.0
h = 0.001

t, X, Y, W, G = rk4_completo4(x1_0, x2_0, x3_0, x4_0,t_0, t_final, h, f1, f2, f3, f4)

fig, axes = plt.subplots()
axes.plot(Y,X,"r-",linewidth='1', label = "Masa" )

axes.plot([0],[0],"ko",markersize = 3, label = "Origen")
axes.set_title("Orbita", fontsize=15)
#plt.ylim(-2.5,2.5)
axes.minorticks_on()
axes.grid(which='minor', linestyle=':', linewidth='0.5', color='black')
axes.grid(True)
axes.legend(loc = 'best')
axes.set_ylabel("x [m]", fontsize=10)
axes.set_xlabel("y [m]", fontsize=10)
#plt.yscale("log")
plt.show()

print("\n")

################################################################################

print("Problema 3")

delta_t = 5e-4

print("Item a)")
print("Se grafico $\\overline{n}$ vs t para L = 2, considerando ∆x = 0.1. Ademas se realizo el ajuste respectivo donde:")
n_t_1 = Crank_Nicolson(21,2,5.1,5e-4)



n_hat_1 = []
Tiempo_1 = linspace(0.5,5,10)
for i in range(len(n_t_1)):
    n_hat_1.append(sum(n_t_1[i])/21)

X_1,F_X_1 = ajuste_error_final(Tiempo_1,n_hat_1, "$\\overline{n}$, L = 2", "Ajuste","True")

print("Item b)")
print("Se grafico $\\overline{n}$ vs t para L = 4, considerando ∆x = 0.1. Ademas se realizo el ajuste respectivo donde:")
n_t_1 = Crank_Nicolson(41,4,5.1,5e-4)

n_hat_2 = []
Tiempo_2 = linspace(0.5,5,10)
for i in range(len(n_t_1)):
    n_hat_2.append(sum(n_t_1[i])/41)

X_2,F_X_2 = ajuste_error_final(Tiempo_2,n_hat_2, "$\\overline{n}$, L = 4", "Ajuste","True")

print("Item c)")
print("Se grafico $\\overline{n}$ vs t para L = 4 y L = 2, manteniendo el mismo ∆x")
print("Se puede apreciar que la grafica para distintos valores de L es bastante distinta.")
print("Mientras que para L = 2, la suma de las densidades para el neutron disminuyen con el tiempo, para L = 4 esta suma se incrementa.")
print("Despues de graficar para diversos valores de L entre 1 y 6, puedo afirmar que mientras el valor de L se incrementa el valor de A crece al igual que la inversa de 1/α, el cual toma valores negativos para L menores que 3.3.")
print("Existe un valor de L entre 3.1 y 3.3, para el cual el valor de 1/α se hace 0, es decir la suma de las densidades del neutron se mantiene constante con el tiempo. ")
print("Al analizar los valores de n(x,t) para distintos valores de t y L, se puede concluir que existe un maximo para x = 0 para un mismo t.")
fig, axes = plt.subplots()
axes.plot(X_1,F_X_1 , "-",markersize = 1, label = "Ajuste, L = 2" )
axes.plot(Tiempo_1,n_hat_1 , "ko",markersize = 3, label = "$\\overline{n}$,L = 2" )
axes.plot(X_2,F_X_2 , "-",markersize = 1, label = "L = 4" )
axes.plot(Tiempo_2,n_hat_2 , "ko",markersize = 3, label = "$\\overline{n}$,L = 4" )
axes.set_title("$\\overline{n}$ vs tiempo", fontsize=15)
axes.set_ylabel("$\\overline{n}$", fontsize=10)
axes.set_xlabel("tiempo [s]", fontsize=10)
axes.minorticks_on()
axes.grid(which='minor', linestyle=':', linewidth='0.2', color='black')
axes.grid(True)
axes.legend(loc = 'best')
plt.yscale("log")
plt.show()

print("\n")
