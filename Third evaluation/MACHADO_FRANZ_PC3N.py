#! /usr/bin/python3
from numpy import *
import matplotlib.pyplot as plt # from pylab import plot,show
import warnings
warnings.filterwarnings("ignore")

#################################### Fin del encabezado
from math import *
from MACHADO_modulo import *
################################################################################
print("================================================================================")
print("Problema 1")

print("Item a)")

def f1(x1, x2, x3, t):
	return -10*x1 + 10*x2

def f2(x1, x2, x3, t):
	return 28*x1 - x2 - x1*x3

def f3(x1, x2, x3, t):
	return -2.6667*x3 + x1*x2

t_0 = 0
t_final = 30
x1_0 = 5
x2_0 = 5
x3_0 = 5
h = 0.0001

t,x,y,z = rk4_completo3(x1_0, x2_0, x3_0, t_0, t_final, h, f1, f2, f3)

#plt.plot(t,x)
#plt.plot(t,y)
#plt.plot(t,z)
#plt.plot(x,y)
#plt.plot(x,z)
#plt.show()

print("Se resolvio el sistema de ecuaciones mediante el metodo RK4")

print("Item b)")

fig, axes = plt.subplots(3,1)
fig.suptitle("x,y,z vs. t", fontsize = 15)
axes[0].plot(t,x, "r-",markersize = 1.5, label = "x(t)" )
axes[1].plot(t,y, "g-",markersize = 1.5, label = "y(t)" )
axes[2].plot(t,z, "b-",markersize = 1.5, label = "z(t)" )

for i in range(3):
	#axes[i].set_title("Ajuste Lineal", fontsize=15)
	#plt.ylim(-2.5,2.5)
	axes[i].minorticks_on()
	axes[i].grid(which='minor', linestyle=':', linewidth='0.5', color='black')
	axes[i].grid(True)
	axes[i].legend(loc = 'best')
axes[0].set_ylabel("x(t)", fontsize=10)
axes[1].set_ylabel("y(t)", fontsize=10)
axes[2].set_ylabel("z(t)", fontsize=10)
axes[2].set_xlabel("t", fontsize = 10)
plt.show()

print("Se grafico las soluciones x,y,z en funcion de t")

print("Item c)")

fig, axes = plt.subplots(1,2)
#fig.suptitle("", fontsize = 15)
axes[0].plot(x,y, "r-",linewidth='0.5', label = "x vs y" )
axes[1].plot(x,z, "b-",linewidth='0.5', label = "x vs z" )


for i in range(2):
	#axes[i].set_title("Ajuste Lineal", fontsize=15)
	#plt.ylim(-2.5,2.5)
	axes[i].minorticks_on()
	axes[i].grid(which='minor', linestyle=':', linewidth='0.5', color='black')
	axes[i].grid(True)
	axes[i].legend(loc = 'best')
axes[0].set_title("Proyeccion xy", fontsize=15)
axes[0].set_ylabel("y", fontsize=10)
axes[0].set_xlabel("x", fontsize = 10)
axes[1].set_title("Proyeccion xz", fontsize=15)
axes[1].set_ylabel("z", fontsize=10)
axes[1].set_xlabel("x", fontsize = 10)
plt.show()

print("Se grafico la proyeccion de la solucion x,y,x en el plano xy y zx")

print("================================================================================")
print("Problema 2")

print("Itema b)")
g = 9.80665 #m/s^2
l = 0.4 #m


def f1(x1, x2, x3, x4, t):
	return x3

def f2(x1, x2, x3, x4, t):
	return x4

def f3(x1, x2, x3, x4, t):
	return -(x3**2*sin(2*x1-2*x2) + 2*x4**2*sin(x1-x2) + (g/l)*(sin(x1-2*x2) + 3*sin(x1)))/(3-cos(2*x1-2*x2))

def f4(x1, x2, x3, x4, t):
	return (4*x3**2*sin(x1-x2) + x4**2*sin(2*x1-2*x2) + 2*(g/l)*(sin(2*x1-x2) - sin(x2)))/(3-cos(2*x1-2*x2))



t_0 = 0
t_final = 100
x1_0 = pi/2
x2_0 = pi/2
x3_0 = 0
x4_0 = 0
h = 0.001

t, theta_1, theta_2, omega_1, omega_2 = rk4_completo4(x1_0, x2_0, x3_0, x4_0,t_0, t_final, h, f1, f2, f3, f4)

T = []
V = []
E = []
x1 = []
y1 = []
x2 = []
y2 = []

for i in range(len(t)):

	a = 0.5*(l**2)*( (omega_2[i])**2 + 2*(omega_1[i])*(omega_2[i])*cos(theta_1[i] - theta_2[i]) + 2*(omega_1[i])**2 )
	b = -g*l*(2*cos(theta_1[i]) + cos(theta_2[i]))
	T.append(a)
	V.append(b)
	E.append(a+b)
	x1.append(l*sin(theta_1[i]))
	y1.append(-l*cos(theta_1[i]))
	x2.append(l*sin(theta_1[i]) + l*sin(theta_2[i]))
	y2.append(-l*cos(theta_1[i]) - l*cos(theta_2[i]))



fig, axes = plt.subplots()
axes.plot(x1,y1,"-",linewidth='0.5', label = "Masa 1" )
axes.plot(x2,y2,"-",linewidth='0.5', label = "Masa 2" )
axes.plot([0],[0],"ko",markersize = 3, label = "Origen")
axes.set_title("Pendulo doble", fontsize=15)
#plt.ylim(-2.5,2.5)
axes.minorticks_on()
axes.grid(which='minor', linestyle=':', linewidth='0.5', color='black')
axes.grid(True)
axes.legend(loc = 'best')
axes.set_ylabel("y [m]", fontsize=10)
axes.set_xlabel("x [m]", fontsize=10)
#plt.yscale("log")
plt.show()

print("Se resolvio el sistema de ecuaciones mediante el metodo RK4")

print("Item c")

fig, axes = plt.subplots(3,1)
fig.suptitle("T,V,E vs. t", fontsize = 15)
axes[0].plot(t,T, "r-",linewidth='0.7', label = "T(t)" )
axes[1].plot(t,V, "g-",linewidth='0.7', label = "V(t)" )
axes[2].plot(t,E, "b-",linewidth='0.7', label = "E(t)" )

for i in range(3):
	axes[i].minorticks_on()
	axes[i].grid(which='minor', linestyle=':', linewidth='0.5', color='black')
	axes[i].grid(True)
	axes[i].legend(loc = 'best')
axes[0].set_ylabel("T(t) [J]", fontsize=10)
axes[1].set_ylabel("V(t) [J]", fontsize=10)
axes[2].set_ylabel("E(t) [J]", fontsize=10)
axes[2].set_xlabel("t [s]", fontsize = 10)
plt.ylim(-0.5,0.5)
plt.show()

print("Se grafico las soluciones T,V,E en funcion de t")

print("Como podemos apreciar, la energia total en la grafica anterior, en escalas apreciables se mantien constante. Pero si hacemos un zoom podemos ver que esto no es asi, esto es principalmente atribuido al error en nuestros calulos, los cuales son muy pequeños es por eso que el orden en que se aprecia este cambio es de 10^-7 ")


fig, axes = plt.subplots()
axes.plot(t,E,"-",linewidth='0.5', label = "E vs t" )
axes.set_title("Energia vs tiempo", fontsize=15)
#plt.ylim(-2.5,2.5)
axes.minorticks_on()
axes.grid(which='minor', linestyle=':', linewidth='0.5', color='black')
axes.grid(True)
axes.legend(loc = 'best')
axes.set_ylabel("E(t)) [J]", fontsize=10)
axes.set_xlabel("t [s]", fontsize=10)
#plt.yscale("log")
plt.show()



print("================================================================================")
print("Problema 3")

rho = 6.22*(1e28)  #m^-3
k_b = 1.3806488*(1e-23) #j*k^-1
V = 1e-04 #m^3
theta_D = 428 #K

def Constante1(T):
	return 9*V*rho*k_b*(T/theta_D)**3

def Constante2(T):
	return (theta_D/T)

def funcion(x):
	return ((x**4)*(exp(x)))/(exp(x)-1)**2

print("Item a)")
cero = 1e-15
T = 30 #K

print("Trapecio n = 100")
X = linspace(cero,Constante2(30),101) #array con 101 puntos, 100 intervalos
integral = M_trapecio(funcion,X)
print("La solucion en base a este metodo es:", integral*Constante1(T))

print("Simpson 3/8 compuesto")
print("Dado que este metodo necesita de un numero de intervalos multiplo de tres, se procedera a evaluar con 99 y 102 intervalos. Tambien se comparara el valor que se obtiene con 100 intervalos")
X = array([-2,4])
integral = M_simpson38(cero,Constante2(30),99,funcion) # el numero de intervalos debe ser multiplo de tres, es por eso que uso 102 en vez de 100
print("La solucion en base a este metodo es (99 intervalos):", integral*Constante1(T))
X = array([-2,4])
integral = M_simpson38(cero,Constante2(30),100,funcion) # el numero de intervalos debe ser multiplo de tres, es por eso que uso 102 en vez de 100
print("La solucion en base a este metodo es (100 intervalos):", integral*Constante1(T))
X = array([-2,4])
integral = M_simpson38(cero,Constante2(30),102,funcion) # el numero de intervalos debe ser multiplo de tres, es por eso que uso 102 en vez de 100
print("La solucion en base a este metodo es (102 intervalos):", integral*Constante1(T))
print("Es claro notar que con 100 intervalos el resltado es menos preciso comparado con 99 y 102 intervalos.")
print("Item b)")


print("Se grafico Cv vs T para un rango de temperaturas entre [5K;500K]")

print("Item c)")

print("Podemos estimar que para una temperatura de 0K, el valor para la capacidad calorifica deberia ser muy proximo a cero.")

C_v1 = []
C_v2 = []
T_array = []
for T in linspace(5,500,100):
	X = linspace(cero,Constante2(T),101)
	a = M_trapecio(funcion,X)
	b = M_simpson38(cero,Constante2(T),102,funcion)
	C_v1.append( a*Constante1(T))
	C_v2.append( b*Constante1(T))
	T_array.append(T)

fig, axes = plt.subplots()
axes.plot(T_array,C_v1,"--*",markersize = 4, label = "Cv-Trapecio" )
axes.plot(T_array,C_v2,"--o",markersize = 4, label = "Cv-Simpson(3/8)" )
axes.set_title("Capacidad Calorifica vs. Temperatura", fontsize=15)
#plt.ylim(-2.5,2.5)
axes.minorticks_on()
axes.grid(which='minor', linestyle=':', linewidth='0.5', color='black')
axes.grid(True)
axes.legend(loc = 'best')
axes.set_ylabel("Cv(T) [J/K]", fontsize=10)
axes.set_xlabel("T [K]", fontsize=10)
#plt.yscale("log")
plt.show()


#fig, axes = plt.subplots()

#Funcion_lagrange(T_array,C_v2,"Funcion lagrange")
#axes.plot(T_array,C_v2,"--o",markersize = 3, label = "Cv-Simpson(3/8)")
#axes.set_title("Interpolacion", fontsize=15)
#plt.show()

#print("Interpolacion por lagrange")
#print(lagrange_completo(T_array,C_v2,0.0))
