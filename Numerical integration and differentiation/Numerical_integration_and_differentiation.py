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

print("Simpson 3/8")
X = array([-2,4])
integral = M_simpson_simple(fun1,X,"3/8")
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
