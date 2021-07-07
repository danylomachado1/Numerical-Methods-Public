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
'''
maxIter = 1000

lenX = lenY = 24
delta = 6

Ttop = 100
Tbottom = 0
Tleft = 75
Tright = 50

Tguess = 0


colorinterpolation = 50
colourMap = plt.cm.jet #colourMap = plt.cm.coolwarm

X, Y = meshgrid(arange(0, lenX + delta ,delta), arange(0, lenY + delta,delta))

#print(X,Y)
T = empty((len(X), len(Y)))
T.fill(Tguess)

#print(T)

#condiciones de contorno
T[(len(Y)-1):, :] = Ttop
T[:1, :] = Tbottom
T[:, (len(X)-1):] = Tright
T[:, :1] = Tleft

#print(T)

for iteration in range(0, maxIter):
    for i in range(1, len(X)-1):
        for j in range(1, len(Y)-1):
            T[i, j] = 0.25 * (T[i+1][j] + T[i-1][j] + T[i][j+1] + T[i][j-1])



print(T)

plt.title("Mapa de temperatura")
plt.contourf(X, Y, T, colorinterpolation, cmap=colourMap)
plt.colorbar()
plt.show()

################################################################################

lenX = lenY = 24
delta = 6

Ttop = 100
Tbottom = 0
Tleft = 75
Tright = 50

Tguess = 0


colorinterpolation = 50
colourMap = plt.cm.jet #colourMap = plt.cm.coolwarm

X, Y = meshgrid(arange(0, lenX + delta ,delta), arange(0, lenY + delta,delta))

#print(X,Y)
T = empty((len(X), len(Y)))
T.fill(Tguess)

#print(T)

#condiciones de contorno
T[(len(Y)-1):, :] = Ttop
T[:1, :] = Tbottom
T[:, (len(X)-1):] = Tright
T[:, :1] = Tleft

#print(T)

A = zeros(((len(T)-2)**2,(len(T)-2)**2), dtype = float)
C = zeros((len(T)-2)**2, dtype = float)


def ubc(i,j):
    if (i==1 and j==1) :
        return "esquina1"
    elif (i==1 and j==len(Y)-2) :
        return "esquina2"
    elif (i==len(X)-2 and j==len(Y)-2) :
        return "esquina3"
    elif (i==len(X)-2 and j==1) :
        return "esquina4"

    elif (i==1 and (j in range(2,len(Y)-2)) ):
        return "Tbottom"

    elif (i==len(X)-2 and (j in range(2,len(Y)-2)) ):
        return "Ttop"

    elif (j==1 and (i in range(2,len(X)-2)) ):
        return "Tleft"

    elif (j==len(Y)-2 and (i in range(2,len(X)-2)) ):
        return "Tright"
    else:
        return "interior"


A[0] = array([4.,-1.,0.,-1.,0.,0.,0.,0.,0.])
A[1] = array([-1.,4.,-1.,0.,-1.,0.,0.,0.,0.])
A[2] = array([0.,-1.,4.,0.,0.,-1.,0.,0.,0.])
A[3] = array([-1.,0.,0.,4.,-1.,0.,-1.,0.,0.])
A[4] = array([0.,-1.,0.,-1.,4.,-1.,0.,-1.,0.])
A[5] = array([0.,0.,-1.,0.,-1.,4.,0.,0.,-1.])
A[6] = array([0.,0.,0.,-1.,0.,0.,4.,-1.,0.])
A[7] = array([0.,0.,0.,0.,-1.,0.,-1.,4.,-1.])
A[8] = array([0.,0.,0.,0.,0.,-1.,0.,-1.,4.])
C = array([75.,0.,50.,75.,0.,50.,175.,100.,150.])
T_x = zeros((len(T)-2)**2, dtype = float)
T_x = gaussSeidel(A,C,T_x,1e-06)
T_x = T_x[0]
m = 0
for i in range(1, len(X)-1):
    for j in range(1, len(Y)-1):
        T[i,j] = T_x[m]
        m = m+1
print(T)

plt.title("Mapa de temperatura")
plt.contourf(X, Y, T, colorinterpolation, cmap=colourMap)
plt.colorbar()
plt.show()

################################################################################
# 1-neuman y 3-dirichlet

maxIter = 5000

lenX = lenY = 24
delta = 6

Ttop = 100
Tbottom = 0
Tleft = 75
Tright = 50

Tguess = 0


colorinterpolation = 50
colourMap = plt.cm.jet #colourMap = plt.cm.coolwarm

X, Y = meshgrid(arange(0, lenX + delta ,delta), arange(0, lenY + delta,delta))

#print(X,Y)
T = empty((len(X), len(Y)))
T.fill(Tguess)

#print(T)

#condiciones de contorno
T[(len(Y)-1):, :] = Ttop
T[:1, :] = Tbottom
T[:, (len(X)-1):] = Tright
T[:, :1] = Tleft

#print(T)

for iteration in range(0, maxIter):
    for i in range(0, len(X)-1):
        for j in range(1, len(Y)-1):
            if i != 0:
                T[i, j] = 0.25 * (T[i+1][j] + T[i-1][j] + T[i][j+1] + T[i][j-1])
            if i == 0:
                if j != 1 and j!= len(Y)-1:
                    T[i, j] = 0.25 * (T[i+1][j] + T[i-1][j] + 2*T[i][j+1])
                else:
                    T[i, j] = 0.25 * (T[i+1][j] + T[i-1][j] + 2*T[i][j+1])

            #if i == 0:
            #    T[i, j] = 0.25 * (T[i][j+1] + T[i][j-1] + 2*T[i+1][j])



print(T)

plt.title("Mapa de temperatura")
plt.contourf(X, Y, T, colorinterpolation, cmap=colourMap)
plt.colorbar()
plt.show()

################################################################################


lenX = lenY = 24
delta = 6

Ttop = 100
Tbottom = 0
Tleft = 75
Tright = 50

Tguess = 0


colorinterpolation = 50
colourMap = plt.cm.jet #colourMap = plt.cm.coolwarm

X, Y = meshgrid(arange(0, lenX + delta ,delta), arange(0, lenY + delta,delta))

#print(X,Y)
T = empty((len(X), len(Y)))
T.fill(Tguess)

#print(T)

#condiciones de contorno
T[(len(Y)-1):, :] = Ttop
T[:1, :] = Tbottom
T[:, (len(X)-1):] = Tright
T[:, :1] = Tleft

#print(T)

A = zeros(((len(T)-2)**2 + 3,(len(T)-2)**2 +3), dtype = float)
C = zeros((len(T)-2)**2 + 3, dtype = float)

A[0] = array([4.,-1.,0.,-2.,0.,0.,0.,0.,0.,0.,0.,0.])
A[1] = array([-1.,4.,-1.,0.,-2.,0.,0.,0.,0.,0.,0.,0.])
A[2] = array([0.,-1.,4.,0.,0.,-2.,0.,0.,0.,0.,0.,0.])
A[3] = array([-1.,0.,0.,4.,-1.,0.,-1.,0.,0.,0.,0.,0.])
A[4] = array([0.,-1.,0.,-1.,4.,-1.,0.,-1.,0.,0.,0.,0.])
A[5] = array([0.,0.,-1.,0.,-1.,4.,0.,0.,-1.,0.,0.,0.])
A[6] = array([0.,0.,0.,-1.,0.,0.,4.,-1.,0.,-1.,0.,0.])
A[7] = array([0.,0.,0.,0.,-1.,0.,-1.,4.,-1.,0.,-1.,0.])
A[8] = array([0.,0.,0.,0.,0.,-1.,0.,-1.,4.,0.,0.,-1.])
A[9] = array([0.,0.,0.,0.,0.,0.,-1.,0.,0.,4.,-1.,0.])
A[10] = array([0.,0.,0.,0.,0.,0.,0.,-1.,0.,-1.,4.,-1.])
A[11] = array([0.,0.,0.,0.,0.,0.,0.,0.,-1.,0.,-1.,4.])
C = array([75.,0.,50.,75.,0.,50.,75.,0.,50.,175.,100.,150.])


T_x = zeros((len(T)-2)**2+3, dtype = float)
T_x = gaussSeidel(A,C,T_x,1e-06)
T_x = T_x[0]
m = 0
for i in range(0, len(X)-1):
    for j in range(1, len(Y)-1):
        T[i,j] = T_x[m]
        m = m+1
print(T)

plt.title("Mapa de temperatura")
plt.contourf(X, Y, T, colorinterpolation, cmap=colourMap)
plt.colorbar()
plt.show()
'''
################################################################################
#2-neuman, 2 dirichlet
'''
maxIter = 5000

lenX = lenY = 24
delta = 6

Ttop = 100
Tbottom = 0
Tleft = 75
Tright = 50

Tguess = 0


colorinterpolation = 50
colourMap = plt.cm.jet #colourMap = plt.cm.coolwarm

X, Y = meshgrid(arange(0, lenX + delta ,delta), arange(0, lenY + delta,delta))

#print(X,Y)
T = empty((len(X), len(Y)))
T.fill(Tguess)

#print(T)

#condiciones de contorno
T[(len(Y)-1):, :] = Ttop
T[:1, :] = Tbottom
T[:, (len(X)-1):] = Tright
T[:, :1] = Tleft

print(T)

for iteration in range(0, maxIter):
    for i in range(0, len(X)-1):
        for j in range(1, len(Y)-1):
            if i != 0:
                T[i, j] = 0.25 * (T[i+1][j] + T[i-1][j] + T[i][j+1] + T[i][j-1])
            if i == 0:
                if j != 1 and j!= len(Y)-1:
                    T[i, j] = 0.25 * (T[i+1][j] + T[i-1][j] + 2*T[i][j+1])
                else:
                    T[i, j] = 0.25 * (T[i+1][j] + T[i-1][j] + 2*T[i][j+1])

            #if i == 0:
            #    T[i, j] = 0.25 * (T[i][j+1] + T[i][j-1] + 2*T[i+1][j])



print(T)

plt.title("Mapa de temperatura")
plt.contourf(X, Y, T, colorinterpolation, cmap=colourMap)
plt.colorbar()
plt.show()
'''
################################################################################


lenX = lenY = 24
delta = 6

Ttop = 100
Tbottom = 0
Tleft = 75
Tright = 50

Tguess = 0


colorinterpolation = 50
colourMap = plt.cm.jet #colourMap = plt.cm.coolwarm

X, Y = meshgrid(arange(0, lenX + delta ,delta), arange(0, lenY + delta,delta))

#print(X,Y)
T = empty((len(X), len(Y)))
T.fill(Tguess)

#print(T)

#condiciones de contorno
T[(len(Y)-1):, :] = Ttop
T[:1, :] = Tbottom
T[:, (len(X)-1):] = Tright
T[:, :1] = Tleft


T[4] = [0.,25.,50.,75.,100.]
T[:,4] = [0.,25.,50.,75.,100.]
#print(T[4])
#print(T[:,4])
#print(T)

A = zeros(((len(T)-2)**2 + 3 + 4 ,(len(T)-2)**2 + 3 + 4), dtype = float)
C = zeros((len(T)-2)**2 + 3 + 4, dtype = float)

A[0] = array([4.,-2.,0.,0.,-2.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.])
A[1] = array([-1.,4.,-1.,0.,0.,-2.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.])
A[2] = array([0.,-1.,4.,-1.,0.,0.,-2.,0.,0.,0.,0.,0.,0.,0.,0.,0.])
A[3] = array([0.,0.,-1.,4.,0.,0.,0.,-2.,0.,0.,0.,0.,0.,0.,0.,0.])
A[4] = array([-1.,0.,0.,0.,4.,-2.,0.,0.,-1.,0.,0.,0.,0.,0.,0.,0.])
A[5] = array([0.,-1.,0.,0.,-1.,4.,-1.,0.,0.,-1.,0.,0.,0.,0.,0.,0.])
A[6] = array([0.,0.,-1.,0.,0.,-1.,4.,-1.,0.,0.,-1.,0.,0.,0.,0.,0.])
A[7] = array([0.,0.,0.,-1.,0.,0.,-1.,4.,0.,0.,0.,-1.,0.,0.,0.,0.])
A[8] = array([0.,0.,0.,0.,-1.,0.,0.,0.,4.,-2.,0.,0.,-1.,0.,0.,0.])
A[9] = array([0.,0.,0.,0.,0.,-1.,0.,0.,-1.,4.,-1.,0.,0.,-1.,0.,0.])
A[10] = array([0.,0.,0.,0.,0.,0.,-1.,0.,0.,-1.,4.,-1.,0.,0.,-1.,0.])
A[11] = array([0.,0.,0.,0.,0.,0.,0.,-1.,0.,0.,-1.,4.,0.,0.,0.,-1.])
A[12] = array([0.,0.,0.,0.,0.,0.,0.,0.,-1.,0.,0.,0.,4.,-2.,0.,0.])
A[13] = array([0.,0.,0.,0.,0.,0.,0.,0.,0.,-1.,0.,0.,-1.,4.,-1.,0.])
A[14] = array([0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,-1.,0.,0.,-1.,4.,-1.])
A[15] = array([0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,-1.,0.,0.,-1.,4.])
C = array([0.,0.,0.,0.,0.,0.,0.,25.,0.,0.,0.,50.,0.,25.,50.,150.])


T_x = zeros((len(T)-2)**2 + 3 + 4, dtype = float)
T_x = gaussSeidel(A,C,T_x,1e-06)
T_x = T_x[0]
#T_x = linalg.solve(A,C)

m = 0
for i in range(0, len(X)-1):
    for j in range(0, len(Y)-1):
        T[i,j] = T_x[m]
        m = m+1
print(T)

plt.title("Mapa de temperatura")
plt.contourf(X, Y, T, colorinterpolation, cmap=colourMap)
plt.colorbar()
plt.show()
