#! /usr/bin/python3
from numpy import *
import matplotlib.pyplot as plt # from pylab import plot,show
import warnings
warnings.filterwarnings("ignore")

#################################### Fin del encabezado
import scipy.constants as constants #para la constante g
from math import *
################################################################################
################################################################################
def SD_forward_L(L,C):
    #N:nuemro de ecuaciones
    N,N = L.shape
    #L:matriz triangular inferior (NxN)
    #C:Matriz de coeficientes independientes (Nx1)
    #R:Matriz de resultados (Nx1)
    R = zeros(N, dtype = float)
    for j in range(N):
        Temp = 0
        if j > 0:
            for k in range(j):
                Temp = Temp + L[j,k]*R[k]
        R[j] = (C[j] - Temp)/L[j,j]

    return R

def SI_backward_U(U,C):
    #N:nuemro de ecuaciones
    N,N = U.shape
    #L:matriz triangular inferior (NxN)
    #C:Matriz de coeficientes independientes (Nx1)
    #R:Matriz de resultados (Nx1)
    R = zeros(N, dtype = float)
    for j in range(N):
        Temp = 0
        if j > 0:
            for k in range(j):
                Temp = Temp + U[(N-1)-j,(N-1)-k]*R[(N-1)-k]
        R[(N-1)-j] = (C[(N-1)-j] - Temp)/U[(N-1)-j,(N-1)-j]

    return R

def unir_matrices(A,C):
    N,N = A.shape
    A_C = zeros((N,N+1), dtype = float)

    for i in range(N):
        for j in range(N+1):
            if j == N:
                A_C[i,j] = C[i]
            if j<N:
                A_C[i,j] = A[i,j]
    return A_C


def gauss_Seidel(A, b):
    x = linalg.solve(A,b)
    return x

def separar_matrices(A_C):
    N,M = A_C.shape
    A = zeros((N,N), dtype = float)
    C = zeros((N), dtype = float)

    for i in range(N):
        for j in range(N+1):
            if j == N:
                C[i] = A_C[i,j]
            if j<N:
                A[i,j] = A_C[i,j]
    return A,C



def E_Gauss(A,C):

    if len(where(diag(A) == 0)[0]) > 0:
        A,C = pivoteo(A,C)


    A_C = unir_matrices(A,C)
    N,M = A_C.shape

    for j in range(M):
        for i in range(N):
            if i>j:
                division = A_C[i,j]/A_C[j,j]
                for k in range(M):
                    A_C[i,k]=A_C[i,k]-division*A_C[j,k]
                    #usando esto, nos devuelve una matriz lista para ser usada en SI_backward_U(A_C)

    U,C = separar_matrices(A_C)
    X = SI_backward_U(U,C)

    return X

def Pivoteo_P(A_C):
    N,M = A_C.shape

    if N==M-1 :
        for k in range(N-1):
            mayor = 0
            filam = k
            for p in range(k-1,N-1):
                if mayor<abs(A_C[p,k]):
                    mayor = abs(A_C[p,k])
                    filam = p
            if mayor == 0:
                print("No funciono, pruebe otra vez")
            else:
                if filam != k:
                    for j in range(M):
                        aux = A_C[k,j]
                        A_C[k,j] = A_C[filam,j]
                        A_C[filam,j] = aux
    return A_C #matriz pivoteada

def pivoteo(A,B):
    again=1
    while again==1:
        N=len(A)
        auxA=zeros((1,N))
        auxB=0
        again=0
        for i in range(0,N,1):
            if A[i,i]==0:
                again=1
                k=0
                while k!=-1 and k<N:
                    if k!=i:
                        if A[k,i]!=0:
                            auxA[0]=copy(A[i])
                            A[i]=copy(A[k])
                            A[k]=copy(auxA[0])
                            auxB=B[i]
                            B[i]=B[k]
                            B[k]=auxB
                            k=-2
                    k=k+1
    return(A,B)     #Retorna las matrices A y B pivoteadas


def Regresion(X,Y):
    n = X.shape[0]
    sumx = 0; sumxy = 0; st = 0
    sumy = 0; sumx2 = 0; sr = 0

    for i in range(n):
        sumx = sumx + X[i]
        sumy = sumy + Y[i]
        sumxy = sumxy + X[i]*Y[i]
        sumx2 = sumx2 + X[i]**2

    xm = sumx/n
    ym = sumy/n
    b = (n*sumxy-sumx*sumy)/(n*sumx2-sumx**2)
    a = ym - b*xm

    for i in range(n):
        st = st + (Y[i] - ym)**2
        sr = sr + (Y[i]-b*X[i]-a)**2

    syx = (sr/(n-2))**0.5
    r2 = (st-sr)/st

    return a,b,syx,r2

def Regresion_polinomial_M(X,Y,orden):
    N = X.shape[0]
    a = zeros((orden+1,orden+2), dtype = float)
    for i in range(orden+1):
        for j in range(i+1):
            k = i+j
            sum = 0
            for l in range(N):
                sum = sum +(X[l])**k
            print(i,j,sum)
            a[i,j] = sum
            a[j,i] = sum
        sum = 0
        for l in range(N):
            sum = sum + (Y[l])*((X[l])**i)
        a[i,orden+1] = sum
    return a

#X = array([0,1,2,3,4,5])
#Y = array([2.1,7.7,13.6,27.2,40.9,61.1])

def funcion(x,a,b):
    return a*x*exp(b*x)

def funcion_a(x,a,b):
    return x*exp(b*x)

def funcion_b(x,a,b):
    return a*(x**2)*exp(b*x)

def Matriz_Z(X,Y,a,b,funcion_a,funcion_b):
    N = X.shape[0]
    Z = zeros((N,2), dtype = float)
    for i in range(N):
        Z[i,0] = funcion_a(X[i],a,b)
        Z[i,1] = funcion_b(X[i],a,b)
    return Z

def Matriz_D(X,Y,a,b,funcion):
    N = X.shape[0]
    D = zeros(N, dtype = float)
    for i in range(N):
        D[i] = Y[i] - funcion(X[i],a,b)
    return D

def Regresion_no_lineal(X,Y,a_0,b_0,funcion,funcion_a,funcion_b):
    a = a_0
    b = b_0
    a_b = array([a,b])
    delta_A = array([2,2])
    residual = linalg.norm(delta_A-a_b)
    i = 0
    while residual > 1e-06:

        #print(residual)
        Z = Matriz_Z(X,Y,a,b,funcion_a,funcion_b)
        #print(Z)
        D = Matriz_D(X,Y,a,b,funcion)
        #print(D)

        A = dot(transpose(Z),Z)
        C = dot(transpose(Z),D)
        #print(A)
        #print(C)

        #delta_A = linalg.solve(A,C)
        delta_A = E_Gauss(A,C)
        #print(delta_A)
        a_b_0 = array([a,b])

        a = a + delta_A[0]
        b = b + delta_A[1]
        a_b_1 = array([a,b])
        residual = linalg.norm(a_b_1-a_b_0)
        i = i+1

    return a_b_1[0],a_b_1[1]
################################################################################
"""
X = array([1,2,3,4,5,6,7])
Y = array([0.5,2.5,2.0,4.0,3.5,6.0,5.5])

a,b,syx,r2 = Regresion(X,Y)

x = linspace(0,10,100)
f_x = []
for x_i in x:
    f_x.append(b*x_i + a)

text = '$r^{2}$ = ' + str(round(r2,3))

fig, axes = plt.subplots()
axes.plot(X,Y , "ro",markersize = 3, label = "Datos" )
axes.plot(x,f_x, "-",markersize = 3, label = "Ajuste" )
axes.set_title("Ajuste Lineal", fontsize=15)
#plt.ylim(-2.5,2.5)
axes.minorticks_on()
axes.grid(which='minor', linestyle=':', linewidth='0.5', color='black')
axes.grid(True)
axes.legend(loc = 'best')
axes.set_ylabel("f(x)", fontsize=10)
axes.set_xlabel("x", fontsize=10)
plt.text(6, 1.5, text, {'color': 'black', 'fontsize': 20})
plt.show()



Time = array([1,2,3,4,5,6,7,8,9,10,11,12,13,14,15])
M_v = array([10.00,16.30,23.00,27.50,31.00,35.60,39.00,41.50,42.90,45.00,46.00,45.50,46.00,49.00,50.00])
M_v1 = array([8.953,16.405,22.607,27.769,32.065,35.641,38.617,41.095,43.156,44.872,46.301,47.49,48.479,49.303,49.988])
M_v2 = array([11.24,18.57,23.729,27.556,30.509,32.855,34.766,36.351,37.687,38.829,39.816,40.678,41.437,42.11,42.712])

fig, axes = plt.subplots()
axes.plot(Time,M_v , "--o",markersize = 3, label = "Medido" )
axes.plot(Time,M_v1, "--o",markersize = 3, label = "Modelo 1" )
axes.plot(Time,M_v2, "--o",markersize = 3, label = "Modelo 2" )
axes.set_title("Ajuste Lineal", fontsize=15)
#plt.ylim(-2.5,2.5)
axes.minorticks_on()
axes.grid(which='minor', linestyle=':', linewidth='0.5', color='black')
axes.grid(True)
axes.legend(loc = 'best')
axes.set_ylabel("f(x)", fontsize=10)
axes.set_xlabel("x", fontsize=10)
plt.show()

a1,b1,syx1,r2_1 = Regresion(M_v,M_v1)

x = linspace(0,60,200)
f_x = []
for x_i in x:
    f_x.append(b1*x_i + a1)

text1 = '$r^{2}$ = ' + str(round(r2_1,3))
text2 = 'a1,b1 = ' + str(round(a1,3)) + ','+str(round(b1,3))


fig, axes = plt.subplots()
axes.plot(M_v,M_v1, "ro",markersize = 3, label = "Datos" )
axes.plot(x,f_x, "-",markersize = 3, label = "Ajuste" )
axes.set_title("Ajuste Lineal", fontsize=15)
#plt.ylim(-2.5,2.5)
axes.minorticks_on()
axes.grid(which='minor', linestyle=':', linewidth='0.5', color='black')
axes.grid(True)
axes.legend(loc = 'best')
axes.set_ylabel("f(x)", fontsize=10)
axes.set_xlabel("x", fontsize=10)
plt.text(15, 5, text1, {'color': 'black', 'fontsize': 10})
plt.text(15, 3, text2, {'color': 'black', 'fontsize': 10})
plt.show()


a2,b2,syx2,r2_2 = Regresion(M_v,M_v1)


x = linspace(0,60,200)
f_x = []
for x_i in x:
    f_x.append(b2*x_i + a2)

text1 = '$r^{2}$ = ' + str(round(r2_2,3))
text2 = 'a2,b2 = ' + str(round(a2,3)) + ','+str(round(b2,3))

fig, axes = plt.subplots()
axes.plot(M_v,M_v1, "ro",markersize = 3, label = "Datos" )
axes.plot(x,f_x, "-",markersize = 3, label = "Ajuste" )
axes.set_title("Ajuste Lineal", fontsize=15)
#plt.ylim(-2.5,2.5)
axes.minorticks_on()
axes.grid(which='minor', linestyle=':', linewidth='0.5', color='black')
axes.grid(True)
axes.legend(loc = 'best')
axes.set_ylabel("f(x)", fontsize=10)
axes.set_xlabel("x", fontsize=10)
plt.text(15, 5, text1, {'color': 'black', 'fontsize': 10})
plt.text(15, 3, text2, {'color': 'black', 'fontsize': 10})
plt.show()
"""
################################################################################
print("Ejercicio 1")
print("La ecuacion liearizada queda como")
print("y^0.5 =(1/b) + (a/b)*(x^-0.5)")
print("donde: a_t = (1/b) y b_t = (a/b)")

X = array([0.5,1,2,3,4])
Y = array([10.4,5.8,3.3,2.4,2.0])

print("Tenemos que encontrar los valores para x^-0.5 e y^0.5")
X_t = copy(X)
Y_t = copy(Y)

for i in range(len(X)):
    X_t[i] = 1/sqrt(X[i])
    Y_t[i] = sqrt(Y[i])

a_t,b_t,syx,r2 = Regresion(X_t,Y_t)

x = linspace(0.5,3,100)
f_x = []
for x_i in x:
    f_x.append(a_t + b_t*x_i)

text1 = '$r^{2}$ = ' + str(round(r2,3))
text2 = 'a_t,b_t = ' + str(round(a_t,3)) + ','+str(round(b_t,3))

fig, axes = plt.subplots()
axes.plot(X_t,Y_t , "ro",markersize = 3, label = "Datos" )
axes.plot(x,f_x, "-",markersize = 3, label = "Ajuste" )
axes.set_title("Ajuste Lineal", fontsize=15)
#plt.ylim(-2.5,2.5)
axes.minorticks_on()
axes.grid(which='minor', linestyle=':', linewidth='0.5', color='black')
axes.grid(True)
axes.legend(loc = 'best')
axes.set_ylabel("f(x)", fontsize=10)
axes.set_xlabel("x", fontsize=10)
plt.text(2, 2, text1, {'color': 'black', 'fontsize': 10})
plt.text(2, 1.5, text2, {'color': 'black', 'fontsize': 10})
plt.show()

print('$r^{2}$ = ' + str(round(r2,5)))
print(text2)
print("entonces:")
print("a,b = ", a_t/b_t,1/a_t)

################################################################################
print("\n")
print("Ejercicio 2")
print("La ecuacion liearizada queda como")
print("ln(y/x) = ln(a) + b*x")
print("donde: a_t = ln(a) y b_t = b")

X = array([0.1,0.2,0.4,0.6,0.9,1.3,1.5,1.7,1.8])
Y = array([0.75,1.25,1.45,1.25,0.85,0.55,0.35,0.28,0.18])

print("Tenemos que encontrar los valores para ln(y/x) e x")
X_t = copy(X)
Y_t = copy(Y)

for i in range(len(X)):
    X_t[i] = X[i]
    Y_t[i] = log(Y[i]/X[i])

a_t,b_t,syx,r2 = Regresion(X_t,Y_t)

x = linspace(0.01,3,100)
f_x = []
for x_i in x:
    f_x.append(a_t + b_t*x_i)

text1 = '$r^{2}$ = ' + str(round(r2,3))
text2 = 'a_t,b_t = ' + str(round(a_t,3)) + ','+str(round(b_t,3))

fig, axes = plt.subplots()
axes.plot(X_t,Y_t , "ro",markersize = 3, label = "Datos" )
axes.plot(x,f_x, "-",markersize = 3, label = "Ajuste" )
axes.set_title("Ajuste Lineal", fontsize=15)
#plt.ylim(-2.5,2.5)
axes.minorticks_on()
axes.grid(which='minor', linestyle=':', linewidth='0.5', color='black')
axes.grid(True)
axes.legend(loc = 'best')
axes.set_ylabel("f(x)", fontsize=10)
axes.set_xlabel("x", fontsize=10)
plt.text(2, 2, text1, {'color': 'black', 'fontsize': 10})
plt.text(2, 1.5, text2, {'color': 'black', 'fontsize': 10})
plt.show()

a = exp(a_t)
b = b_t
print(text1)
print(text2)


print("entonces:")
print("a,b = ", a,b)


#X = array([0.1,0.2,0.4,0.6,0.9,1.3,1.5,1.7,1.8])
#Y = array([0.75,1.25,1.45,1.25,0.85,0.55,0.35,0.28,0.18])

#x = linspace(0.01,2,200)
#f_x = []
#for x_i in x:
#    f_x.append(a*x_i*exp(b*x_i))

#fig, axes = plt.subplots()
#axes.plot(X,Y , "ro",markersize = 3, label = "Datos" )
#axes.plot(x,f_x, "-",markersize = 3, label = "Ajuste" )
#axes.set_title("Ajuste luego de linializar", fontsize=15)
#plt.ylim(-2.5,2.5)
#axes.minorticks_on()
#axes.grid(which='minor', linestyle=':', linewidth='0.5', color='black')
#axes.grid(True)
#axes.legend(loc = 'best')
#axes.set_ylabel("f(x)", fontsize=10)
#axes.set_xlabel("x", fontsize=10)
#plt.show()




X = array([0.1,0.2,0.4,0.6,0.9,1.3,1.5,1.7,1.8])
Y = array([0.75,1.25,1.45,1.25,0.85,0.55,0.35,0.28,0.18])
a_0 = 5
b_0 = 1

a_mod,b_mod = Regresion_no_lineal(X,Y,a_0,b_0,funcion,funcion_a,funcion_b)

x = linspace(0.01,2,200)
f_x_1 = []
f_x_2 = []
for x_i in x:
    f_x_1.append(a*x_i*exp(b*x_i))
    f_x_2.append(a_mod*x_i*exp(b_mod*x_i))

fig, axes = plt.subplots()
axes.plot(X,Y , "ro",markersize = 3, label = "Datos" )
axes.plot(x,f_x_1, "-",markersize = 3, label = "Ajuste lineal" )
axes.plot(x,f_x_2, "-",markersize = 3, label = "Ajuste no lineal" )
axes.set_title("Ajuste", fontsize=15)
#plt.ylim(-2.5,2.5)
axes.minorticks_on()
axes.grid(which='minor', linestyle=':', linewidth='0.5', color='black')
axes.grid(True)
axes.legend(loc = 'best')
axes.set_ylabel("f(x)", fontsize=10)
axes.set_xlabel("x", fontsize=10)
plt.show()

err_1 = []
err_2 = []
for i in range(len(X)):
    err_1.append((Y[i]-a*X[i]*exp(b*X[i]))**2)
    err_2.append((Y[i]-a_mod*X[i]*exp(b_mod*X[i])))

cuantificaciondelerror1 = (sqrt(sum(err_1)/(len(X)-1)))
cuantificaciondelerror2 = (sqrt(sum(err_2)/(len(X)-1)))

print("error_1,error_2 : ", cuantificaciondelerror1,cuantificaciondelerror2)
