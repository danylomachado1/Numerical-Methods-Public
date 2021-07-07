#! /usr/bin/python3
from numpy import *
import matplotlib.pyplot as plt # from pylab import plot,show
import warnings
warnings.filterwarnings("ignore")

#################################### Fin del encabezado
import scipy.constants as constants #para la constante g


def SD_forward_L(L,C):
    #N:nuemro de ecuaciones
    N,N = L.shape
    #L:matriz triangular inferior (NxN)
    #C:Matriz de coeficientes independientes (Nx1)
    #R:Matriz de resultados (Nx1)
    R = zeros((N,1), dtype = float)
    for j in range(N):
        Temp = 0
        if j > 0:
            for k in range(j):
                Temp = Temp + L[j,k]*R[k,0]
        R[j,0] = (C[j,0] - Temp)/L[j,j]

    return R

def SI_backward_U(U,C):
    #N:nuemro de ecuaciones
    N,N = U.shape
    #L:matriz triangular inferior (NxN)
    #C:Matriz de coeficientes independientes (Nx1)
    #R:Matriz de resultados (Nx1)
    R = zeros((N,1), dtype = float)
    for j in range(N):
        Temp = 0
        if j > 0:
            for k in range(j):
                Temp = Temp + U[(N-1)-j,(N-1)-k]*R[(N-1)-k,0]
        R[(N-1)-j,0] = (C[(N-1)-j,0] - Temp)/U[(N-1)-j,(N-1)-j]

    return R

def unir_matrices(A,C):
    N,N = A.shape
    A_C = zeros((N,N+1), dtype = float)

    for i in range(N):
        for j in range(N+1):
            if j == N:
                A_C[i,j] = C[i,0]
            if j<N:
                A_C[i,j] = A[i,j]
    return A_C

def separar_matrices(A_C):
    N,M = A_C.shape
    A = zeros((N,N), dtype = float)
    C = zeros((N,1), dtype = float)

    for i in range(N):
        for j in range(N+1):
            if j == N:
                C[i,0] = A_C[i,j]
            if j<N:
                A[i,j] = A_C[i,j]
    return A,C



def E_Gauss(A,C):
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

def D_LU_U(A):
    #Descomposicion LU con los 1 en la matriz U
    N,N = A.shape
    L = zeros((N,N), dtype = float)
    U = zeros((N,N), dtype = float)
    for k in range(N):
        U[k,k] = 1
        suma = 0
        for p in range(k-1):
            suma = suma + L[k,p]*U[p,k]
        L[k,k] = A[k,k]-suma

        for i in range(k,N):
            suma = 0
            for r in range(k-1):
                suma = suma + L[i,r]*U[r,k]
            L[i,k]=A[i,k] - suma
        for j in range(k,N):
            suma = 0
            for s in range(k-1):
                suma = suma + L[k,s]*U[s,j]
            U[k,j]=(A[k,j]-suma)/L[k,k]

    return L,U


def D_LU_L(A):
    #Descomposicion LU con los 1 en la matriz L
    N,N = A.shape

    L = zeros((N,N), dtype = float)
    U = zeros((N,N), dtype = float)

    for i in range(N):
        L[i,i] = 1.

    for i in range(N):  # Para cada "marco" superior izquierdo
        for j in range(i, N):    # Rellena la U a partir de la fila
            suma = 0.0
            for k in range(0, j):
                suma = suma + L[i,k] * U[k,j]
            U[i,j] = A[i,j] - suma

        for j in range(i+1, N):  # Rellena la L a partir de la columna
            suma = 0.0
            for k in range(0, i):
                suma = suma + L[j,k]*U[k,i]
            L[j,i] = (A[j,i] - suma) / U[i,i]

    return L, U


def Cholesky(A):
    #Descomposicion LU con los 1 en la matriz L
    N,N = A.shape
    L = zeros((N,N), dtype = float)
    U = zeros((N,N), dtype = float)

    for k in range(N):
        suma = 0
        for p in range(k):
            suma = suma +L[k,p]*U[p,k]
        L[k,k]=sqrt(A[k,k]-suma)
        U[k,k]=L[k,k]

        for i in range(k,N):
            suma = 0
            for r in range(k):
                suma = suma + L[i,r]*U[r,k]
            L[i,k] = (A[i,k]-suma)/L[k,k]
        for j in range(k,N):
            suma = 0
            for s in range(k):
                suma = suma + L[k,s]*U[s,j]
            U[k,j]=(A[k,j]-suma)/L[k,k]
    return L, U





V_0 = 5
R=1

def Matriz_corrientes(N,R,V_0):
    if N % 2 != 0:
        N = int(input("Introdusca un valor valido, N debe ser par."))

    if R < 0:
        R = float(input("Introdusca un valor valido, R debe ser positivo."))

    A = zeros((N+1,N+1), dtype = float)
    C = zeros((N+1,1), dtype = float)

    C[0,0] = V_0/R

    for i in range(N+1):
        if i == 0:
            A[i,i] = N/2 +1

        if i == N:
            A[0,i] = -1
            A[i,0] = 1

        if (i < N ) and (i > 0):
            if i % 2 != 0:
                A[0,i] = -1
                A[i,0] = 1
            else:
                A[0,i] = 0
                A[i,0] = 0

    for j in range(1,N+1):
        if j < N:
            A[j,j]=-3
            A[j,j-1]=1
            A[j,j+1]=1
        if j == N:
            A[j,j-1]=1
            A[j,j]=-3

    return A,C

def corrientes_voltajes(I,R,V_0):
        #I matriz que contiene a los valores de cooreintes asociados a cada malla
    N,M = I.shape  #M es uno
    V = zeros((N,M), dtype = float)
    V[0,0] = V_0

    V[1,0] = V_0 - R*I[1,0]   #V_1

    for i in range(2,N):
        if i % 2 == 0:
            V[i,0] = V[i-2,0]-R*(I[0,0]-I[i-1,0])
        else:
            V[i,0] = V[i-2,0]-R*I[i-1]

    return(V)

def graficador_voltaje(V,N):
    fig, axes = plt.subplots()
    label_0 = "Voltaje, N = " + str(N)

    axes.plot(V, "*",markersize = 3, label = label_0 )   ### se podria poner en vez de "o" "--o" para que se visualice la linea que sigue pero se hace dificil la visualizacion de los otros puntos, asi mismo, el valor de markersize tambien puede incrementarse.


    axes.minorticks_on()
    axes.grid(which='minor', linestyle=':', linewidth='0.5', color='black')
    axes.grid(True)
    axes.legend(loc = 'upper right')
    axes.set_ylabel("voltaje", fontsize=10)
    axes.set_xlabel("nodo ", fontsize=10)
    axes.set_title("Voltaje en cada nodo", fontsize=15)

    ######### visualization #################
    #plt.yscale("log")     #comentar para visualozar en escala normar
    plt.show()
#def cholesky():
#    return algo


##########################main###########################3
###################################################################################
#parte 1
print("Parte 1:")
#a) resolver por el metodo de la sustitucion backward:
print("a) Resolver por el metodo de la sustitucion backward")
U = array([[1,2,1],[0,-4,1],[0,0,-2]])
C = array([[5],[2],[4]])
R = SI_backward_U(U,C)
U_C = unir_matrices(U,C)
print("Se presenta el siguiente sistema de ecuaciones:")
print(U_C)
print("Este sistema tiene como solucion a:")
print(R)
print("\n")
#b) resolver por el metodo de la sustitucion forward:
print("b) Resolver por el metodo de la sustitucion forward")
L = array([[2,0,0],[1,4,0],[4,3,3]])
C = array([[4],[2],[5]])
R = SD_forward_L(L,C)
L_C = unir_matrices(L,C)
print("Se presenta el siguiente sistema de ecuaciones:")
print(L_C)
print("Este sistema tiene como solucion a:")
print(R)
print("\n")
###########################################################################################
#Parte 2
print("Parte 2:")
print("Primero se se encontro las similtudes entre los sitemas de ecuaciones para N=2,4,6, donde, por induccion se encontro el sistema de ecuaciones para las corrientes para N par")
print("para N = 6:")
V_0 = 5
R = 1.0
N = 6
A,C = Matriz_corrientes(N,R,V_0)
print("Se resuelvr para las corrientes, las cuales estan relacionadas con los voltajes")
I = E_Gauss(A,C)
print(I)

print("En base a las corrientes se encuentran los voltajes en cada nodo")
V = corrientes_voltajes(I,R,V_0)
for i in range(N+1):
    print("V_",i," = ",V[i,0])
#graficador_voltaje(V,N)
print("\n")

print("para N = 100:") #para N=1000 se suele demorar casi 5 minutos con el metodo de eliminacion de Gauss
V_0 = 5
R = 1.0
N = 100
A,C = Matriz_corrientes(N,R,V_0)
print("Se resuelvr para las corrientes, las cuales estan relacionadas con los voltajes")
I = E_Gauss(A,C)
print(I)

print("En base a las corrientes se encuentran los voltajes en cada nodo")
V = corrientes_voltajes(I,R,V_0)
for i in range(N+1):
    print("V_",i," = ",V[i,0])
#graficador_voltaje(V,N)
print("\n")
#########################################################################################3
#Parte 3
print("Parte 3:")
g = constants.g
k = 10

#m_1 = float(imput("ingrese el valor de la masa numero 1:"))
#m_2 = float(imput("ingrese el valor de la masa numero 1:"))
#m_3 = float(imput("ingrese el valor de la masa numero 1:"))

m_1 = 1
m_2 = 2
m_3 = 3

print("se resuelve para las posiciones en el estado estacionario")
A = array([[3*k,-2*k,0],[-2*k,3*k,-k],[0,-k,k]])
C = array([[m_1*g],[m_2*g],[m_3*g]])

A_C = unir_matrices(A,C)
print(A_C)

X = E_Gauss(A,C)
print("Donde obtenemos que X =")
print(X)
