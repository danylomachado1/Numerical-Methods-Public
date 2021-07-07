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
#Metodos Directos
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


def Cholesky(A): #para resolver con Cholesky la matriz debe ser positiva definida y simetrica
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
################################################################################
################################################################################
def Confirma_simetria(A):
    N,M = A.shape
    for i in range(N):
        for j in range(N):
            if (A[i][j] != A[j][i]):
                return False
    return True
################################################################################
################################################################################
#Metodos Iterativos
def jacobi(A,b,N,tol):
    x = zeros(len(A[0]))

    D = diag(A)
    R = A - diagflat(D)

    j = 0
    err = []
    iter = 0     #iteraciones
    for i in range(N):
         x2 = (b - dot(R,x)) / D
         delta = linalg.norm(x - x2)
         err.append(delta)
         j = j+1
         iter = iter +1     #iteraciones
         if delta < tol:
             return x2,err,iter
         x = x2

    return x,err,iter

################################################################################
def gaussSeidel(A, b, x, tol):
    A_C = unir_matrices(A,b)
    if len(where(diag(A) == 0)[0]) > 0:
        A_C = Pivoteo_P(A_C)
    A,b = separar_matrices(A_C)
    N,M = A.shape
    Iter_max = 1000000
    xprev = zeros(N)
    err = []
    iter = 0  #iteraciones
    for i in range(Iter_max):
        for j in range(N):
            xprev[j] = x[j]
        for j in range(N):
            suma = 0.0
            for k in range(N):
                if (k != j):
                    suma = suma + A[j][k] * x[k]
            x[j] = (b[j] - suma) / A[j][j]

        diff1norm = 0.0
        oldnorm = 0.0
        for j in range(N):
            diff1norm = diff1norm + abs(x[j] - xprev[j])
            oldnorm = oldnorm + abs(xprev[j])
        if oldnorm == 0.0:
            oldnorm = 1.0
        norm = diff1norm / oldnorm
        err.append(norm)
        iter = iter + 1  #iteraciones
        if (norm < tol) and i != 0:
            x_2 = x
            return x_2,err, iter
    print("Iter_max")
    return x, err, iter


################################################################################
def Metodo_SOR(A, b, omega, initial_guess, convergence_criteria):
    A_C = unir_matrices(A,b)
    if len(where(diag(A) == 0)[0]) > 0:
        A_C = Pivoteo_P(A_C)
    A,b = separar_matrices(A_C)

    phi = initial_guess[:]
    residual = linalg.norm(dot(A, phi) - b)
    err = []
    iter  = 0       #iteraciones
    while residual > convergence_criteria:
        for i in range(A.shape[0]):
            sigma = 0
            for j in range(A.shape[1]):
                if j != i:
                    sigma += A[i][j] * phi[j]

            phi[i] = (1 - omega) * phi[i] + (omega / A[i][i]) * (b[i] - sigma)


        residual = linalg.norm(dot(A, phi) - b)
        err.append(residual)
        iter = iter + 1

    return phi, err, iter      #iteraciones

################################################################################

def Maximo_descenso(A, b, x):   #para resolver con este metodo la matriz debe ser positiva definida y simetrica
    err = []
    r = dot(A,x)-b
    #p = r
    p = -r
    rsold = dot(transpose(r), r)

    #rsnew = dot(transpose(r), r)
    iter = 0   #iteraciones
    residual = linalg.norm(dot(A, x) - b)
    while residual > 1e-6:
        Ar = dot(A, r)
        alpha = rsold / dot(transpose(r), Ar)
        x = x + dot(alpha, p)

        r = dot(A,x)-b
        rsnew = dot(transpose(r), r)
        residual = linalg.norm(dot(A, x) - b)
        err.append(residual)
        #print(rsnew)
        p = -r
        rsold = rsnew
        iter = iter +1      #iteraciones

    return x,err, iter


################################################################################
################################################################################
def Gradiente_conjudado(A, b, x): #para resolver con este metodo la matriz debe ser positiva definida y simetrica
    err = []
    r = dot(A,x)-b
    #p = r
    p = -r
    rsold = dot(transpose(r), r)

    residual = linalg.norm(dot(A, x) - b)
    #rsnew = 1
    iter = 0
    while residual > 1e-6:

    #for i in range(len(b)):
        Ap = dot(A, p)
        alpha = rsold / dot(transpose(p), Ap)
        x = x + dot(alpha, p)

        r = r + dot(alpha, Ap)
        rsnew = dot(transpose(r), r)
        residual = linalg.norm(dot(A, x) - b)
        #print(residual)

        err.append(residual)
        #print(x)
        #if sqrt(rsnew) < 1e-8:
        #    break

        iter = iter +1
        p = -r + (rsnew/rsold)*p
        rsold = rsnew
    return x,err,iter
################################################################################
Datos_x = array([2.,3.,4.,5.,6.])
Datos_y = array([2.,6.,5.,5.,6.])

def Vandermonde_polinomio(Datos_x,Datos_y):
    N = len(Datos_x)  #numero de puntos
    M_v = zeros((N,N), dtype = float)
    for i in range(N):
        for j in range(N):
            M_v[i,j] = Datos_x[i]**j
    return M_v

def Funcion_polinomio(Datos_x,Datos_y,coeficientes):
    #fig, axes = plt.subplots()
    N = len(coeficientes)
    x = linspace(Datos_x[0],Datos_x[-1],40000)
    a = 0
    for i in range(N):
        a = a + dot(x**i,coeficientes[i])
    axes.plot(x,a , "-",markersize = 1, label = "Funcion polinomial" )
    axes.plot(Datos_x,Datos_y,"ro", label = "Datos")
    axes.minorticks_on()
    axes.grid(which='minor', linestyle=':', linewidth='0.5', color='black')
    axes.grid(True)
    axes.legend(loc = 'best')
    axes.set_ylabel("y", fontsize=10)
    axes.set_xlabel("x ", fontsize=10)
    #axes.set_title("Interpolacion: Polinomio", fontsize=15)
    #plt.show()

def Funcion_polinomio(coeficientes,x):
    N = len(coeficientes)
    a = 0
    for i in range(N):
        a = a +  dot(x**i,coeficientes[i])
    return a

def grafica_datos_funcion(Datos_x,Datos_y,coeficientes,Funcion,lab):
    #fig, axes = plt.subplots()
    f_x = []
    x = linspace(Datos_x[0],Datos_x[-1],200)
    for x_i in x:
        y = Funcion(coeficientes,x_i)
        f_x.append(y)

    axes.plot(x,f_x, "--",markersize = 0.5, label = lab )   ### se podria poner en vez de "o" "--o" para que se visualice la linea que sigue pero se hace dificil la visualizacion de los otros puntos, asi mismo, el valor de markersize tambien puede incrementarse.
    #axes.plot(Datos_x,Datos_y,"ro",markersize = 4, label = "Datos")
    axes.minorticks_on()
    axes.grid(which='minor', linestyle=':', linewidth='0.5', color='black')
    axes.grid(True)
    axes.legend(loc = 'best')
    axes.set_ylabel("y", fontsize=10)
    axes.set_xlabel("x ", fontsize=10)
    #axes.set_title("Interpolacion: Polinomio", fontsize=15)
    ########## visualization #################
    #plt.show()

""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

def lagrange_k(Datos_x,k,x):
    N = len(Datos_x)
    a = 1
    for i in range(N):
        if i != k:
            a = a*(x-Datos_x[i])/(Datos_x[k]-Datos_x[i])
    return a



def lagrange_completo(Datos_x,Datos_y,x):
    N = len(Datos_x)
    a = 0
    for i in range(N):
        a = a + Datos_y[i]*lagrange_k(Datos_x,i,x)
    return a



def Funcion_lagrange(Datos_x,Datos_y,lab):
    #fig, axes = plt.subplots()
    N = len(Datos_x)
    x = linspace(Datos_x[0],Datos_x[-1],200)
    f_x = []
    for x_i in x:
        y = lagrange_completo(Datos_x,Datos_y,x_i)
        f_x.append(y)

    axes.plot(x,f_x , ":",markersize = 2, label = lab )
    #axes.plot(Datos_x,Datos_y,"ro", label = "Datos")
    axes.minorticks_on()
    axes.grid(which='minor', linestyle=':', linewidth='0.5', color='black')
    axes.grid(True)
    axes.legend(loc = 'best')
    axes.set_ylabel("y", fontsize=10)
    axes.set_xlabel("x", fontsize=10)
    #axes.set_title("Interpolacion: Lagrange", fontsize=15)
    #plt.show()

""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

def newton_k(Datos_x,k,x):
    N = len(Datos_x)
    a = 1
    for i in range(N):
        if i == 0:
            a = 1
        if i < k:
            a = a*(x-Datos_x[i])
    return a

def Vandermonde_newton(Datos_x,Datos_y):
    N = len(Datos_x)  #numero de puntos
    M_v = zeros((N,N), dtype = float)
    for i in range(N):
        for j in range(N):
            if j<=i:
                if j == 0:
                    M_v[i,j] = 1.0
                else:
                    M_v[i,j] = newton_k(Datos_x,j,Datos_x[i])
            else :
                M_v[i,j] = 0.0
    return M_v

def newton_completo(Datos_x,Datos_y,coeficientes,x):
    N = len(Datos_x)
    a = 0
    for i in range(N):
        a = a + coeficientes[i]*newton_k(Datos_x,i,x)
    return a

def Funcion_newton(Datos_x,Datos_y,coeficientes):
    #fig, axes = plt.subplots()
    N = len(Datos_x)
    x = linspace(Datos_x[0],Datos_x[-1],200)
    f_x = []
    for x_i in x:
        y = newton_completo(Datos_x,Datos_y,coeficientes,x_i)
        f_x.append(y)

    axes.plot(x,f_x , "-.",markersize = 1, label = "Funcion newton" )
    #axes.plot(Datos_x,Datos_y,"ro", label = "Datos")
    axes.minorticks_on()
    axes.grid(which='minor', linestyle=':', linewidth='0.5', color='black')
    axes.grid(True)
    axes.legend(loc = 'best')
    axes.set_ylabel("y", fontsize=10)
    axes.set_xlabel("x", fontsize=10)
    #axes.set_title("Interpolacion: Newton", fontsize=15)
    #plt.show()

################################################################################
def funcion_1(x):
    return atan(20*sin(12*x)) + (1/50)*exp(3*x)*sin(300*x)

def puntos_chebyshev(a,b,n):
    Puntos_c = []
    t = 0
    for i in range(n):
        Puntos_c.append(0.5*(a+b) + 0.5*(b-a)*cos((i*pi)/(n-1)))
        t = t+1
    #print(t)
    return Puntos_c

def puntos_equidistantes(a,b,n):
    Puntos_e = []
    for i in range(n):
        Puntos_e.append(i/(n-1))
    return Puntos_e

def M_spline_2(Datos_x_e,Datos_y_e,N_Pts):
    A = zeros((3*(N_Pts-1),3*(N_Pts-1)), dtype = float)
    C = zeros((3*(N_Pts-1)), dtype = float)

    A[0,0] = 1
    for i in range(3*(N_Pts-1)):
        if i%3 == 1:
            A[i,i-1] = (Datos_x_e[int((i-1)/3)])**2
            A[i,i] = Datos_x_e[int((i-1)/3)]
            A[i,i+1] = 1
            #C[i] = funcion_2(Datos_x_e[int((i-1)/3)])
            C[i] = Datos_y_e[int((i-1)/3)]
        if i%3 == 2:
            A[i,i-2] = (Datos_x_e[int((i+1)/3)])**2
            A[i,i-1] = Datos_x_e[int((i+1)/3)]
            A[i,i] = 1
            #C[i] = funcion_2(Datos_x_e[int((i+1)/3)])
            C[i] = Datos_y_e[int((i+1)/3)]
        if i != 0 and i%3 == 0:
            A[i,i-3] = 2*Datos_x_e[int(i/3)]
            A[i,i-2] = 1
            A[i,i] = -2*Datos_x_e[int(i/3)]
            A[i,i+1] = -1
            C[i] = 0

    return A,C


def M_spline_3(Datos_x_e,Datos_y_e,N_Pts):
    A = zeros((4*(N_Pts-1),4*(N_Pts-1)), dtype = float)
    C = zeros((4*(N_Pts-1)), dtype = float)

    #A[0,0] = 1
    A[0,0] = 6*Datos_x_e[0]
    A[0,1] = 2

    A[4*(N_Pts-1)-1,4*(N_Pts-1)-4] = 6*Datos_x_e[-1]
    A[4*(N_Pts-1)-1,4*(N_Pts-1)-3] = 2

    for i in range(4*(N_Pts-1)):

        if i%4 == 1:
            A[i,i-1] = (Datos_x_e[int((i-1)/4)])**3
            A[i,i] = Datos_x_e[int((i-1)/4)]**2
            A[i,i+1] = Datos_x_e[int((i-1)/4)]
            A[i,i+2] = 1
            #C[i] = funcion_2(Datos_x_e[int((i-1)/3)])
            C[i] = Datos_y_e[int((i-1)/4)]
        if i%4 == 2:
            A[i,i-2] = (Datos_x_e[int((i+2)/4)])**3
            A[i,i-1] = Datos_x_e[int((i+2)/4)]**2
            A[i,i] = Datos_x_e[int((i+2)/4)]
            A[i,i+1] = 1
            #C[i] = funcion_2(Datos_x_e[int((i+1)/3)])
            C[i] = Datos_y_e[int((i+2)/4)]

        if i!= 4*(N_Pts-1)-1 and i%4 == 3:
            A[i,i-3] = 3*(Datos_x_e[int((i+1)/4)])**2
            A[i,i-2] = 2 * Datos_x_e[int((i+1)/4)]
            A[i,i-1] = 1
            A[i,i] = 0

            A[i,i+1] = -3*(Datos_x_e[int((i+1)/4)])**2
            A[i,i+2] = -2 * Datos_x_e[int((i+1)/4)]
            A[i,i+3] = -1
            A[i,i+4] = 0
            #C[i] = funcion_2(Datos_x_e[int((i+1)/3)])
            C[i] = 0

        if i != 0 and i%4 == 0:
            A[i,i-4] = 6*(Datos_x_e[int((i)/4)])
            A[i,i-3] = 2
            #A[i,i] = 1
            #A[i,i+1] = 0

            A[i,i] = -6*(Datos_x_e[int((i)/4)])
            A[i,i+1] = -2
            #A[i,i+3] = -1
            #A[i,i+4] = 0
            C[i] = 0

    return A,C

###############################################################################3
def funcion_2(x):
    return (sin(x))**2

################################################################################
###################################Main#########################################
################################################################################
"""
Datos_x = array([2.,3.,4.,5.,6.])
Datos_y = array([2.,6.,5.,5.,6.])

M_v_polinomio = Vandermonde_polinomio(Datos_x ,Datos_y)
M_v_newton = Vandermonde_newton(Datos_x ,Datos_y)
coeficientes_polinomio = E_Gauss(M_v_polinomio, Datos_y)
coeficientes_newton = E_Gauss(M_v_newton, Datos_y)

fig, axes = plt.subplots()
grafica_datos_funcion(Datos_x,Datos_y,coeficientes_polinomio,Funcion_polinomio,"funcion polinomial")
Funcion_lagrange(Datos_x,Datos_y)
Funcion_newton(Datos_x ,Datos_y,coeficientes_newton)
axes.set_title("Interpolacion: Polinomio", fontsize=15)
plt.show()

#################################

Datos_x = array([0.,1.,2.,3.,4.,5.,6.])
Datos_y = array([3.,5.,6.,5.,4.,4.,5.])

M_v_polinomio = Vandermonde_polinomio(Datos_x ,Datos_y)
M_v_newton = Vandermonde_newton(Datos_x ,Datos_y)
coeficientes_polinomio = E_Gauss(M_v_polinomio, Datos_y)
coeficientes_newton = E_Gauss(M_v_newton, Datos_y)

fig, axes = plt.subplots()
grafica_datos_funcion(Datos_x,Datos_y,coeficientes_polinomio,Funcion_polinomio,"funcion polinomial")
Funcion_lagrange(Datos_x,Datos_y)
Funcion_newton(Datos_x ,Datos_y,coeficientes_newton)
axes.set_title("Interpolacion: Polinomio", fontsize=15)
plt.show()

#################################

Datos_x = array([-2.0,0.0,1.0])
Datos_y = array([-27.0,-1.0,0.0])

M_v_polinomio = Vandermonde_polinomio(Datos_x ,Datos_y)
M_v_newton = Vandermonde_newton(Datos_x ,Datos_y)
coeficientes_polinomio = E_Gauss(M_v_polinomio, Datos_y)
coeficientes_newton = E_Gauss(M_v_newton, Datos_y)

fig, axes = plt.subplots()
grafica_datos_funcion(Datos_x,Datos_y,coeficientes_polinomio,Funcion_polinomio,"funcion polinomial")
Funcion_lagrange(Datos_x,Datos_y)
Funcion_newton(Datos_x ,Datos_y,coeficientes_newton)
axes.set_title("Interpolacion: Polinomio", fontsize=15)
plt.show()
"""
########################################
print("Parte monomios, lagrage y newton")
Datos_x = array([0.0,1.0,2.0,5.5,11.0,13.0,16.0,18.0])
Datos_y = array([0.5,3.134,5.3,9.9,10.2,9.35,7.2,6.2])

M_v_polinomio = Vandermonde_polinomio(Datos_x ,Datos_y)
M_v_newton = Vandermonde_newton(Datos_x ,Datos_y)

coeficientes_polinomio = E_Gauss(M_v_polinomio, Datos_y)
coeficientes_newton = E_Gauss(M_v_newton, Datos_y)

fig, axes = plt.subplots()
axes.plot(Datos_x,Datos_y,"ro",markersize = 6, label = "Datos")
grafica_datos_funcion(Datos_x,Datos_y,coeficientes_polinomio,Funcion_polinomio,"funcion polinomial")
Funcion_lagrange(Datos_x,Datos_y,"Funcion lagrange")
Funcion_newton(Datos_x ,Datos_y,coeficientes_newton)
axes.set_title("Interpolacion: Polinomio", fontsize=15)
plt.show()

print("Interpolacion por polinomios")
print(Funcion_polinomio(coeficientes_polinomio,8.0))
print("Interpolacion por lagrange")
print(lagrange_completo(Datos_x,Datos_y,8.0))
print("Interpolacion por newtom")
print(newton_completo(Datos_x,Datos_y,coeficientes_newton,8.0))

#unido = unir_matrices(M_v_polinomio,Datos_y)
#pivoteado = Pivoteo_P(unido)

#M_v_polinomio, Datos_y = separar_matrices(pivoteado)
#coeficientes_polinomio = E_Gauss(M_v_polinomio, Datos_y)
#residual_convergence = 1e-8
#omega = 1.5 #Relaxation factor debe ser mayor a 1
#initial_guess = zeros(len(Datos_x)) #numero de incongnitas
#coeficientes_polinomio, err,iter = Metodo_SOR(M_v_polinomio,Datos_y, omega, initial_guess, residual_convergence)

#########################################3#

#Datos_x_e = puntos_equidistantes(0,1,100)
print("Parte lagrange y puntos_chebyshev")
Datos_x_e = linspace(0,1,100)
Datos_x_c = puntos_chebyshev(0,1,100)

Datos_y_e = []
Datos_y_c = []

for i in range(len(Datos_x_e)):
    Datos_y_e.append(funcion_1(Datos_x_e[i]))

for i in range(len(Datos_x_c)):
    Datos_y_c.append(funcion_1(Datos_x_c[i]))


fig, axes = plt.subplots()

x = linspace(0,1,500)
f_x = []
err_e = []
err_c = []
x_err = []
for x_i in x:
    y = funcion_1(x_i)
    if y != 0 and funcion_1(x_i) != lagrange_completo(Datos_x_e,Datos_y_e,x_i) and funcion_1(x_i)!=lagrange_completo(Datos_x_c,Datos_y_c,x_i):
        err_e.append(abs((funcion_1(x_i)-lagrange_completo(Datos_x_e,Datos_y_e,x_i))))
        err_c.append(abs((funcion_1(x_i)-lagrange_completo(Datos_x_c,Datos_y_c,x_i))))
        x_err.append(x_i)
    f_x.append(y)
axes.plot(x,f_x , "-",markersize = 1, label = "Funcion" )
axes.plot(Datos_x_e,Datos_y_e,"ro",markersize = 4, label = "Datos_e")
axes.plot(Datos_x_c,Datos_y_c,"go",markersize = 4, label = "Datos_c")
Funcion_lagrange(Datos_x_e ,Datos_y_e,"Funcion lagrange_e")
Funcion_lagrange(Datos_x_c ,Datos_y_c,"Funcion lagrange_c")
axes.set_title("Interpolacion: Polinomio", fontsize=15)
plt.ylim(-2.5,2.5)
plt.show()

fig, axes = plt.subplots()
axes.plot(x_err,err_e,":",markersize = 2, label = "error_e")
axes.plot(x_err,err_c,"-.",markersize = 2, label = "error_c")
axes.minorticks_on()
axes.grid(which='minor', linestyle=':', linewidth='0.5', color='black')
axes.grid(True)
axes.legend(loc = 'best')
axes.set_ylabel("error relativo", fontsize=10)
axes.set_xlabel("x", fontsize=10)
#plt.ylim(0,2)
plt.yscale("log")
plt.show()

################################################################################
print("Parte n veces puntos_chebyshev")

Datos_x_c_100 = puntos_chebyshev(0,1,100)
Datos_x_c_200 = puntos_chebyshev(0,1,200)
Datos_x_c_300 = puntos_chebyshev(0,1,300)
Datos_x_c_400 = puntos_chebyshev(0,1,400)
Datos_x_c_500 = puntos_chebyshev(0,1,500)
Datos_x_c_600 = puntos_chebyshev(0,1,600)
Datos_x_c_700 = puntos_chebyshev(0,1,700)
Datos_x_c_800 = puntos_chebyshev(0,1,800)
Datos_x_c_900 = puntos_chebyshev(0,1,900)
Datos_x_c_1000 = puntos_chebyshev(0,1,1000)

Datos_y_c_100 = []
Datos_y_c_200 = []
Datos_y_c_300 = []
Datos_y_c_400 = []
Datos_y_c_500 = []
Datos_y_c_600 = []
Datos_y_c_700 = []
Datos_y_c_800 = []
Datos_y_c_900 = []
Datos_y_c_1000 = []

for i in range(len(Datos_x_c_100)):
    Datos_y_c_100.append(funcion_1(Datos_x_c_100[i]))

for i in range(len(Datos_x_c_200)):
    Datos_y_c_200.append(funcion_1(Datos_x_c_200[i]))

for i in range(len(Datos_x_c_300)):
    Datos_y_c_300.append(funcion_1(Datos_x_c_300[i]))

for i in range(len(Datos_x_c_400)):
    Datos_y_c_400.append(funcion_1(Datos_x_c_400[i]))

for i in range(len(Datos_x_c_500)):
    Datos_y_c_500.append(funcion_1(Datos_x_c_500[i]))

for i in range(len(Datos_x_c_600)):
    Datos_y_c_600.append(funcion_1(Datos_x_c_600[i]))

for i in range(len(Datos_x_c_700)):
    Datos_y_c_700.append(funcion_1(Datos_x_c_700[i]))

for i in range(len(Datos_x_c_800)):
    Datos_y_c_800.append(funcion_1(Datos_x_c_800[i]))

for i in range(len(Datos_x_c_900)):
    Datos_y_c_900.append(funcion_1(Datos_x_c_900[i]))

for i in range(len(Datos_x_c_1000)):
    Datos_y_c_1000.append(funcion_1(Datos_x_c_1000[i]))



x = linspace(0,1,100)
f_x = []
err_c_100 = []
err_c_200 = []
err_c_300 = []
err_c_400 = []
err_c_500 = []
err_c_600 = []
err_c_700 = []
err_c_800 = []
err_c_900 = []
err_c_1000 = []
x_err = []

for x_i in x:
    y = funcion_1(x_i)
    #if y!=0:
    err_c_100.append(abs((funcion_1(x_i)-lagrange_completo(Datos_x_c_100,Datos_y_c_100,x_i))))
    err_c_200.append(abs((funcion_1(x_i)-lagrange_completo(Datos_x_c_200,Datos_y_c_200,x_i))))
    err_c_300.append(abs((funcion_1(x_i)-lagrange_completo(Datos_x_c_300,Datos_y_c_300,x_i))))
    err_c_400.append(abs((funcion_1(x_i)-lagrange_completo(Datos_x_c_400,Datos_y_c_400,x_i))))
    err_c_500.append(abs((funcion_1(x_i)-lagrange_completo(Datos_x_c_500,Datos_y_c_500,x_i))))
    err_c_600.append(abs((funcion_1(x_i)-lagrange_completo(Datos_x_c_600,Datos_y_c_600,x_i))))
    err_c_700.append(abs((funcion_1(x_i)-lagrange_completo(Datos_x_c_700,Datos_y_c_700,x_i))))
    err_c_800.append(abs((funcion_1(x_i)-lagrange_completo(Datos_x_c_800,Datos_y_c_800,x_i))))
    err_c_900.append(abs((funcion_1(x_i)-lagrange_completo(Datos_x_c_900,Datos_y_c_900,x_i))))
    err_c_1000.append(abs((funcion_1(x_i)-lagrange_completo(Datos_x_c_1000,Datos_y_c_1000,x_i))))
    x_err.append(x_i)

    f_x.append(y)

xd = [amax(err_c_100),amax(err_c_200),amax(err_c_300),amax(err_c_400),amax(err_c_500),amax(err_c_600)]
print(xd)

fig, axes = plt.subplots()
axes.plot(x,f_x , "-",markersize = 1, label = "Funcion" )
#axes.plot(Datos_x_c_200,Datos_y_c_200,"ro",markersize = 2, label = "Datos_c_200")
#axes.plot(Datos_x_c_500,Datos_y_c_500,"go",markersize = 2, label = "Datos_c_500")
#axes.plot(Datos_x_c_1000,Datos_y_c_1000,"bo",markersize = 2, label = "Datos_c_500")
Funcion_lagrange(Datos_x_c_200 ,Datos_y_c_200,"Funcion lagrange_c_200")
Funcion_lagrange(Datos_x_c_500 ,Datos_y_c_500,"Funcion lagrange_c_500")
Funcion_lagrange(Datos_x_c_1000 ,Datos_y_c_1000,"Funcion lagrange_c_1000")
axes.set_title("Interpolacion: Polinomio Lagrange, 100,500,1000", fontsize=15)
plt.ylim(-2.5,2.5)
plt.show()

fig, axes = plt.subplots()
axes.plot(xd, "--o",markersize = 1, label = "Error maximo" )
#axes.plot(x_err,err_c_200,":",markersize = 2, label = "error_e")
#axes.plot(x_err,err_c_500,"-.",markersize = 2, label = "error_c")
#axes.plot(x_err,err_c_1000,"-",markersize = 2, label = "error_c")
axes.minorticks_on()
axes.grid(which='minor', linestyle=':', linewidth='0.5', color='black')
axes.grid(True)
axes.legend(loc = 'best')
axes.set_ylabel("Error", fontsize=10)
axes.set_xlabel("N (x+1)*100", fontsize=10)
axes.set_title("Error Maximo", fontsize=15)
#plt.ylim(0,2)
plt.yscale("log")
plt.show()

################################################################################
#spline lineal
print("parte spline lineal")
def funcion_2(x):
    return (sin(x))**2

def linear_function(funcion,x_1,x_2,x):
    m = (funcion(x_2)-funcion(x_1))/(x_2-x_1)
    return funcion(x_1)+m*(x-x_1)

Datos_x_e = linspace(0,10,10)
Datos_y_e = []

P_to = 3.4

for i in range(len(Datos_x_e)):
    Datos_y_e.append(funcion_2(Datos_x_e[i]))


for i in range(1,len(Datos_x_e)):
    if P_to>Datos_x_e[i-1] and P_to<Datos_x_e[i]:
        x_l = Datos_x_e[i-1]
        x_u = Datos_x_e[i]

linear_function(funcion_2,x_l,x_u,P_to)

x = linspace(0,10,100)
f_x = []
for x_i in x:
    f_x.append(funcion_2(x_i))


fig, axes = plt.subplots()
axes.plot(x,f_x , "-",markersize = 1, label = "Funcion" )
axes.plot(Datos_x_e,Datos_y_e,"--o",markersize = 2, label = "Datos_e")
axes.set_title("Interpolacion: Spline lineal", fontsize=15)
#plt.ylim(-2.5,2.5)
axes.minorticks_on()
axes.grid(which='minor', linestyle=':', linewidth='0.5', color='black')
axes.grid(True)
axes.legend(loc = 'best')
axes.set_ylabel("f(x)", fontsize=10)
axes.set_xlabel("x", fontsize=10)
plt.show()

################################################################################
#spline cuadratico
print("Parte spline cuadratico")
N_Pts = 20
Datos_x_e = linspace(0,10,N_Pts)
Datos_y_e = []

for i in range(len(Datos_x_e)):
    Datos_y_e.append(funcion_2(Datos_x_e[i]))

#Datos_x_e = [3,4.5,7,9]
#Datos_y_e = [2.5,1,2.5,0.5]


A,C = M_spline_2(Datos_x_e,Datos_y_e,N_Pts)
A,C = pivoteo(A,C)
#guess = zeros(C.shape)
#coeficientes, err,iter =gaussSeidel(A, C, guess, 1e-04)
#coeficientes = E_Gauss(A,C)
#Descomposicion LU


coeficientes = E_Gauss(A,C)
#coeficientes = gauss_Seidel(A,C)

f_x_sp = []
X = []
for i in range(1,len(Datos_x_e)):
    x = linspace(Datos_x_e[i-1],Datos_x_e[i],10)
    for x_i in x:
        f_x_sp.append(coeficientes[3*i-3]*(x_i**2)+coeficientes[3*i-2]*(x_i)+coeficientes[3*i-1])
        X.append(x_i)


x = linspace(0,10,100)
f_x = []
for x_i in x:
    f_x.append(funcion_2(x_i))


fig, axes = plt.subplots()
axes.plot(X,f_x_sp , "-",markersize = 1, label = "Funcion sp" )
#axes.plot(x,f_x,"--o",markersize = 2, label = "Funcion")
axes.plot(Datos_x_e,Datos_y_e,"--o",markersize = 2, label = "Funcion")
axes.set_title("Interpolacion: Spline cuadratica", fontsize=15)
#plt.ylim(-2.5,2.5)
axes.minorticks_on()
axes.grid(which='minor', linestyle=':', linewidth='0.5', color='black')
axes.grid(True)
axes.legend(loc = 'best')
axes.set_ylabel("f(x)", fontsize=10)
axes.set_xlabel("x", fontsize=10)
plt.show()

################################################################################
#spline cubica
print("Parte spline 3")
N_Pts = 8
Datos_x_e = linspace(0,10,N_Pts)
Datos_y_e = []

Datos_x_c = puntos_chebyshev(0,10,N_Pts)
Datos_y_c = []


for i in range(len(Datos_x_e)):
    Datos_y_e.append(funcion_2(Datos_x_e[i]))

for i in range(len(Datos_x_c)):
    Datos_y_c.append(funcion_2(Datos_x_c[i]))

A,C = M_spline_3(Datos_x_e,Datos_y_e,N_Pts)
A_c,C_c = M_spline_3(Datos_x_c,Datos_y_c,N_Pts)

A,C = pivoteo(A,C)
#A_c,C_c = pivoteo(A_c,C_c)


coeficientes = E_Gauss(A,C)
coeficientes_c = gauss_Seidel(A_c,C_c)

f_x_sp = []
X = []
for i in range(1,len(Datos_x_e)):
    x = linspace(Datos_x_e[i-1],Datos_x_e[i],10)
    for x_i in x:
        f_x_sp.append(coeficientes[4*i-4]*(x_i**3)+coeficientes[4*i-3]*(x_i**2)+coeficientes[4*i-2]*(x_i)+coeficientes[4*i-1])
        X.append(x_i)

f_x_sp_c = []
X_c = []
for i in range(1,len(Datos_x_c)):
    x = linspace(Datos_x_c[i-1],Datos_x_c[i],10)
    for x_i in x:
        f_x_sp_c.append(coeficientes_c[4*i-4]*(x_i**3)+coeficientes_c[4*i-3]*(x_i**2)+coeficientes_c[4*i-2]*(x_i)+coeficientes_c[4*i-1])
        X_c.append(x_i)


x = linspace(0,10,100)
f_x = []
for x_i in x:
    f_x.append(funcion_2(x_i))


fig, axes = plt.subplots()
axes.plot(X,f_x_sp , "--o",markersize = 3, label = "Funcion sp" )
axes.plot(X_c,f_x_sp_c , "--*",markersize = 3, label = "Funcion sp_chebyshev" )
axes.plot(x,f_x,"-",markersize = 2, label = "Funcion")
Funcion_lagrange(Datos_x_e ,Datos_y_e,"Funcion lagrange_e")
#axes.plot(Datos_x_e,Datos_y_e,"--o",markersize = 2, label = "Funcion")
axes.set_title("Interpolacion: Spline, Lagrange", fontsize=15)
#plt.ylim(-2.5,2.5)
axes.minorticks_on()
axes.grid(which='minor', linestyle=':', linewidth='0.5', color='black')
axes.grid(True)
axes.legend(loc = 'best')
axes.set_ylabel("f(x)", fontsize=10)
axes.set_xlabel("x", fontsize=10)
plt.show()
