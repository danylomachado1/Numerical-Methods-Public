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
            #oldnorm = oldnorm + abs(x[j])
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


def Maximo_descenso(A, b, x):
    err = []
    r = dot(A,x)-b
    #p = r
    p = -r
    rsold = dot(transpose(r), r)

    #rsnew = dot(transpose(r), r)
    iter = 0   #iteraciones
    residual = linalg.norm(dot(A, x) - b)
    while residual > 1e-8:
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


#ejemplo
#A = array([[4,1],[1,3]])
#b = array([1,2])
#x = array([2,1])
#array([0.09090909, 0.63636364])

################################################################################
################################################################################
def Gradiente_conjudado(A, b, x):
    err = []
    r = dot(A,x)-b
    #p = r
    p = -r
    rsold = dot(transpose(r), r)

    residual = linalg.norm(dot(A, x) - b)
    #rsnew = 1
    iter = 0
    while residual > 1e-8:

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

def Confirma_simetria(A):
    N,M = A.shape
    for i in range(N):
        for j in range(N):
            if (A[i][j] != A[j][i]):
                return False
    return True

################################################################################

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
    x = linspace(Datos_x[0],Datos_x[-1]+0.5,100)
    f_x = []
    for x_i in x:
        y = newton_completo(Datos_x,Datos_y,coeficientes,x_i)
        f_x.append(y)

    axes.plot(x,f_x , "-.",markersize = 1, label = "Funcion Newton" )
    #axes.plot(Datos_x,Datos_y,"ro", label = "Datos")
    axes.minorticks_on()
    axes.grid(which='minor', linestyle=':', linewidth='0.5', color='black')
    axes.grid(True)
    axes.legend(loc = 'best')

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
################################################################################
################################################################################
###############################################################################3
#problema 1
print("El sistema de ecuacion a solucionar es el siguiente:")
print("\n")
print("(k_1+k_2)*x_1 - k_2*x_2 = 0")
print("-k_2*x_1 + (k_2+k_3)*x_2 - k_3*x_3 = 0")
print("-k_3+x_2 + (k_3 + k_4)*x_3 - k_4*x_4 = 0")
print("-k_4*x_3 + k_4*x_4 = F")
print("\n")

A = zeros((4,4), dtype = float)
C = zeros(4, dtype = float)
k_1=150
k_2=50
k_3=75
k_4=225
F=2000
A[0] = [k_1+k_2,-k_2,0,0]
A[1] = [-k_2,(k_2+k_3),-k_3,0]
A[2] = [0,-k_3,(k_4+k_3),-k_4]
A[3] = [0,0,-k_4,k_4]
C = [0,0,0,F]
print("A")
print(A)
print("C")
print(C)
print("\n")
print("gaussSeidel")
guess = zeros(len(C))
x, err,iter = gaussSeidel(A, C, guess, 1e-08)
print("los valores hallados para [x_1,x_2,x_3,x_4] :",x)
print("El numero de iteraciones fue: ", iter)

#N = 1000
#tol = 1e-8
#x,err,iter = jacobi(A,C,N,tol)
#print(x,iter)

#print("Sor")
#residual_convergence = 1e-8
#omega = 1.5 #Relaxation factor debe ser mayor a 1
#initial_guess = zeros(len(C)) #numero de incongnitas
#phi, err,iter = Metodo_SOR(A,C, omega, initial_guess, residual_convergence)
#print(phi,iter)

#print("veamos si la matriz de coeficientes es definida positiva")
#print(A)
#print("los valores propios de A son")
#print(linalg.eigvals(A))

#validar_vp = where(linalg.eigvals(A) < 0)

#if len(validar_vp[0]) == 0 and Confirma_simetria(A) :
    #print("Maximodescenso")
    #x = zeros(len(C))
    #x, err, iter = Maximo_descenso(A,C,x)
    #print(x,iter)

    #print("Gradiente conjudado")
    #x = zeros(len(C))
    #x, err,iter = Gradiente_conjudado(A,C,x)
    #print(x,iter)

K = range(1,len(err)+1)


print("La ecuacion liearizada queda como")
print("ln(err) = ln(beta) + alfa*ln(K)")
print("donde: a_t = ln(beta) y b_t = alfa")
print("Tenemos que encontrar los valores para ln(err) y ln(K)")
X_t = copy(K)
Y_t = copy(err)

for i in range(len(err)):
    X_t[i] = log(K[i])
    Y_t[i] = log(err[i])

a_t,b_t,syx,r2 = Regresion(X_t,Y_t)

text1 = '$r^{2}$ = ' + str(round(r2,5))
text2 = 'a_t,b_t = ' + str(round(a_t,3)) + ','+str(round(b_t,3))

beta = exp(a_t)
alfa = b_t

text3 = '$r$ = ' + str(round(sqrt(r2),5))
text4 = '$alfa , beta$ = ' + str(round(alfa,3))+','+str(round(beta,3))
print(text1)
print("Entonces el coeficiente de correlacion es:",sqrt(r2))
print(text2)
print("entonces:")
print("beta,alfa = ", beta, alfa)

X_linearizado = linspace(K[0],K[-1],300)
f_x_linearizado = []
for x_i in X_linearizado:
    f_x_linearizado.append(beta*x_i**alfa)


fig, axes = plt.subplots()
axes.plot(K,err,"ro",markersize = 1, label = "Error obtenido: Gauss Seidel")
axes.plot(X_linearizado,f_x_linearizado , "-",markersize = 1, label = "Linearizacion" )
axes.set_title("Interpolacion: Problema 1", fontsize=15)
axes.set_ylabel("error", fontsize=10)
axes.set_xlabel("iteracion", fontsize=10)
axes.minorticks_on()
axes.grid(which='minor', linestyle=':', linewidth='0.5', color='black')
axes.grid(True)
axes.legend(loc = 'best')
plt.text(75, 5, text3, {'color': 'black', 'fontsize': 10})
plt.text(75, 1, text4, {'color': 'black', 'fontsize': 10})
#plt.ylim(10)
plt.yscale("log")
plt.show()
print("\n")


#problema 2
print("Problema 2\n")
#lectura de Datos
datos = open("datos.dat","r")
data = datos.readlines()
datos.close()
velocidad = []
delta_P = []
#print(data)
for line in data:
    sline = line.split()
    #print(sline)
    if len(sline) != 0:
        if sline[0] != "#":
            velocidad.append(float(sline[0]))
            delta_P.append(float(sline[1]))

#convertir a array

velocidad = array(velocidad)
delta_P = array(delta_P)

M_v_newton = Vandermonde_newton(delta_P,velocidad)
coeficientes_newton = E_Gauss(M_v_newton,velocidad)

print("Los coeficientes de nuetro polinomio interpolante, usando los polinomios de Newton, son:")
print(coeficientes_newton)

A,C = M_spline_2(delta_P,velocidad,len(delta_P))
A,C = pivoteo(A,C)

coeficientes_spline2 = E_Gauss(A,C)

print("\n")
print("Los coeficientes de cada polinomio de grado 2, por intervalos, son:")
print(coeficientes_spline2)
print("Donde, los tres primeros valores son los coeficientes del primero polinomio de grado dos, los siguientes tres corresponden al siguiente polinomio y asi sucesivamente")

f_x_sp = []
X_sp = []
for i in range(1,len(delta_P)):
    x = linspace(delta_P[i-1],delta_P[i],10)
    for x_i in x:
        f_x_sp.append(coeficientes_spline2[3*i-3]*(x_i**2)+coeficientes_spline2[3*i-2]*(x_i)+coeficientes_spline2[3*i-1])
        X_sp.append(x_i)

print("\n")
print("La ecuacion liearizada queda como")
print("ln(v) = ln(a) + b*ln(delta_P)")
print("donde: a_t = ln(a) y b_t = b")
print("Tenemos que encontrar los valores para ln(v) y ln(delta_P)")
X_t = copy(delta_P)
Y_t = copy(velocidad)

for i in range(len(delta_P)):
    X_t[i] = log(delta_P[i])
    Y_t[i] = log(velocidad[i])

a_t,b_t,syx,r2 = Regresion(X_t,Y_t)

text1 = '$r^{2}$ = ' + str(round(r2,7))
text2 = 'a_t,b_t = ' + str(round(a_t,3)) + ','+str(round(b_t,3))
a = exp(a_t)
b = b_t
print(text1)
print(text2)
print("entonces:")
print("a,b = ", a,b)

X_linearizado = linspace(delta_P[0],delta_P[-1]+0.5,100)
f_x_linearizado = []
for x_i in X_linearizado:
    f_x_linearizado.append(a*x_i**b)


fig, axes = plt.subplots()
axes.plot(delta_P,velocidad,"ro",markersize = 6, label = "Datos")
axes.plot(X_sp,f_x_sp , "--",markersize = 1, label = "Funcion sp" )
axes.plot(X_linearizado,f_x_linearizado , ":",markersize = 1, label = "Linearizacion" )
Funcion_newton(delta_P,velocidad,coeficientes_newton)
#Funcion_lagrange(delta_P,velocidad,"Funcion Lagrange")
axes.set_title("Interpolacion: Problema 2", fontsize=15)
axes.set_ylabel("Velocidad (m/s)", fontsize=10)
axes.set_xlabel("Caida de presion (mm Hg) ", fontsize=10)
#plt.ylim(10)
#plt.yscale("log")
plt.show()

print("\n")

#Problema 3
print("Problema 3")
def funcion_p3(x):
    return exp(-x**2)*sin(50*x)

Datos_x_e = linspace(0,1,20)
Datos_x_c = puntos_chebyshev(0,1,20)

Datos_y_e = []
Datos_y_c = []

for i in range(len(Datos_x_e)):
    Datos_y_e.append(funcion_p3(Datos_x_e[i]))

for i in range(len(Datos_x_c)):
    Datos_y_c.append(funcion_p3(Datos_x_c[i]))

fig, axes = plt.subplots()

x = linspace(0,1,100)
f_x = []
err_e = []
err_c = []
x_err = []
for x_i in x:
    y = funcion_p3(x_i)
    if y != 0 and funcion_p3(x_i) != lagrange_completo(Datos_x_e,Datos_y_e,x_i) and funcion_p3(x_i)!=lagrange_completo(Datos_x_c,Datos_y_c,x_i):
        err_e.append(abs((funcion_p3(x_i)-lagrange_completo(Datos_x_e,Datos_y_e,x_i))))
        err_c.append(abs((funcion_p3(x_i)-lagrange_completo(Datos_x_c,Datos_y_c,x_i))))
        x_err.append(x_i)
    f_x.append(y)
axes.plot(x,f_x , "-",markersize = 1, label = "Funcion" )
axes.plot(Datos_x_e,Datos_y_e,"ro",markersize = 4, label = "P_equidistantes")
axes.plot(Datos_x_c,Datos_y_c,"go",markersize = 4, label = "P_Chebyshev")
Funcion_lagrange(Datos_x_e ,Datos_y_e,"F. Lagrange_equidistante")
Funcion_lagrange(Datos_x_c ,Datos_y_c,"F. Lagrange_Chebyshev")
axes.set_title("Problema 3: Polinomio Interpolante", fontsize=15)
plt.ylim(-2.5,2.5)
plt.show()

fig, axes = plt.subplots()
axes.plot(x_err,err_e,":",markersize = 2, label = "Error_equidistante")
axes.plot(x_err,err_c,"-.",markersize = 2, label = "Error_Chebyshev")
axes.minorticks_on()
axes.grid(which='minor', linestyle=':', linewidth='0.5', color='black')
axes.grid(True)
axes.legend(loc = 'best')
axes.set_ylabel("Error relativo", fontsize=10)
axes.set_xlabel("x", fontsize=10)
axes.set_title("Errores relativos", fontsize=15)
#plt.ylim(0,2)
plt.yscale("log")
plt.show()
