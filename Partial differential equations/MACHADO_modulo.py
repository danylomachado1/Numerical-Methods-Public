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
################################################################################
def metodo_busquedaincremental(funcion,x_l,x_u,paso_x,iter_max): #,error_aproximado)
    print("Se da inicio al metodo de la busqueda incremental con un paso de 1e-06\n")

    if funcion(x_l)*funcion(x_u) >= 0:
        print("El metodo de la busqueda incremental no es aplicable en este intervalo [",x_l,",",x_u,"]")
        return None
    a_n = x_l
    b_n = x_u

    for n in range(1,iter_max+1):
        m_n = b_n
        m_n_old = m_n

        m_n = a_n + paso_x  #aplicacion del metodo
        f_m_n = funcion(m_n)

        ########################################
        if funcion(a_n)*f_m_n < 0:
            a_n = a_n
            b_n = m_n

        elif funcion(b_n)*f_m_n < 0:
            a_n = m_n
            b_n = b_n

        elif f_m_n == 0:
            print("Se encontro una solucion exacta luego de ",n,"iteraciones\n")
            return m_n
        #elif error < error_aproximado:
        #    print("se encontro una solucion aproximada")
        #    return a_n+paso_x
        else:
            print("El metodo de la busqueda incremental falló")
            return None

    print("se llego al maximo de iteraciones\n")
    #print("se consiguio un error de",error)
    return m_n

################################################################################
def metodo_biseccion(funcion,x_l,x_u,iter_max,error_aproximado):

    print("Se da inicio al metodo de la biseccion\n")

    if funcion(x_l)*funcion(x_u) >= 0:
        print("El metodo de la biseccion no es aplicable en este intervalo [",x_l,",",x_u,"]")
        return None

    a_n = x_l
    b_n = x_u
    err = []

    for n in range(1,iter_max+1):

        m_n = b_n
        m_n_old = m_n

        m_n = (a_n + b_n)/2    #aplicacion del metodo
        f_m_n = funcion(m_n)

        if m_n != 0:
            error = abs((m_n-m_n_old)/m_n)
            err.append(error)

        #print(error,error_aproximado)
        #############################

        if funcion(a_n)*f_m_n < 0:
            a_n = a_n
            b_n = m_n

        if funcion(b_n)*f_m_n < 0:
            a_n = m_n
            b_n = b_n

        if f_m_n == 0:          #condicion para la solucion exacta
            print("Se encontro una solucion exacta luego de ",n,"iteraciones\n")
            return m_n,err

        if error < error_aproximado:    #cota para el error
            print("se encontro una solucion aproximada luego de ",n,"iteraciones\n")
            return m_n,err
        #else:
        #    print("El metodo de la biseccion falló")
        #    return None

    print("se llego al maximo de iteraciones")
    print("se consiguio un error de",error)
    return (a_n+b_n)/2,err

################################################################################
def metodo_falsaposicion(funcion,x_l,x_u,iter_max,error_aproximado):

    print("Se da inicio al metodo de la falsa posicion\n")

    if funcion(x_l)*funcion(x_u) >= 0:
        print("El metodo de la falsa posicion no es aplicable en este intervalo [",x_l,",",x_u,"]")
        return None

    a_n = x_l
    b_n = x_u
    err = []

    for n in range(1,iter_max+1):

        m_n = b_n
        m_n_old = m_n

        m_n = b_n - (a_n-b_n)*funcion(b_n)/(funcion(a_n)-funcion(b_n))   #aplicacion del metodo
        f_m_n = funcion(m_n)


        if m_n != 0:
            error = abs((m_n-a_n)/m_n) #cambion m_n_old por a_n
            err.append(error)

        #print(m_n,m_n_old,abs((m_n-m_n_old)/m_n),error)
        ##################################

        if funcion(a_n)*f_m_n < 0:
            a_n = a_n
            b_n = m_n

        if funcion(b_n)*f_m_n < 0:
            a_n = m_n
            b_n = b_n

        if f_m_n == 0:
            print("Se encontro una solucion exacta luego de ",n,"iteraciones\n")
            return m_n,err

        if error < error_aproximado:    #cota para el error
            print("se encontro una solucion aproximada luego de ",n,"iteraciones\n")
            return m_n,err

    print("se llego al maximo de iteraciones")
    print("se consiguio un error de",error)
    return a_n - (b_n-a_n)*funcion(a_n)/(funcion(b_n)-funcion(a_n)),err

################################################################################
def metodo_newtonraphson(funcion,funcion_derivada,x_0,iter_max,error_aproximado):

    print("Se da inicio al metodo de newton raphson\n")

    xn = x_0
    err = []
    for n in range(1,iter_max+1):

        xn_old = xn

        fxn = funcion(xn)
        Dfxn = funcion_derivada(xn)
        xn = xn - fxn/Dfxn   #apicacion del metodo
        if xn != 0:
            error = abs((xn-xn_old)/xn)
            err.append(error)
        #print(xn,fxn,error)

        if error < error_aproximado:   #cota para el error
            print('Se encontro la solucion despues de ',n,'iteraciones\n')
            return xn,err

        if Dfxn == 0:     #condicion del metodo
            print('Derivada igual a cero. No se encontro solucion.')
            return None

    print('Se excedio el numero maximo de iteraciones. No se encontro solucion.')
    return None
################################################################################
def metodo_secante(funcion,x_l,x_u,iter_max,error_aproximado): #x_l,x_u> raiz

    print("Se da inicio al metodo de la secante\n")


    a_n = x_l
    b_n = x_u

    err =[]

    for n in range(1,iter_max+1):
        m_n = b_n - funcion(b_n)*(a_n - b_n)/(funcion(a_n) - funcion(b_n)) #aplicacion del metodo
        f_m_n = funcion(m_n)

        if m_n != 0:
            error = abs((m_n-a_n)/m_n)
            err.append(error)

        ##############################3

            a_n = m_n

        if f_m_n == 0:
            print("Se encontro solucion exacta luego de ",n,"iteraciones\n")
            return m_n,err
        if error < error_aproximado:        #cota para el error
            print("se encontro una solucion aproximada luego de ",n,"iteraciones\n")
            return m_n,err

    print("se llego al maximo de iteraciones")
    print("se consiguio un error de",error)
    return a_n - funcion(a_n)*(b_n - a_n)/(funcion(b_n) - funcion(a_n)),err

################################################################################
#funciona para funciones de la forma f(x) = x + g(x), lo que analizamos con este metodo es la funcion g(x)

def  metodo_puntofijo(funcion,x_0,iter_max,error_aproximado):

    print("Se da inicio al metodo del punto fijo\n")

    a_n = x_0
    err = []
    for n in range(1, iter_max+1):
        p = funcion(a_n)     #aplicacion del metodo
        error = abs((p-a_n)/p)
        err.append(error)

        #print(a_n,p,error)
        if error < error_aproximado:       #cota para el error
            print("Se encontro una solucion aproximada luego de ",n,"iteraciones\n")
            return p,err
        a_n = p
    print("el metodo fallo despues de ",iter_max,"iteraciones")

def graficador_a(funcion,a,b,lab,ylab,xlab,y1,y2):

        fig, axes = plt.subplots()
        x = linspace(a,b,3000)
        f_x = []
        cero = []
        for x_i in x:
            y = funcion(x_i)
            f_x.append(y)
            cero.append(0)


        axes.plot(x,f_x , "-",markersize = 0.5, label = lab )   ### se podria poner en vez de "o" "--o" para que se visualice la linea que sigue pero se hace dificil la visualizacion de los otros puntos, asi mismo, el valor de markersize tambien puede incrementarse.
        #axes.plot(x,cero,"k-",markersize = 0.5, label = "linea cero")
        axes.minorticks_on()
        axes.grid(which='minor', linestyle=':', linewidth='0.5', color='black')
        axes.grid(True)
        axes.legend(loc = 'upper right')
        axes.set_ylabel(ylab, fontsize=10)
        axes.set_xlabel(xlab, fontsize=10)
        axes.set_title("Funcion de interes", fontsize=15)
        plt.ylim(y1,y2)
        #plt.yscale("log")
        #plt.xscale("log")
        ########## visualization #################
        plt.show()
        #plt.yscale("log")    # se puede cambiar para visualizar mejor
        #plt.xscale("log")
        #plt.savefig('exercise_a.png', dpi=800)
        #plt.close()

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

def ajuste_error(K,err,lab1, lab2):
    #lab1 = "Error obtenido: metodo"
    #lab2 = "Linearizacion-metodo"
    X_t = copy(K)
    Y_t = copy(err)

    for i in range(len(err)):
        X_t[i] = log(K[i])
        Y_t[i] = log(err[i])

    a_t,b_t,syx,r2 = Regresion(X_t,Y_t)

    text1 = '$r^{2}$ = ' + str(round(r2,5))
    text2 = 'a_t,b_t = ' + str(round(a_t,3)) + ','+str(round(b_t,3))

    alfa = exp(a_t)
    beta = b_t

    text3 = '$r$ = ' + str(round(sqrt(r2),5))
    text4 = '$alfa , beta$ = ' + str(round(alfa,3))+','+str(round(beta,3))

    lab2 = lab2 + "\nalfa, beta = "+ str(round(alfa,3))+','+str(round(beta,3))+"\nr ="+ str(round(sqrt(r2),5))

    print(text1)
    print("Entonces el coeficiente de correlacion es:",sqrt(r2))
    print(text2)
    print("entonces:")
    print("alfa,beta = ", alfa, beta)

    X_linearizado = linspace(K[0],K[-1],300)
    f_x_linearizado = []
    for x_i in X_linearizado:
        f_x_linearizado.append(alfa*x_i**beta)


    fig, axes = plt.subplots()
    axes.plot(K,err,"ro",markersize = 2, label = lab1)
    axes.plot(X_linearizado,f_x_linearizado , "r-",markersize = 1, label = lab2  )
    axes.set_title("Ajuste del error", fontsize=15)
    axes.set_ylabel("error", fontsize=10)
    axes.set_xlabel("iteracion", fontsize=10)
    axes.minorticks_on()
    axes.grid(which='minor', linestyle=':', linewidth='0.2', color='black')
    axes.grid(True)
    axes.legend(loc = 'best')
    #plt.text(K[0]+5, 0.1, text3, {'color': 'black', 'fontsize': 10})
    #plt.text(K[0]+5, 0.2, text4, {'color': 'black', 'fontsize': 10})
    #plt.ylim(10)
    plt.yscale("log")
    plt.show()

def ajuste_multivariable(Y,X_1,X_2):
    n = len(Y)
    sum_Y = sum(Y)
    sum_X_1 = sum(X_1)
    sum_X_2 = sum(X_2)
    sum_X_1_X_1 = sum(X_1**2)
    sum_X_1_X_2 = sum(X_1*X_2)
    sum_X_2_X_2 = sum(X_2**2)
    sum_X_1_Y = sum(X_1*Y)
    sum_X_2_Y = sum(X_2*Y)

    N_variables = 2
    A = zeros((N_variables+1,N_variables+1), dtype = float)
    C = zeros(N_variables+1, dtype = float)

    A[0,0] = n      ; A[0,1] = sum_X_1    ; A[0,2] = sum_X_2    ;C[0] = sum_Y
    A[1,0] = sum_X_1; A[1,1] = sum_X_1_X_1; A[1,2] = sum_X_1_X_2;C[1] = sum_X_1_Y
    A[2,0] = sum_X_2; A[2,1] = sum_X_1_X_2; A[2,2] = sum_X_2_X_2;C[2] = sum_X_2_Y

    #gaussSeidel(A, b, x, tol)
    x = E_Gauss(A, C)
    alpha_0 = exp(x[0])
    alpha_1 = x[1]
    alpha_2 = x[2]

    return alpha_0,alpha_1,alpha_2


################################################################################


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
    fig, axes = plt.subplots()
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




################################################################################
#Tarea 7
def M_trapecio(funcion,X):
    # n = numero de intervalos
    sum = 0
    if len(X) == 2:
        return (X[-1]-X[0])*((funcion(X[-1])+funcion(X[0]))/2)


    else:
        for i in range(1,int(len(X))):
            sum = sum + (X[i]-X[i-1])*(funcion(X[i-1])+funcion(X[i]))/2
        return sum

def M_trapecio_datos(X,Y):
    sum = 0
    for i in range(1,int(len(X))):
        sum = sum + (X[i]-X[i-1])*(Y[i-1]+Y[i])/2
    return sum

def M_simpson_simple(funcion,X,opc):
    if opc == "1/3":
        h = (X[1]-X[0])/2
        return 2*h*(funcion(X[0])+4*funcion(h+X[0])+funcion(X[1]))/6

    if opc == "3/8":
        h = (X[1]-X[0])/3
        return 3*h*(funcion(X[0])+3*(funcion(h+X[0])+funcion(2*h+X[0]))+funcion(X[1]))/8

def M_simpson13(a, b, n, f ):
    h = (b - a)/n        #n debe ser impar
    x = list()
    fx = list()
    i = 0
    while i<= n:
        x.append(a + i * h)
        fx.append(f(x[i]))
        i += 1

    sum = 0
    i = 0
    while i<= n:
        if i == 0 or i == n:
            sum+= fx[i]
        elif i % 2 != 0:
            sum+= 4 * fx[i]
        else:
            sum+= 2 * fx[i]
        i+= 1
    sum = sum * (h / 3)
    return sum

def M_simpson38(a, b, n, func):
	h = (b - a)/n         #n debe ser multiplo de 3
	sum = func(a) + func(b)
	for i in range(1, n):
		if (i % 3 == 0):
			sum = sum + 2*func(a + i*h)
		else:
			sum = sum + 3*func(a + i*h)

	return ((3*h)/8)*sum

################################################################################
#Tarea 8

P3_GL_r  = array([-sqrt(3/5),0,sqrt(3/5)])
P3_GRL_r = array([-1.00000,-0.289898,0.689898])
P3_GLL_r = array([-1.000000000000000,0.000000000000000,1.000000000000000])
P3_GL_w  = array([5/9,8/9,5/9])
P3_GRL_w = array([0.222222,1.0249717,0.7528061])
P3_GLL_w = array([0.333333333333333,1.333333333333333,0.333333333333333])

P4_GL_r  = array([-sqrt((3/7) + (2/7)*sqrt(6/5)),-sqrt((3/7) - (2/7)*sqrt(6/5)),sqrt((3/7) - (2/7)*sqrt(6/5)),sqrt((3/7) + (2/7)*sqrt(6/5))])
P4_GRL_r = array([-1.000000,-0.575319,0.181066,0.822824])
P4_GLL_r = array([-1.000000000000000,-0.447213595499958,0.447213595499958,1.000000000000000])
P4_GL_w  = array([(18-sqrt(30))/36,(18+sqrt(30))/36,(18+sqrt(30))/36,(18-sqrt(30))/36])
P4_GRL_w = array([0.125000,0.657689,0.776387,0.440924])
P4_GLL_w = array([0.166666666666667,0.833333333333333,0.833333333333333,0.166666666666667])

P5_GL_r  = array([-(1/3)*sqrt(5 + 2*sqrt(10/7)),-(1/3)*sqrt(5 - 2*sqrt(10/7)),0,(1/3)*sqrt(5 - 2*sqrt(10/7)),(1/3)*sqrt(5 + 2*sqrt(10/7))])
P5_GRL_r = array([-1.000000,-0.720480,-0.167181,0.446314,0.885792])
P5_GLL_r = array([-1.000000000000000,-0.654653670707977,0.000000000000000,0.654653670707977,1.000000000000000])
P5_GL_w  = array([(322-13*sqrt(70))/900,(322+13*sqrt(70))/900,128/225,(322+13*sqrt(70))/900,(322-13*sqrt(70))/900])
P5_GRL_w = array([0.080000,0.446208,0.623653,0.562712,0.287427])
P5_GLL_w = array([0.100000000000000,0.544444444444444,0.711111111111111,0.544444444444444,0.100000000000000])


def fun_tarea8(x):
    return 4/(1+x**2)

def fun_A_B(x,A,B):
    return fun_tarea8(((B-A)/2)*x+(A+B)/2)

def GL(fun_A_B,A,B,etiqueta):
    if etiqueta == "3":
        sum = 0
        for i in range(3):
            sum = sum + ((B-A)/2)*P3_GL_w[i]*fun_A_B(P3_GL_r[i],A,B)
        return sum

    if etiqueta == "4":
        sum = 0
        for i in range(4):
            sum = sum + ((B-A)/2)*P4_GL_w[i]*fun_A_B(P4_GL_r[i],A,B)
        return sum

    if etiqueta == "5":
        sum = 0
        for i in range(5):
            sum = sum + ((B-A)/2)*P5_GL_w[i]*fun_A_B(P5_GL_r[i],A,B)
        return sum

def GRL(fun_A_B,A,B,etiqueta):
	if etiqueta == "3":
		sum = 0
		for i in range(3):
			sum = sum + ((B-A)/2)*P3_GRL_w[i]*fun_A_B(P3_GRL_r[i],A,B)
		return sum

	if etiqueta == "4":
		sum = 0
		for i in range(4):
			sum = sum + ((B-A)/2)*P4_GRL_w[i]*fun_A_B(P4_GRL_r[i],A,B)
		return sum

	if etiqueta == "5":
		sum = 0
		for i in range(5):
			sum = sum + ((B-A)/2)*P5_GRL_w[i]*fun_A_B(P5_GRL_r[i],A,B)
		return sum

def GLL(fun_A_B,A,B,etiqueta):
    if etiqueta == "3":
        sum = 0
        for i in range(3):
            sum = sum + ((B-A)/2)*P3_GLL_w[i]*fun_A_B(P3_GLL_r[i],A,B)
        return sum

    if etiqueta == "4":
        sum = 0
        for i in range(4):
            sum = sum + ((B-A)/2)*P4_GLL_w[i]*fun_A_B(P4_GLL_r[i],A,B)
        return sum

    if etiqueta == "5":
        sum = 0
        for i in range(5):
            sum = sum + ((B-A)/2)*P5_GLL_w[i]*fun_A_B(P5_GLL_r[i],A,B)
        return sum

def P_derivada_forward(X,T,i):
    h = T[i+1]-T[i]
    a = (-X[i+2]+4*X[i+1]-3*X[i])/(2*h)
    return a

def P_derivada_backward(X,T,i):
    h = T[i]-T[i-1]
    a = (X[i-2]-4*X[i-1]+3*X[i])/(2*h)
    return a

def P_derivada_central(X,T,i):
    h = T[i]-T[i-1]
    a = (X[i+1]-X[i-1])/(2*h)
    return a

def S_derivada_forward(X,T,i):
    h = T[i+1]-T[i]
    a = (-X[i+3]+4*X[i+2]-5*X[i+1]+2*X[i])/(h**2)
    return a

def S_derivada_backward(X,T,i):
    h = T[i]-T[i-1]
    a = (-X[i-3]+4*X[i-2]-5*X[i-1]+2*X[i])/(h**2)
    return a

def S_derivada_central(X,T,i):
    h = T[i]-T[i-1]
    a = (X[i+1]-2*X[i]+X[i-1])/(h**2)
    return a

################################################################################
#Tarea 9


#El primer metodo para la resolucion de ecuacines diferenciales, es el mas inexacto como veremos mas adelante
def Metodo_euler(y0, t0, t_final, h, f):
	lista_t = []
	lista_y = []

	y = y0         #valor inicial del problema, en este caso y(0) = 1
	t = t0        #punto inical y(t) = y0

	while t <= t_final+h:		# para incluir t_final
		lista_t.append(round(t,6))
		lista_y.append(y)

		y = y + h * f(y,t)    #usamos la definicion del metodo, de manera iterativa iremos llenando la lista_y
		t = t + h

	return lista_t, lista_y

def Metodo_euler_sistema_ex(y1_0,y2_0,t0,t_final,h,f1,f2):
    lista_t = []
    lista_y1 = []
    lista_y2 = []

    y1 = y1_0
    y2 = y2_0
    t = t0

    while t <= t_final:
        lista_t.append(round(t,6))
        lista_y1.append(y1)
        lista_y2.append(y2)
        y1 = y1 + h*f1(y1,y2)
        y2 = y2 + h*f2(y1,y2)
        t = t + h

    return lista_t, lista_y1, lista_y2

def Metodo_euler_sistema_im(y1_0,y2_0,t0,t_final,h):
    lista_t = []
    lista_y1 = []
    lista_y2 = []

    y1 = y1_0
    y2 = y2_0
    t = t0

    while t <= t_final:
        lista_t.append(round(t,6))
        lista_y1.append(y1)
        lista_y2.append(y2)

        A = zeros((2,2), dtype = float)
        C = zeros(2, dtype = float)
        A[0,0] = 1-999*h
        A[0,1] = -1999*h
        A[1,0] = 1000*h
        A[1,1] = 1+2000*h
        C[0] = y1
        C[1] = y2
        Y = E_Gauss(A, C)

        y1 = Y[0]
        y2 = Y[1]
        t = t + h

    return lista_t, lista_y1, lista_y2


#Metodo rk2 modificado de euler
def Metodo_Heund(y0,t0, t_final, h, f):
	lista_t  = []
	lista_y = []

	y = y0        #valor inicial del problema, en este caso y(0) = 1
	t = t0        #punto inicial   y(t) = y0

	while t <= t_final+h:		# para incluir t_final
		lista_t.append(round(t,6))
		lista_y.append(y)

		k1 = f(y,t)
		k2 = f(y + h*k1, t+h)

		y = y + (0.5*k1 + 0.5*k2)*h
		t = t + h

	return lista_t, lista_y

def Metodo_Heund_correccion(y0,t0, t_final, h, f):
    lista_t  = []
    lista_y = []
    y = y0        #valor inicial del problema, en este caso y(0) = 1
    t = t0        #punto inicial   y(t) = y0
    while t <= t_final+h:
        lista_t.append(round(t,6))
        lista_y.append(y)
        f0 = f(y,t)
        y_i = y + f(y,t)*h
        err = 10
        while err > 1e-6:
            y_s = y + (f0 + f(y_i,t+h))*h/2
            y_a = y_i
            y_i = y_s
            err = (abs(y_s-y_a))/y_s
            #print(y_s,err)

        y = y_s
        t = t + h
    return lista_t, lista_y

def Metodo_Heund_modificado(y_back,y0,t0, t_final, h, f):
    lista_t  = []
    lista_y = []
    yb = y_back
    y = y0        #valor inicial del problema, en este caso y(0) = 1
    t = t0        #punto inicial   y(t) = y0
    while t <= t_final:
        lista_t.append(round(t,6))
        lista_y.append(y)
        f0 = f(y,t)
        y_i = yb + f(y,t)*2*h
        err = 10
        while err > 1e-4:
            y_s = y + (f0 + f(y_i,t+h))*h/2
            y_a = y_i
            y_i = y_s
            err = (abs(y_s-y_a))/y_s
            #print(y_s,err)

        yb = y
        y = y_s
        t = t + h
    return lista_t, lista_y


#Metodo rk2 modificado de euler
def Metodo_puntomedio(y0,t0, t_final, h, f):
	lista_t  = []
	lista_y = []

	y = y0        #valor inicial del problema, en este caso y(0) = 1
	t = t0        #punto inicial   y(t) = y0

	while t <= t_final+h:		# para incluir t_final
		lista_t.append(round(t,6))
		lista_y.append(y)

		k1 = f(y,t)
		k2 = f(y + 0.5*h*k1, t+0.5*h)

		y = y + h * k2
		t = t + h

	return lista_t, lista_y

#Metodo rk2 modificado de euler
def Metodo_Ralston(y0,t0, t_final, h, f):
	lista_t  = []
	lista_y = []

	y = y0        #valor inicial del problema, en este caso y(0) = 1
	t = t0        #punto inicial   y(t) = y0

	while t <= t_final+h:		# para incluir t_final
		lista_t.append(round(t,6))
		lista_y.append(y)

		k1 = f(y,t)
		k2 = f(y + (3/4)*h*k1, t+(3/4)*h)

		y = y + ((1/3)*k1 + (2/3)*k2)*h
		t = t + h

	return lista_t, lista_y

def Metodo_rk3(y0,t0, t_final, h, f):
	lista_t  = []
	lista_y = []

	y = y0
	t = t0

	while t <= t_final+h:		# para incluir t_final
		lista_t.append(t)
		lista_y.append(y)

		k1 = f(y,t)
		k2 = f(y + 0.5*h*k1, t+0.5*h)
		k3 = f(y - k1*h + 2*k2*h, t+h)

		y += h/6. * (k1 + 4.*k2 + k3)
		t += h

	return lista_t, lista_y


def Metodo_rk4(y0,t0, t_final, h, f):
	lista_t  = []
	lista_y = []

	y = y0
	t = t0

	while t <= t_final+h:		# para incluir t_final
		lista_t.append(t)
		lista_y.append(y)

		k1 = f(y,t)
		k2 = f(y + 0.5*h*k1, t+0.5*h)
		k3 = f(y + 0.5*h*k2, t+0.5*h)
		k4 = f(y + h*k3, t+h)

		y += h/6. * (k1 + 2.*k2 + 2.*k3 + k4)
		t += h

	return lista_t, lista_y

################################################################################
#PC3

def rk4_completo3(x1_0, x2_0, x3_0, t_0, t_final, h, f1, f2, f3):
    lista_t = []
    lista_x1 = []
    lista_x2 = []
    lista_x3 = []

    x1 = x1_0
    x2 = x2_0         #valor inicial del problema, en este caso y(0) = 1
    x3 = x3_0
    t = t_0       #punto inicial   y(t) = y0

    while t <= t_final:
        # para incluir t_final
        lista_t.append(round(t,6))
        lista_x1.append(x1)
        lista_x2.append(x2)
        lista_x3.append(x3)
        k1=h*f1(x1, x2, x3, t)
        l1=h*f2(x1, x2, x3, t)
        m1=h*f3(x1, x2, x3, t)
        k2=h*f1(x1+k1/2, x2+l1/2, x3+m1/2, t+h/2)
        l2=h*f2(x1+k1/2, x2+l1/2, x3+m1/2, t+h/2)
        m2=h*f3(x1+k1/2, x2+l1/2, x3+m1/2, t+h/2)
        k3=h*f1(x1+k2/2, x2+l2/2, x3+m2/2, t+h/2)
        l3=h*f2(x1+k2/2, x2+l2/2, x3+m2/2, t+h/2)
        m3=h*f3(x1+k2/2, x2+l2/2, x3+m2/2, t+h/2)
        k4=h*f1(x1+k3, x2+l3, x3+m3, t+h)
        l4=h*f2(x1+k3, x2+l3, x3+m3, t+h)
        m4=h*f3(x1+k3, x2+l3, x3+m3, t+h)
        x1+=(k1+2*k2+2*k3+k4)/6
        x2+=(l1+2*l2+2*l3+l4)/6
        x3+=(m1+2*m2+2*m3+m4)/6
        t+=h

    return lista_t, lista_x1, lista_x2, lista_x3

def rk4_completo4(x1_0, x2_0, x3_0, x4_0,t_0, t_final, h, f1, f2, f3, f4):
    lista_t = []
    lista_x1 = []
    lista_x2 = []
    lista_x3 = []
    lista_x4 = []

    x1 = x1_0
    x2 = x2_0         #valor inicial del problema, en este caso y(0) = 1
    x3 = x3_0
    x4 = x4_0
    t = t_0       #punto inicial   y(t) = y0

    while t <= t_final:
        # para incluir t_final
        lista_t.append(round(t,6))
        lista_x1.append(x1)
        lista_x2.append(x2)
        lista_x3.append(x3)
        lista_x4.append(x4)
        k1=h*f1(x1, x2, x3, x4, t)
        l1=h*f2(x1, x2, x3, x4, t)
        m1=h*f3(x1, x2, x3, x4, t)
        n1=h*f4(x1, x2, x3, x4, t)
        k2=h*f1(x1+k1/2, x2+l1/2, x3+m1/2, x4+n1/2, t+h/2)
        l2=h*f2(x1+k1/2, x2+l1/2, x3+m1/2, x4+n1/2, t+h/2)
        m2=h*f3(x1+k1/2, x2+l1/2, x3+m1/2, x4+n1/2, t+h/2)
        n2=h*f4(x1+k1/2, x2+l1/2, x3+m1/2, x4+n1/2, t+h/2)
        k3=h*f1(x1+k2/2, x2+l2/2, x3+m2/2, x4+n2/2, t+h/2)
        l3=h*f2(x1+k2/2, x2+l2/2, x3+m2/2, x4+n2/2, t+h/2)
        m3=h*f3(x1+k2/2, x2+l2/2, x3+m2/2, x4+n2/2, t+h/2)
        n3=h*f4(x1+k2/2, x2+l2/2, x3+m2/2, x4+n2/2, t+h/2)
        k4=h*f1(x1+k3, x2+l3, x3+m3, x4+n3, t+h)
        l4=h*f2(x1+k3, x2+l3, x3+m3, x4+n3, t+h)
        m4=h*f3(x1+k3, x2+l3, x3+m3, x4+n3, t+h)
        n4=h*f4(x1+k3, x2+l3, x3+m3, x4+n3, t+h)
        x1+=(k1+2*k2+2*k3+k4)/6
        x2+=(l1+2*l2+2*l3+l4)/6
        x3+=(m1+2*m2+2*m3+m4)/6
        x4+=(n1+2*n2+2*n3+n4)/6
        t+=h

    return lista_t, lista_x1, lista_x2, lista_x3, lista_x4

def rk4_completo2(x1_0, x2_0, t_0, t_final, h, f1, f2):
    lista_t = []
    lista_x1 = []
    lista_x2 = []


    x1 = x1_0
    x2 = x2_0         #valor inicial del problema, en este caso y(0) = 1

    t = t_0       #punto inicial   y(t) = y0

    while t <= t_final:
        # para incluir t_final
        lista_t.append(round(t,6))
        lista_x1.append(x1)
        lista_x2.append(x2)

        k1=h*f1(x1, x2, t)
        l1=h*f2(x1, x2, t)

        k2=h*f1(x1+k1/2, x2+l1/2, t+h/2)
        l2=h*f2(x1+k1/2, x2+l1/2, t+h/2)

        k3=h*f1(x1+k2/2, x2+l2/2, t+h/2)
        l3=h*f2(x1+k2/2, x2+l2/2, t+h/2)

        k4=h*f1(x1+k3, x2+l3, t+h)
        l4=h*f2(x1+k3, x2+l3, t+h)

        x1+=(k1+2*k2+2*k3+k4)/6
        x2+=(l1+2*l2+2*l3+l4)/6

        t+=h

    return lista_t, lista_x1, lista_x2

################################################################################
#Tarea 10

def Dif_finitas_1(xi,xf,n,T1,T2,T_a):
    x = linspace(xi,xf,n)
    h = (xf-xi)/(n-1)
    A = zeros((n-2,n-2), dtype = float)
    C = zeros(n-2, dtype = float)
    h_p = 0.01

    for i in range(n-2):
        if i == 0:
            A[i,i] = (2+h_p*h**2)
            A[i,i+1] = -1
            C[i] = h_p*h**2*20+T1

        if i == n-3:
            A[i,i] = (2+0.01*h**2)
            A[i,i-1] = -1
            C[i] = h_p*h**2*20+T2

        if i != 0 and i != n-3:
            A[i,i] = (2+h_p*h**2)
            A[i,i-1] = -1
            A[i,i+1] = -1
            C[i] = h_p*h**2*T_a  #ta = 20 temperatura ambiente

    T = zeros(len(A))
    T = gaussSeidel(A,C,T,1e-06)
    #print(T)
    x = linspace(xi+h,xf-h,n-2)
    T = T[0]
    #print(T)
    return T,x

def Dif_finitas_2(xi,xf,n,T1_p,T2,T_inf):  #n: numero de puntos, #intervalo = n - 1
    x = linspace(xi,xf,n)
    h = (xf-xi)/(n-1)
    A = zeros((n-1,n-1), dtype = float)
    C = zeros(n-1, dtype = float)
    h_p = 0.01

    for i in range(n-1):
        if i == 0:
            A[i,i] = (2+h_p*h**2)
            A[i,i+1] = -2
            C[i] = h_p*h**2*T_inf - 2*h*T1_p

        if i == n-2:
            A[i,i] = (2+h_p*h**2)
            A[i,i-1] = -1
            C[i] = h_p*h**2*T_inf+T2


        if i != 0 and i != n-2:
            A[i,i] = (2+h_p*h**2)
            A[i,i-1] = -1
            A[i,i+1] = -1
            C[i] = h_p*h**2*T_inf

    T = zeros(len(A))
    T = gaussSeidel(A,C,T,1e-06)
    x = linspace(xi,xf-h,n-1)
    T = T[0]
    return T,x

def Metodo_disparo(x1_0, x2_0,guess1,guess2, t_0, t_final, h, f1, f2):

    lista_ta, lista_x1a, lista_x2a = rk4_completo2(x1_0, guess1, t_0, t_final, h, f1, f2)
    lista_tb, lista_x1b, lista_x2b = rk4_completo2(x1_0, guess2, t_0, t_final, h, f1, f2)

    z1 = guess1
    z2 = guess2
    x2_0_1 = lista_x1a[-1]
    x2_0_2 = lista_x1b[-1]

    z0 = z1 + ((z2-z1)/(x2_0_2-x2_0_1))*(x2_0 - x2_0_1)

    lista_t, lista_x1, lista_x2 = rk4_completo2(x1_0, z0, t_0, t_final, h, f1, f2)

    return lista_t, lista_x1, lista_x1a, lista_x1b


#def Metodo_potencia(mat, start, maxit):
def Metodo_potencia(A, X, iter_max):
    """
    Diferencia entre norm y max
    """
    result = X
    for i in range(iter_max):
        result = dot(A,result)
        result = result/linalg.norm(result)
        #result = result/max(result)
    return result



def check(A, result):

    prd = dot(A,result)
    eigval = prd[0]/result[0]
    print('valor propio hallado :' , eigval)
    print('vector propio hallado :' , result)
    [eigs, vecs] = linalg.eig(A)
    abseigs = list(abs(eigs))
    ind = abseigs.index(max(abseigs))
    print('valor propio mas alto :', eigs[ind])
    print('vector propio mas alto :', vecs[ind])
