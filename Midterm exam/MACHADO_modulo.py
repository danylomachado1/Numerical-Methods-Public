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
