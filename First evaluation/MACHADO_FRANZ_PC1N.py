#! /usr/bin/python3
from numpy import *
import matplotlib.pyplot as plt # from pylab import plot,show
import warnings
warnings.filterwarnings("ignore")


#########################################################
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

        else:
            print("El metodo de la busqueda incremental falló")
            return None

    print("se llego al maximo de iteraciones\n")
    #print("se consiguio un error de",error)
    return m_n

#############################################################################################################def metodo_biseccion(funcion,x_l,x_u,iter_max,error_aproximado):

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


    print("se llego al maximo de iteraciones")
    print("se consiguio un error de",error)
    return (a_n+b_n)/2,err

#############################################################################################################

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
            error = abs((m_n-a_n)/m_n)
            err.append(error)



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

#############################################################################################################
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
#############################################################################################################

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

#############################################################################################################
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

#############################################################################################################


def graficador_a(funcion,a,b,lab):

        fig, axes = plt.subplots()
        x = linspace(a,b,40000)
        f_x = []
        cero = []
        for x_i in x:
            y = funcion(x_i)
            f_x.append(y)
            cero.append(0)


        axes.plot(x,f_x , "-",markersize = 0.5, label = lab )   ### se podria poner en vez de "o" "--o" para que se visualice la linea que sigue pero se hace dificil la visualizacion de los otros puntos, asi mismo, el valor de markersize tambien puede incrementarse.
        axes.plot(x,cero,"k-",markersize = 0.5, label = "linea cero")
        axes.minorticks_on()
        axes.grid(which='minor', linestyle=':', linewidth='0.5', color='black')
        axes.grid(True)
        axes.legend(loc = 'upper right')
        axes.set_ylabel("f(x) (N)", fontsize=10)
        axes.set_xlabel("x (m)", fontsize=10)
        axes.set_title("Funcion de interes", fontsize=15)

        ########## visualization #################
        plt.show()
        #plt.yscale("log")    # se puede cambiar para visualizar mejor
        #plt.xscale("log")
        #plt.savefig('exercise_a.png', dpi=800)
        #plt.close()

def graficador_b(funcion,a,b,lab):

        fig, axes = plt.subplots()
        x = linspace(a,b,40000)
        f_x = []
        cero = []
        for x_i in x:
            y = funcion(x_i)
            f_x.append(y)
            cero.append(0)


        axes.plot(x,f_x , "-",markersize = 0.5, label = lab )   ### se podria poner en vez de "o" "--o" para que se visualice la linea que sigue pero se hace dificil la visualizacion de los otros puntos, asi mismo, el valor de markersize tambien puede incrementarse.
        axes.plot(x,cero,"k-",markersize = 0.5, label = "linea cero")
        axes.minorticks_on()
        axes.grid(which='minor', linestyle=':', linewidth='0.5', color='black')
        axes.grid(True)
        axes.legend(loc = 'upper right')
        axes.set_ylabel("f(h)", fontsize=10)
        axes.set_xlabel("h (m)", fontsize=10)
        axes.set_title("Funcion de interes", fontsize=15)

        ########## visualization #################
        plt.show()
        #plt.yscale("log")    # se puede cambiar para visualizar mejor
        #plt.xscale("log")
        #plt.savefig('exercise_a.png', dpi=800)
        #plt.close()
##########################################################
##########################################################
def graficador_tres(funcion_1,funcion_2,funcion_3,a,b):

        fig, axes = plt.subplots()
        x = linspace(a,b,40000)
        f1_x = []
        f2_x = []
        f3_x = []
        cero = []
        for x_i in x:
            y_1 = funcion_1(x_i)
            y_2 = funcion_2(x_i)

            y_3 = funcion_3(x_i)
            f3_x.append(y_3)
            f1_x.append(y_1)
            f2_x.append(y_2)

            cero.append(0)


        axes.plot(x,f1_x , "-",markersize = 0.5, label = "funcion: F_x" )   ### se podria poner en vez de "o" "--o" para que se visualice la linea que sigue pero se hace dificil la visualizacion de los otros puntos, asi mismo, el valor de markersize tambien puede incrementarse.
        axes.plot(x,f2_x , "-",markersize = 0.5, label = "funcion: F_y" )
        axes.plot(x,f3_x , "-",markersize = 0.5, label = "funcion: F_total" )
        axes.plot(x,cero,"k-",markersize = 0.5, label = "linea cero")
        axes.minorticks_on()
        axes.grid(which='minor', linestyle=':', linewidth='0.5', color='black')
        axes.grid(True)
        axes.legend(loc = 'center right')
        axes.set_ylabel("f(x) (N)", fontsize=10)
        axes.set_xlabel("x (m)", fontsize=10)
        axes.set_title("Fuerza electrica", fontsize=15)

        ########## visualization #################
        plt.show()
        #plt.yscale("log")    # se puede cambiar para visualizar mejor
        #plt.xscale("log")
        #plt.savefig('exercise_a.png', dpi=800)
        #plt.close()

#########################################################
def graficador_error(err_1,err_2,err_3):
    fig, axes = plt.subplots()

    axes.plot(err_1, "--o",markersize = 4, label = "Error : Biseccion" )   ### se podria poner en vez de "o" "--o" para que se visualice la linea que sigue pero se hace dificil la visualizacion de los otros puntos, asi mismo, el valor de markersize tambien puede incrementarse.
    axes.plot(err_2, "--*",markersize = 4, label = "Error : Secante" )
    axes.plot(err_3, "--h",markersize = 4, label = "Error : Newton Raphson" )


    axes.minorticks_on()
    axes.grid(which='minor', linestyle=':', linewidth='0.5', color='black')
    axes.grid(True)
    axes.legend(loc = 'upper right')
    axes.set_ylabel("error aproximado", fontsize=10)
    axes.set_xlabel("iteraciones ", fontsize=10)
    axes.set_title("error vs iteracion", fontsize=15)

    ######### visualization #################
    plt.yscale("log")     #comentar para visualozar en escala normar
    plt.show()
#########################################################
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

def Resolucion_LU(L,U,C):
    print("Primero se hallara la matriz columna 'D' por sustititucion en L*D = C")
    D = SD_forward_L(L,C)
    print(D)
    print("Ahora se procede a hallar la solucion del sistema A*R = C")
    R = SI_backward_U(U,D)

    return R


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



##########################main#####################################################
#A partir de esta seccion comenza a ejecutarse los ejercicios de la practica
###################################################################################
#parte 1
print("#############################################################################")
print("Pregunta 1:")
#a) resolver por el metodo de la sustitucion backward:
print("a) Resolver por el metodo de eliminacion de Gauss")
V_0 = 5
R = 1
A = array([[-4,1,1,1],[1,-3,0,1],[1,0,-3,1],[1,1,1,-4]])
C = array([[-V_0],[0],[-V_0],[0]])
R = E_Gauss(A,C)
A_C = unir_matrices(A,C)
print("Se presenta el siguiente sistema de ecuaciones:")
print(A_C)
print("Este sistema tiene como solucion a:")
print(R)
print("\n")
#b) resolver por el metodo de la descomposicion LU:
print("b) Resolver por el metodo de la descomposicion LU")
V_0 = 5
R = 1
A = array([[-4,1,1,1],[1,-3,0,1],[1,0,-3,1],[1,1,1,-4]])
C = array([[-V_0],[0],[-V_0],[0]])
L,U = D_LU_L(A)

print("La matriz A mediante descomposicion LU se descompone en:")
print("L:")
print(L,"\n")
print("U:")
print(U,"\n")

A_C = unir_matrices(A,C)
print("Se presenta el siguiente sistema de ecuaciones:")
print(A_C)

R = Resolucion_LU(L,U,C)

print("Este sistema tiene como solucion a:")
print(R)
print("\n")
#b) resolver por el metodo de la descomposicion de Cholesky:
print("c) Resolver por el metodo de la descomposicion de Cholesky")
V_0 = 5
R = 1
print("para aplicar la descomposicion por Cholesky, debemos asegurarnos que la matriz sea definida positiva")
A_m = array([[-4,1,1,1],[1,-3,0,1],[1,0,-3,1],[1,1,1,-4]])
C_m = array([[-V_0],[0],[-V_0],[0]])
A = -A_m
C = -C_m
print(A)
print("los valores propios de esta matriz son: 1,3,5,5")
print("Por lo tanto este sistema puede ser resuelto bajo la descomposicion de Cholesky")

L,U = Cholesky(A)

print("La matriz A mediante descomposicion de Cholesky se descompone en:")
print("L:")
print(L,"\n")
print("U:")
print(U,"\n")

A_C = unir_matrices(A,C)
print("Se presenta el siguiente sistema de ecuaciones:")
print(A_C)

R = Resolucion_LU(L,U,C)

print("Este sistema tiene como solucion a:")
print(R)
print("\n")
###########################################################################################
#Parte 2
print("#############################################################################")
print("Parte 2:")
#a) resolver por el metodo de la sustitucion backward:
print("a) Resolver por el metodo de eliminacion de Gauss")
m_1 = 2
m_2 = 0.5
m_3 = 0.3
k_1 = 0.5
k_2 = 0.5
k_3 = 0.5
k_4 = 0.5
A = array([[(k_1+k_2)/m_1,-k_2/m_1,0],[-k_2/m_2,(k_2+k_3)/m_2,-k_3/m_2],[0,-k_3/m_3,(k_3+k_4)/m_3]])
C = array([[-1],[1.2],[1.3]])
R = E_Gauss(A,C)
A_C = unir_matrices(A,C)
print("Se presenta el siguiente sistema de ecuaciones:")
print(A_C)
print("Este sistema tiene como solucion a:")
print(R)
print("\n")
#b) resolver por el metodo de la descomposicion LU:
print("b) Resolver por el metodo de la descomposicion LU")
m_1 = 2
m_2 = 0.5
m_3 = 0.3
k_1 = 0.5
k_2 = 0.5
k_3 = 0.5
k_4 = 0.5
A = array([[(k_1+k_2)/m_1,-k_2/m_1,0],[-k_2/m_2,(k_2+k_3)/m_2,-k_3/m_2],[0,-k_3/m_3,(k_3+k_4)/m_3]])
C = array([[-1],[1.2],[1.3]])
L,U = D_LU_L(A)

print("La matriz A mediante descomposicion LU se descompone en:")
print("L:")
print(L,"\n")
print("U:")
print(U,"\n")

A_C = unir_matrices(A,C)
print("Se presenta el siguiente sistema de ecuaciones:")
print(A_C)

R = Resolucion_LU(L,U,C)

print("Este sistema tiene como solucion a:")
print(R)
print("\n")
#b) resolver por el metodo de la descomposicion de Cholesky:
print("c) Resolver por el metodo de la descomposicion de Cholesky")
m_1 = 2
m_2 = 0.5
m_3 = 0.3
k_1 = 0.5
k_2 = 0.5
k_3 = 0.5
k_4 = 0.5
A = array([[(k_1+k_2)/m_1,-k_2/m_1,0],[-k_2/m_2,(k_2+k_3)/m_2,-k_3/m_2],[0,-k_3/m_3,(k_3+k_4)/m_3]])
C = array([[-1],[1.2],[1.3]])
print(A)
print("los valores propios de esta matriz son: 0.286, 1.409,4.139")
print("Por lo tanto este sistema puede ser resuelto bajo la descomposicion de Cholesky")
#print("Los valores propios de la matriz a descomponer posee valores propios no positivos, por lo tanto no es definida positiva.")
#print("Sus valores propios son: ")
L,U = Cholesky(A)

print("La matriz A mediante descomposicion de Cholesky se descompone en:")
print("L:")
print(L,"\n")
print("U:")
print(U,"\n")

A_C = unir_matrices(A,C)
print("Se presenta el siguiente sistema de ecuaciones:")
print(A_C)

R = Resolucion_LU(L,U,C)

print("Este sistema tiene como solucion a:")
print(R)
print("\n")
#########################################################################################3
#Parte 3
print("#############################################################################")
print("Parte 3:")
print("A la hora de calcular F_y usando una calculadroa de integrales, obtuve un error, pero al usar una herramienta computacional, obtuve un valor numerico el cual usare.")
k = 8.9875517873681764*1e09
a = 2
q = 1*1e-04
Q = 2*1e-05
D_C = (2*k*Q*q*a)
def Fuerza_x(x):
    return (D_C*0.48917*x)/((x**2+a**2)**1.5)
def Fuerza_y(x):
    return (D_C*(-0.738155))/(x**2+a**2)**1.5
def Fuerza_z(x):
    return (D_C*(+0.233843))/(x**2+a**2)**1.5
def Fuerza_total(x):
    return sqrt((Fuerza_x(x))**2 + (Fuerza_y(x))**2 +(Fuerza_z(x))**2)


graficador_tres(Fuerza_x,Fuerza_y,Fuerza_total,-10,10)

def Funcion_x(x):
    #return Fuerza_x(x)-3.12
    return Fuerza_x(x)-1.56
def Derivada_Funcion_x(x):
    return (D_C*0.48917)*( (1/((x**2+a**2)**1.5)) -((3*x**2)/((x**2+a**2)**2.5)) )
def Funcion_y(x):
    #return Fuerza_y(x)-2.17
    return Fuerza_y(x)+2.17
def Derivada_Funcion_y(x):
    return (D_C*-0.738155)*( ((-3*x)/((x**2+a**2)**2.5)) )

graficador_a(Funcion_x,0,5,"Funcion_x")
#print("Luego de Graficar,podemos darnos cuenta de que existen dos soluciones, la primera se encuentra entre los puntos 0 y 1, y la segunda entre 1.5 y 2.5")
print("Luego de Graficar,podemos darnos cuenta de que existen dos soluciones, la primera se encuentra entre los puntos 0 y 1, y la segunda entre 3.5 y 4.5")
print("Procederemos a calcular la primera solucion para Funcion_y(x)")
solucion1,err1 = metodo_biseccion(Funcion_x,0,1,100,1e-06)

print("el valor hallado para 'x' fue:",solucion1," \n")

solucion2,err2 = metodo_secante(Funcion_x,0,1,100,1e-06)

print("el valor hallado para 'x' fue:",solucion2," \n")

solucion3,err3 = metodo_newtonraphson(Funcion_x,Derivada_Funcion_x,0,100,1e-06)

print("el valor hallado para 'x' fue:",solucion3," \n")


graficador_error(err1,err2,err3)
print("el primer valor escogido para x es:",round(solucion3,6),",el cual nos da que F_x = 1.56\n")
#############################################################################################
print("Procederemos a calcular la segunda solucion para Funcion_x(x)")
solucion1,err1 = metodo_biseccion(Funcion_x,3.5,4.5,100,1e-06)

print("el valor hallado para 'x' fue:",solucion1," \n")

solucion2,err2 = metodo_secante(Funcion_x,3.5,4.5,100,1e-06)

print("el valor hallado para 'x' fue:",solucion2," \n")

solucion3,err3 = metodo_newtonraphson(Funcion_x,Derivada_Funcion_x,3.5,100,1e-06)

print("el valor hallado para 'x' fue:",solucion3," \n")

graficador_error(err1,err2,err3)
print("el segundo valor escogido para x es:",round(solucion3,6),",el cual nos da que F_x = 1.56\n")
####################################################################
graficador_a(Funcion_y,-5,5,"Funcion_y")
#print("Luego de Graficar,podemos darnos cuenta de que existen dos soluciones, la primera se encuentra entre los puntos 0 y 1, y la segunda entre 4 y 5")
print("Luego de Graficar,podemos darnos cuenta de que existen dos soluciones, la primera se encuentra entre los puntos -3 y -2, y la segunda entre 2 y 3")
print("Procederemos a calcular la primera solucion para Funcion_y(x)")
solucion1,err1 = metodo_biseccion(Funcion_y,-3,-2,100,1e-06) #revisar

print("el valor hallado para 'x' fue:",solucion1," \n")

solucion2,err2 = metodo_secante(Funcion_y,-3,-2,100,1e-06) #revisar

print("el valor hallado para 'x' fue:",solucion2," \n")

solucion3,err3 = metodo_newtonraphson(Funcion_y,Derivada_Funcion_y,-2,100,1e-06)

print("el valor hallado para 'x' fue:",solucion3," \n")


graficador_error(err1,err2,err3)
print("el primer valor escogido para x es:",round(solucion3,6),",el cual nos da que F_y = -2.17")
####################################################################
print("Procederemos a calcular la segunda solucion para Funcion_y(x)")
solucion1,err1 = metodo_biseccion(Funcion_y,2,3,100,1e-06)

print("el valor hallado para 'x' fue:",solucion1," \n")

solucion2,err2 = metodo_secante(Funcion_y,2,3,100,1e-06)

print("el valor hallado para 'x' fue:",solucion2," \n")

solucion3,err3 = metodo_newtonraphson(Funcion_y,Derivada_Funcion_y,3,100,1e-06)

print("el valor hallado para 'x' fue:",solucion3," \n")

graficador_error(err1,err2,err3)
print("el segundo valor escogido para x es:",round(solucion3,6),",el cual nos da que F_y = -2.17")

#########################################################################################3
#Parte 4
print("#############################################################################")
print("Parte 4:")
print("Al igualar el empuje con el peso, tenemos que 5*h^3-15*h^2+16 = 0")
def Funcion_h(x):
    return 5*x**3 - 15*x**2 + 16
def Derivada_Funcion_h(x):
    return 15*x**2-30*x
graficador_b(Funcion_h,1,2,"Funcion_h")
print("Luego de Graficar,podemos darnos cuenta de que la solucion se encuentra entre 1 y 2")
print("Procederemos a calcular la primera solucion para Funcion_y(x)")
solucion1,err1 = metodo_biseccion(Funcion_h,1,2,100,1e-06)

print("el valor hallado para 'x' fue:",solucion1," \n")

solucion2,err2 = metodo_secante(Funcion_h,1,2,100,1e-06)

print("el valor hallado para 'x' fue:",solucion2," \n")

solucion3,err3 = metodo_newtonraphson(Funcion_h,Derivada_Funcion_h,1,100,1e-06)

print("el valor hallado para 'x' fue:",solucion3," \n")



graficador_error(err1,err2,err3)

print("Como vemos en la grafica de errores, el metodo de Newton Raphson es el que nos da menor error")
print("el valor escogido para h es:",round(solucion3,6))
