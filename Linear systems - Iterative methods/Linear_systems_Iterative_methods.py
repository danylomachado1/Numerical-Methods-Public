#! /usr/bin/python3
from numpy import *
import matplotlib.pyplot as plt # from pylab import plot,show
import warnings
warnings.filterwarnings("ignore")

#################################### Fin del encabezado
import scipy.constants as constants #para la constante g
################################################################################
def Confirma_simetria(A):
    N,M = A.shape
    for i in range(N):
        for j in range(N):
            if (A[i][j] != A[j][i]):
                return False
    return True

def graficador_error_cinco(err_1,err_2,err_3,err_4,err_5):
    fig, axes = plt.subplots()

    axes.plot(err_1, "--o",markersize = 4, label = "Error : Jacobi" )   ### se podria poner en vez de "o" "--o" para que se visualice la linea que sigue pero se hace dificil la visualizacion de los otros puntos, asi mismo, el valor de markersize tambien puede incrementarse.
    axes.plot(err_2, "--*",markersize = 4, label = "Error : Gauss Seidel" )
    axes.plot(err_3, "--h",markersize = 4, label = "Error : Sor" )
    axes.plot(err_4, "--+",markersize = 4, label = "Error : Maximo descenso" )
    axes.plot(err_5, "--1",markersize = 4, label = "Error : Gradiente conjudado" )


    axes.minorticks_on()
    axes.grid(which='minor', linestyle=':', linewidth='0.5', color='black')
    axes.grid(True)
    axes.legend(loc = 'best')
    axes.set_ylabel("Error", fontsize=10)
    axes.set_xlabel("iteracion ", fontsize=10)
    axes.set_title("Error: diferentes metodos", fontsize=15)

    ######### visualization #################
    plt.yscale("log")
    #plt.xscale("log")     #comentar para visualozar en escala normar
    plt.show()

def graficador_error_tres(err_1,err_2,err_3):
    fig, axes = plt.subplots()

    axes.plot(err_1, "--o",markersize = 4, label = "Error : Jacobi" )   ### se podria poner en vez de "o" "--o" para que se visualice la linea que sigue pero se hace dificil la visualizacion de los otros puntos, asi mismo, el valor de markersize tambien puede incrementarse.
    axes.plot(err_2, "--*",markersize = 4, label = "Error : Gauss Seidel" )
    axes.plot(err_3, "--h",markersize = 4, label = "Error : Sor" )



    axes.minorticks_on()
    axes.grid(which='minor', linestyle=':', linewidth='0.5', color='black')
    axes.grid(True)
    axes.legend(loc = 'best')
    axes.set_ylabel("Error", fontsize=10)
    axes.set_xlabel("iteracion ", fontsize=10)
    axes.set_title("Error: diferentes metodos", fontsize=15)

    ######### visualization #################
    plt.yscale("log")     #comentar para visualozar en escala normar
    #plt.xscale("log")
    plt.show()


def Matriz_voltajes(N,R,V_0):
    #if N % 2 != 0:
    #    N = int(input("Introdusca un valor valido, N debe ser par."))

    #if R < 0:
    #    R = float(input("Introdusca un valor valido, R debe ser positivo."))

    A = zeros((N,N), dtype = float)
    C = zeros((N), dtype = float)

    C[0] = V_0
    C[1] = V_0

    A[0,0] = 3
    A[0,1] = -1
    A[0,2] = -1

    A[1,0] = -1
    A[1,1] = 4
    A[1,2] = -1
    A[1,3] = -1

    A[N-1,N-1] = 3
    A[N-1,N-2] = -1
    A[N-1,N-3] = -1

    A[N-2,N-1] = -1
    A[N-2,N-2] = 4
    A[N-2,N-3] = -1
    A[N-2,N-4] = -1


    for i in range(2,N-2):
        A[i,i] = 4
        A[i,i-1] = -1
        A[i,i-2] = -1
        A[i,i+1] = -1
        A[i,i+2] = -1


    return A,C

################################################################################

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

#A = array([[4,1],[1,3]])
#b = array([1,2])
#N = 100
#tol = 1e-8

#x,err = jacobi(A,b,N,tol)
################################################################################
def gaussSeidel(A, b, x, tol):
    N,M = A.shape
    Iter_max = 100000
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

    return x_2, err, iter

#matrix2 = array([[3.0, 1.0], [2.0, 6.0]])
#vector2 = array([5.0, 9.0])
#guess = array([0.0, 0.0])
#x, err=gaussSeidel(matrix2, vector2, guess, 1e-06)

################################################################################
def Metodo_SOR(A, b, omega, initial_guess, convergence_criteria):
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



#residual_convergence = 1e-8
#omega = 0.5 #Relaxation factor debe ser mayor a 1

#A = array([[4,-1,-6,0],[-5,-4,10,8],[0,9,4,-2],[1,0,-7,5]])
#b = array([2,21,-12,-6])
#initial_guess = zeros(4)

#phi, err = Metodo_SOR(A, b, omega, initial_guess, residual_convergence)
#print(phi)



################################################################################

def Maximo_descenso(A, b, x):
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

#ejemplo
#A = array([[4,1],[1,3]])
#b = array([1,2])
#x = array([2,1])
#array([0.09090909, 0.63636364])

################################################################################
print("Parte 1:")
#a) resolver por el metodo de la sustitucion backward:
print("a)")

U = array([[1,2,1],[0,-4,1],[0,0,-2]])
C = array([5,2,4])

print("Jacobi")
N = 100
tol = 1e-8
x,err1,iter = jacobi(U,C,N,tol)
print(x,iter)

print("Gauss Seidel")
guess = array([0.0, 0.0, 0.0])
x, err2,iter =gaussSeidel(U, C, guess, 1e-06)
print(x,iter)

print("Sor")
residual_convergence = 1e-8
omega = 1.5 #Relaxation factor debe ser mayor a 1
initial_guess = zeros(3) #numero de incongnitas
phi, err3,iter = Metodo_SOR(U,C, omega, initial_guess, residual_convergence)
print(phi,iter)

print("veamos si la matriz de coeficientes es definida positiva")
print(U)
print("los valores propios de U son")
print(linalg.eigvals(U))

validar_vp = where(linalg.eigvals(U) < 0)

if len(validar_vp[0]) == 0 and Confirma_simetria(U) :
    print("Maximodescenso")
    x = zeros(3)
    x, err4, iter = Maximo_descenso(U,C,x)
    print(x,iter)

    print("Gradiente conjudado")
    x = zeros(3)
    x, err5,iter = Gradiente_conjudado(U,C,x)
    print(x,iter)

#graficador_error(err1,err2,err3,err4,err5)
if len(validar_vp[0]) == 0 and Confirma_simetria(U):
    graficador_error_cinco(err1,err2,err3,err4,err5)
else:
    graficador_error_tres(err1,err2,err3)


print("b)")

L = array([[2,0,0],[1,4,0],[4,3,3]])
C = array([4,2,5])

print("Jacobi")
N = 100
tol = 1e-8
x,err1,iter = jacobi(L,C,N,tol)
print(x,iter)

print("Gauss Seidel")

guess = array([0.0, 0.0, 0.0])
x, err2,iter =gaussSeidel(L, C, guess, 1e-06)
print(x,iter)

print("Sor")
residual_convergence = 1e-8
omega = 1.5 #Relaxation factor debe ser mayor a 1
initial_guess = zeros(3) #numero de incongnitas
phi, err3,iter = Metodo_SOR(L,C, omega, initial_guess, residual_convergence)
print(phi,iter)

print("veamos si la matriz de coeficientes es definida positiva")
print(L)
print("los valores propios de L son")
print(linalg.eigvals(L))

validar_vp = where(linalg.eigvals(L) < 0)

if len(validar_vp[0]) == 0 and Confirma_simetria(L):
    print("Maximodescenso")
    x = zeros(3)#
    x, err4, iter = Maximo_descenso(L,C,x)
    print(x,iter)

    print("Gradiente conjudado")
    x = x = array([4,2,5])
    x, err5,iter = Gradiente_conjudado(L,C,x)
    print(x,iter)

if len(validar_vp[0]) == 0 and Confirma_simetria(L):
    graficador_error_cinco(err1,err2,err3,err4,err5)
else:
    graficador_error_tres(err1,err2,err3)

print("Parte 2:")
print("a")

V_0 = 5
R = 1.0
N_v = 6
A,C = Matriz_voltajes(N_v,R,V_0)
print("jacobi")
N = 1000
tol = 1e-8
x,err1,iter = jacobi(A,C,N,tol)
print(x,iter)

print("gaussSeidel")
guess = zeros(N_v)
x, err2,iter =gaussSeidel(A, C, guess, 1e-06)
print(x,iter)

print("Sor")
residual_convergence = 1e-8
omega = 1.5 #Relaxation factor debe ser mayor a 1
initial_guess = zeros(N_v) #numero de incongnitas
phi, err3,iter = Metodo_SOR(A,C, omega, initial_guess, residual_convergence)
print(phi,iter)

print("veamos si la matriz de coeficientes es definida positiva")
print(A)
print("los valores propios de L son")
print(linalg.eigvals(A))

validar_vp = where(linalg.eigvals(A) < 0)

if len(validar_vp[0]) == 0 and Confirma_simetria(A):
    print("Maximodescenso")
    x = zeros(N_v)#
    x, err4, iter = Maximo_descenso(A,C,x)
    print(x,iter)

    print("Gradiente conjudado")
    x = zeros(N_v)
    x, err5,iter = Gradiente_conjudado(A,C,x)
    print(x,iter)

if len(validar_vp[0]) == 0 and Confirma_simetria(A):
    graficador_error_cinco(err1,err2,err3,err4,err5)
else:
    graficador_error_tres(err1,err2,err3)

print("b")

V_0 = 5
R = 1.0
N_v = 100
A,C = Matriz_voltajes(N_v,R,V_0)
print("jacobi")
N = 100000
tol = 1e-6
x,err1,iter = jacobi(A,C,N,tol)
print(x,iter)

print("gaussSeidel")
guess = zeros(N_v)
x, err2,iter =gaussSeidel(A, C, guess, 1e-06)
print(x,iter)

print("Sor")
residual_convergence = 1e-6
omega = 1.5 #Relaxation factor debe ser mayor a 1
initial_guess = zeros(N_v) #numero de incongnitas
phi, err3,iter = Metodo_SOR(A,C, omega, initial_guess, residual_convergence)
print(phi,iter)

print("veamos si la matriz de coeficientes es definida positiva")
print(A)
print("los valores propios de L son")
print(linalg.eigvals(A))

validar_vp = where(linalg.eigvals(A) < 0)

if len(validar_vp[0]) == 0 and Confirma_simetria(A):
    print("Maximodescenso")
    x = zeros(N_v)#
    x, err4, iter = Maximo_descenso(A,C,x)
    print(x,iter)

    print("Gradiente conjudado")
    x = zeros(N_v)
    x, err5,iter = Gradiente_conjudado(A,C,x)
    print(x,iter)

if len(validar_vp[0]) == 0 and Confirma_simetria(A):
    graficador_error_cinco(err1,err2,err3,err4,err5)
else:
    graficador_error_tres(err1,err2,err3)

print("Parte 3")

g = constants.g
k = 10

m_1 = 1
m_2 = 2
m_3 = 3

print("se resuelve para las posiciones en el estado estacionario")
A = array([[3*k,-2*k,0],[-2*k,3*k,-k],[0,-k,k]])
C = array([m_1*g,m_2*g,m_3*g])

print("jacobi")
N = 100000
tol = 1e-6
x,err1,iter = jacobi(A,C,N,tol)
print(x,iter)

print("gaussSeidel")
guess = zeros(3)
x, err2,iter =gaussSeidel(A, C, guess, 1e-06)
print(x,iter)

print("Sor")
residual_convergence = 1e-6
omega = 1.5 #Relaxation factor debe ser mayor a 1
initial_guess = zeros(3) #numero de incongnitas
phi, err3,iter = Metodo_SOR(A,C, omega, initial_guess, residual_convergence)
print(phi,iter)

print("veamos si la matriz de coeficientes es definida positiva")
print(A)
print("los valores propios de L son")
print(linalg.eigvals(A))

validar_vp = where(linalg.eigvals(A) < 0)

if len(validar_vp[0]) == 0 and Confirma_simetria(A):
    print("Maximodescenso")
    x = zeros(3)#
    x, err4, iter = Maximo_descenso(A,C,x)
    print(x,iter)

    print("Gradiente conjudado")
    x = zeros(3)
    x, err5,iter = Gradiente_conjudado(A,C,x)
    print(x,iter)

if len(validar_vp[0]) == 0 and Confirma_simetria(A):
    graficador_error_cinco(err1,err2,err3,err4,err5)
else:
    graficador_error_tres(err1,err2,err3)
