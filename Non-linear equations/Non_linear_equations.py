#! /usr/bin/python3
from numpy import *
import matplotlib.pyplot as plt # from pylab import plot,show
import warnings
warnings.filterwarnings("ignore")

#################################### Fin del encabezado
from math import *   #se encarga de dar las funciones matematicas basicas, como exp(), sin(), cos()
import scipy.constants as constants    #se encarga de darme algunas constantes


def funcion_tarea(x):
    return 5*exp(-x)+x-5

def funcion_tarea_derivada(x):
    return-5*exp(-x)+1

def funcion_tarea_puntofijo(x):
    return 5-5*exp(-x)

def conversor(x):
    longitud_de_onda = ((constants.h)*(constants.c))/((constants.k)*x)
    return longitud_de_onda
#############################################################################################################
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


def graficador(funcion,a,b):

        fig, axes = plt.subplots()
        x = linspace(a,b,40000)
        f_x = []
        cero = []
        for x_i in x:
            y = funcion(x_i)
            f_x.append(y)
            cero.append(0)


        axes.plot(x,f_x , "-",markersize = 0.5, label = "funcion: f(x)" )   ### se podria poner en vez de "o" "--o" para que se visualice la linea que sigue pero se hace dificil la visualizacion de los otros puntos, asi mismo, el valor de markersize tambien puede incrementarse.
        axes.plot(x,cero,"k-",markersize = 0.5, label = "linea cero")
        axes.minorticks_on()
        axes.grid(which='minor', linestyle=':', linewidth='0.5', color='black')
        axes.grid(True)
        axes.legend(loc = 'center right')
        axes.set_ylabel("f(x)", fontsize=10)
        axes.set_xlabel("x ", fontsize=10)
        axes.set_title("Funcion de interes", fontsize=15)

        ########## visualization #################
        plt.show()
        #plt.yscale("log")    # se puede cambiar para visualizar mejor
        #plt.xscale("log")
        #plt.savefig('exercise_a.png', dpi=800)
        #plt.close()

def graficador_error(err_1,err_2,err_3,err_4,err_5):
    fig, axes = plt.subplots()

    axes.plot(err_1, "--o",markersize = 4, label = "Error : Biseccion" )   ### se podria poner en vez de "o" "--o" para que se visualice la linea que sigue pero se hace dificil la visualizacion de los otros puntos, asi mismo, el valor de markersize tambien puede incrementarse.
    axes.plot(err_2, "--*",markersize = 4, label = "Error : Falsa posicion" )
    axes.plot(err_3, "--h",markersize = 4, label = "Error : Newton Raphson" )
    axes.plot(err_4, "--H",markersize = 4, label = "Error : Secante" )
    axes.plot(err_5, "--X",markersize = 4, label = "Error : Punto fijo" )

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
#############################################################################################################
#main program:

#comenzaremos graficando la funcion, la cual debemos encontrar su raiz
graficador(funcion_tarea,-2,10)
#con esta grafica nos podemos dar la idea de que la raiz se encuentra entre 4 y 5
print("como se ve en la grafica de la funcion de interes, la raiz se encuentra entre 4 y 5")




#############################################################################################################
###Antes de proceder con los metodos, veamos cual es el valor que nos ofrece la libreria de scipy.constants para la constante de wien
print("El valor que se muestra en la referencia para 'x' es: 4.965114231744276")
print("El valor que se muestra en la referencia para la constante de wien es:", constants.Wien,"\n")

solucion1 = metodo_busquedaincremental(funcion_tarea,4,5,1e-06,1000000)
#4.965115000134902
print("el valor hallado para 'x' fue:",solucion1)
print("el valor hallado para 'Lambda' fue:",round(conversor(solucion1),12),"\n")
solucion2,err2 = metodo_biseccion(funcion_tarea,4,5,100,1e-06) #revisar
#4.965114231744276
print("el valor hallado para 'x' fue:",solucion2)
print("el valor hallado para 'Lambda' fue:",round(conversor(solucion2),12),"\n")
solucion3,err3 = metodo_falsaposicion(funcion_tarea,4,5,100,1e-06)
#4.965114231744276
print("el valor hallado para 'x' fue:",solucion3)
print("el valor hallado para 'Lambda' fue:",round(conversor(solucion3),12),"\n")
solucion4,err4 = metodo_newtonraphson(funcion_tarea,funcion_tarea_derivada,4,100,1e-06)
#4.965114231744276
print("el valor hallado para 'x' fue:",solucion4)
print("el valor hallado para 'Lambda' fue:",round(conversor(solucion4),12),"\n")
solucion5,err5 = metodo_secante(funcion_tarea,4,5,100,1e-06) #revisar
#4.965114231744276,
print("el valor hallado para 'x' fue:",solucion5)
print("el valor hallado para 'Lambda' fue:",round(conversor(solucion5),12),"\n")
solucion6,err6 = metodo_puntofijo(funcion_tarea_puntofijo,4,100,1e-06)
#4.96511414525846
print("el valor hallado para 'x' fue:",solucion6)
print("el valor hallado para 'Lambda' fue:",round(conversor(solucion6),12),"\n")

print("Como podemos ver todas los metodos nos dan un valor cercano para esta constante")

graficador_error(err2,err3,err4,err5,err6)

print("El metodo que mas convendria usar es el de newto raphson, puesto consideramos el numero de iteraciones y el error alcanzado al termino de las mismas.")
