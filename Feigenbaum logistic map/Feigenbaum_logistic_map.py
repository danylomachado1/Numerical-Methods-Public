#! /usr/bin/python3
from numpy import *
import matplotlib.pyplot as plt # from pylab import plot,show
import warnings
warnings.filterwarnings("ignore")

#################################### Fin del encabezado

def exercise_a():

    fig, axes = plt.subplots()

    r = 1  ## Constante

    options = [0.45, 0.50, 0.55]

    ######## indice ##################

    k = zeros((1001,), dtype=int)

    for i in range(1000):
        k[i+1]= i+1
    ##################################

    xlabel = ["x_0 = 0.45", "x_0 = 0.50", "x_0 = 0.55"]
    markerlabel = [ "o","-", "v"]
    m = 0

    for x_0 in options:
        x = zeros(1001)  ## Array de ceros, contendra todos los valores de iteracion
        x[0] = x_0
        for i in range(1000):
            x[i+1] = r*x[i]*(1-x[i])
        label = xlabel[m]
        marker = markerlabel[m]
        axes.plot(k, x , marker ,markersize = 0.5, label = label )   ### se podria poner en vez de "o" "--o" para que se visualice la linea que sigue pero se hace dificil la visualizacion de los otros puntos, asi mismo, el valor de markersize tambien puede incrementarse.
        m = m+1

    axes.minorticks_on()
    axes.grid(which='minor', linestyle=':', linewidth='0.5', color='black')
    axes.grid(True)
    axes.legend(loc = 'center right')
    axes.set_ylabel("x_(k+1)", fontsize=10)
    axes.set_xlabel("Index (k) ", fontsize=10)
    axes.set_title("Evolution", fontsize=15)

    ########## visualization #################
    plt.yscale("log")    # se puede cambiar para visualizar mejor
    plt.xscale("log")
    plt.show()

    #plt.savefig('exercise_a.png', dpi=800)
    #plt.close()
    ##########################################

def exercise_b():

    fig, axes = plt.subplots()
    r = linspace(1,4,3001)           ### inicalmente el programa funciona con 30 valores, pero para visualizar la grafica mejor, incremente el numero de puntos a 3000.

    oname = ("Bifurcacion.dat")
    outfile = open(oname,'w')

    for const in r:
        x = zeros(3001)  ## Array de ceros, contendra todos los valores de iteracion
        y = zeros(3001)
        x[0] = 0.5
        y[0] = const
        for i in range(3000):
            x[i+1] = const*x[i]*(1-x[i])
            y[i+1] = const
            if i > 1000:
                s = "%s %s\n" %(y[i+1], x[i+1])
                outfile.write(s)

    x,y = [],[]
    for line in open("Bifurcacion.dat","r"):
        values =[ float(s) for s in line.split()]
        x.append(values[0])
        y.append(values[1])



    axes.plot(x,y, ",", color = "red")
    axes.minorticks_on()

    axes.grid(which='minor', linestyle=':', linewidth='0.3', color='black')
    axes.grid(True)
    axes.set_ylabel("x_(k+1)", fontsize=10)
    axes.set_xlabel("r", fontsize=10)
    axes.set_title("Bifurcacion", fontsize=15)
    ###plt.yscale("log")

    ########## visualization #################
    plt.show()
    #plt.savefig('exercise_b.png', dpi=800)
    #plt.close()
    ##########################################


###### main program ##########################

exercise_a()  # esta parte suele hacerse rapido

exercise_b()  # esta parte suele tardar

##############################################
