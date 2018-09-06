#!/usr/bin/env python
import matplotlib.cm as cm
import sys
import numpy as np
import matplotlib.pyplot as plt
sys.path.append('/Users/marcelino/Things_EDS/acceleration/')
import accelerationmethods as am

fname = sys.argv[1]
ftype = sys.argv[2]
dimensions = sys.argv[3]

if dimensions=='1':
    if ftype=='c':
        beta = float(sys.argv[4])
        a = float(sys.argv[5])
        b = float(sys.argv[6])
        saving = sys.argv[7]
        data = np.load(fname)
        fgrid = data[0:-3]
        tamano = np.size(fgrid)
        begin = data[-3]
        end = data[-2]
        xgrid = np.linspace(begin, end, tamano)
        commitor = am.commone(xgrid, fgrid, beta, a, b, None, 1)[0]
        plt.figure()
        plt.plot(xgrid, commitor, linewidth=2.0)
        plt.xlabel('Coordenada Colectiva')
        plt.ylabel('Probabilidad')
        plt.title('Probabilidades de compromiso')
        plt.grid()
        if saving=='no':
            plt.show()
        else:
            plt.savefig('graph')
    elif ftype=='m':
        beta = float(sys.argv[4])
        porcentaje = float(sys.argv[5])
        minima = [float(sys.argv[y]) for y in range(6,len(sys.argv)-1)]
        savo = sys.argv[-1]
        data = np.load(fname)
        fes = data[0:-3]
        tamano = np.size(fes)
        begin = data[-3]
        end = data[-2]
        xgrid = np.linspace(begin, end, tamano)
        if savo=='no':
            dati = None
        else:
            dati = 1
        am.metaestable(xgrid, fes, beta, porcentaje, minima, dati)
    elif ftype=='s':
        a = float(sys.argv[4])
        b = float(sys.argv[5])
        dynamic = np.load(fname)
        pasos = am.local_escape(dynamic, [a,b])
        print(pasos)
    else:
        beta = float(sys.argv[4])
        binos = int(sys.argv[5])
        ndynamics = int(sys.argv[6])
        dt = float(sys.argv[7])
        steps = int(sys.argv[8])
        mu = float(sys.argv[9])
        std = float(sys.argv[10])
        xo = float(sys.argv[11])
        data = np.load(fname)
        fgrid = data[0:-3]
        tamano = np.size(fgrid)
        a = data[-3]
        b = data[-2]
        euler = am.emone_para(a, b, fgrid, beta, binos, ndynamics, dt, steps, mu, std, xo)
else:
    if ftype=='c':
        beta = float(sys.argv[4])
        a_x = float(sys.argv[5])
        a_y = float(sys.argv[6])
        b_x = float(sys.argv[7])
        b_y = float(sys.argv[8])
        saving = sys.argv[9]
        ast = [a_x,a_y]
        bst = [b_x,b_y]
        domain = np.load('domains.npy')
        a = domain[0]
        b = domain[1]
        c = domain[2]
        d = domain[3]
        fes_matrix = np.load(fname)
        tamano = np.size(fes_matrix[0,:])
        xgrid = np.linspace(a, b, tamano)
        ygrid = np.linspace(c, d, tamano)
        fes_comm = am.commtwo(xgrid, ygrid, fes_matrix, beta, ast, bst, 1, None)[0]
        fig = plt.figure()
        ax = fig.add_subplot(111)
        extent = [a,b,c,d]
        ax.contourf(fes_comm, np.size(fes_matrix[:,0]), cmap=plt.cm.jet, extent=extent)
        plt.colorbar(ax.contourf(fes_comm, np.size(fes_matrix[:,0]), cmap=plt.cm.jet, extent=extent))
        plt.grid()
        plt.xlabel('Coordenada x')
        plt.ylabel('Coordenada y')
        plt.title('Probabilidades de compromiso')
        if saving=='no':
            plt.show()
        else:
            plt.savefig('graph.png')
    elif ftype=='m':
        beta = float(sys.argv[4])
        porcentaje = float(sys.argv[5])
        minima = [float(sys.argv[y]) for y in range(6,len(sys.argv)-1)]
        howmany = int(len(minima)/2.0)
        r = np.zeros((2,howmany))
        for te in range(howmany):
            r[0,te] = minima[te]
            r[1,te] = minima[te+1]
        lista = [r[:,y] for y in range(howmany)]
        savo = sys.argv[-1]
        fes = np.load(fname)
        tamano = np.size(fes[0,:])
        datos = np.load('domains.npy')
        a = datos[0]
        b = datos[1]
        c = datos[2]
        d = datos[3]
        xgrid = np.linspace(a, b, tamano)
        ygrid = np.linspace(c, d, tamano)
        if savo=='no':
            dati = None
        else:
            dati = 1
        am.metaestable_2D(xgrid, ygrid, fes, beta, porcentaje, lista, dati)

    elif ftype=='s':
        number = float(sys.argv[4])
        dinamica = np.load(fname)
        pasos = am.local_escape_2D(dinamica,number)
        print(pasos)
    else:
        beta = float(sys.argv[4])
        binos = int(sys.arg[5])
        ndynamics = int(sys.arg[6])
        dt = float(sys.argv[7])
        steps = float(sys.argv[8])
        mu = float(sys.argv[9])
        std = float(sys.argv[10])
        xo = float(sys.argv[11])
        yo = float(sys.argv[12])
        fgrid = np.load(fname)
        domain = np.load('domains.npy')
        a = domain[0]
        b = domain[1]
        c = domain[2]
        d = domain[3]
        emtwo_para(a, b, c, d, fgrid, beta, binos, ndynamics, dt, steps, mu, std, xo, yo)
