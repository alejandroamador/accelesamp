#!/usr/bin/env python
import matplotlib.cm as cm
import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


fname = sys.argv[1]
ftype = sys.argv[2]
dimensions = sys.argv[3]
saving = sys.argv[4]

if dimensions=='1':
    if ftype=='f':
        data = np.load(fname)
        a = data[-3]
        b = data[-2]
        fesgrid = data[0:-3]
        tamano = np.size(fesgrid)
        xgrid = np.linspace(a, b, tamano)
        plt.figure()
        plt.plot(xgrid, fesgrid, linewidth=2.0)
#        plt.axvline(x=-6, linestyle=':', linewidth=2, color='k')
#        plt.axvline(x=6, linestyle=':', linewidth=2, color='k')
#        plt.axvline(x=-4.25, linestyle=':', linewidth=2, color='k')
#        plt.axvline(x=-0.81, linestyle=':', linewidth=2, color='k')
#        plt.axvline(x=1.11, linestyle=':', linewidth=2, color='k')
#        plt.axvline(x=3.32, linestyle=':', linewidth=2, color='k')
        plt.title('Panorama energetico')
        plt.xlabel('Coordenada Colectiva')
        plt.ylabel('Energia')
        plt.grid()
        if saving=='no':
            plt.show()
        else:
            plt.savefig('graph.png')
    elif ftype=='p':
        tamano = int(sys.argv[5])
        data = np.load(fname)
        pasotes = data[0:tamano]
        tiempo = [u for u in range(tamano)]
        plt.figure()
        plt.plot(tiempo, pasotes)
        plt.xlabel('Tiempo')
        plt.ylabel('Coordenada Colectiva')
        plt.title('Dinamica en el tiempo')
        plt.grid()
        if saving=='no':
            plt.show()
        else:
            plt.savefig('graph.png')
    elif ftype=='c':
        data = np.load(fname)
        a = data[-3]
        b = data[-2]
        fesgrid = data[0:-3]
        tamano = np.size(fesgrid)
        xgrid = np.linspace(a, b, tamano)
        plt.figure()
        plt.plot(xgrid, fesgrid, linewidth=2.0)
        plt.xlabel('Coordenada Colectiva')
        plt.ylabel('Probabilidad')
        plt.title('Probabilidades de compromiso')
        plt.grid()
        if saving=='no':
            plt.show()
        else:
            plt.savefig('graph.png')
    elif ftype=='d':
        data = np.load(fname)
        a = data[-2]
        b = data[-1]
        disgrid = data[0:-2]
        tamano = np.size(disgrid)
        xgrid = np.linspace(a, b, tamano)
        plt.figure()
        plt.plot(xgrid, disgrid, linewidth=2.0)
        plt.title('Distribucion en el canonico')
        plt.xlabel('Coordenada Colectiva')
        plt.ylabel('Probabilidad')
        plt.grid()
        if saving=='no':
            plt.show()
        else:
            plt.savefig('graph.png')
    elif ftype=='co':
        data = np.load(fname)
        binos = int(data[1,-1])
        dynamic = data[0,1:-1]
        pesos = np.ones_like(dynamic)/float(len(dynamic))
        plt.figure()
        plt.grid()
        plt.hist(dynamic, weights=pesos, bins=binos)
        plt.title('Histograma de dinamica sobre la FES')
        plt.xlabel('Coordenada Colectiva')
        plt.ylabel('Probabilidad')
        if saving=='no':
            plt.show()
        else:
            plt.savefig('graph.png')
    else:
        data = np.load(fname)
        binos = int(data[-1])
        dynamic = data[1:-1]
        pesos = np.ones_like(dynamic)/float(len(dynamic))
        plt.figure()
        plt.grid()
        plt.hist(dynamic, weights=pesos, bins=binos)
        plt.title('Histograma de dinamica sobre la FES')
        plt.xlabel('Coordenada Colectiva')
        plt.ylabel('Probabilidad')
        if saving=='no':
            plt.show()
        else:
            plt.savefig('graph.png')
else:
    if ftype=='f' or ftype=='c' or ftype=='d':
        domain = np.load('domains.npy')
        fes_matrix = np.load(fname)
        a = domain[0]
        b = domain[1]
        c = domain[2]
        d = domain[3]
        if ftype=='f':
            dsize = np.size(fes_matrix[0,:])
            fig = plt.figure()
            ax = fig.add_subplot(111)
            extent = [a, b, c, d]
            ax.contourf(fes_matrix, dsize, cmap=plt.cm.jet, extent=extent)
            plt.colorbar(ax.contourf(fes_matrix, dsize, cmap=plt.cm.jet, extent=extent))
            plt.grid()
            plt.xlabel('Coordenada x')
            plt.ylabel('Coordenada y')
            plt.title('Panorama energetico')
            if saving=='no':
                plt.show()
            else:
                plt.savefig('graph.png')

        elif ftype=='d':
            dsize = np.size(fes_matrix[0,:])
            fig = plt.figure()
            ax = fig.add_subplot(111)
            extent = [a, b, c, d]
            ax.contourf(fes_matrix.T, dsize, cmap=plt.cm.jet, extent=extent)
            plt.colorbar(ax.contourf(fes_matrix.T, dsize, cmap=plt.cm.jet, extent=extent))
            plt.grid()
            plt.xlabel('Coordenada x')
            plt.ylabel('Coordenada y')
            plt.title('Probabilidad')
            if saving=='no':
                plt.show()
            else:
                plt.savefig('graph.png')

        elif ftype=='c':
            dsize = np.size(fes_matrix[0,:])
            fig = plt.figure()
            ax = fig.add_subplot(111)
            extent = [a,b,c,d]
            ax.contourf(fes_matrix, dsize, cmap=plt.cm.jet, extent=extent)
            plt.colorbar(ax.contourf(fes_matrix, dsize, cmap=plt.cm.jet, extent=extent))
            plt.grid()
            plt.xlabel('Coordenada x')
            plt.ylabel('Coordenada y')
            plt.title('Probabilidades de compromiso')
            if saving=='no':
                plt.show()
            else:
                plt.savefig('graph.png')

    else:
        if ftype=='p':
            tamano = int(sys.argv[-1])
            domain = np.load('domains.npy')
            data = np.load(fname)
            exx = data[0,1:tamano]
            eyy = data[1,1:tamano]
            ex = exx[::50]
            ey = eyy[::50]
            tiempo = np.linspace(1,len(exx),len(ex))
            a = domain[0]
            b = domain[1]
            c = domain[2]
            d = domain[3]
            fig = plt.figure()
            #ax = fig.add_subplot(111)
            #extent = [a, b, c, d]
            #ax.contourf(ndist_normal.T, dsize, cmap=plt.cm.jet, extent=extent)
            #plt.colorbar(ax.contourf(ndist_normal.T, dsize, cmap=plt.cm.jet, extent=extent))
            plt.scatter(ex,ey,c=tiempo, cmap=plt.cm.jet, alpha=0.5)
            plt.grid()
            plt.colorbar()
            plt.xlabel('Coordenada x')
            plt.ylabel('Coordenada y')
            plt.title('Dinamica')
            if saving=='no':
                plt.show()
            else:
                plt.savefig('graph.png')

        else:
            domain = np.load('domains.npy')
            data = np.load(fname)
            ex = data[0,1:-1]
            ey = data[1,1:-1]
            binos = int(data[1,-1])
            ndist, x, y = np.histogram2d(ex,ey, bins=binos)
            a = x[0]
            b = x[-1]
            c = y[0]
            d = y[-1]
            ndist_normal = (1.0/np.sum(ndist))*ndist
            dsize = np.size(x)-1

            fig = plt.figure()
            ax = fig.add_subplot(111)
            extent = [a, b, c, d]
            ax.contourf(ndist_normal.T, dsize, cmap=plt.cm.jet, extent=extent)
            plt.colorbar(ax.contourf(ndist_normal.T, dsize, cmap=plt.cm.jet, extent=extent))
            plt.grid()
            plt.xlabel('Coordenada x')
            plt.ylabel('coordenada y')
            plt.title('Probabilidad')
            if saving=='no':
                plt.show()
            else:
                plt.savefig('graph.png')

