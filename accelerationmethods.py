"""
Module with four methods for molecular dynamics acceleration
meta dynamics, variationally enhanced, gaussian method and
flooding. Each method is optimized (only a few use parallel
computing) and works in 1D and 2D.
"""

from tempfile import TemporaryFile
import matplotlib.pyplot as plt
import os
import multiprocessing as mp
import time
import math
import numpy as np
from scipy import integrate
import scipy.linalg as nps
import scipy.misc as sci

################################
#########EULER_MARUYAMA#########
################################
#Gives location of point in array.
def point_location(array_in, array_ds, point):

    location = int(round(float(point-array_in)/array_ds))

    return location

#Array derivative.
def devone_array(ds, fesgrid, location):

    val_f = fesgrid[location]
    try:
        increment = fesgrid[location+1]
        derivative = float(increment-val_f)/ds
    except IndexError:
        increment = fesgrid[location-1]
        derivative = float(val_f-increment)/ds

    return derivative

#Gives an euler-maruyama dynamic simulation.
def emone(a, b, fes, beta, binos, dt, steps, mu, std, xo, name, data=None, seed=None):
    '''
    |Takes limits of domain, fes in array or parameterized format, thermodynamic beta, step time, number
    |of steps, mean and std deviation for normal distribution (noise), and starting condition.
    |
    |Writes file with positions over time.
    '''

    local_st = np.random.RandomState(seed)
    if data==None:
        try:
            size_space = len(fes)
            dynamic = np.zeros(steps+1)
            ds = float(b-a)/(size_space-1)
            sigma = np.sqrt(2.0/beta)
            noise = sigma*np.sqrt(dt)
            xo_ar = point_location(a, ds, xo)
            pot = fes[xo_ar]
            pot_dx = devone_array(ds, fes, xo_ar)

            for m in range(steps):
                dynamic[m] = xo
                r = local_st.normal(mu, std)
                boundary = xo - (dt*pot_dx)+(noise*r)

                if a<=boundary<=b:
                     xn = boundary
                else:
                     xn = xo

                xn_ar = point_location(a, ds, xn)
                potn = fes[xn_ar]
                potn_dx = devone_array(ds, fes, xn_ar)
                transition_one = abs(xn-xo+dt*pot_dx)**2-abs(xo-xn+dt*potn_dx)**2
                transition_two = (4.0/beta)*dt
                transition_term = np.exp(transition_one/transition_two)
                density = np.exp(beta*(pot-potn))
                val = transition_term*density

                if val > 1:
                    pot, pot_dx = potn, potn_dx
                    xo = xn
                else:
                    if val > local_st.uniform():
                        pot, pot_dx = potn, potn_dx
                        xo = xn
            dynamic[-1] = binos
            np.save(name, dynamic)


        except TypeError:
            dynamic = np.zeros(steps+1)
            sigma = np.sqrt(2.0/beta)
            noise = sigma*np.sqrt(dt)
            pot = fes(xo)
            pot_dx = derivative_point(fes, xo)

            for m in range(steps):
                dynamic[m] = xo
                r = local_st.normal(mu, std)
                boundary = xo - (dt*pot_dx)+(noise*r)

                if a<=boundary<=b:
                    xn = boundary
                else:
                    xn = xo

                potn = fes(xn)
                potn_dx = derivative_point(fes, xn)
                transition_one = abs(xn-xo+dt*pot_dx)**2-abs(xo-xn+dt*potn_dx)**2
                transition_two = (4.0/beta)*dt
                transition_term = np.exp(transition_one/transition_two)
                density = np.exp(beta*(pot-potn))
                val = transition_term*density

                if val > 1:
                    pot, pot_dx = potn, potn_dx
                    xo = xn
                else:
                    if val > local_st.uniform():
                        pot, pot_dx = potn, potn_dx
                        xo = xn

            dynamic[-1] = binos
            np.save(name, dynamic)

    else:
        try:
            size_space = len(fes)
            dynamic = np.zeros(steps)
            ds = float(b-a)/(size_space-1)
            sigma = np.sqrt(2.0/beta)
            noise = sigma*np.sqrt(dt)
            xo_ar = point_location(a, ds, xo)
            pot = fes[xo_ar]
            pot_dx = devone_array(ds, fes, xo_ar)

            for m in range(steps):
                dynamic[m] = xo
                r = local_st.normal(mu, std)
                boundary = xo - (dt*pot_dx)+(noise*r)

                if a<=boundary<=b:
                     xn = boundary
                else:
                     xn = xo

                xn_ar = point_location(a, ds, xn)
                potn = fes[xn_ar]
                potn_dx = devone_array(ds, fes, xn_ar)
                transition_one = abs(xn-xo+dt*pot_dx)**2-abs(xo-xn+dt*potn_dx)**2
                transition_two = (4.0/beta)*dt
                transition_term = np.exp(transition_one/transition_two)
                density = np.exp(beta*(pot-potn))
                val = transition_term*density

                if val > 1:
                    pot, pot_dx = potn, potn_dx
                    xo = xn
                else:
                    if val > local_st.uniform():
                        pot, pot_dx = potn, potn_dx
                        xo = xn
            return dynamic

        except TypeError:
            dynamic = np.zeros(steps)
            sigma = np.sqrt(2.0/beta)
            noise = sigma*np.sqrt(dt)
            pot = fes(xo)
            pot_dx = derivative_point(fes, xo)

            for m in range(steps):
                dynamic[m] = xo
                r = local_st.normal(mu, std)
                boundary = xo - (dt*pot_dx)+(noise*r)

                if a<=boundary<=b:
                    xn = boundary
                else:
                    xn = xo

                potn = fes(xn)
                potn_dx = derivative_point(fes, xn)
                transition_one = abs(xn-xo+dt*pot_dx)**2-abs(xo-xn+dt*potn_dx)**2
                transition_two = (4.0/beta)*dt
                transition_term = np.exp(transition_one/transition_two)
                density = np.exp(beta*(pot-potn))
                val = transition_term*density

                if val > 1:
                    pot, pot_dx = potn, potn_dx
                    xo = xn
                else:
                    if val > local_st.uniform():
                        pot, pot_dx = potn, potn_dx
                        xo = xn
            return dynamic


#Gives several euler-maruyama dynamic simulations appended into one, uses parallel computing.
def emone_para(a, b, fes, beta, binos, ndynamics, dt, steps, mu, std, xo, data=None, oficial=None):

    processes = []
    if oficial==None:
        for j in range(ndynamics):
            namej = "emone_results_"+str(j)
            pj = mp.Process(target=emone, args=(a, b, fes, beta, binos, dt, steps, mu, std, xo, namej))
            processes.append(pj)
            pj.start()
        for j in range(ndynamics):
            processes[j].join()

        super_dynamic = np.array([])
        for k in range(ndynamics):
            namek = "emone_results_"+str(k)+".npy"
            nowfalse = np.load(namek)
            now = nowfalse[0:-1]
            super_dynamic = np.concatenate((super_dynamic, now))
        for k in range(ndynamics):
            nameu = "emone_results_"+str(k)+".npy"
            os.remove(nameu)

        if data==None:
            super_dynamic = np.concatenate((super_dynamic, np.array([binos])))
            np.save("emone_results",super_dynamic)
        else:
            return super_dynamic
    else:
        ndynamico = ndynamics + 1
        for j in range(ndynamico):
            namej = "emone_results_"+str(j)
            pj = mp.Process(target=emone, args=(a, b, fes, beta, binos, dt, steps, mu, std, xo, namej))
            processes.append(pj)
            pj.start()
        for j in range(ndynamico):
            processes[j].join()

        shorty = np.array([])
        tempori = np.load("emone_results_"+str(ndynamics)+".npy")
        shorty = np.concatenate((shorty,tempori))
        os.remove("emone_results_"+str(ndynamics)+".npy")
        super_dynamic = np.array([])
        for k in range(ndynamics):
            namek = "emone_results_"+str(k)+".npy"
            nowfalse = np.load(namek)
            now = nowfalse[0:-1]
            super_dynamic = np.concatenate((super_dynamic, now))
        for k in range(ndynamics):
            nameu = "emone_results_"+str(k)+".npy"
            os.remove(nameu)

        return super_dynamic, shorty

#Array 2D derivative. Entries seem to be inverted, pay no atention to that. coco
def devtwo_array(ds, fesgrid, xloc, yloc):

    val_f = fesgrid[xloc, yloc]


    try:
        incre_x = fesgrid[xloc, yloc+1]
        decre_x = fesgrid[xloc, yloc-1]
        partial_x = float(incre_x-decre_x)/(2*ds)
    except IndexError:
        try:
            increment_x = fesgrid[xloc, yloc+1]
            partial_x = float(increment_x-val_f)/ds
        except IndexError:
            increment_x = fesgrid[xloc, yloc-1]
            partial_x = float(val_f-increment_x)/ds

    try:
        incre_y = fesgrid[xloc+1, yloc]
        decre_y = fesgrid[xloc-1, yloc]
        partial_y = float(incre_y-decre_y)/(2*ds)
    except IndexError:
        try:
            increment_y = fesgrid[xloc+1, yloc]
            partial_y = float(increment_y-val_f)/ds
        except IndexError:
            increment_y = fesgrid[xloc-1, yloc]
            partial_y = float(val_f-increment_y)/ds

    return partial_x, partial_y

#Gives euler-maruyama dynamic simulation in 2D.
def emtwo(a, b, c, d, fes, beta, binos, dt, steps, mu, std, xo, yo, name, data=None, seed=None):

    local_st = np.random.RandomState(seed)
    if data==None:
        try:
            size_grids = len(fes[0,:])
            dynamic = np.zeros((2, steps+1))
            ds = float(b-a)/(size_grids-1)
            sigma = np.sqrt(2.0/beta)
            noise = sigma*np.sqrt(dt)
            xo_ar, yo_ar = point_location_2D(a, c, ds, [xo,yo])
            pot = fes[xo_ar, yo_ar]
            pot_dx, pot_dy = devtwo_array(ds, fes, xo_ar, yo_ar)

            for m in range(steps):
                dynamic[0,m] = xo
                dynamic[1,m] = yo
                rx = local_st.normal(mu, std)
                ry = local_st.normal(mu, std)
                xboundary = xo-(dt*pot_dx)+(noise*rx)
                yboundary = yo-(dt*pot_dy)+(noise*ry)

                if a<=xboundary<=b:
                    xn = xboundary
                else:
                    xn = xo

                if c<=yboundary<=d:
                    yn = yboundary
                else:
                    yn = yo

                xn_ar, yn_ar = point_location_2D(a, c, ds, [xn, yn])
                potn = fes[xn_ar, yn_ar]
                potn_dx, potn_dy = devtwo_array(ds, fes, xn_ar, yn_ar)
                transition_one = (xn-xo+dt*pot_dx)**2+(yn-yo+dt*pot_dy)**2-(xo-xn+dt*potn_dx)**2-(yo-yn+dt*potn_dy)**2
                transition_two = (4.0/beta)*dt
                transition_term = np.exp(transition_one/transition_two)
                density = np.exp(beta*(pot-potn))
                val = transition_term*density
                if val > 1:
                    pot, pot_dx, pot_dy = potn, potn_dx, potn_dy
                    xo, yo = xn, yn
                else:
                    if val > local_st.uniform():
                        pot, pot_dx, pot_dy = potn, potn_dx, potn_dy
                        xo, yo = xn, yn

            dynamic[0,-1] = int(binos)
            np.save(name, dynamic)


        except TypeError:
            dynamic = np.zeros((2,steps+1))
            sigma = np.sqrt(2/beta)
            noise = sigma*np.sqrt(dt)
            pot = fes(xo,yo)
            pot_dx, pot_dy = partial_point(fes, xo, yo)

            for m in range(steps):
                dynamic[0,m] = xo
                dynamic[1,m] = yo
                rx = local_st.normal(mu, std)
                ry = local_st.normal(mu, std)
                xboundary = xo-(dt*pot_dx)+(noise*rx)
                yboundary = yo-(dt*pot_dy)+(noise*ry)

                if a<=xboundary<=b:
                    xn = xboundary
                else:
                    xn = xo

                if c<=yboundary<=d:
                    yn = yboundary
                else:
                    yn = yo

                potn = fes(xn, yn)
                potn_dx, potn_dy = partial_point(fes, xn, yn)
                transition_one = (xn-xo+dt*pot_dx)**2+(yn-yo+dt*pot_dy)**2-(xo-xn+dt*potn_dx)**2-(yo-yn+dt*potn_dy)**2
                transition_two = (4.0/beta)*dt
                transition_term = np.exp(transition_one/transition_two)
                density = np.exp(beta*(pot-potn))
                val = transition_term*density
                if val > local_st.uniform() or val>1:
                    pot, pot_dx, pot_dy = potn, potn_dx, potn_dy
                    xo, yo = xn, yn

            dynamic[0,-1] = int(binos)
            np.save(name, dynamic)

    else:
        try:
            size_grids = len(fes[0,:])
            dynamics = np.zeros((2, steps+1))
            ds = float(b-a)/(size_grids-1)
            sigma = np.sqrt(2/beta)
            noise = sigma*np.sqrt(dt)
            xo_ar, yo_ar = point_location_2D(a, c, ds, [xo,yo])
            pot = fes[xo_ar, yo_ar]
            pot_dx, pot_dy = devtwo_array(ds, fes, xo_ar, yo_ar)

            for m in range(steps):
                dynamics[m,0] = xo
                dynamics[m,1] = yo
                rx = local_st.normal(mu, std)
                ry = local_st.normal(mu, std)
                xboundary = xo-(dt*pot_dx)+(noise*rx)
                yboundary = yo-(dt*pot_dy)+(noise*ry)

                if a<=xboundary<=b:
                    xn = xboundary
                else:
                    xn = xo

                if c<=yboundary<=d:
                    yn = yboundary
                else:
                    yn = yo

                xn_ar, yn_ar = point_location_2D(a, c, ds, [xn, yn])
                potn = fes[xn_ar, yn_ar]
                potn_dx, potn_dy = devtwo_array(ds, fes, xn_ar, yn_ar)
                transition_one = (xn-xo+dt*pot_dx)**2+(yn-yo+dt*pot_dy)**2-(xo-xn+dt*potn_dx)**2-(yo-yn+dt*potn_dy)**2
                transition_two = (4.0/beta)*dt
                transition_term = np.exp(transition_one/transition_two)
                density = np.exp(beta*(pot-potn))
                val = transition_term*density
                if val > 1:
                    pot, pot_dx, pot_dy = potn, potn_dx, potn_dy
                    xo, yo = xn, yn
                else:
                    if val > local_st.uniform():
                        pot, pot_dx, pot_dy = potn, potn_dx, potn_dy
                        xo, yo = xn, yn

            return dynamics

        except TypeError:
            sigma = np.sqrt(2/beta)
            noise = sigma*np.sqrt(dt)
            pot = fes(xo,yo)
            pot_dx, pot_dy = partial_point(fes, xo, yo)
            dynamics = np.zeros((2, steps))

            for m in range(steps):
                dynamics[m,0] = xo
                dynamics[m,1] = yo
                rx = local_st.normal(mu, std)
                ry = local_st.normal(mu, std)
                xboundary = xo-(dt*pot_dx)+(noise*rx)
                yboundary = yo-(dt*pot_dy)+(noise*ry)

                if a<=xboundary<=b:
                    xn = xboundary
                else:
                    xn = xo

                if c<=yboundary<=d:
                    yn = yboundary
                else:
                    yn = yo

                potn = fes(xn, yn)
                potn_dx, potn_dy = partial_point(fes, xn, yn)
                transition_one = (xn-xo+dt*pot_dx)**2+(yn-yo+dt*pot_dy)**2-(xo-xn+dt*potn_dx)**2-(yo-yn+dt*potn_dy)**2
                transition_two = (4.0/beta)*dt
                transition_term = np.exp(transition_one/transition_two)
                density = np.exp(beta*(pot-potn))
                val = transition_term*density
                if val > local_st.uniform() or val>1:
                    pot, pot_dx, pot_dy = potn, potn_dx, potn_dy
                    xo, yo = xn, yn

            return dynamics


#Several langevin dynamics appended into one.
def emtwo_para(a, b, c, d, fes, beta, binos, ndynamics, dt, steps, mu, std, xo, yo, data=None, oficial=None):

    processes = []
    if oficial==None:
        for j in range(ndynamics):
            namej = "emtwo_results_"+str(j)
            pj = mp.Process(target=emtwo, args=(a, b, c, d, fes, beta, binos, dt, steps, mu, std, xo, yo, namej))
            processes.append(pj)
            pj.start()
        for j in range(ndynamics):
            processes[j].join()

        super_dynamic = np.zeros((2,1))
        for k in range(ndynamics):
            namek = "emtwo_results_"+str(k)+".npy"
            nowfalse = np.load(namek)
            now = nowfalse[:,0:-1]
            super_dynamic = np.concatenate((super_dynamic, now), axis=1)
        for k in range(ndynamics):
            nameu = "emtwo_results_"+str(k)+".npy"
            os.remove(nameu)

        if data==None:
            bini = np.zeros((2,1))
            bini[1,0] = binos
            super_dynamic = np.concatenate((super_dynamic,bini), axis=1)
            np.save("emtwo_results", super_dynamic)
        else:
            return super_dynamic
    else:
        ndynamico = ndynamics+1
        for j in range(ndynamico):
            namej = "emtwo_results_"+str(j)
            pj = mp.Process(target=emtwo, args=(a, b, c, d, fes, beta, binos, dt, steps, mu, std, xo, yo, namej))
            processes.append(pj)
            pj.start()
        for j in range(ndynamico):
            processes[j].join()

        shorty = np.zeros((2,1))
        coquito = np.load("emtwo_results_"+str(ndynamics)+".npy")
        shorty = np.concatenate((shorty, coquito), axis=1)
        os.remove("emtwo_results_"+str(ndynamics)+".npy")
        super_dynamic = np.zeros((2,1))
        for k in range(ndynamics):
            namek = "emtwo_results_"+str(k)+".npy"
            nowfalse = np.load(namek)
            now = nowfalse[:,0:-1]
            super_dynamic = np.concatenate((super_dynamic, now), axis=1)
        for k in range(ndynamics):
            nameu = "emtwo_results_"+str(k)+".npy"
            os.remove(nameu)

        return super_dynamic, shorty


##############################################################
######################METADYNAMICS############################
##############################################################

#Ordinary derivative from function, at a point.
def derivative_point(function, point):

    ds = 1e-9
    increment = function(point+ds)
    val_f = function(point)
    derivative = float(increment-val_f)/(ds)

    return derivative

#Gives contribution to FES aproximation for a point in the dynamics.
def contribution_point(xgrid, fes_sample, point, gaussian_height, delta_sigma):

    x_res = xgrid[1]-xgrid[0]
    x_in = xgrid[0]
    three_sigma_array = int(round(((6.0*delta_sigma)/(x_res))))
    w = gaussian_height
    point_array = point_location(x_in, x_res, point)
    center_point = xgrid[point_array]
    exp_center = np.exp(-float((center_point-point)**2)/(2*(delta_sigma**2)))
    fes_sample[point_array] = fes_sample[point_array] + w*exp_center

    for i in range(1, three_sigma_array):
        if point_array-i>=0:
            left_point = xgrid[point_array-i]
            exp_left = np.exp(-float(abs(left_point-point)**2)/(2*(delta_sigma**2)))
            contribution_left = w*exp_left
            fes_sample[point_array-i] = fes_sample[point_array-i] + contribution_left
            continue
        else:
            continue

    for l in range(1, three_sigma_array):
        try:
            right_point = xgrid[point_array+l]
            exp_right = np.exp(-float(abs(right_point-point)**2)/(2*(delta_sigma**2)))
            contribution_right = w*exp_right
            fes_sample[point_array+l] = fes_sample[point_array+l] + contribution_right
            continue
        except IndexError:
            continue

    return fes_sample

#Gives aproximation of FES, given fes and initial position, one dimension.
def mdyone(x_grid, potential, steps, initial_position, delta_sigma, beta, dt, gaussian_height,alpha=None, data=None):

    '''|This function takes as entries the domain of your FES (as an array), the parameterized FES, number of steps of simulation, initial
       |position in configuration space, delta sigma, height of gaussian and delta x.
       |
       |By default returns file with data corresponding to FES reconstruction, unless last argument is 1 then returns the data without saving it'''

    start = time.time()

    if alpha==None:
        alfa = 0.5
    else:
        alfa = alpha

    size_space = np.size(x_grid)
    a = x_grid[0]
    b = x_grid[-1]
    ds = x_grid[2]-x_grid[1]
    sigma = np.sqrt(2.0/beta)
    noise = sigma*np.sqrt(dt)
    xo = initial_position
    fes_sample = np.zeros(size_space)
    xo_ar = point_location(a, ds, xo)
    pot_dx = derivative_point(potential, xo) + devone_array(ds, fes_sample, xo_ar)
    pot = potential(xo)+fes_sample[xo_ar]
    dinamica = []
    cont = 0

    for i in xrange(1, steps+1):
        cont = 0
        dinamica.append(xo)
        r = np.random.normal(0,1)
        boundary = xo-(pot_dx*dt)+(noise*r)

        if a<=boundary<=b:
            xn = boundary
        else:
            xn = xo
            cont = 1

        xn_ar = point_location(a, ds, xn)
        potn = potential(xn) + fes_sample[xn_ar]
        potn_dx = derivative_point(potential, xn) + devone_array(ds, fes_sample, xn_ar)
        transition_one = abs(xn-xo+dt*pot_dx)**2-abs(xo-xn+dt*potn_dx)**2
        transition_two = (4.0/beta)*dt
        transition_term = np.exp(transition_one/transition_two)
        density = np.exp(beta*(pot-potn))
        val = transition_term*density

        if val > 1:
            pot, pot_dx = potn, potn_dx
            xo = xn
        else:
            if val > np.random.uniform():
                pot, pot_dx = potn, potn_dx
                xo = xn

        if cont==0:
            fes_sample = contribution_point(x_grid, fes_sample, xo, gaussian_height, delta_sigma)
        else:
            continue

    end = time.time()
    duration = (1.0*(end-start))/60


    fes_samplet = np.zeros(size_space+3)
    for j in range(size_space):
        fes_samplet[j] = -fes_sample[j]
    fes_samplet[-3] = a
    fes_samplet[-2] = b
    fes_samplet[-1] = duration
    if data==None:
        np.save("mdyone_results", fes_samplet)
        np.save("mdyone_dynamic", dinamica)
    else:
        return fes_sample, duration

#Gives location of point in grid.

def point_location_2D(array_in_x, array_in_y, array_ds, point):

    location_x = int(round(float(point[0]-array_in_x)/array_ds))
    location_y = int(round(float(point[1]-array_in_y)/array_ds))

    return location_x, location_y

#Numerical partial derivative from function, at a point.
def partial_point(function, x, y):

    dx = 1e-9
    dy = 1e-9
    increment_x = function(x+dx, y)
    increment_y = function(x, y+dy)
    val_f = function(x, y)

    partial_x = float(increment_x-val_f)/(dx)
    partial_y = float(increment_y-val_f)/(dy)

    return partial_x, partial_y

#Gives the Gaussian term at specified time and position, two dimensions.
def gaussian_term_2D(normal_x, normal_y, pos_x, pos_y, gaussian_height, delta_sigma):

    W = gaussian_height
    sum_term = 0
    sofar = np.size(pos_x)

    for i in xrange(sofar):
        exp_x = np.exp(-float(abs(normal_x-pos_x[i])**2)/(2*(delta_sigma**2)))
        exp_y = np.exp(-float(abs(normal_y-pos_y[i])**2)/(2*(delta_sigma**2)))
        sum_term = sum_term + exp_x*exp_y
    return sum_term*W

#Gives contribution to FES aproximation for a point in the dynamics.
def contribution_point_2D(xgrid, ygrid, fes_sample, point, gaussian_height, delta_sigma):

    size = np.size(xgrid)
    x_res = xgrid[1]-xgrid[0]
    y_res = ygrid[1]-ygrid[0]
    x_in = xgrid[0]
    y_in = ygrid[0]
    three_sigma_x = int(round(((5.0*delta_sigma)/(x_res))))
    three_sigma_y = int(round(((5.0*delta_sigma)/(y_res))))
    w = gaussian_height
    array_x, array_y = point_location_2D(x_in, y_in, x_res, point)

    center_x = xgrid[array_x]
    exp_center_x = np.exp(-float(abs(center_x-point[0])**2)/(2*(delta_sigma**2)))
    for j in range(0, three_sigma_y):
        if array_y-j>=0:
            left_point_y = ygrid[array_y-j]
            exp_left_y = np.exp(-float(abs(left_point_y-point[1])**2)/(2*(delta_sigma**2)))
            fes_sample[array_y-j][array_x] = fes_sample[array_y-j][array_x] + w*exp_center_x*exp_left_y
            continue
        else:
            continue
    for l in range(1, three_sigma_y):
        try:
            right_point_y = ygrid[array_y+l]
            exp_right_y = np.exp(-float(abs(right_point_y-point[1])**2)/(2*(delta_sigma**2)))
            fes_sample[array_y+l][array_x] = fes_sample[array_y+l][array_x] + w*exp_center_x*exp_right_y
        except IndexError:
            continue

    for i in range(1, three_sigma_x):
        if array_x-i>=0:
            left_point_x = xgrid[array_x-i]
            exp_left_x = np.exp(-float(abs(left_point_x-point[0])**2)/(2*(delta_sigma**2)))
            for j in range(0, three_sigma_y):
                if array_y-j>=0:
                    left_point_y = ygrid[array_y-j]
                    exp_left_y = np.exp(-float(abs(left_point_y-point[1])**2)/(2*(delta_sigma**2)))
                    fes_sample[array_y-j][array_x-i] = fes_sample[array_y-j][array_x-i] + w*exp_left_x*exp_left_y
                    continue
                else:
                    continue
            for l in range(1, three_sigma_y):
                try:
                    right_point_y = ygrid[array_y+l]
                    exp_right_y = np.exp(-float(abs(right_point_y-point[1])**2)/(2*(delta_sigma**2)))
                    fes_sample[array_y+l][array_x-i] = fes_sample[array_y+l][array_x-i] + w*exp_left_x*exp_right_y
                except IndexError:
                    continue
            continue
        else:
            continue

    for i in range(1, three_sigma_x):
        try:
            right_point_x = xgrid[array_x+i]
            exp_right_x = np.exp(-float(abs(right_point_x-point[0])**2)/(2*(delta_sigma**2)))
            for j in range(0, three_sigma_y):
                if array_y-j>=0:
                    left_point_y = ygrid[array_y-j]
                    exp_left_y = np.exp(-float(abs(left_point_y-point[1])**2)/(2*(delta_sigma**2)))
                    fes_sample[array_y-j][array_x+i] = fes_sample[array_y-j][array_x+i] + w*exp_right_x*exp_left_y
                    continue
                else:
                    continue
            for l in range(1, three_sigma_y):
                try:
                    right_point_y = ygrid[array_y+l]
                    exp_right_y = np.exp(-float(abs(right_point_y-point[1])**2)/(2*(delta_sigma**2)))
                    fes_sample[array_y+l][array_x+i] = fes_sample[array_y+l][array_x+i] + w*exp_right_x*exp_right_y
                except IndexError:
                    continue
        except IndexError:
            continue

    return fes_sample

#Mixed derivative and value.
def mixed_stuff(function, grid, xo, yo, a, c, ds):
    point = [xo,yo]
    x_ar, y_ar = point_location_2D(a, c, ds, point)
    f = function(xo,yo)
    df = partial_point(function, xo, yo)
    g = grid[y_ar, x_ar]
    dg = devtwo_array(ds, grid, y_ar, x_ar)
    t = f+g
    dt_0 = df[0]+dg[0]
    dt_1 = df[1]+dg[1]

    return t, dt_0, dt_1

#Gives you a sequence of steps, given the potential and initial position, n dimensions.
def mdytwo(xgrid, ygrid, fes, steps, initial_point, delta_sigma, beta, dt, gaussian_height, alpha=None, data=None):

    '''|This function takes as entries the domain of your FES (two arrays: x,y), the parameterized 2D-FES, number of steps of simulation, initial
       |position (tuple e.g. [2,3]), delta sigma, height of gaussian and deltas x,y.
       |
       |Returns a matrix containing values of FES reconstruction. By default saves data in text file, otherwise it just returns matrix.'''

    start = time.time()
    if alpha==None:
        alfa = 0.5
    else:
        alfa = alpha

    size_space = np.size(xgrid)
    a = xgrid[0]
    b = xgrid[-1]
    c = ygrid[0]
    d = ygrid[-1]
    ds = xgrid[1]-xgrid[0]
    sigma = np.sqrt(2/beta)
    noise = sigma*np.sqrt(dt)
    xo, yo = initial_point[0], initial_point[1]
    fes_sample = np.zeros([size_space, size_space])
    impor = mixed_stuff(fes, fes_sample, xo, yo, a, c, ds)
    pot_dx, pot_dy = impor[1], impor[2]
    pot = impor[0]
    pos_x, pos_y = [], []
    cont = 0
    fes_sample = contribution_point_2D(xgrid, ygrid, fes_sample, [xo,yo], gaussian_height, delta_sigma)

    for i in xrange(1, steps+1):
        print(steps-i)
        pos_x.append(xo)
        pos_y.append(yo)
        rx = np.random.normal(0,1)
        ry = np.random.normal(0,1)
        xbound = xo-(dt*pot_dx)+(noise*rx)
        ybound = yo-(dt*pot_dy)+(noise*ry)
        if a<=xbound<=b:
            xn = xbound
        else:
            cont = 1
            xn = xo

        if c<=ybound<=d:
            yn = ybound
        else:
            cont = 1
            yn = yo

        imporn = mixed_stuff(fes, fes_sample, xn, yn, a, c, ds)
        potn = imporn[0]
        potn_dx, potn_dy = imporn[1], imporn[2]
        transition_one = (xn-xo+dt*pot_dx)**2+(yn-yo+dt*pot_dy)**2-(xo-xn+dt*potn_dx)**2-(yo-yn+dt*potn_dy)**2
        transition_two = (4.0/beta)*dt
        transition_term = np.exp(transition_one/transition_two)
        density = np.exp(beta*(pot-potn))
        val = transition_term*density
        if val > np.random.uniform() or val>1:
            pot, pot_dx, pot_dy = potn, potn_dx, potn_dy
            xo, yo = xn, yn
        if cont==0:
            fes_sample = contribution_point_2D(xgrid, ygrid, fes_sample, [xo,yo], gaussian_height, delta_sigma)
        else:
            continue

    end = time.time()
    duration = (1.0*(end-start))/60

    dinamicota = np.zeros((2,steps))
    dinamicota[0,:] = pos_x
    dinamicota[1,:] = pos_y
    if data==None:
        np.save('mdytwo_results', -fes_sample)
        np.save('mdytwo_dynamic', dinamicota)
    else:
        return fes_sample, duration

####################################################################
########################VARIATIONALLY ENHANCED######################
####################################################################

#Given vector of coeficients and one number, returns integer in the series correspondent to that number and the type of function (0=cos, 1=sin).
def correspondent_exp(alpha, entry):

    number_cos_terms = int((np.size(alpha)+1)*(2**-1))

    if entry<number_cos_terms:
        integer = entry
        type_of_exp = 0
    else:
        integer = entry-number_cos_terms+1
        type_of_exp = 1

    return np.array([integer, type_of_exp])

#Defines expansion functions in the closed interval from a to b.
def expansion_function(type_of_exp, k, x, a, b):

    l = float(b-a)/2

    if type_of_exp==0:
        expansion_value = np.cos(float(k*np.pi*x)/(l))
    else:
        expansion_value = np.sin(float(k*np.pi*x)/(l))

    return expansion_value

#Defines biased potential as a fourier expansion. Alpha has the terms corresponding to 0, 1_cos, ..., n_cos, 1_sin, ..., n_sin.
def biased_potential(alpha, x, a, b):

    size_expansion = np.size(alpha)
    coef_and_types = [correspondent_exp(alpha, i) for i in range(0, size_expansion)]
    function_value = 0

    for j in range(0, size_expansion):
        function_value = function_value + alpha[j]*expansion_function(coef_and_types[j][1], coef_and_types[j][0], x, a, b)

    return function_value

#Computes expected value given function, weight and limits of integration.
def expected_value(function, weight, a, b):

    def weight_function(x):
        return function(x)*weight(x)

    exp_val = integrate.quad(weight_function, a, b)[0]

    return exp_val

#Computes biased expected value, given function and dynamic.
def biased_expected(function, dinamica):

    steps = np.size(dinamica)
    expected = 0
    for i in range(steps):
        xnow = dinamica[i]
        value = function(xnow)
        expected = expected+value
    expected = expected/steps
    return expected

#Gives biased covariance
def biased_covariance(f_one, f_two, dinamica):

    def product_f(x):
        return f_one(x)*f_two(x)

    biased_p = biased_expected(product_f,dinamica)
    biased_one_two = biased_expected(f_one,dinamica)*biased_expected(f_two,dinamica)
    covariance = biased_p - biased_one_two

    return covariance

#Gives entry of omegas gradient.
def gradient_entry(alpha, weight, a, b, at, bt, dinamica,entry):

    coef_and_type = correspondent_exp(alpha, entry)

    def entry_expansion_function(x):
        return expansion_function(coef_and_type[1], coef_and_type[0], x, at, bt)

    gradient_entry = expected_value(entry_expansion_function, weight, a, b)-biased_expected(entry_expansion_function, dinamica)

    return gradient_entry

#Gives an entry of omegas hessian.
def hessian_entry(beta,alpha,at, bt, dinamica, entry_one, entry_two):

    terms_one = correspondent_exp(alpha, entry_one)
    terms_two = correspondent_exp(alpha, entry_two)

    def exp_function_one(x):
        return expansion_function(terms_one[1], terms_one[0], x, at, bt)
    def exp_function_two(x):
        return expansion_function(terms_two[1], terms_two[0], x, at, bt)

    hessian_entry = beta*biased_covariance(exp_function_one, exp_function_two, dinamica)

    return hessian_entry

#Gives some entries of derivative.
def der_omega(beta, fes, alpha, weight, a, b, at, bt, dinamica,num_entries, num_process, diag_hessian, gradient_para):

    zeta = num_entries
    size_all = np.size(alpha)
    while num_entries*num_process+zeta>size_all:
        zeta = zeta-1

    if zeta==0:
        pass
    else:
        for i in range(zeta):
            entry = i+num_process*num_entries
            gradient_para[entry] = gradient_entry(alpha, weight, a, b, at, bt, dinamica,entry)
            diag_hessian[entry] = hessian_entry(beta,alpha,at, bt, dinamica,entry, entry)

#Gives derivatives of omega.
def der_omega_para(beta, fes, alpha, weight, a, b, at, bt, dinamica,cores=None):

    if cores==None:
        num_cores = mp.cpu_count()
    else:
        num_cores = cores

    size_space = np.size(alpha)
    z = int(math.ceil((1.0*size_space)/(num_cores)))
    diag_hessian = mp.Array('d', size_space)
    gradient_para = mp.Array('d', size_space)
    processes = []

    for j in range(num_cores):
        pj = mp.Process(target=der_omega, args=(beta, fes, alpha, weight, a, b, at, bt, dinamica,z, j, diag_hessian, gradient_para))
        processes.append(pj)
        pj.start()
    for j in range(num_cores):
        processes[j].join()

    return gradient_para[:], diag_hessian[:]

#comprobation
def compro(xgrid, a, b, fes, alpha, weight, beta):

    def potential_sofar(x):
        return biased_potential(alpha, x, a, b)

    def exp_numerator(x):
        return (np.exp(-beta*(fes(x)+potential_sofar(x))))

    def exp_denominator(x):
        return np.exp(-beta*(fes(x)))

    numerator = integrate.quad(exp_numerator, a, b)[0]
    denominator = integrate.quad(exp_denominator, a, b)[0]
    logi = (1.0/beta)*np.log(numerator/denominator)
    other = expected_value(potential_sofar, weight, a, b)

    return logi+other

#Computes Omega minima.
def omega_min1(fes, target_distribution, a, b, beta):

    def function_one(x):
        return np.exp(-beta*(fes(x)))

    def function_two(x):
        return target_distribution(x)*(fes(x)+(1.0/beta)*np.log(target_distribution(x)))

    term_one = -(1.0/beta)*np.log(integrate.quad(function_one, a, b)[0])
    term_two = -(integrate.quad(function_two, a, b)[0])

    return term_one+term_two

#Variationally enhanced method for sampling.
def varone(x_grid, fes, number_coeficients, iterations, mu, beta, epsilon, binos, ndynamics, dt, steps, xo, distribution=None, cores=None, data=None):

    start = time.time()
    a = x_grid[0]
    b = x_grid[-1]
    at = x_grid[0]-epsilon
    bt = x_grid[-1]+epsilon
    avg_alpha_notnormal = np.zeros(number_coeficients)
    alpha_new = 0
    last_pos = xo
    toda_tray = np.array([])


    if distribution==None:
        def target_distribution(x):
            return 1.0/(bt-at)
    else:
        target_distribution=distribution

    meta = omega_min1(fes, target_distribution, at, bt, beta)
    bajado = np.zeros(iterations)
    print meta
    for i in range(0, iterations):
        avg_alpha_notnormal = avg_alpha_notnormal + alpha_new
        avg_alpha = (1.0/(i+1))*avg_alpha_notnormal

        def biased(x):
            value = fes(x)+biased_potential(avg_alpha, x, at, bt)
            return value
        trayectos = emone_para(at, bt, biased, beta, binos, ndynamics, dt, steps, 0, 1, last_pos, 1, 1)
        mini_trayec = trayectos[0][0:-1]
        seguir = trayectos[1][0:-1]
        toda_tray = np.concatenate((toda_tray, seguir))

        a_local = min(mini_trayec)
        b_local = max(mini_trayec)
        last_pos = mini_trayec[-1]

        derivatives = der_omega_para(beta, fes, avg_alpha, target_distribution, a_local, b_local, at, bt, mini_trayec,cores)
        gradient, hessian = derivatives[0], derivatives[1]
        alpha_new = alpha_new - mu*(gradient+hessian*(alpha_new-avg_alpha))
        p = compro(x_grid, at, bt, fes, avg_alpha, target_distribution, beta)
        bajado[i] = p
        print (a_local, b_local, p, i)

    def zv(x):
        value = np.exp(-beta*(fes(x)+biased_potential(avg_alpha, x, at, bt)))
        return value

    z_v = integrate.quad(zv, at, bt)[0]
    constant = (1.0/beta)*np.log(z_v)
    fes_sample = [-biased_potential(avg_alpha, i, at, bt)-(1.0/beta)*np.log(target_distribution(i))-constant for i in x_grid]
    end = time.time()
    duration = (1.0*(end-start))/60


    np.save('varibajada',bajado)
    tamano = np.size(fes_sample)
    fesgrido = np.zeros(tamano+3)
    fesgrido[-3] = a
    fesgrido[-2] = b
    fesgrido[-1] = duration
    for k in range(tamano):
        fesgrido[k] = fes_sample[k]

    if data==None:
        np.save("varone_results", fesgrido)
        np.save("varone_dynamic", toda_tray)
    else:
        return fes_sample, duration

#Defines expansion functions in an R2 rectangle [a, b]x[c, d]. (1=coscos, 2=cossin, 3=sincos, 4=sinsin)
def expansion_function_2D(type_of_exp, k_x, k_y, x, y, a, b, c, d):

    l_x = float(b-a)/2
    l_y = float(d-c)/2

    if type_of_exp==0:
        expansion_value = np.cos(float(k_x*np.pi*x)/(l_x))*np.cos(float(k_y*np.pi*y)/(l_y))
    elif type_of_exp==1:
        expansion_value = np.cos(float((k_x)*np.pi*x)/(l_x))*np.sin(float((k_y+1)*np.pi*y)/(l_y))
    elif type_of_exp==2:
        expansion_value = np.sin(float((k_x+1)*np.pi*x)/(l_x))*np.cos(float((k_y)*np.pi*y)/(l_y))
    else:
        expansion_value = np.sin(float((k_x+1)*np.pi*x)/(l_x))*np.sin(float((k_y+1)*np.pi*y)/(l_y))

    return expansion_value

#Given size of a matrix and a number, returns correspondent place in matrix.
def number_to_location(matrix_size, number):

    matrix_half = int(np.sqrt(matrix_size))
    if number%matrix_half==0:
        row = number/matrix_half-1
        column = matrix_half-1
    else:
        row = number/matrix_half
        column = number%matrix_half-1

    return row, column

#Given matrix of coefficients and one number, returns integers in the series correspondent to that number and the type of function.
def correspondent_exp_2D(coef_matrix, entry):

    number_cos_cos = np.size(coef_matrix[0])
    number_cos_sin = np.size(coef_matrix[1])
    number_sin_cos = np.size(coef_matrix[2])
    number_sin_sin = np.size(coef_matrix[3])

    if entry<=number_cos_cos:
        integers = number_to_location(number_cos_cos, entry)
        type_of_exp = 0

    elif number_cos_cos<entry<=number_cos_cos+number_cos_sin:
        integers = number_to_location(number_cos_sin, entry-number_cos_cos)
        type_of_exp = 1

    elif number_cos_cos+number_cos_sin<entry<=number_cos_cos+number_cos_sin+number_sin_cos:
        integers = number_to_location(number_sin_cos, entry-number_cos_cos-number_cos_sin)
        type_of_exp = 2

    else:
        integers = number_to_location(number_sin_sin, entry-number_cos_cos-number_cos_sin-number_sin_cos)
        type_of_exp = 3

    return integers, type_of_exp

#Defines biased potential as a 2D Fourier expansion, coef_matrix has the terms corresponding to cos_cos, cos_sin, sin_cos, sin_sin.
def biased_potential_2D(coef_matrix, x, y, a, b, c, d):

    size_expansion = np.size(coef_matrix[1])+np.size(coef_matrix[2])+np.size(coef_matrix[3])+np.size(coef_matrix[0])
    function_value = 0

    for j in range(size_expansion):
        coef_and_type = correspondent_exp_2D(coef_matrix, j+1)
        coef_x = coef_and_type[0][0]
        coef_y = coef_and_type[0][1]
        typeof = coef_and_type[1]
        function_value = function_value + coef_matrix[typeof][coef_x,coef_y]*expansion_function_2D(typeof, coef_x, coef_y, x, y, a, b, c, d)

    return function_value

#Computes expected value given function, weight and limits of integration.
def expected_value_2D(function, weight, a, b, c, d):

    def weight_function(x, y):
        return function(x, y)*weight(x, y)

    exp_val = integrate.dblquad(weight_function, a, b, lambda x: c, lambda x: d)[0]

    return exp_val

#Computes biased expected value
def biased_expected_2D(function, dinamica):
    steps = np.size(dinamica[0,:])
    expected = 0
    for i in range(steps):
        xnow = dinamica[0,i]
        ynow = dinamica[1,i]
        value = function(xnow,ynow)
        expected = expected + value
    expected = expected/steps
    return expected

#Gives biased covariance of two functions
def biased_covariance_2D(f_one, f_two, dinamica):

    def p_functions(x, y):
        return f_one(x, y)*f_two(x, y)

    biased_product = biased_expected_2D(p_functions, dinamica)
    biased_one_two = biased_expected_2D(f_one, dinamica)*biased_expected_2D(f_two, dinamica)
    covariance = biased_product - biased_one_two

    return covariance

#Gives entry of omegas gradient.
def gradient_entry_2D(coef_matrix, weight, al, bl, cl, dl, a, b, c, d, dinamica,entry):

    coef_and_type = correspondent_exp_2D(coef_matrix, entry)

    def entry_expansion_function(x, y):
        return expansion_function_2D(coef_and_type[1], coef_and_type[0][0], coef_and_type[0][1], x, y, a, b, c, d)

    gradient_entry = expected_value_2D(entry_expansion_function, weight, al, bl, cl, dl)-biased_expected_2D(entry_expansion_function, dinamica)

    return gradient_entry

#Gives entry of omegas hessian.
def hessian_entry_2D(beta,coef_matrix, al, bl, cl, dl, a, b, c, d, dinamica,entry_one, entry_two):

    terms_one = correspondent_exp_2D(coef_matrix, entry_one)
    terms_two = correspondent_exp_2D(coef_matrix, entry_two)

    def exp_function_one(x, y):
        return expansion_function_2D(terms_one[1], terms_one[0][0], terms_one[0][1], x, y, a, b, c, d)

    def exp_function_two(x, y):
        return expansion_function_2D(terms_two[1], terms_two[0][0], terms_two[0][1], x, y, a, b, c, d)

    hessian_entry = beta*biased_covariance_2D(exp_function_one, exp_function_two, dinamica)

    return hessian_entry

#Gives some entries of derivative.
def der_omega_2D(beta, fes, coef_matrix, weight, al, bl, cl, dl, a ,b, c, d, dinamica,num_entries, num_process, hess, grad, size_all):

    zeta = num_entries
    while num_entries*num_process+zeta>size_all:
        zeta = zeta-1

    if zeta==0:
        pass
    else:
        for i in range(zeta):
            entry = i+num_process*num_entries
            grad[entry] = gradient_entry_2D(coef_matrix, weight, al, bl, cl, dl, a, b, c, d, dinamica,entry+1)
            hess[entry] = hessian_entry_2D(beta, coef_matrix, al, bl, cl, dl, a, b, c, d, dinamica,entry, entry+1)

#Gives derivfatives of omega.
def der_omega_para_2D(beta, fes, coef_matrix, weight, al, bl, cl, dl, a, b, c, d, dinamica,cores=None):

    dim_cc = int(np.size(coef_matrix[0][0,:]))
    dim_cs = int(np.size(coef_matrix[1][0,:]))
    dim_sc = int(np.size(coef_matrix[2][0,:]))
    dim_ss = int(np.size(coef_matrix[3][0,:]))
    number_cc = dim_cc**2
    number_cs = dim_cs**2
    number_sc = dim_sc**2
    number_ss = dim_ss**2

    size_space = number_cc+number_cs+number_sc+number_ss

    if cores==None:
        num_cores = mp.cpu_count()
    else:
        num_cores = cores

    z = int(math.ceil((1.0*size_space)/(num_cores)))
    hess_para = mp.Array('d', size_space)
    grad_para = mp.Array('d', size_space)
    processes = []
    for j in range(num_cores):
        pj = mp.Process(target=der_omega_2D, args=(beta, fes, coef_matrix, weight, al, bl, cl, dl, a, b, c, d, dinamica,z, j, hess_para, grad_para, size_space))
        processes.append(pj)
        pj.start()
    for j in range(num_cores):
        processes[j].join()

    grad_cc = np.matrix(np.zeros((dim_cc, dim_cc)))
    grad_cs = np.matrix(np.zeros((dim_cs, dim_cs)))
    grad_sc = np.matrix(np.zeros((dim_sc, dim_sc)))
    grad_ss = np.matrix(np.zeros((dim_ss, dim_ss)))
    gradient_omega = [grad_cc, grad_cs, grad_sc, grad_ss]
    hess_cc = np.matrix(np.zeros((dim_cc, dim_cc)))
    hess_cs = np.matrix(np.zeros((dim_cs, dim_cs)))
    hess_sc = np.matrix(np.zeros((dim_sc, dim_sc)))
    hess_ss = np.matrix(np.zeros((dim_ss, dim_ss)))
    hessian_omega = [hess_cc, hess_cs, hess_sc, hess_ss]

    for i in range(size_space):
        coef_and_type = correspondent_exp_2D(coef_matrix, i+1)
        coef_x = coef_and_type[0][0]
        coef_y = coef_and_type[0][1]
        typeof = coef_and_type[1]
        gradient_omega[typeof][coef_x, coef_y] = grad_para[i]
        hessian_omega[typeof][coef_x, coef_y] = hess_para[i]

    return gradient_omega, hessian_omega

#Computes omega min
def omega_min(fes, distribution, a, b, c, d, beta):

    def fone(x,y):
        return np.exp(-beta*(fes(x,y)))

    def ftwo(x,y):
        return distribution(x,y)*(fes(x,y)+(1.0/beta)*np.log(distribution(x,y)))

    term1 = -(1.0/beta)*np.log(integrate.dblquad(fone, a, b, lambda x: c, lambda x: d)[0])
    term2 = -(integrate.dblquad(ftwo, a, b, lambda x: c, lambda x: d)[0])

    return term1+term2

#Comprobacion
def compro_2D(xgrid, ygrid, a, b, c, d, fes, coef_matrix, weight, beta):
    def psofar(x,y):
        return biased_potential_2D(coef_matrix, x, y, a, b, c, d)
    def exp_numerator(x,y):
        return np.exp(-beta*(fes(x,y)+psofar(x,y)))
    def exp_denominator(x,y):
        return np.exp(-beta*(fes(x,y)))

    numerator = integrate.dblquad(exp_numerator, a, b, lambda x: c, lambda x: d)[0]
    denominator = integrate.dblquad(exp_denominator, a, b, lambda x: c, lambda x: d)[0]
    logi = (1.0/beta)*np.log(numerator/denominator)
    other = expected_value_2D(psofar, weight, a, b, c, d)

    return logi+other

#Variationally enhanced method 2D.
def vartwo(xgrid, ygrid, fes, array_number_coef, iterations, mu, beta, xepsilon, yepsilon, binos, ndynamics, dtt, steps, xo, yo, distribution=None, cores=None, data=None):

    start = time.time()
    a = xgrid[0]
    b = xgrid[-1]
    c = ygrid[0]
    d = ygrid[-1]
    at = a-xepsilon
    bt = b+xepsilon
    ct = c-yepsilon
    dt = d+yepsilon
    dim_cc = int(array_number_coef[0])
    dim_cs = int(array_number_coef[1])
    dim_sc = int(array_number_coef[2])
    dim_ss = int(array_number_coef[3])
    number_cc = dim_cc**2
    number_cs = dim_cs**2
    number_sc = dim_sc**2
    number_ss = dim_ss**2
    number_all = number_cc+number_cs+number_sc+number_ss
    cc_avg_notnormal = np.matrix(np.zeros((dim_cc, dim_cc)))
    cs_avg_notnormal = np.matrix(np.zeros((dim_cs, dim_cs)))
    sc_avg_notnormal = np.matrix(np.zeros((dim_sc, dim_sc)))
    ss_avg_notnormal = np.matrix(np.zeros((dim_ss, dim_ss)))
    cc_new = np.matrix(np.zeros((dim_cc, dim_cc)))
    cs_new = np.matrix(np.zeros((dim_cs, dim_cs)))
    sc_new = np.matrix(np.zeros((dim_sc, dim_sc)))
    ss_new = np.matrix(np.zeros((dim_ss, dim_ss)))
    last_x = xo
    last_y = yo
    toda_x = []
    toda_y = []

    if distribution==None:
        def target_distribution(x,y):
            return 1.0/((b-a)*(d-c))
    else:
        target_distribution=distribution


    meta = omega_min(fes, target_distribution, a, b, c, d, beta)
    print(meta)
    for i in range(iterations):
        cc_avg_notnormal = cc_avg_notnormal + cc_new
        cs_avg_notnormal = cs_avg_notnormal + cs_new
        sc_avg_notnormal = sc_avg_notnormal + sc_new
        ss_avg_notnormal = ss_avg_notnormal + ss_new
        cc_avg = (1.0/(i+1))*cc_avg_notnormal
        cs_avg = (1.0/(i+1))*cs_avg_notnormal
        sc_avg = (1.0/(i+1))*sc_avg_notnormal
        ss_avg = (1.0/(i+1))*ss_avg_notnormal
        alpha_avg_matrix = [cc_avg, cs_avg, sc_avg, ss_avg]

        def biased(x,y):
            value = fes(x,y)+biased_potential_2D(alpha_avg_matrix,x,y,at,bt,ct,dt)
            return value
        trayectos = emtwo_para(at, bt, ct, dt, biased, beta, binos, ndynamics, dtt, steps, 0, 1, last_x, last_y, 1,1)
        mini_trayec = trayectos[0]
        seguir = trayectos[1]
        seguir_x, seguir_y = seguir[0,1:-1], seguir[1,1:-1]
        toda_x, toda_y = np.concatenate((toda_x, seguir_x)), np.concatenate((toda_y, seguir_y))
        a_loc, b_loc = min(mini_trayec[0,1:-1]), max(mini_trayec[0,1:-1])
        c_loc, d_loc = min(mini_trayec[1,1:-1]), max(mini_trayec[1,1:-1])
        last_x, last_y = mini_trayec[0,-1], mini_trayec[1,-1]

        derivatives = der_omega_para_2D(beta, fes, alpha_avg_matrix, target_distribution, a_loc, b_loc, c_loc, d_loc, at, bt, ct, dt, mini_trayec,cores)
        gradient, hessian = derivatives[0], derivatives[1]
        cc_new = cc_new - mu*(gradient[0]+hessian[0]*(cc_new-cc_avg))
        cs_new = cs_new - mu*(gradient[1]+hessian[1]*(cs_new-cs_avg))
        sc_new = sc_new - mu*(gradient[2]+hessian[2]*(sc_new-sc_avg))
        ss_new = ss_new - mu*(gradient[3]+hessian[3]*(ss_new-ss_avg))

        please = compro_2D(xgrid, ygrid, a, b, c, d, fes, alpha_avg_matrix, target_distribution, beta)
        print(a_loc, b_loc, c_loc, d_loc, please, i)

    def zv(x,y):
        value = np.exp(-beta*(fes(x,y)+biased_potential_2D(alpha_avg_matrix, x, y, a-xepsilon, b+xepsilon, c-yepsilon, d+yepsilon)))
        return value

    tomono = np.size(toda_x)
    toda_tray = np.zeros((2,tomono+1))
    toda_tray[0,-1] = binos
    for ra in range(tomono):
        toda_tray[0,ra] = toda_x[ra]
        toda_tray[1,ra] = toda_y[ra]
    z_v = integrate.dblquad(zv, a-xepsilon, b+xepsilon, lambda x: c-yepsilon, lambda x: d+yepsilon)[0]
    constant = (1.0/beta)*np.log(z_v)
    fes_sample = np.zeros([len(xgrid), len(ygrid)])
    for i in range(len(xgrid)):
        for j in range(len(ygrid)):
            contri_pot = -biased_potential_2D(alpha_avg_matrix, xgrid[i], ygrid[j], a-xepsilon, b+xepsilon, c-yepsilon, d+yepsilon)
            fes_sample[j][i] = contri_pot-(1.0/beta)*np.log(target_distribution(xgrid[i], ygrid[j]))-constant
    end = time.time()
    duration = float(end-start)/60

    if data==None:
        np.save('vartwo_results', fes_sample)
        np.save('vartwo_dynamic', toda_tray)
    else:
        return fes_sample, duration

##################################################
####################FLOODING######################
##################################################

#Gives locacion of point in 1D bin space.
def pointloc1(xin, dx, point, limit):

    binxp = int(float(point-xin)/dx)

    if binxp<=limit:
        binx = binxp
    else:
        binx = limit

    return binx

#Reweight function.
def reone(xgrid, fes, delta_fes, beta, binos, ndynamics, dt, steps, xo, namae):

    mu = 0
    std = 1
    a = xgrid[0]
    b = xgrid[-1]
    dx = xgrid[1]-xgrid[0]
    size_space = len(xgrid)
    def total_pot(x):
        value = delta_fes(x)+fes(x)
        return value
    trajecto = emone_para(a, b, total_pot, beta, binos, ndynamics+1, dt, steps, mu, std, xo, 1, 1)
    rew_simulation = trajecto[0]
    salvado = trajecto[1]
    np.save(namae,salvado)
    ndist, x = np.histogram(rew_simulation, bins=int(binos))
    ndist_normal = (1.0/np.sum(ndist))*ndist
    dsize = np.size(x)-1
    dx = x[1]-x[0]
    xeo = x[0]
    expos = np.zeros(dsize)
    contador = np.zeros(dsize)
    expos2 = np.zeros(dsize)
    expos3 = np.zeros(dsize)
    for i in range(steps):
        xnow = rew_simulation[i]
        xwhere = pointloc1(xeo, dx, xnow, dsize-1)
        exp = delta_fes(xnow)
        exp2 = (delta_fes(xnow))**2
        expos[xwhere] = expos[xwhere]+exp
        expos2[xwhere] = expos2[xwhere]+exp2
        contador[xwhere] = contador[xwhere]+1
    for k in range(dsize):
        if contador[k]==0:
            continue
        else:
            expos[k] = expos[k]/contador[k]
            expos2[k] = expos2[k]/contador[k]
    for l in range(dsize):
        expos3[l] = np.exp(beta*expos[l]+((beta**2)/2.0)*(expos2[l]-(expos[l])**2))
    expos3 = (1.0/np.sum(expos3))*expos3
    z = (ndist_normal*expos3)
    f = np.zeros(dsize)
    for t in range(dsize):
        if z[t]!=0:
            f[t] = -(1.0/beta)*np.log(z[t])
        else:
            if t==0:
                f[t] = f[t+1]
            else:
                try:
                    f[t] = (f[t-1]+f[t+1])/2.0
                except IndexError:
                    f[t] = f[t-1]

    return f

#Gives the ducts and corresponding coefficients "lambda" for a given FES from a to b.
def flood_coefs(fes, a, b, minima):

    ducts = minima
    number_ducts = np.size(ducts)
    ducts_and_coefficients = np.zeros((number_ducts, 2))

    for i in range(0, number_ducts):
        n_duct = ducts[i]
        h = 1e-7
        coefficient = (fes(n_duct+h)-2*fes(n_duct)+fes(n_duct-h))/(h*h)
        ducts_and_coefficients[i,0] = ducts[i]
        ducts_and_coefficients[i,1] = (coefficient/2)

    return ducts_and_coefficients

#Constant flooding method in 1D.
def fldone(xgrid, fes, minima, flood_strengths, floods_coefficients, beta, binos, ndynamics, dt, steps, xo, data=None):

    '''|Takes array corresponding to domain, parameterized FES and list containing flooding strengths for
       |all minima from left to right.
       |
       |Returns array (same dim as domain) containing FES reconstruction values. By default saves results in file.'''

    start = time.time()
    a = xgrid[0]
    b = xgrid[-1]
    dx = xgrid[1]-xgrid[0]
    numero_ducts = np.size(flood_strengths)
    size_space = np.size(xgrid)
    fes_sample = np.zeros(size_space)

    def flooding(x):
        flood_potential = 0
        for i in range(0, numero_ducts):
            centro = floods_coefficients[i,0]
            lambdo = floods_coefficients[i,1]
            flood_potential = flood_potential + (flood_strengths[i])*np.exp(-(float(lambdo)/flood_strengths[i])*((x-centro)**2))
        return flood_potential

    fes_reweight = reone(xgrid, fes, flooding, beta, binos, ndynamics, dt, steps, xo, "fldone_dynamic")
    end = time.time()
    duration = float(end-start)/60

    tamano = np.size(fes_reweight)
    fesgrido = np.zeros(tamano+3)
    fesgrido[-3] = a
    fesgrido[-2] = b
    fesgrido[-1] = duration
    for k in range(tamano):
        fesgrido[k] = fes_reweight[k]

    if data==None:
        np.save("fldone_results", fesgrido)
    else:
        return fes_sample, duration

#Gives location of point in bin.
def pointloc(xin, yin, dx, dy, point, limit):

    binxp = int(float(point[0]-xin)/dx)
    binyp = int(float(point[1]-yin)/dy)

    if binxp<=limit:
        binx = binxp
    else:
        binx = limit

    if binyp<=limit:
        biny = binyp
    else:
        biny = limit

    return binx, biny

#2D reweight function
def retwo(xgrid, ygrid, fes, delta_fes, beta, binos, ndynamics, dt, steps, xo, yo, namae):

    mu = 0
    std = 1
    a = xgrid[0]
    b = xgrid[-1]
    c = ygrid[0]
    d = ygrid[-1]
    ds = xgrid[1]-xgrid[0]
    xsize = np.size(xgrid)
    def total_pot(x,y):
        value = delta_fes(x,y) + fes(x,y)
        return value
    trajecto = emtwo_para(a, b, c, d, total_pot, beta, binos, ndynamics, dt, steps, mu, std, xo, yo, 1, 1)
    rew_simulation = trajecto[0]
    salvado = trajecto[1]
    np.save(namae, salvado)
    ex = rew_simulation[0,:]
    ey = rew_simulation[1,:]
    ndist, x, y = np.histogram2d(ex,ey,bins=binos)
    ndist_normal = (1.0/np.sum(ndist))*ndist
    dsize = np.size(x)-1
    dx = x[1]-x[0]
    dy = y[1]-y[0]
    xeo = x[0]
    yeo = y[0]
    expos = np.zeros((dsize,dsize))
    expos2 = np.zeros((dsize,dsize))
    expos3 = np.zeros((dsize,dsize))
    contador = np.zeros((dsize,dsize))
    for i in range(steps):
        xnow = ex[i]
        ynow = ey[i]
        xwhere, ywhere = pointloc(xeo, yeo, dx, dy, [xnow,ynow], dsize-1)
        exp = delta_fes(xnow,ynow)
        exp2 = exp**2
        expos[xwhere,ywhere] = expos[xwhere,ywhere]+exp
        expos2[xwhere,ywhere] = expos2[xwhere,ywhere]+exp
        contador[xwhere,ywhere] = contador[xwhere,ywhere]+1
    for k in range(dsize):
        for l in range(dsize):
            if contador[k,l]==0:
                continue
            else:
                expos[k,l] = expos[k,l]/contador[k,l]
                expos2[k,l] = expos2[k,l]/contador[k,l]
    for m in range(dsize):
        for n in range(dsize):
            expos3[m,n] = np.exp(beta*expos[m,n]+((beta**2)/2.0)*(expos2[m,n]+expos[m,n]**2))
    expos3 = (1.0/np.sum(expos3))*expos3
    z = (ndist_normal*expos3)
    f = np.zeros(np.shape(z))
    for i in range(dsize):
        for j in range(dsize):
            if z[i,j]!=0:
                f[i,j] = -(1.0/beta)*np.log(z[i,j])
            else:
                normalization = np.zeros((3,3))
                contador = 0
                for l in range(-1, 2):
                    for m in range(-1, 2):
                        if i+l<0 or j+m<0:
                            normalization[l+1, m+1] = 0
                        else:
                            try:
                                normalization[l+1, m+1] = f[i+l, j+m]
                                contador = contador+1
                            except IndexError:
                                normalization[l+1, m+1] = 0
                                continue
                f[i,j] = np.sum(normalization)/contador
    return f.T

#Constant flooding method for 2D.
def fldtwo(xgrid, ygrid, fes, minima, flood_strengths, fld_coef, beta, binos, ndynamics, dt, steps, xo, yo, data=None):


    '''
    |Takes arrays of domain (R2 rectangle), parameterized FES, array with corresponding flooding strenghts
    |for all minima (from top to bottom, and left to right, in the grid).
    |
    |Returns matrix corresponding to reconstructed FES. By default saves results in text file.
    '''

    start = time.time()
    a = xgrid[0]
    b = xgrid[-1]
    c = ygrid[0]
    d = ygrid[-1]
    ds = xgrid[1]-xgrid[0]
    xsize = len(xgrid)
    ysize = len(ygrid)

    def delta_v(x,y):
        centro_x = fld_coef[0]
        centro_y = fld_coef[1]
        lambdo_x = fld_coef[2]
        lambdo_y = fld_coef[3]
        value = (flood_strengths)*np.exp(-(1.0/flood_strengths)*((lambdo_x*(x-centro_x)**2)+(lambdo_y*(y-centro_y)**2)))
        return value

    fes_reweight = retwo(xgrid, ygrid, fes, delta_v, beta, binos, ndynamics, dt, steps, xo, yo, "fldtwo_dynamic")
    end = time.time()
    duration = float(end-start)/60

    if data==None:
        np.save('fldtwo_results', fes_reweight)
    else:
        return fes_reweight, duration


############################################
#################GAUSSIAN###################
############################################

def gsone(x_grid, fes, sigma_o, option, beta, binos, ndynamics, dt, steps, xo, data=None):

    '''
    |Takes array corresponding to domain, parameterized FES, sigma_o and option to boost.
    |
    |Returns array (same dim. as domain) with FES reconstruction values. By default saves results in file.
    '''

    start = time.time()
    a = x_grid[0]
    b = x_grid[-1]
    dx = x_grid[1]-x_grid[0]
    size_space = np.size(x_grid)
    fes_almost = np.load('./emone_results.npy')
    xgrid_sample = fes_almost[0:-1]
    fes_sample = [fes(x) for x in xgrid_sample]
    avg_pot = np.mean(fes_sample)
    sigma_pot = np.std(fes_sample)
    max_pot = np.amax(fes_sample)
    min_pot = np.amin(fes_sample)
    fes_reweight = np.zeros(size_space)

    if option==1:
        ko_prime = (float(sigma_o)/sigma_pot)*(float(max_pot-min_pot)/(max_pot-avg_pot))
        ko = min(1.0, ko_prime)
        energy = max_pot
    else:
        ko_biprime = (1-float(sigma_o)/sigma_pot)*(float(max_pot-min_pot)/(avg_pot-min_pot))
        if 0<ko_biprime<=1:
            ko = ko_biprime
        else:
            ko = 1.0
        energy = min_pot+float(max_pot-min_pot)/ko

    k = ko*(1.0/(max_pot-min_pot))
    def delta_v(x):
        if fes(x)>=energy:
            return 0
        else:
            increment = float(k*((energy-fes(x))**2))/2

            return increment

    fes_reweight = reone(x_grid, fes, delta_v, beta, binos, ndynamics, dt, steps, xo, "gsone_dynamic")
    end = time.time()
    duration = float(end-start)/60

    tamano = np.size(fes_reweight)
    fesgrido = np.zeros(tamano+3)
    fesgrido[-3] = a
    fesgrido[-2] = b
    fesgrido[-1] = duration
    for k in range(tamano):
        fesgrido[k] = fes_reweight[k]

    if data==None:
        np.save("gsone_results", fesgrido)
    else:
        return fes_sample, duration


def gstwo(xgrid, ygrid, fes, sigma_o, option, beta, binos, ndynamics, dt, steps, mu, std, xo, yo, data=None):

    '''
    |Takes grids corresponding to domain (rectangle in R2), parameterized FES, sigma0 and the boost option.
    |
    |By default returns file with data corresponding to FES reconstruction, unless last argument is 1 then returns the data without saving it.
    '''

    start = time.time()
    a = xgrid[0]
    b = xgrid[-1]
    c = ygrid[0]
    d = ygrid[-1]
    ds = xgrid[1]-xgrid[0]
    size_x = np.size(xgrid)

    fes_almost = np.load('./emtwo_results.npy')
    xgrid_sample = fes_almost[0,0:-1]
    ygrid_sample = fes_almost[1,0:-1]
    fes_sample = np.zeros(np.size(xgrid_sample))
    for i in range(np.size(xgrid_sample)):
        xo, yo = xgrid_sample[i], ygrid_sample[i]
        fes_sample[i] = fes(xo, yo)
    avg_pot = np.mean(fes_sample)
    sigma_pot = np.std(fes_sample)
    max_pot = np.amax(fes_sample)
    min_pot = np.amin(fes_sample)

    if option==1:
        ko_prime = (float(sigma_o)/sigma_pot)*(float(max_pot-min_pot)/(max_pot-avg_pot))
        ko = min(1.0, ko_prime)
        energy = max_pot
    else:
        ko_biprime = (1-float(sigma_o)/sigma_pot)*(float(max_pot-min_pot)/(avg_pot-min_pot))
        if 0<ko_biprime<=1:
            ko = ko_biprime
        else:
            ko = 1.0
        energy = min_pot+float(max_pot-min_pot)/ko

    k = ko*(1.0/(max_pot-min_pot))
    def delta_v(x,y):
        if fes(x,y)>=energy:
            increment = 0
        else:
            increment = float(k*((energy-fes(x,y))**2))/2
        return increment

    fes_reweight = retwo(xgrid, ygrid, fes, delta_v, beta, binos, ndynamics, dt, steps, xo, yo, "gstwo_dynamic")
    end = time.time()
    duration = float(end-start)/60

    if data==None:
        np.save('gstwo_results', fes_reweight)
    else:
        return fes_reweight, duration

#####################################
############COMMITORS################
#####################################

#Gives location of point in array.
def point_array(array_in, array_ds, point):

    location = int(round(float(point-array_in)/array_ds))

    return location

#To obtain the commitor from a to b, given the transition matrix, linear equation method.
def commitor_alg(t, a, b):

    size_space = np.size(t[0, :])
    coef_matrix = np.zeros(np.shape(t))
    b_vector = np.zeros(size_space)

    coef_matrix[a-1][a-1] = 1
    coef_matrix[b-1][b-1] = 1
    b_vector[b-1] = 1

    for i in xrange(size_space):
        if i!=(a-1) and i!=(b-1):
            coef_matrix[i, :] = t[i, :]
            coef_matrix[i, i] = coef_matrix[i, i]-1

    comm_a_b = np.linalg.solve(coef_matrix, b_vector)

    return comm_a_b

#Power Method for obtaining dominant eigenvector.
def power_method(matrix, iterations):

    size_space = np.size(matrix[0,:])
    eigenv = [np.random.random() for i in range(0, size_space)]
    z = np.zeros(size_space)

    for i in xrange(iterations):
        z = np.dot(matrix, eigenv)
        eigenv = z*((np.linalg.norm(z))**(-1))

    return eigenv

#To obtain the commitor from a to b, given the transition matrix, eigenvector method.
def commitor_eig(t, a, b):

    abs_m = np.zeros(np.shape(t))
    size_space = np.size(t[0,:])
    comm_a_b = np.zeros(size_space)

    for i in xrange(size_space):
        for j in xrange(size_space):
            if i==(a-1) or i==(b-1):
                if j==i:
                    abs_m[i][j] = 1
                else:
                    pass
            else:
                abs_m[i][j] = t[i][j]

    eigenvec = power_method(abs_m, 10000)

    for k in xrange(0, size_space):
        comm_a_b[k] = float(eigenvec[k]-eigenvec[a-1])/(eigenvec[b-1]-eigenvec[a-1])

    return comm_a_b

#To obtain probability of going from state i to state j, given potential grid u.
def proba_change_state_1D(u, i, j, beta):

    normalization = np.zeros(3)
    y_i = i
    y_j = j

    if abs(y_i-y_j)>1:
        proba_i_j = 0
    else:
        proba_not_normal = min(1, np.exp(-beta*(u[y_j]-u[y_i])))

        for m in range(-1, 2):
            if y_i+m<0:
                normalization[m+1] = 0
            else:
                try:
                    normalization[m+1] = min(1, np.exp(-beta*(u[y_i+m]-u[y_i])))
                except IndexError:
                    normalization[m+1] = 0
                    continue

        proba_i_j = float(proba_not_normal)/np.sum(normalization)

    return proba_i_j

#To obtain transition matrix given the potential grid, this gives the full matrix.
def trans_matrix_1D(u, beta):

    size_space = np.size(u)
    t_matrix = np.zeros((size_space, size_space))
    for j in xrange(0,2):
        t_matrix[0][j] = proba_change_state_1D(u, 0, j, beta)

    for j in xrange(-1,1):
        t_matrix[size_space-1][size_space-1+j] = proba_change_state_1D(u, size_space-1, size_space-1+j, beta)

    for i in xrange(1, size_space-1):
        for k in xrange(-1,2):
            t_matrix[i][i+k] = proba_change_state_1D(u, i, i+k, beta)

    return t_matrix

#Gives commitor probabilities given two states in coordinate form.
def commone(xgrid, fes, beta, a, b, method=None, data=None):
    '''
    |Takes grid of domain, parameterized FES/array of values in xgrid, thermodynamic beta, states a and b
    |in coordinate form (commitor from a to b), method by default is algebraic.
    |
    |By default writes file with data corresponding to commitor probabilities.
    '''

    start = time.time()

    try:
        temp = len(fes)
        delta = xgrid[1]-xgrid[0]
        astate = point_array(xgrid[0], delta, a)
        bstate = point_array(xgrid[0], delta, b)
        fes_trans = trans_matrix_1D(fes, beta)

        if method==None:
            fes_comm = commitor_alg(fes_trans, astate, bstate)
        else:
            fes_comm = commitor_eig(fes_trans, astate, bstate)

    except TypeError:
        delta = xgrid[1]-xgrid[0]
        astate = point_array(xgrid[0], delta, a)
        bstate = point_array(xgrid[0], delta, b)
        fes_grid = [fes(i) for i in xgrid]
        fes_trans = trans_matrix_1D(fes_grid, beta)

        if method==None:
            fes_comm = commitor_alg(fes_trans, astate, bstate)
        else:
            fes_comm = commitor_eig(fes_trans, astate, bstate)

    end = time.time()
    duration = float(end-start)/60

    tamano = np.size(fes_comm)
    fesgrido = np.zeros(tamano+3)
    fesgrido[-3] = xgrid[0]
    fesgrido[-2] = xgrid[-1]
    fesgrido[-1] = duration
    for k in range(tamano):
        fesgrido[k] = fes_comm[k]

    if data==None:
        np.save("commone_results", fesgrido)
    else:
        return fesgrido[0:-3], duration

#Gives state number corresponding to coordinate.
def point_array_2D(a, b, c, d, delta, pointx, pointy):

    squares = int(round(float(b+delta-a)/delta))
    y_contribution = int(round(float(pointy-c)/delta))
    x_contribution = int(round(float(pointx-a)/delta))+1

    return y_contribution*squares+x_contribution

#To obtain probability of going from state i to state j, given potential grid u.
def trans_matrix_2D(u, beta):

    size_space = np.size(u)
    t_matrix = np.zeros((size_space, size_space))
    size_grid = np.size(u[0,:])

    for i in xrange(size_space):
        normalization = np.zeros((3, 3))
        x_i = i/size_grid
        y_i = i%size_grid

        for l in range(-1, 2):
            for m in range(-1, 2):
                if x_i+l<0 or y_i+m<0:
                    normalization[l+1, m+1] = 0
                else:
                    try:
                        normalization[l+1, m+1] = min(1, np.exp(-beta*(u[x_i+l, y_i+m]-u[x_i, y_i])))
                    except IndexError:
                        normalization[l+1, m+1] = 0
                        continue

        normi = 1.0/np.sum(normalization)
        for l in range(-1, 2):
            for m in range(-1,2):
                flat_state = size_grid*(x_i+l)+y_i+m
                if flat_state<0:
                    continue
                else:
                    try:
                        t_matrix[i, flat_state] = normi*normalization[l+1, m+1]
                    except IndexError:
                        continue

    return t_matrix

#Gives commitor probabilities in two dimensions.
def commtwo(xgrid, ygrid, fes, beta, a, b, data=None, method=None):
    '''
    |Takes grids of domain, parameterized/matrix FES, thermodynamic beta, states a and b in tuple form
    |[a_x, a_y].
    |
    |By default writes file with data corresponding to commitor probabilities.
    '''

    start = time.time()
    size_grid = np.size(xgrid)
    delta = xgrid[1]-xgrid[0]

    try:
        temp = np.size(fes)
        fes_grid = np.zeros([size_grid, size_grid])
        for i in range(size_grid):
            for j in range(size_grid):
                fes_grid[j][i] = fes[j][i]
    except TypeError:
        fes_grid = np.zeros([size_grid, size_grid])
        for i in range(size_grid):
            for j in range(size_grid):
                value = fes(xgrid[i], ygrid[j])
                fes_grid[j][i] = value

    astate = point_array_2D(xgrid[0], xgrid[-1], ygrid[0], ygrid[-1], delta, a[0], a[1])
    bstate = point_array_2D(xgrid[0], xgrid[-1], ygrid[0], ygrid[-1], delta, b[0], b[1])
    fes_trans = trans_matrix_2D(fes_grid, beta)

    if method==None:
        fes_comm = commitor_alg(fes_trans, astate, bstate)
    else:
        fes_comm = commitor_eig(fes_trans, astate, bstate)

    fes_comm_matrix = np.reshape(fes_comm, (size_grid, size_grid))

    end = time.time()
    duration = float(end-start)/60

    if data==None:
        np.save('commtwo_results', fes_comm_matrix)
    else:
        return fes_comm_matrix, duration


######################################################
#####################MISC#############################
######################################################

#This function gives an array with original fes values.
def orone(xgrid, fes, data=None):
    '''
    |Takes grid of domain and parameterized FES.
    |
    |Returns array of fes values.
    '''

    a = xgrid[0]
    b = xgrid[-1]
    tamano = np.size(xgrid)
    fesgrid = np.zeros(tamano+3)
    for k in range(tamano):
        ex = xgrid[k]
        fesgrid[k] = fes(ex)

    fesgrid[-3] = a
    fesgrid[-2] = b
    if data==None:
        np.save("orone_results", fesgrid)
    else:
        return fesgrid

#Writes file with dimensions of problem.
def dimension_file(xgrid, ygrid, fes):
    a = xgrid[0]
    b = xgrid[-1]
    c = ygrid[0]
    d = ygrid[-1]
    ds = xgrid[1]-xgrid[0]
    tamano = np.size(xgrid)
    dimensions = [a,b,c,d,tamano, ds]
    np.save('domains',dimensions)

#Gives a matrix of two dimensional fes.
def ortwo(xgrid, ygrid, fes, data=None):
    '''
    |Takes grids of domain and parameterized fes.
    |
    |Returns matrix with fes values in two dimensions.
    '''

    size = np.size(xgrid)
    fes_matrix = np.zeros((size, size))
    for i in range(size):
        for j in range(size):
            xo, yo = xgrid[i], ygrid[j]
            fes_matrix[j,i] = fes(xo, yo)

    if data==None:
        np.save('ortwo_results',fes_matrix)
    else:
        return fes_matrix

#This function gives an array with original distribution fes values.
def disone(xgrid, bins, fes, beta, data=None):
    '''
    |Takes grid of domain and parameterized FES.
    |
    |Returns array of fes values.
    '''

    a = xgrid[0]
    b = xgrid[-1]
    dx = float(b-a)/bins
    fesgrid = np.zeros(bins+2)
    fesgrid[-1] = b
    fesgrid[-2] = a
    def distri(x):
        value = np.exp(-beta*fes(x))
        return value

    normalization = 1.0/integrate.quad(distri, a, b)[0]
    for i in range(bins):
        fesgrid[i] = normalization*integrate.quad(distri, a+i*dx, a+(i+1)*dx)[0]

    if data==None:
        np.save("disone_results", fesgrid)
    else:
        return fesgrid

#Gives a matrix of two dimensional fes distribution.
def distwo(xgrid, ygrid, bins, fes, beta, data=None):
    '''
    |Takes grids of domain and parameterized fes.
    |
    |Returns matrix with fes values in two dimensions.
    '''

    a = xgrid[0]
    b = xgrid[-1]
    c = ygrid[0]
    d = ygrid[-1]
    ds = float(b-a)/bins
    fesgrid = np.zeros((bins, bins))
    def distri(x,y):
        value = np.exp(-beta*fes(x,y))
        return value

    normalization = 1.0/integrate.dblquad(distri, a, b, lambda x: c, lambda x: d)[0]
    for i in range(bins):
        for j in range(bins):
            fesgrid[j,i] = normalization*integrate.dblquad(distri, a+i*ds, a+(i+1)*ds, lambda x: (c+j*ds), lambda x: (c+(j+1)*ds))[0]

    coco = np.sum(fesgrid)
    if data==None:
        np.save('distwo_results', fesgrid)
    else:
        return fes_matrix

#Gives number of steps needed to escape from a local minima in 1D.
def local_escape(dynamic, limits):
    contador = 0
    check = 0
    a = limits[0]
    b = limits[1]
    while contador<1000:
        maybe = dynamic[check]
        check = check+1
        if a<=maybe<=b:
            continue
        else:
            contador = contador+1
    return check

#Gives number of steps needed to escape from a local minima in 2D.
def local_escape_2D(dynamic,number):

    referencia = np.load('metaestables.npy')
    dominio = np.load('domains.npy')
    a = dominio[0]
    b = dominio[1]
    c = dominio[2]
    ds = dominio[5]
    contador = 0
    check = 0
    while contador<10000:
        point = dynamic[:,check+1]
        point_loc = point_location_2D(a, c, ds, point)
        mx, my = point_loc[0], point_loc[1]
        value = referencia[my,mx]
        check = check+1
        if value==number:
            continue
        else:
            contador = contador+1
    return check

#Sign structure function.
def sign_structure(eigenvecs, theta, x):
    tamano = len(eigenvecs)
    signs = np.zeros(tamano+1)
    for k in range(tamano):
        value = eigenvecs[k][x]
        if value > theta:
            signs[k] = 1
            continue
        elif value < -theta:
            continue
        else:
            signs[-1] = np.random.uniform()
            break
    return signs

#Gives dominant eigenvectors of operator to infinity.
def dominance(xgrid, fes_grid, beta):
        operator = trans_matrix_1D(fes_grid, beta)
        dominant_eigen = []
        for i in range(10):#20
            operator = np.dot(operator,operator)
        eigen = np.linalg.eig(operator)
        eigenvalues = eigen[0]
        eigenvectors = eigen[1]
        print(eigenvalues[0:5])
        for j in range(100):
            check = eigenvalues[j]
            if check > 0.9:
                dominant_eigen.append(np.real(eigenvectors[:,j]))
            else:
                break

        tomono = len(dominant_eigen)
        for t in range(tomono):
            maxo = np.max(abs(dominant_eigen[t]))
            dominant_eigen[t] = dominant_eigen[t]/maxo
            if dominant_eigen[t][0] < 0:
                dominant_eigen[t] = -dominant_eigen[t]
            else:
                continue
        return dominant_eigen

#Given initial point and percentage of error, it returns set of quasiconstant values with respect to initial point.
def change_finder(array, start, percentage):
    actual = array[start]
    tamano = len(array)-1
    top = abs(percentage*actual)
    a = int
    b = int
    check_left = False
    check_right = False
    count_left = start
    count_right = start
    while check_left==False:
        maybe_left = array[count_left]
        if count_left==0 or abs(maybe_left-actual)>top:
            check_left=True
        else:
            count_left = count_left-1
            continue
    while check_right==False:
        maybe_right = array[count_right]
        if count_right==tamano or abs(maybe_right-actual)>top:
            check_right=True
        else:
            count_right = count_right + 1
            continue
    a = count_left
    b = count_right

    return [a,b]

#Gives number of metaestable sets and corresponding domains.
def metaestable(xgrid, fes, beta, porcentaje, minima, data=None):

    try:
        tamanot = np.size(fes)
        fes_grid = np.zeros(tamanot)
        for k in range(tamanot):
            fes_grid[k] = fes[k]
    except TypeError:
        fes_grid = [fes(u) for u in xgrid]

    if data==None:
        tamano = np.size(xgrid)
        ds = xgrid[1]-xgrid[0]
        a = xgrid[0]
        dominant_eig = dominance(xgrid, fes_grid, beta)
        eignum = int(len(dominant_eig))
        min_loc = [point_location(a, ds, point) for point in minima]
        min_num = len(min_loc)
        loc_all = np.zeros((eignum,min_num,2))
        for j in range(eignum):
            for t in range(min_num):
                loc = change_finder(dominant_eig[j], min_loc[t], porcentaje)
                loc_a = loc[0]
                loc_b = loc[1]
                loc_all[j,t,0] = loc_a
                loc_all[j,t,1] = loc_b
        locations = np.zeros((2,min_num))
        for r in range(min_num):
            locations[0,r] = xgrid[int(np.max(loc_all[:,r,0]))]
            locations[1,r] = xgrid[int(np.min(loc_all[:,r,1]))]
        print(locations)
        plt.figure()
        for u in range(min_num):
            plt.axvline(x=locations[0,u], linewidth=1.5, color = 'k', linestyle=':')
            plt.axvline(x=locations[1,u], linewidth=1.5, color = 'k', linestyle=':')
        for t in range(eignum):
            plt.plot(xgrid, dominant_eig[t], linewidth=1.5)
        plt.grid()
        plt.show()
    elif data=='help':
        tamano = np.size(xgrid)
        ds = xgrid[1]-xgrid[0]
        a = xgrid[0]
        dominant_eig = dominance(xgrid, fes_grid, beta)
        eignum = int(len(dominant_eig))
        min_loc = [point_location(a, ds, point) for point in minima]
        min_num = len(min_loc)
        loc_all = np.zeros((eignum,min_num,2))
        for j in range(eignum):
            for t in range(min_num):
                loc = change_finder(dominant_eig[j], min_loc[t], porcentaje)
                loc_a = loc[0]
                loc_b = loc[1]
                loc_all[j,t,0] = loc_a
                loc_all[j,t,1] = loc_b
        locations = np.zeros((2,min_num))
        for r in range(min_num):
            locations[0,r] = xgrid[int(np.max(loc_all[:,r,0]))]
            locations[1,r] = xgrid[int(np.min(loc_all[:,r,1]))]
        return locations
    else:
        tamano = np.size(xgrid)
        ds = xgrid[1]-xgrid[0]
        a = xgrid[0]
        dominant_eig = dominance(xgrid, fes_grid, beta)
        eignum = int(len(dominant_eig))
        min_loc = [point_location(a, ds, point) for point in minima]
        min_num = len(min_loc)
        loc_all = np.zeros((eignum,min_num,2))
        for j in range(eignum):
            for t in range(min_num):
                loc = change_finder(dominant_eig[j], min_loc[t], porcentaje)
                loc_a = loc[0]
                loc_b = loc[1]
                loc_all[j,t,0] = loc_a
                loc_all[j,t,1] = loc_b
        locations = np.zeros((2,min_num))
        for r in range(min_num):
            locations[0,r] = xgrid[int(np.max(loc_all[:,r,0]))]
            locations[1,r] = xgrid[int(np.min(loc_all[:,r,1]))]
        print(locations)
        plt.figure()
        for u in range(min_num):
            plt.axvline(x=locations[0,u], linewidth=1.5, color = 'k', linestyle=':')
            plt.axvline(x=locations[1,u], linewidth=1.5, color = 'k', linestyle=':')
        for t in range(eignum):
            plt.plot(xgrid, dominant_eig[t], linewidth=1.5)
        plt.grid()
        plt.xlabel("Coordenada Colectiva")
        plt.ylabel("Eigenvalores Dominantes")
        plt.title("Conjuntos metaestables")
        plt.savefig('graph.png')


#Gives dominant eigenvectors in two dimensions.
def dominance_2D(fes_grid, beta):
    operator = trans_matrix_2D(fes_grid, beta)
    dominant_eigen = []
    for i in range(7):
        operator = np.dot(operator,operator)
    eigen = np.linalg.eig(operator)
    eigenvalues = eigen[0]
    eigenvectors = eigen[1]
    print(eigenvalues[0:10])
    for j in range(100):
        check = eigenvalues[j]
        if check > 0.9:
            dominant_eigen.append(np.real(eigenvectors[:,j]))
        else:
            break
    tomono = len(dominant_eigen)
    for t in range(tomono):
        maxo = np.max(abs(dominant_eigen[t]))
        dominant_eigen[t] = dominant_eigen[t]/maxo
    return dominant_eigen


#Finds changes (duh).
def change_finder_2D(grid, start, percentage):
    actual_x = int(start[0])
    actual_y = int(start[1])
    actual_value = grid[actual_x, actual_y]
    tamano = np.size(grid[0,:])
    top = abs(actual_value*percentage)
    checador = np.zeros((tamano,tamano))
    for i in range(tamano):
        for j in range(tamano):
            value = grid[i,j]
            if abs(value-actual_value)<top:
                checador[i,j] = 1
            else:
                checador[i,j] = 0
    return checador

#Gives metaestable sets in two dimensions.
def metaestable_2D(xgrid, ygrid, fes, beta, porcentaje, minima, data=None):
    tamanot = int
    try:
        tamanot = np.size(fes[0,:])
        fes_grid = np.zeros((tamanot, tamanot))
        for k in range(tamanot):
            for g in range(tamanot):
                fes_grid[k,g] = fes[k,g]
    except TypeError:
        tamanot = np.size(xgrid)
        fes_grid = np.zeros((tamanot,tamanot))
        for k in range(tamanot):
            for g in range(tamanot):
                xo, yo = xgrid[k], ygrid[g]
                fes_grid[g,k] = fes(xo,yo)
    tamano = np.size(xgrid)
    ds = xgrid[1]-xgrid[0]
    a = xgrid[0]
    b = xgrid[-1]
    c = ygrid[0]
    d = ygrid[-1]
    dominant_eig = dominance_2D(fes_grid, beta)
    eignum = int(len(dominant_eig))
    min_num = len(minima)
    min_loc = np.zeros((2,min_num))
    for g in range(min_num):
        xo, yo = point_location_2D(a, c, ds, minima[g])
        min_loc[0,g] = yo
        min_loc[1,g] = xo

    loc_all = np.zeros((tamano,tamano,min_num,eignum))
    for j in range(eignum):
        eig_matrix = np.reshape(dominant_eig[j], (tamano, tamano))
        for k in range(min_num):
            xi, yi = min_loc[0,k], min_loc[1,k]
            loc = change_finder_2D(eig_matrix, [xi, yi], porcentaje)
            loc_all[:,:,k,j] = loc

    contours = np.zeros((tamano,tamano,min_num))
    for t in range(min_num):
        for ay in range(tamano):
            for by in range(tamano):
                checa = sum(loc_all[ay,by,t,:])
                if checa==eignum:
                    contours[ay,by,t] = 1
                else:
                    contours[ay,by,t] = 0

    eigos = np.zeros((tamano,tamano,eignum))
    for t in range(eignum):
        eigos[:,:,t] = np.reshape(dominant_eig[t], (tamano,tamano))
    dsize = np.size(contours[0,:])
    extent = [a, b, c, d]
    fes_uni = contours[:,:,0]
    for t in range(1,min_num):
        fes_uni = fes_uni+contours[:,:,t]
#    fig, axes = plt.subplots(nrows=3, ncols=2)
#    salidota = np.zeros(np.shape(fes_uni))
#    for g in range(min_num):
#        for a in range(tamano):
#            for b in range(tamano):
#                valot = contours[a,b,g]
#                if valot==1:
#                    salidota[a,b] = g+1
#                else:
#                    continue

    plt.figure()
    plt.contourf(fes_uni, dsize, cmap=plt.cm.jet, extent=extent)
    plt.grid()
    plt.savefig('co5')
    exit()

    axes[1, 1].contourf(fes_uni, dsize, cmap=plt.cm.jet, extent=extent)
    axes[1, 1].set_title('Conjuntos metaestables')
    axes[1, 1].grid()
    axes[0, 0].contourf(eigos[:,:,0], dsize, cmap=plt.cm.jet, extent=extent)
    axes[0, 0].set_title('Primer eigenvector')
    axes[0, 0].grid()
    axes[0, 1].contourf(eigos[:,:,1], dsize, cmap=plt.cm.jet, extent=extent)
    axes[0, 1].set_title('Segundo eigenvector')
    axes[0, 1].grid()
    axes[1, 0].contourf(eigos[:,:,2], dsize, cmap=plt.cm.jet, extent=extent)
    axes[1, 0].set_title('Tercer eigenvector')
    axes[1, 0].grid()
    fig.tight_layout()
    if data==None:
        plt.show()
    else:
        plt.savefig('graph.png')
        np.save('metaestables', salidota)


###########################################################################
###########################################################################
###########################################################################
###########################################################################
###########################################################################
###########################################################################

#Kind of reweight pero sobre la FES, no la distribucion.
def no_rew(xgrid, fes, binos,dinamica):

    a = xgrid[0]
    b = xgrid[-1]
    dx = xgrid[1]-xgrid[0]
    size_space = len(xgrid)
    fes_array = np.zeros(binos)
    steps = np.size(dinamica)
    contador = np.zeros(binos)
    for i in range(steps):
        xnow = dinamica[i]
        xwhere = point_location(a, dx, xnow)
        value = fes(xnow)
        fes_array[xwhere] = fes_array[xwhere]+value
        contador[xwhere] = contador[xwhere]+1
    for k in range(binos):
        if contador[k]==0:
            continue
        else:
            fes_array[k] = fes_array[k]/float(contador[k])
    f = np.zeros(binos)
    for t in range(binos):
        if fes_array[t]!=0:
            f[t] = fes_array[t]
        else:
            if t==0:
                f[t] = f[t+1]
            else:
                try:
                    f[t] = (f[t-1]+f[t+1])/2.0
                except IndexError:
                    f[t] = f[t-1]
    return f

#Metadinamica diferente.
def mdyone_new(x_grid, potential, steps, initial_position, delta_sigma, beta, dt, binos, alpha=None, height=None, data=None):

    '''|This function takes as entries the domain of your FES (as an array), the parameterized FES, number of steps of simulation, initial
       |position in configuration space, delta sigma, height of gaussian and delta x.
       |
       |By default returns file with data corresponding to FES reconstruction, unless last argument is 1 then returns the data without saving it'''

    start = time.time()

    if alpha==None:
        alfa = 0.5
    else:
        alfa = alpha

    if height==None:
        fms = 0
        for i in range(len(x_grid)):
            force = derivative_point(potential, x_grid[i])
            fms = fms + force**2
        fms = float(fms)/len(x_grid)
        gaussian_height = delta_sigma*alfa*(np.e**(-0.5))*np.sqrt(fms)
    else:
        gaussian_height = height

    size_space = np.size(x_grid)
    a = x_grid[0]
    b = x_grid[-1]
    ds = x_grid[2]-x_grid[1]
    sigma = np.sqrt(2.0/beta)
    noise = sigma*np.sqrt(dt)
    xo = initial_position
    fes_sample = np.zeros(size_space)
    fes_porfa = np.zeros(size_space)
    fes_sample = contribution_point(x_grid, fes_sample, xo, gaussian_height, delta_sigma)
    xo_ar = point_location(a, ds, xo)
    pot_dx = derivative_point(potential, xo) + devone_array(ds, fes_sample, xo_ar)
    pot = potential(xo)+fes_sample[xo_ar]
    dinamica = []

    for i in xrange(1, steps+1):
        dinamica.append(xo)
        r = np.random.normal(0,1)
        boundary = xo-(pot_dx*dt)+(noise*r)

        if a<=boundary<=b:
            xn = boundary
        else:
            xn = xo+(pot_dx*dt)-(noise*r)

        xn_ar = point_location(a, ds, xn)
        potn = potential(xn) + fes_sample[xn_ar]
        potn_dx = derivative_point(potential, xn) + devone_array(ds, fes_sample, xn_ar)
        transition_one = abs(xn-xo+dt*pot_dx)**2-abs(xo-xn+dt*potn_dx)**2
        transition_two = (4.0/beta)*dt
        transition_term = np.exp(transition_one/transition_two)
        density = np.exp(beta*(pot-potn))
        val = transition_term*density

        if val > np.random.uniform():
            pot, pot_dx = potn, potn_dx
            xo = xn
        else:
            continue

        fes_sample = contribution_point(x_grid, fes_sample, xo, gaussian_height, delta_sigma)

    end = time.time()
    duration = (1.0*(end-start))/60

    fes_porfa = no_rew(x_grid, potential, binos,dinamica)
    fes_samplot = np.zeros(size_space+3)
    for j in range(size_space):
        fes_samplot[j] = fes_porfa[j]
    fes_samplot[-3] = a
    fes_samplot[-2] = b
    fes_samplot[-1] = duration
    fes_samplet = np.zeros(size_space+3)
    for j in range(size_space):
        fes_samplet[j] = -fes_sample[j]
    fes_samplet[-3] = a
    fes_samplet[-2] = b
    fes_samplet[-1] = duration
    if data==None:
        np.save("mdyone_new_results", fes_samplot)
        np.save("mdyone_old_results", fes_samplet)
        np.save("mdyone_new_dynamic", dinamica)
    else:
        return fes_sample, duration


#Norew in 2D.
def no_rew_2D(xgrid, ygrid, fes, binos, dinamica):
    a = xgrid[0]
    b = xgrid[-1]
    c = ygrid[0]
    d = ygrid[-1]
    ds = xgrid[1]-xgrid[0]
    xsize = np.size(xgrid)
    ds_bin = (b-a)/(binos-1)
    contador = np.zeros((binos,binos))
    fes_array = np.zeros((binos,binos))
    ex = dinamica[0,:]
    ey = dinamica[1,:]
    steps = np.size(ex)
    for i in range(steps):
        xnow = ex[i]
        ynow = ey[i]
        xwhere, ywhere = point_location_2D(a, c, ds_bin, [xnow,ynow])
        value = fes(xnow,ynow)
        fes_array[xwhere,ywhere] = fes_array[xwhere,ywhere]+value
        contador[xwhere,ywhere] = contador[xwhere,ywhere]+1
    for k in range(binos):
        for l in range(binos):
            if contador[k,l]==0:
                continue
            else:
                fes_array[k,l] = float(fes_array[k,l])/contador[k,l]
    f = np.zeros(np.shape(fes_array))
    for i in range(binos):
        for j in range(binos):
            if fes_array[i,j]!=0:
                f[i,j] = fes_array[i,j]
            else:
                normalization = np.zeros((3,3))
                contador = 0
                for l in range(-1, 2):
                    for m in range(-1, 2):
                        if i+l<0 or j+m<0:
                            normalization[l+1, m+1] = 0
                        else:
                            try:
                                normalization[l+1, m+1] = f[i+l, j+m]
                                contador = contador+1
                            except IndexError:
                                normalization[l+1, m+1] = 0
                                continue
                f[i,j] = np.sum(normalization)/contador
    return f.T

#Gives you a sequence of steps, given the potential and initial position, n dimensions.
def mdytwo_new(xgrid, ygrid, fes, steps, initial_point, delta_sigma, beta, dt, binos, alpha=None, height=None, data=None):

    '''|This function takes as entries the domain of your FES (two arrays: x,y), the parameterized 2D-FES, number of steps of simulation, initial
       |position (tuple e.g. [2,3]), delta sigma, height of gaussian and deltas x,y.
       |
       |Returns a matrix containing values of FES reconstruction. By default saves data in text file, otherwise it just returns matrix.'''

    start = time.time()
    if alpha==None:
        alfa = 0.5
    else:
        alfa = alpha

    if height==None:
        api = xgrid[0]
        bpi = xgrid[-1]
        cpi = ygrid[0]
        dpi = ygrid[-1]
        xgrido = np.arange(api, bpi, 0.1)
        ygrido = np.arange(cpi, dpi, 0.1)
        fms = 0
        for i in range(len(xgrido)):
            for j in range(len(ygrido)):
                force_x, force_y = partial_point(fes, xgrido[i], ygrido[j])
                fms = fms + force_x**2 + force_y**2
        fms = float(fms)/(len(xgrido)*len(ygrido))
        gaussian_height = delta_sigma*alfa*(np.e**(-0.5))*np.sqrt(fms)
    else:
        gaussian_height = height

    size_space = np.size(xgrid)
    a = xgrid[0]
    b = xgrid[-1]
    c = ygrid[0]
    d = ygrid[-1]
    ds = xgrid[1]-xgrid[0]
    sigma = np.sqrt(2/beta)
    noise = sigma*np.sqrt(dt)
    xo, yo = initial_point[0], initial_point[1]
    fes_sample = np.zeros([size_space, size_space])
    impor = mixed_stuff(fes, fes_sample, xo, yo, a, c, ds)
    pot_dx, pot_dy = impor[1], impor[2]
    pot = impor[0]
    pos_x, pos_y = [], []

    for i in xrange(1, steps+1):
        print(steps-i)
        fes_sample = contribution_point_2D(xgrid, ygrid, fes_sample, [xo,yo], gaussian_height, delta_sigma)
        pos_x.append(xo)
        pos_y.append(yo)
        rx = np.random.normal(0,1)
        ry = np.random.normal(0,1)
        xbound = xo-(dt*pot_dx)+(noise*rx)
        ybound = yo-(dt*pot_dy)+(noise*ry)
        if a<=xbound<=b:
            xn = xbound
        else:
            xn = xo+(dt*pot_dx)-(noise*rx)

        if c<=ybound<=d:
            yn = ybound
        else:
            yn = yo+(dt*pot_dy)-(noise*ry)

        imporn = mixed_stuff(fes, fes_sample, xn, yn, a, c, ds)
        potn = imporn[0]
        potn_dx, potn_dy = imporn[1], imporn[2]
        transition_one = (xn-xo+dt*pot_dx)**2+(yn-yo+dt*pot_dy)**2-(xo-xn+dt*potn_dx)**2-(yo-yn+dt*potn_dy)**2
        transition_two = (4.0/beta)*dt
        transition_term = np.exp(transition_one/transition_two)
        density = np.exp(beta*(pot-potn))
        val = transition_term*density
        if val > np.random.uniform() or val>1:
            pot, pot_dx, pot_dy = potn, potn_dx, potn_dy
            xo, yo = xn, yn


    end = time.time()
    duration = (1.0*(end-start))/60

    dinamicota = np.zeros((2,steps))
    dinamicota[0,:] = pos_x
    dinamicota[1,:] = pos_y
    f_new = no_rew_2D(xgrid, ygrid, fes, binos, dinamicota)
    if data==None:
        np.save('mdytwo_new_results',f_new)
        np.save('mdytwo_old_results', -fes_sample)
        np.save('mdytwo_dynamic', dinamicota)
    else:
        return fes_sample, duration

#COLECTIVAS
################################################################
################################################################
################################################################
################################################################

#Encontrar raices.

#Proyeccion sobre x
def proy_x(xgrid,c,d,fes,beta):
    a = xgrid[0]
    b = xgrid[-1]
    tamano = np.size(xgrid)
    grido = np.zeros(tamano)
    def distri(x,y):
        value = np.exp(-beta*fes(x,y))
        return value
    normal = 1.0/integrate.dblquad(distri, a, b, lambda x: c, lambda x: d)[0]
    def distri_n(x,y):
        value = normal*np.exp(-beta*fes(x,y))
        return value
    for t in xrange(tamano):
        xo = xgrid[t]
        def distri_nl(y):
            value = distri_n(xo,y)
            return value
        value = integrate.quad(distri_nl, c, d)[0]
        grido[t] = value
    grido = (1.0/np.sum(grido))*grido
    grido = -(1.0/beta)*np.log(grido)
    return grido

#Proyeccion sobre y
def proy_y(ygrid,a,b,fes,beta):
    c = ygrid[0]
    d = ygrid[-1]
    tamano = np.size(ygrid)
    grido = np.zeros(tamano)
    def distri(x,y):
        value = np.exp(-beta*fes(x,y))
        return value
    normal = 1.0/integrate.dblquad(distri, a, b, lambda x: c, lambda x: d)[0]
    def distri_n(x,y):
        value = normal*np.exp(-beta*fes(x,y))
        return value
    for t in xrange(tamano):
        yo = ygrid[t]
        def distri_nl(x):
            value = distri_n(x,yo)
            return value
        value = integrate.quad(distri_nl, a, b)[0]
        grido[t] = value
    grido = -(1.0/beta)*np.log(grido)
    return grido

#Given FES and 1D metadynamic bias computes total force.
def mixed_colec(fes, fes_sample, xo, yo, a, ds):
    x_ar = point_location(a, ds, xo)
    f = fes(xo,yo)
    df = partial_point(fes, xo, yo)
    g = fes_sample[x_ar]
    dg = devone_array(ds, fes_sample, x_ar)
    t = f+g
    dt_0 = df[0]+dg
    dt_1 = df[1]
    return t, dt_0, dt_1

#Given trajectory in 2D it returns FES in colective variable.
def col_fes(trajecto, beta, binos, data=None):
    rew_simulation = trajecto[0,:]
    ndist, x = np.histogram(rew_simulation, bins=int(binos))
    ndist_normal = (1.0/np.sum(ndist))*ndist
    dsize = np.size(x)-1
    dx = x[1]-x[0]
    xeo = x[0]
    a = x[0]
    b = x[-1]
    partition = np.zeros(dsize)
    z = (ndist_normal)
    f = np.zeros(dsize+3)
    for t in range(dsize):
        if z[t]!=0:
            f[t] = -(1.0/beta)*np.log(z[t])
        else:
            if t==0:
                f[t] = f[t+1]
            else:
                try:
                    f[t] = (f[t-1]+f[t+1])/2.0
                except IndexError:
                    f[t] = f[t-1]
    f[-3] = a
    f[-2] = b
    if data==None:
        np.save('colfes',f)
    else:
        return f

#Prueba de promedio
def promedio(xgrid,ygrid,fes):
    a = xgrid[0]
    b = xgrid[-1]
    c = ygrid[0]
    d = ygrid[-1]
    sizo = np.size(xgrid)
    colec = np.zeros(sizo)
    for i in range(sizo):
        xo = xgrid[i]
        value = 0
        for j in range(sizo):
            yo = ygrid[j]
            value = value + fes(xo,yo)
        value = (1.0/sizo)*value
        colec[i] = value
    return colec

#Metadinamica con dos coordenadas, una es colectiva.
def mdytwo_colec(xgrid, ygrid, fes, steps, initial_point, delta_sigma, beta, dt, alpha=None, height=None, data=None):

    '''|This function takes as entries the domain of your FES (two arrays: x,y), the parameterized 2D-FES, number of steps of simulation, initial
       |position (tuple e.g. [2,3]), delta sigma, height of gaussian and deltas x,y.
       |
       |Returns a matrix containing values of FES reconstruction. By default saves data in text file, otherwise it just returns matrix.'''

    start = time.time()
    if alpha==None:
        alfa = 0.5
    else:
        alfa = alpha

    if height==None:
        api = xgrid[0]
        bpi = xgrid[-1]
        cpi = ygrid[0]
        dpi = ygrid[-1]
        xgrido = np.arange(api, bpi, 0.1)
        ygrido = np.arange(cpi, dpi, 0.1)
        fms = 0
        for i in range(len(xgrido)):
            for j in range(len(ygrido)):
                force_x, force_y = partial_point(fes, xgrido[i], ygrido[j])
                fms = fms + force_x**2 + force_y**2
        fms = float(fms)/(len(xgrido)*len(ygrido))
        gaussian_height = delta_sigma*alfa*(np.e**(-0.5))*np.sqrt(fms)
    else:
        gaussian_height = height

    size_space = np.size(xgrid)
    a = xgrid[0]
    b = xgrid[-1]
    c = ygrid[0]
    d = ygrid[-1]
    ds = xgrid[1]-xgrid[0]
    sigma = np.sqrt(2/beta)
    noise = sigma*np.sqrt(dt)
    xo, yo = initial_point[0], initial_point[1]
    fes_sample = np.zeros(size_space)
    impor = mixed_colec(fes, fes_sample, xo, yo, a, ds)
    pot_dx, pot_dy = impor[1], impor[2]
    pot = impor[0]
    pos_x, pos_y = [], []

    for i in xrange(1, steps+1):
        print(steps-i)
        fes_sample = contribution_point(xgrid, fes_sample, xo, gaussian_height, delta_sigma)
        pos_x.append(xo)
        pos_y.append(yo)
        rx = np.random.normal(0,1)
        ry = np.random.normal(0,1)
        xbound = xo-(dt*pot_dx)+(noise*rx)
        ybound = yo-(dt*pot_dy)+(noise*ry)
        if a<=xbound<=b:
            xn = xbound
        else:
            xn = xo+(dt*pot_dx)-(noise*rx)

        if c<=ybound<=d:
            yn = ybound
        else:
            yn = yo+(dt*pot_dy)-(noise*ry)

        imporn = mixed_colec(fes, fes_sample, xo, yo, a, ds)
        potn = imporn[0]
        potn_dx, potn_dy = imporn[1], imporn[2]
        transition_one = (xn-xo+dt*pot_dx)**2+(yn-yo+dt*pot_dy)**2-(xo-xn+dt*potn_dx)**2-(yo-yn+dt*potn_dy)**2
        transition_two = (4.0/beta)*dt
        transition_term = np.exp(transition_one/transition_two)
        density = np.exp(beta*(pot-potn))
        val = transition_term*density
        if val > np.random.uniform() or val>1:
            pot, pot_dx, pot_dy = potn, potn_dx, potn_dy
            xo, yo = xn, yn


    end = time.time()
    duration = (1.0*(end-start))/60

    dinamicota = np.zeros((2,steps))
    dinamicota[0,:] = pos_x
    dinamicota[1,:] = pos_y
    if data==None:
        np.save('mdytwo_colec_results', -fes_sample)
        np.save('mdytwo_colec_dynamic', dinamicota)
    else:
        return fes_sample, duration

