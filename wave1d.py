#
# 1d shallow water model
#
# solves
# dh/dt + D du/dx = 0
# du/dt + g dh/dx + f*u = 0
# 
# staggered discretiztation in space and central in time
#
# o -> o -> o -> o ->   # staggering
# L u  h u  h u  h  R   # element
# 0 1  2 3  4 5  6  7   # index in state vector
#
# m=1/2, 3/2, ...
#  u[n+1,m] + 0.5 g dt/dx ( h[n+1,m+1/2] - h[n+1,m-1/2]) + 0.5 dt f u[n+1,m] 
#= u[n  ,m] - 0.5 g dt/dx ( h[n  ,m+1/2] - h[n  ,m-1/2]) - 0.5 dt f u[n  ,m]
# m=1,2,3,...
#  h[n+1,m] + 0.5 D dt/dx ( u[n+1,m+1/2] - u[n+1,m-1/2])  
#= h[n  ,m] - 0.5 D dt/dx ( u[n  ,m+1/2] - u[n  ,m-1/2])

#%%

from enum import auto
import numpy as np
import pandas as pd ###
from scipy.sparse import spdiags
from scipy.sparse.linalg import spsolve
from scipy.signal import find_peaks ###
from scipy.signal import argrelextrema ###
from sklearn.cluster import KMeans ### 
import matplotlib.pyplot as plt
import timeseries
import dateutil 
import datetime

minutes_to_seconds=60.
hours_to_seconds=60.*60.
days_to_seconds=24.*60.*60.

def settings():
    s=dict() #hashmap to  use s['g'] as s.g in matlab
    # Constants
    s['g']=9.81 # acceleration of gravity
    s['D']=20.0 # Depth
    s['f']=1/(0.06*days_to_seconds) # damping time scale
    L=100.e3 # length of the estuary
    s['L']=L
    n=100 #number of cells
    s['n']=n    
    # Grid(staggered water levels at 0 (boundary) dx 2dx ... (n-1)dx
    #      velocities at dx/2, 3dx/2, (n-1/2)dx
    dx=L/(n+0.5)
    s['dx']=dx
    x_h = np.linspace(0,L-dx,n)
    s['x_h'] = x_h
    s['x_u'] = x_h+0.5    
    # initial condition
    s['h_0'] = np.zeros(n)
    s['u_0'] = np.zeros(n)    
    # time
    t_f=2.*days_to_seconds #end of simulation
    dt=10.*minutes_to_seconds
    s['dt']=dt
    reftime=dateutil.parser.parse("201312050000") #times in secs relative
    s['reftime']=reftime
    t=dt*np.arange(np.round(t_f/dt)) + dt ### time step shift
    s['t']=t
    #boundary (western water level)
    #1) simple function
    #s['h_left'] = 2.5 * np.sin(2.0*np.pi/(12.*hours_to_seconds)*t)
    #2) read from file
    (bound_times,bound_values)=timeseries.read_series('tide_cadzand.txt')
    bound_t=np.zeros(len(bound_times))
    for i in np.arange(len(bound_times)):
        bound_t[i]=(bound_times[i]-reftime).total_seconds()
    s['h_left'] = np.interp(t,bound_t,bound_values)  



    s['sigma_N'] = 0.2
    s['alpha'] = np.exp(-s['dt']/(6*hours_to_seconds))
    s['sigma_forecast'] = s['sigma_N'] *np.sqrt(1 - s['alpha']**2 )
#    s['AR_forcing'] = AR_process(sigma_w = s['sigma_forecast'], alpha=s['alpha'], size=len(t))
#    s['h_left'] = np.interp(t,bound_t,bound_values) + s['AR_forcing']

    return s

def initialize(settings): #return (h,u,t) at initial time 
    #compute initial fields and cache some things for speed
    h_0=settings['h_0']
    u_0=settings['u_0']
    n=settings['n']
    x=np.zeros(2*n) #order h[0],u[0],...h[n],u[n], changed order!
    x[0::2]=h_0[:]
    x[1::2]=u_0[:]
    #time
    t=settings['t']
    reftime=settings['reftime']
    dt=settings['dt']
    times=[]
    second=datetime.timedelta(seconds=1)
    for i in np.arange(len(t)):
        times.append(reftime+i*int(dt)*second)
    settings['times']=times
    #initialize coefficients
    # create matrices in form A*x_new=B*x+alpha 
    # A and B are tri-diagonal sparse matrices 
    Adata=np.zeros((3,2*n)) #order h[0],u[0],...h[n],u[n]  
    Bdata=np.zeros((3,2*n))
    #left boundary
    Adata[1,0]=1.
    #right boundary
    Adata[1,2*n-1]=1.
    # i=1,3,5,... du/dt  + g dh/sx + f u = 0
    #  u[n+1,m] + 0.5 g dt/dx ( h[n+1,m+1/2] - h[n+1,m-1/2]) + 0.5 dt f u[n+1,m] 
    #= u[n  ,m] - 0.5 g dt/dx ( h[n  ,m+1/2] - h[n  ,m-1/2]) - 0.5 dt f u[n  ,m]
    g=settings['g'];dx=settings['dx'];f=settings['f']
    temp1=0.5*g*dt/dx
    temp2=0.5*f*dt
    for i in np.arange(1,2*n-1,2):
        Adata[0,i-1]= -temp1
        Adata[1,i  ]= 1.0 + temp2
        Adata[2,i+1]= +temp1
        Bdata[0,i-1]= +temp1
        Bdata[1,i  ]= 1.0 - temp2
        Bdata[2,i+1]= -temp1
    # i=2,4,6,... dh/dt + D du/dx = 0
    #  h[n+1,m] + 0.5 D dt/dx ( u[n+1,m+1/2] - u[n+1,m-1/2])  
    #= h[n  ,m] - 0.5 D dt/dx ( u[n  ,m+1/2] - u[n  ,m-1/2])
    D=settings['D']
    temp1=0.5*D*dt/dx
    for i in np.arange(2,2*n,2):
        Adata[0,i-1]= -temp1
        Adata[1,i  ]= 1.0
        Adata[2,i+1]= +temp1
        Bdata[0,i-1]= +temp1
        Bdata[1,i  ]= 1.0
        Bdata[2,i+1]= -temp1    
    # build sparse matrix
    A=spdiags(Adata,np.array([-1,0,1]),2*n,2*n)
    B=spdiags(Bdata,np.array([-1,0,1]),2*n,2*n)
    A=A.tocsr()
    B=B.tocsr()
    settings['A']=A #cache for later use
    settings['B']=B

    return (x,t[0])

def plot_state(fig,x,i,s, obs_h, ilocs, P):

    # copmute confidence intervals
    trP = np.diag(P)
    # 2* sigma as upper and lower bound so confidence interval of 95.45%
    upperbound = 2* np.sqrt(trP[:-1])
    lowerbound = -1*upperbound
    
    #plot all waterlevels and velocities at one time
    fig.clear()
    xh=s['x_h']
    ax1=fig.add_subplot(211)
    ax1.plot(xh,x[0::2], 'k')
    # plot the sample mean of the obervations
    ax1.scatter(xh[ilocs//2], obs_h, c='r')
    # plot confidence intervals
    ax1.plot(xh, x[0::2]+ upperbound[0::2], '--b' ,alpha=.5)
    ax1.plot(xh, x[0::2]+ lowerbound[0::2], '--b', alpha=.5)
    ax1.set_ylabel('h')
    ax1.set_ylim([-4,4])

    xu=s['x_u']
    ax2=fig.add_subplot(212)
    ax2.plot(xu,x[1::2])
    ax2.plot(xh, x[1::2] + upperbound[1::2], '--b', alpha=.5)
    ax2.plot(xh, x[1::2] +lowerbound[1::2], '--b', alpha=.5)
    ax2.set_ylabel('u')
    ax2.set_ylim([-3,3])

    #plt.savefig("fig_map_%3.3d.png"%i)
    plt.draw()
    plt.pause(0.01)
    
def plot_series(t,series_data,s,obs_data):
    
    
    # plot timeseries from model and observations
    loc_names=s['loc_names']
    nseries=len(loc_names)
    for i in range(nseries):
        fig,ax=plt.subplots()
        ax.plot(t,series_data[i,:],'b-')
        ax.set_title(loc_names[i])
        ax.set_xlabel('time')
        ntimes=min(len(t),obs_data.shape[1])
        ax.plot(t[0:ntimes],obs_data[i,0:ntimes],'k-')
        # plt.savefig(("%s.png"%loc_names[i]).replace(' ','_'))
    
def plop(s_data, o_data):
    
    n = 5 # five cities, heights
    
    bias = np.zeros(n)
    rmse = np.zeros(n)
    med = np.zeros(n)
    
    for i in range(n):
        bias[i] = np.sqrt(np.sum(np.abs(s_d[i,:]-o_d[i,1:]))/len(s_d[i,:]))
        rmse[i] = np.sqrt(np.sum((s_d[i,:]-o_d[i,1:])**2)/len(s_d[i,:]))
        med[i] = np.median(s_d[i,:]-o_d[i,1:])
        
    return bias, rmse, med

def AR_process(sigma_w, alpha, size, startup=1000):

    N = int(size + startup)

    ## iid normal
    white_noise = np.random.normal(loc=0, scale=sigma_w, size=N)

    AR = np.zeros_like(white_noise)
    AR[0] = white_noise[0]
    for i in range(N-1):
        AR[i+1] = alpha*AR[i] + white_noise[i+1]

    return AR[startup:] ## Returns the AR process with the correct length after startup

def timestep(x,i,settings): #return (h,u) one timestep later
    # take one timestep
    temp=x.copy() 
    A=settings['A']
    B=settings['B']
    rhs=B.dot(temp) #B*x
    rhs[0]= settings['h_left'][i]  #left boundary
    newx=spsolve(A,rhs)
    return newx

#%%

if __name__ == '__main__':
    #t_t, o_t, s_d, o_d, s = simulate()
    #plot_series(t_t,s_d,s,o_d)
    #plt.show()




    #################################
    ####
    #### LOADING OBSERVATIONS
    ####
    #################################

    s = settings()
    m0, T = initialize(s)

    dx = s['dx']
    L = s['L']
    xlocs_waterlevel=np.array([0.0*L,0.25*L,0.5*L,0.75*L,0.99*L])
    xlocs_velocity=np.array([0.0*L,0.25*L,0.5*L,0.75*L])
    ilocs=np.hstack((np.round((xlocs_waterlevel)/dx)*2,np.round((xlocs_velocity-0.5*dx)/dx)*2+1)).astype(int) #indices of waterlevel locations in x
    loc_names=[]
    names=['Cadzand','Vlissingen','Terneuzen','Hansweert','Bath']
    for i in range(len(xlocs_waterlevel)):
        loc_names.append('Waterlevel at x=%f km %s'%(0.001*xlocs_waterlevel[i],names[i]))
    for i in range(len(xlocs_velocity)):
        loc_names.append('Velocity at x=%f km %s'%(0.001*xlocs_velocity[i],names[i]))
    s['xlocs_waterlevel']=xlocs_waterlevel
    s['xlocs_velocity']=xlocs_velocity
    s['ilocs']=ilocs
    s['loc_names']=loc_names
    #load observations
    (obs_times,obs_values)=timeseries.read_series('tide_cadzand.txt')
    observed_data=np.zeros((len(ilocs),len(obs_times)))
    observed_data[0,:]=obs_values[:]
    (obs_times,obs_values)=timeseries.read_series('tide_vlissingen.txt')
    observed_data[1,:]=obs_values[:]
    (obs_times,obs_values)=timeseries.read_series('tide_terneuzen.txt')
    observed_data[2,:]=obs_values[:]
    (obs_times,obs_values)=timeseries.read_series('tide_hansweert.txt')
    observed_data[3,:]=obs_values[:]
    (obs_times,obs_values)=timeseries.read_series('tide_bath.txt')
    observed_data[4,:]=obs_values[:]


    ##############################################
    ####
    #### PICK WHICH OBSERVATIONS TO USE IN ASSIMILATION
    ####
    ##############################################
    #  change the indices for a different set of observations
    s['obs_ilocs']=ilocs[1:4]
    s['obs_xlocs'] = xlocs_waterlevel[1:4]
    s['obs_loc_names'] = loc_names[1:4]

    observed_data_used = observed_data[1:4  , 1:]

    #%%

    #################################
    ####
    #### INITIALIZATION KALMAN FILTER
    ####
    #################################



    s['ensize'] = 500 # number of ensembles
    t=s['t'][:] #[:40] # numpy array
    times=s['times'][:] #[:40] # datetime array

    sigmaens = .2
    # ensemble noise
    k = np.ones(200)
    # k[1::2] = 2 # variance of height and velocity

    # uncorrelated initial ensemble covariance
    #P0 = sigmaens*k*np.eye(s['n']*2) # initial ensemble covariance 

    ## Correlated ensemble covariance start
    # Autocorrelation matrix
    autocorr0 = np.zeros(shape=(2*s['n'], 2*s['n']))
    for i in range(s['n']):
        if i<20 and i%2 == 0:
            temp = np.ones(2*s['n'] - i) * (1- i/20)
            autocorr0 = autocorr0 + np.diag(temp, k=i) + np.diag(temp, k=-1*i)
    
    P0 = autocorr0 * sigmaens
    print(P0)


    y = np.random.normal(0, 1, size=(s['n']*2, len(t)+1, s['ensize'])) ## standard normal, gridpoints for h x ensemble size
    eps0 = m0.reshape(-1,1) + np.linalg.cholesky(P0).dot(y[:,0,:])
    x0 = np.mean(eps0, axis=1)

    # series_data=np.zeros((len(ilocs),len(t)))

    #################################
    ####
    #### Data Assimilation Step
    ####
    #################################

    # sigmaobs = np.copy(s['sigma_N'])
    sigmaobs = 0.05
    R = np.eye(len(s['obs_ilocs']))*sigmaobs
    v = np.zeros(shape=(len(s['obs_ilocs']), len(times), s['ensize']))
    SR = np.linalg.cholesky(R)

    # creation of noise for the virtual observations
    for i in range(len(times)):
        v[:, i, :] = SR.dot(np.random.normal(0, 1, size=(len(s['obs_ilocs']), s['ensize'])))
    # v = np.linalg.cholesky(R).dot(np.random.normal(0, 1, size=(4, len(times)-1, s['ensize'])))

    # virtual observations
    z_virt = observed_data_used.reshape(len(s['obs_ilocs']), -1, 1) - v # lmao, is alleen maar huilen dit hoor

    ##################################
    ###### Kalman Filter Initialization
    ##################################

    H = np.zeros(shape=(len(s['obs_ilocs']),2*s['n'] + 1))

    ## using only the last 4 observations so not at Cadzand
    for j, iloc in enumerate(s['obs_ilocs']):
        H[j, iloc] = 1

    x_array= np.zeros(shape=(2*s['n']+1, len(t)))
    series_data = np.zeros(shape=(len(s['ilocs']), len(t)))

    K_array = np.zeros(shape=(2*s['n'] + 1, len(s['obs_ilocs']), len(t)))

    #################################
    ####
    #### Ensemble Forecast Step
    ####
    #################################

    sigma_forecast = s['sigma_forecast']

    w_N = np.random.normal(0,sigma_forecast,size=(len(t), s['ensize'])) # only noise on boundary, time uncorrelated!

    G = np.eye(2*s['n']+1)

    N_0_arr = np.zeros(s['ensize'])
    for j in range(s['ensize']):
        N_0 = AR_process(sigma_forecast, s['alpha'], size=1000)[-1]
    N_0_arr[j] = N_0

    # initial ensemble
    eps = np.vstack([eps0, N_0_arr])

    fig1,ax1 = plt.subplots()

    for k in range(len(t)):
        print('timestep %d'%k)

        # ensemble forecast step at time k
        epsf = np.zeros_like(eps)
        for i in range(s['ensize']):
            matvec = timestep(eps[:-1,i],k,s)

            
            epsf[:-1,i] = matvec # model
            
            epsf[0,i] = epsf[0, i] + eps[-1,i] # add noise to first/boundary element
            # epsf[0, i] = s['h_left'][k] + eps[-1, i]
            
            epsf[-1,i] = s['alpha']*eps[-1, i]  + w_N[k,i] # add AR process to last element
            
        # sample mean
        xf = np.mean(epsf, axis=1)

        # forecast error
        ef = epsf - xf.reshape(-1,1)   


        # More efficient calculation of K
        # factorize forecast variance estimate Pf=L L'
        Lf = 1/np.sqrt(s['ensize'] -1) * ef
        # H L
        Psi = np.matmul(H, Lf)

        D = np.matmul(Psi, Psi.T) + R
        K = np.matmul(np.matmul(Lf, Psi.T), np.linalg.inv(D))

        # Psi @ Psi.T + R
        temp0 = np.matmul(Psi, Psi.T) + R
        # temp1 = I - Psi.T @ (Psi @ Psi.T + R)^-1 @ Psi
        temp1 = np.eye(s['ensize']) - np.matmul( np.matmul(Psi.T, np.linalg.inv(temp0)), Psi)
        # ensemble variance estimate P = Lf @ temp1 @ Lf.T
        P = np.matmul(np.matmul(Lf, temp1 ), Lf.T)

        # # forecast covariance
        # Pf = np.zeros((s['n']*2 +1, s['n']*2 +1))

        # for j in range(s['ensize']):
        #     Pf = Pf + np.outer(ef[:, j], ef[:,j])/(s['ensize']-1)
            
        # # kalman filter creation
        # D = np.matmul(np.matmul(H,Pf), H.T) + R 

        # K = np.matmul(np.matmul(Pf, H.T), np.linalg.inv(D))


        # if k%10 == 0:
        #     plt.figure()
        #     plt.plot(K)

        K_array[:, :, k] = K

        # innovation
        innov = z_virt[:, k, :] - np.matmul(H, epsf)

        # measurement update
        eps = epsf + np.matmul(K, innov)

        # sample mean of the estimate
        x = np.mean(eps, axis=1)

        z_obs = np.mean(z_virt[:, k, :], axis=1)
        plot_state(fig1, x[:-1], k, s, z_obs, s['obs_ilocs'], P)    


        x_array[:, k] = x
        series_data[:,k] = x_array[s['ilocs'],k]

    plot_series(times,series_data,s,observed_data)

    # plt.plot(x_array[:-1,::10])
    plt.show()
