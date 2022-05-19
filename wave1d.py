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
# hello

#%%

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
    # s['AR_forcing'] = AR_forcing(sigma_w = s['sigma_w'], alpha=s['alpha'], length=len(t))
    # s['h_left'] = np.interp(t,bound_t,bound_values) + s['AR_forcing']



    return s

def timestep(x,i,settings): #return (h,u) one timestep later
    # take one timestep
    temp=x.copy() 
    A=settings['A']
    B=settings['B']
    rhs=B.dot(temp) #B*x
    rhs[0]=settings['h_left'][i] #left boundary
    newx=spsolve(A,rhs)
    return newx

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

def plot_state(fig,x,i,s):

    '''
    #plot all waterlevels and velocities at one time
    fig.clear()
    xh=s['x_h']
    ax1=fig.add_subplot(211)
    ax1.plot(xh,x[0::2])
    ax1.set_ylabel('h')
    ax1.set_ylim([-3,3])
    xu=s['x_u']
    ax2=fig.add_subplot(212)
    ax2.plot(xu,x[1::2])
    ax2.set_ylabel('u')
    ax2.set_ylim([-3,3])
    plt.savefig("fig_map_%3.3d.png"%i)
    plt.draw()
    #plt.pause(0.2)
    '''
    
def plot_series(t,series_data,s,obs_data):
    
    
    # plot timeseries from model and observations
    loc_names=s['loc_names']
    nseries=len(loc_names)
    for i in range(4):
        fig,ax=plt.subplots()
        ax.plot(t,series_data[i,:],'b-')
        ax.set_title(loc_names[i])
        ax.set_xlabel('time')
        ntimes=min(len(t),obs_data.shape[1])
        ax.plot(t[0:ntimes],obs_data[i,0:ntimes],'k-')
        # plt.savefig(("%s.png"%loc_names[i]).replace(' ','_'))
    
def simulate():
    # for plots
    plt.close('all')
    fig1,ax1 = plt.subplots() #maps: all state vars at one time
    # locations of observations
    s=settings()
    L=s['L']
    dx=s['dx']
    xlocs_waterlevel=np.array([0.0*L,0.25*L,0.5*L,0.75*L,0.99*L])
    xlocs_velocity=np.array([0.0*L,0.25*L,0.5*L,0.75*L])
    ilocs=np.hstack((np.round((xlocs_waterlevel)/dx)*2,np.round((xlocs_velocity-0.5*dx)/dx)*2+1)).astype(int) #indices of waterlevel locations in x
    print(ilocs)
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
    #
    (x,t0)=initialize(s)
    t=s['t'][:] #[:40]
    times=s['times'][:] #[:40]
    series_data=np.zeros((len(ilocs),len(t)))
    for i in np.arange(1,len(t)):
        print('timestep %d'%i)
        x=timestep(x,i,s)
        plot_state(fig1,x,i,s) #show spatial plot; nice but slow
        series_data[:,i]=x[ilocs]
        
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

    plot_series(times,series_data,s,observed_data)
    
    return t, obs_times, series_data, observed_data
    
    
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



#%%
    
s = settings()


#load observations
(obs_times,obs_values)=timeseries.read_series('tide_vlissingen.txt')
observed_data=np.zeros((4,len(obs_times)))
observed_data[0,:]=obs_values[:]
(obs_times,obs_values)=timeseries.read_series('tide_terneuzen.txt')
observed_data[1,:]=obs_values[:]
(obs_times,obs_values)=timeseries.read_series('tide_hansweert.txt')
observed_data[2,:]=obs_values[:]
(obs_times,obs_values)=timeseries.read_series('tide_bath.txt')
observed_data[3,:]=obs_values[:]

observed_data = observed_data[:, 1:]

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



m0, T = initialize(s)

#%%

#### INITIALIZATION KALMAN FILTER
s['ensize'] = 50
t=s['t'][:] #[:40]
times=s['times'][:] #[:40]

sigmaens = 0.1 # Ensemble noise

k = np.ones(200)
k[1::2] = 2
P0 = sigmaens *k* np.eye(s['n']*2) # initial ensemble covariance 


y = np.random.normal(0, 1, size=(s['n']*2, len(t)+1, s['ensize'])) ## standard normal, gridpoints for h x ensemble size
eps0 = m0.reshape(-1,1) + np.linalg.cholesky(P0).dot(y[:,0,:])
x0 = np.mean(eps0, axis=1)

#%%


# series_data=np.zeros((len(ilocs),len(t)))

sigma_forecast = 0.01 #s['sigma_forecast']
w = np.zeros(shape=(2*s['n'], len(t), s['ensize']))
for i in range(2*s['n']):
    for j in range(s['ensize']):
        w[i, :, j] = AR_process(sigma_forecast, s['alpha'], size=len(t))

# sigmaobs = np.copy(s['sigma_N'])
sigmaobs = 0.01
R = np.eye(4)*sigmaobs
v = np.zeros(shape=(4, len(times), s['ensize']))
SR = np.linalg.cholesky(R)
for i in range(len(times)):
    v[:, i, :] = SR.dot(np.random.normal(0, 1, size=(4, s['ensize'])))
# v = np.linalg.cholesky(R).dot(np.random.normal(0, 1, size=(4, len(times)-1, s['ensize'])))

z_virt = observed_data.reshape(4, -1, 1) + v

H = np.zeros(shape=(4,200))
H[:, ilocs[1:5]] = 1


x_array= np.zeros(shape=(2*s['n'], len(t)))
series_data = np.zeros(shape=(len(ilocs), len(t)))

#%%

eps = eps0.copy()
for k in range(len(t)):
    print('timestep %d'%k)

    epsf = np.zeros_like(eps)
    for i in range(s['ensize']):
        epsf[:,i]=timestep(eps[:,i],k,s) + w[:, k, i]

    xf = np.mean(epsf, axis=1)

    ef = epsf - xf.reshape(-1,1)
    
    Pf = np.zeros((s['n']*2, s['n']*2))

    for j in range(s['ensize']):
        Pf = Pf + np.outer(ef[:, j], ef[:,j])/(s['ensize']-1)
    
    print(np.linalg.eigvals(Pf))

    D = np.matmul(np.matmul(H,Pf), H.T) + R
    K = np.matmul(np.matmul(Pf, H.T), np.linalg.inv(D))

    if k%50 == 0:
        plt.figure()
        plt.plot(K)

    innov = z_virt[:, k, :] - np.matmul(H, epsf)

    eps = epsf + np.matmul(K, innov)
    x = np.mean(eps, axis=1)

    e = eps - x.reshape(-1,1)

    P = np.zeros_like(Pf)

    for j in range(s['ensize']):
        P = P + np.outer(e[:, j], e[:,j])/(s['ensize']-1)
    print(np.linalg.eigvals(P))
    eps = x + np.linalg.cholesky(P).dot(y[:,k+1,:])
    break
    x_array[:, k] = x
    series_data[:,k]=x[ilocs,k]


#%%

plot_series(times,series_data,s,observed_data)
plt.show()








# %%
