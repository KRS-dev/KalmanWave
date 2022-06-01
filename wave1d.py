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
from scipy.stats import ks_2samp
from sklearn.cluster import KMeans ### 
import matplotlib.pyplot as plt
import timeseries
import dateutil 
import datetime

minutes_to_seconds=60.
hours_to_seconds=60.*60.
days_to_seconds=24.*60.*60.
np.random.seed(100)

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
    upperbound = 2* np.sqrt(trP)
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
    loc_names=s['loc_names'][0:5]
    loc_names[0] = loc_names[0].replace('0000', '',1)
    loc_names[1] = loc_names[1].replace('0000', '',1)
    loc_names[2] = loc_names[2].replace('0000', '',1)
    loc_names[3] = loc_names[3].replace('0000', '',1)
    loc_names[4] = loc_names[4].replace('0000', '',1)

    
    nseries=len(loc_names)
    ntimes=min(len(t),obs_data.shape[1])
    
    figs,axs = plt.subplots(2,2, figsize=(8,8))
    axs[0,0].plot(t,series_data[1,:],'g-' )
    axs[0,0].plot(t[0:ntimes],obs_data[1,0:ntimes],'k-')
    axs[0,0].set_title(loc_names[1])
    axs[0,0].minorticks_off()
    #axs[0,0].set_xticks(np.arange(0, series_data[1,-1],))
    #axs[0,0].set_xlim(0, 200000)
    #axs[0,0].locator_params(axis='x', nbins=4)
    
    axs[0,1].plot(t,series_data[2,:],'b-' )
    axs[0,1].plot(t[0:ntimes],obs_data[2,0:ntimes],'k-')
    axs[0,1].set_title(loc_names[2])
    axs[0,1].minorticks_off()
    
    axs[1,0].plot(t,series_data[3,:],'m-' )
    axs[1,0].plot(t[0:ntimes],obs_data[3,0:ntimes],'k-')
    axs[1,0].set_title(loc_names[3])
    axs[1,0].minorticks_off()
    
    
    axs[1,1].plot(t,series_data[4,:],'r-' )
    axs[1,1].plot(t[0:ntimes],obs_data[4,0:ntimes],'k-')
    axs[1,1].set_title(loc_names[4])
    axs[1,1].minorticks_off()
    
    axs[1,0].set_xlabel('time (s)')
    axs[1,1].set_xlabel('time (s)')
    axs[1,0].set_ylabel('height (m)')
    axs[0,0].set_ylabel('height (m)')
    
    figs.savefig("simulate.png")
    
    # for i in range(nseries):
        #fig,ax=plt.subplots()
        #ax.plot(t,series_data[i,:],'b-')
        #ax.set_title(loc_names[i])
        #ax.set_xlabel('time')
        #ntimes=min(len(t),obs_data.shape[1])
        #ax.plot(t[0:ntimes],obs_data[i,0:ntimes],'k-')
        #plt.savefig(("%s.png"%loc_names[i]).replace(' ','_'))
def plop(s_data, o_data):
    
    n = 5 # five cities, heights
    
    bias = np.zeros(n)
    rmse = np.zeros(n)
    med = np.zeros(n)
    ks2 = []
    
    ntimes = np.min([s_data.shape[1], o_data.shape[1]])
    
    for i in range(n):
        bias[i] = np.sqrt(np.sum(np.abs(s_data[i,:]-o_data[i,1:]))/ntimes)
        rmse[i] = np.sqrt(np.sum((s_data[i,:]-o_data[i,1:])**2)/ntimes)
        med[i] = np.median(s_data[i,:]-o_data[i,1:])
        ks2.append(ks_2samp(s_data[i,:],o_data[i,1:]))
        
    return bias, rmse, med, ks2

def plot_statistics(statistics):
    fig, axes = plt.subplots(nrows=4, ncols=3, figsize=(12,12), sharey=True)

    n_bins = 10
    statistics_list = ['Bias', 'Median', 'RMSE']

    for i in range(0, 4):
        for j in range(statistics.shape[2]):

            data = statistics[i+1, :, j]
            if j == 2:
                axes[i, j].hist(data, density=True, bins=n_bins, color='skyblue')
            else:
                axes[i, j].hist(data, density=True, bins=n_bins)



            x = np.linspace(np.min(data), np.max(data), 100)

            mean = np.mean(data)
            sigma = np.std(data)

            axes[i,j].plot(x, norm.pdf(x, mean, sigma), 'r-', alpha=0.6, label='norm pdf')
            
            textstr = '\n'.join((
                r'$\mu=%.2f$' % (mean, ),
                r'$\sigma=%.2f$' % (sigma, )))

            # these are matplotlib.patch.Patch properties
            props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)

            # place a text box in upper left in axes coords
            axes[i,j].text(0.05, 0.95, textstr, transform=axes[i,j].transAxes, fontsize=12,
                    verticalalignment='top', bbox=props)


            if j == 0:
                # print(s['loc_names'][i+1])
                axes[i, j].set_ylabel(s['loc_names'][i+1].split(' ')[-1])
            
            if i == 0:
                axes[i,j].set_title(statistics_list[j])

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


def enKF_timestep(k, eps, s):
    
    w_N = s['w_N']

    epsf = np.zeros_like(eps)

    for i in range(s['ensize']):
        matvec = timestep(eps[:-1,i],k,s)
        
        epsf[:-1,i] = matvec # model
        
        epsf[0,i] = epsf[0, i] + eps[-1,i] # add noise to first/boundary element
        
        epsf[-1,i] = s['alpha']*eps[-1, i]  + w_N[k,i] # add AR process to last element
    
    return epsf

def enKF_timestep_no_noise(k, eps, s):
    w_N = s['w_N']

    epsf = np.zeros_like(eps)
    for i in range(s['ensize']):
        matvec = timestep(eps[:-1,i],k,s)
        
        epsf[:-1,i] = matvec # model
        
        # epsf[0,i] = epsf[0, i] + eps[-1,i] # add noise to first/boundary element
        
        epsf[-1,i] = s['alpha']*eps[-1, i]  + w_N[k,i] # add AR process to last element
    
    return epsf

def enKF_measurementstep(k, epsf, s):

    H = s['H']
    R = s['R']
    z_virt = s['z_virt']

    # sample mean
    xf = np.mean(epsf[:-1, :], axis=1)
    # forecast error
    ef = epsf[:-1, :] - xf.reshape(-1,1)
    
    # More efficient calculation of K
    # factorize forecast variance estimate Pf=L L'
    Lf = 1/np.sqrt(s['ensize'] -1) * ef
    # H L
    Psi = H.dot(Lf)

    D = Psi.dot( Psi.T) + R

    K = (Lf.dot(Psi.T)).dot(np.linalg.inv(D))

    # for i, loc in enumerate(s['obs_ilocs']):
    #     K[ loc+20:, i] = 0
    #     K[:loc-20, i] = 0

    # Psi @ Psi.T + R
    temp0 = np.matmul(Psi, Psi.T) + R
    # temp1 = I - Psi.T @ (Psi @ Psi.T + R)^-1 @ Psi
    temp1 = np.eye(s['ensize']) - np.matmul( np.matmul(Psi.T, np.linalg.inv(temp0)), Psi)
    # ensemble variance estimate P = Lf @ temp1 @ Lf.T
    P = np.matmul(np.matmul(Lf, temp1 ), Lf.T)
    
    # if k%40 == 0:
    #     plt.figure()
    #     plt.plot(K)
    #     plt.legend(s['obs_loc_names'])

    # innovation
    innov = z_virt[:, k, :] - np.matmul(H, epsf[:-1,:])

    # measurement update
    eps = np.zeros_like(epsf)
    eps[:-1, :] = epsf[:-1, :] + np.matmul(K, innov)
    eps[-1, :] = epsf[-1, :]

    return eps, K, P

def enKF(eps0, prediction_timei, s):

    eps = eps0.copy()
    t = s['t']
    z_virt = s['z_virt']
    x_array= np.zeros(shape=(2*s['n']+1, len(t)))
    series_data = np.zeros(shape=(len(s['ilocs']), len(t)))
    K_array = np.zeros(shape=(2*s['n'] , len(s['obs_ilocs']), len(t)))

    fig1,ax1 = plt.subplots()

    for k in range(len(t)):
        print('timestep %d'%k)

        if k== prediction_timei:
            print('Start Prediction at {}'.format(s['times'][k]))

        #################################
        ####
        #### Ensemble Forecast Step
        ####
        #################################
        if k < prediction_timei:
            epsf = enKF_timestep(k, eps, s)
            #################################
            ####
            #### Data Assimilation Step
            ####
            #################################   
            
            eps, K, P = enKF_measurementstep(k, epsf, s)
            K_array[:, :, k] = K
        else:
            eps = enKF_timestep_no_noise(k, eps, s)
            ef = eps - np.mean(eps, axis=1).reshape(-1,1)
            Lf = 1/np.sqrt(s['ensize']-1) *ef[:-1, :]
            P = Lf.dot(Lf.T)

        # sample mean of the estimate
        x = np.mean(eps, axis=1)

        z_obs = np.mean(z_virt[:, k, :], axis=1)
        plot_state(fig1, x[:-1], k, s, z_obs, s['obs_ilocs'], P)    


        x_array[:, k] = x
        series_data[:,k] = x_array[s['ilocs'],k]
    

    # #################################
    # ####
    # #### TWIN EXPERIMENT
    # ####
    # #################################    

    # xi = np.vstack([eps0, N_0_arr])
    # xi_array = np.zeros((2*s['n']+1, len(t), s['ensize']))
    # series_twin = np.zeros(shape=(len(s['ilocs']), len(t), s['ensize']))
    
    # b_twin_array = np.zeros((5, s['ensize']))
    # r_twin_array = np.zeros((5, s['ensize']))
    # m_twin_array = np.zeros((5, s['ensize']))
    
    # for k in range(len(t)):
        
    #     for i in range(s['ensize']):
    #         matvec = timestep(xi[:-1,i],k,s)
            
    #         xi[:-1,i] = matvec # model
            
    #         xi[0,i] = xi[0,i] + xi[-1,i] # add noise to first/boundary element
            
    #         xi[-1,i] = s['alpha']*xi[-1, i]  + w_N[k,i] # add AR process to last element
        
    #         xi_array[:,k,:] = xi
        
    #         series_twin[:,k,i] = xi[s['ilocs'],i]
            
    
    # b_twin, r_twin, m_twin, _ = plop(np.mean(series_twin, axis=2), observed_data)

    b_kalman, r_kalman, m_kalman, _ = plop(series_data, observed_data)

    # twin_statistics = np.vstack([b_twin, r_twin, m_twin])
    kalman_statistics = np.vstack([b_kalman, r_kalman, m_kalman])
    # df_twin = pd.DataFrame(twin_statistics, index=['bias', 'rmse', 'median'], columns=s['loc_names'][:5])
    # print(df_twin)
    df_kalman = pd.DataFrame(kalman_statistics, index=['bias', 'rmse', 'median'], columns=s['loc_names'][:5] )
    print(df_kalman)

    plt.ion()
    # plot_statistics(twin_statistics)
    # plot_statistics(kalman_statistics)

    plot_series(times,series_data,s,observed_data)
    # plot_series(times, np.mean(series_twin, axis=2), s, observed_data)

    # plt.plot(x_array[:-1,::10])

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
    (obs_times,obs_values)=timeseries.read_series('waterlevel_vlissingen.txt')
    observed_data[1,:]=obs_values[:]
    (obs_times,obs_values)=timeseries.read_series('waterlevel_terneuzen.txt')
    observed_data[2,:]=obs_values[:]
    (obs_times,obs_values)=timeseries.read_series('waterlevel_hansweert.txt')
    observed_data[3,:]=obs_values[:]
    (obs_times,obs_values)=timeseries.read_series('waterlevel_bath.txt')
    observed_data[4,:]=obs_values[:]


    ##############################################
    ####
    #### PICK WHICH OBSERVATIONS TO USE IN ASSIMILATION
    ####
    ##############################################
    #  change the indices for a different set of observations
    s['obs_ilocs']=ilocs[1:5]
    s['obs_xlocs'] = xlocs_waterlevel[1:5]
    s['obs_loc_names'] = loc_names[1:5]

    observed_data_used = observed_data[1:5  , 1:]

    #%%

    #################################
    ####
    #### INITIALIZATION KALMAN FILTER
    ####
    #################################
    
    #################################
    #### Parameters to be varied
    #################################
    
    s['ensize'] = 200                    # number of ensembles: 50, 100, 200, 500
    sigmaens = 0.01                       # ensemble variance:  
    
    k = np.ones(200)
    # k[1::2] = 2                       # variance of height and velocity
    P0 = sigmaens*k*np.eye(s['n']*2)    # initial ensemble covariance (uncorrelated)
    
    ## Correlated ensemble covariance start
    # Autocorrelation matrix
    # autocorr0 = np.zeros(shape=(2*s['n'], 2*s['n']))
    # for i in range(s['n']):
    #     if i<20 and i%2 == 0:
    #         temp = np.ones(2*s['n'] - i) * (1- i/20)
    #         autocorr0 = autocorr0 + np.diag(temp, k=i) + np.diag(temp, k=-1*i)
    # P0 = autocorr0 * sigmaens
    
    sigmaobs = 0.01                     # observation variance:
    # sigmaobs = np.copy(s['sigma_N'])
    
    ############################################################################
    
    #################################
    #### Creation of initial ensemble
    #################################
    
    t=s['t'][:] #[:40] # numpy array
    times=s['times'][:] #[:40] # datetime array

    print(P0)

    y = np.random.normal(0, 1, size=(s['n']*2, len(t)+1, s['ensize'])) ## standard normal, gridpoints for h x ensemble size
    eps0 = m0.reshape(-1,1) + np.linalg.cholesky(P0).dot(y[:,0,:])
    x0 = np.mean(eps0, axis=1)

    # series_data=np.zeros((len(ilocs),len(t)))

    #################################
    #### Data Assimilation Initialization
    #################################

    R = np.eye(len(s['obs_ilocs']))*sigmaobs
    s['R'] = R

    v = np.zeros(shape=(len(s['obs_ilocs']), len(times), s['ensize']))
    SR = np.linalg.cholesky(R)

    # creation of noise for the virtual observations
    for i in range(len(times)):
        v[:, i, :] = SR.dot(np.random.normal(0, 1, size=(len(s['obs_ilocs']), s['ensize'])))
    # v = np.linalg.cholesky(R).dot(np.random.normal(0, 1, size=(4, len(times)-1, s['ensize'])))

    # virtual observations
    z = observed_data_used.reshape(len(s['obs_ilocs']), -1, 1)
    z_virt = z - v # lmao, is alleen maar huilen dit hoor
    s['z_virt'] = z_virt

    ##################################
    ###### Kalman Filter Initialization
    ##################################

    H = np.zeros(shape=(len(s['obs_ilocs']),2*s['n']))
    ## using only the last 4 observations so not at Cadzand
    for j, iloc in enumerate(s['obs_ilocs']):
        H[j, iloc] = 1
    s['H'] = H
    
    ##################################
    ###### Ensemble Forecast Initialization
    ##################################

    sigma_forecast = s['sigma_forecast']

    w_N = np.random.normal(0,sigma_forecast,size=(len(t), s['ensize'])) # only noise on boundary, time uncorrelated!
    s['w_N'] = w_N

    G = np.eye(2*s['n']+1)

    N_0_arr = np.zeros(s['ensize'])
    for j in range(s['ensize']):
        N_0 = AR_process(sigma_forecast, s['alpha'], size=1000)[-1]
    N_0_arr[j] = N_0

    # initial ensemble
    eps = np.vstack([eps0, N_0_arr])

    peak_i = np.argmax(observed_data[4,:]) # peak time indice recorded at Bath
    
    prediction_starttimes = peak_i - 6 * np.arange(1, 7) # Start 1, 2, 3, 4, 5, 6 hours before the peak with predicting


    for sttime in prediction_starttimes:
        enKF(eps, sttime, s)
    
    plt.show()
