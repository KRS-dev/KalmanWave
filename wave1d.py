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
    
    '''
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
        plt.savefig(("%s.png"%loc_names[i]).replace(' ','_'))
    '''
    
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
        bias[i] = np.mean(s_data[i,:]-o_data[i,1:])
        rmse[i] = np.sqrt(np.sum((s_data[i,:]-o_data[i,1:])**2)/len(s_data[i,:]))
        med[i] = np.median(s_data[i,:]-o_data[i,1:])
        
    return bias, rmse, med
    
def peaks_find(s_data, t_data):
    
    n = 5 # five cities, heights
    
    peaks = []
    pos = []
    heights = []
    
    for i in range(n):
        peak = find_peaks(s_data[i,:])[0]
        height = s_data[i,:][peak]
        peak_pos = t_data[peak]
        
        i_n = np.where(height < 0.0)[0]
        
        print(len(i_n))
        
        if len(i_n) >= 1:
            peak = np.delete(peak, i_n)
            height = np.delete(height, i_n)
            peak_pos = t_data[peak]
        
        heights.append(height) 
        peaks.append(peak)
        pos.append(peak_pos)
    
    return heights, peaks, pos
    
def peaks_cluster(nc, p):
    
    km = KMeans(n_clusters=nc)
    p['label'] = km.fit_predict(p[['time']])
    labels = p.label.unique()
    
    times = np.zeros(nc)
    
    j = 0
    for i in labels:
        c = p.loc[p['label'] == i]          ## Extracting each cluster in correct order
        print(c)
        ind = c['height'].idxmax()          ## Index of the maximum height
        ttime = c['time'].loc[ind]          ## The time at the index
        times[j] = ttime
        j = j + 1
  
    return times

#main program
if __name__ == "__main__":
    t_t, o_t, s_d, o_d = simulate()
    
    
    ###################################################################
    
    #### Finding the peaks
    
    data_heights, data_peaks, positions = peaks_find(s_d, t_t)
    
    ####################################################################
    
    #### Clustering the extrema by using K-means
    
    ## Dataframe creation
    
    # Cadzand
    
    Data_c = {'time': positions[0], 'height': data_heights[0]} 
    points_c = pd.DataFrame(Data_c, columns=['time', 'height'])

    # Vlissingen    
    
    Data_v = {'time': positions[1], 'height': data_heights[1]}
    points_v = pd.DataFrame(Data_v, columns=['time', 'height'])
    
    # Terneuzen
    
    Data_t = {'time': positions[2], 'height': data_heights[2]}
    points_t = pd.DataFrame(Data_t, columns=['time', 'height'])
    
    # Hansweert
    
    Data_h = {'time': positions[3], 'height': data_heights[3]}
    points_h = pd.DataFrame(Data_h, columns=['time', 'height'])
    
    # Bath
    
    Data_b = {'time': positions[4], 'height': data_heights[4]}
    points_b = pd.DataFrame(Data_b, columns=['time', 'height'])
    
    ## K-means application on Vlissingen and Terneuzen
    
    n_c = 4
    
    t_vlissingen = peaks_cluster(n_c, points_v)
    t_terneuzen = peaks_cluster(n_c, points_t)
    t_difference = t_terneuzen - t_vlissingen
    v_v_t = 25000./(np.sum(t_difference)/len(t_difference))
    
    #####################################################################
    
    #### Bias, RMSE, Median 
     
    b, r, m = plop(s_d,o_d)
    
    plt.ion()
    
    fig_peaks = plt.figure()
    ax_peaks = fig_peaks.subplots()
    ax_peaks.plot(t_t, s_d[0,:], color = 'k')
    ax_peaks.plot(t_t, s_d[1,:], color = 'm')
    ax_peaks.plot(t_t, s_d[4,:], color = 'b')
    ax_peaks.scatter(positions[0], data_heights[0], color='y', marker = 'o', label = 'maxima Cadzand')
    ax_peaks.scatter(positions[1], data_heights[1], color='g', marker = 'o', label = 'maxima Vlissingen')
    ax_peaks.scatter(positions[4], data_heights[4], color='r', marker = 'o', label = 'maxima Bath')
    plt.title("Peaks")
    ax_peaks.legend()
    
    plt.show()
