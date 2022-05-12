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
        bias[i] = np.sqrt(np.sum(np.abs(s_d[i,:]-o_d[i,1:]))/len(s_d[i,:]))
        rmse[i] = np.sqrt(np.sum((s_d[i,:]-o_d[i,1:])**2)/len(s_d[i,:]))
        med[i] = np.median(s_d[i,:]-o_d[i,1:])
        
    return bias, rmse, med


#main program
if __name__ == "__main__":
    t_t, o_t, s_d, o_d = simulate()
    
    # Vlissingen
    bias_c = np.sqrt(np.sum(np.abs(s_d[0,:]-o_d[0,1:]))/len(s_d[0,:]))
    rmse_c = np.sqrt(np.sum((s_d[0,:]-o_d[0,1:])**2)/len(s_d[0,:]))
    med_c = np.median(s_d[0,:]-o_d[0,1:]) # less sensitive to outliers
    
    # Vlissingen
    bias_v = np.sqrt(np.sum(np.abs(s_d[1,:]-o_d[1,1:]))/len(s_d[1,:]))
    rmse_v = np.sqrt(np.sum((s_d[1,:]-o_d[1,1:])**2)/len(s_d[1,:]))
    med_v = np.median(s_d[1,:]-o_d[1,1:]) # less sensitive to outliers
    
    # Terneuzen
    bias_t = np.sqrt(np.sum(np.abs(s_d[2,:]-o_d[2,1:]))/len(s_d[2,:]))
    rmse_t = np.sqrt(np.sum((s_d[2,:]-o_d[2,1:])**2)/len(s_d[2,:]))
    med_t = np.median(s_d[2,:]-o_d[2,1:]) # less sensitive to outliers
    
    # Hansweert
    bias_h = np.sqrt(np.sum(np.abs(s_d[3,:]-o_d[3,1:]))/len(s_d[3,:]))
    rmse_h = np.sqrt(np.sum((s_d[3,:]-o_d[3,1:])**2)/len(s_d[3,:]))
    med_h = np.median(s_d[3,:]-o_d[3,1:]) # less sensitive to outliers
    
    # Bath
    bias_b = np.sqrt(np.sum(np.abs(s_d[4,:]-o_d[4,1:]))/len(s_d[4,:]))
    rmse_b = np.sqrt(np.sum((s_d[4,:]-o_d[4,1:])**2)/len(s_d[4,:]))
    med_b = np.median(s_d[4,:]-o_d[4,1:]) # less sensitive to outliers
    
    b, r, m = plop(s_d,o_d)
    
    ### finding the peaks
    
    peaks_c = find_peaks(s_d[0,:])[0] # index 0 = Cadzand
    #peaks_c = argrelextrema(s_d[0,:], np.greater)[0]
    heights_c = s_d[0,:][peaks_c]
    peak_pos_c = t_t[peaks_c]

    #peaks_v = find_peaks(s_d[1,:])[0] # index 1 = Vlissingen
    peaks_v = argrelextrema(s_d[1,:], np.greater)[0]
    peaks_v = peaks_v
    heights_v = s_d[1,:][peaks_v]
    peak_pos_v = t_t[peaks_v]
    
    #peaks_t = find_peaks(s_d[2,:])[0] # index 2 = Terneuzen
    peaks_t = argrelextrema(s_d[2,:], np.greater)[0]
    heights_t = s_d[2,:][peaks_t]
    peak_pos_t = t_t[peaks_t]
    
    #peaks_h = find_peaks(s_d[3,:])[0] # index 3 = Hansweert
    peaks_h = argrelextrema(s_d[3,:], np.greater)[0]
    heights_h = s_d[3,:][peaks_h]
    peak_pos_h = t_t[peaks_h]
    
    #peaks_b = find_peaks(s_d[4,:])[0] # index 4 = Bath
    peaks_b = argrelextrema(s_d[4,:], np.greater)[0]
    heights_b = s_d[4,:][peaks_b]
    
    ## delete element with negative heights
    i_neg = np.where(heights_b < 0.0)[0]
    peaks_b = np.delete(peaks_b, i_neg)
    heights_b = np.delete(heights_b, i_neg)
    
    peak_pos_b = t_t[peaks_b]
    
   
    ### Clustering the extrema by using K-means
    
    #### Dataframe creation

    Data_v = {'time': peak_pos_v, 'height': heights_v}
    points_v = pd.DataFrame(Data_v, columns=['time', 'height'])
    
    Data_b = {'time': peak_pos_b, 'height': heights_b}
    points_b = pd.DataFrame(Data_b, columns=['time', 'height'])

    #### K-means application

    kmeans_v = KMeans(n_clusters=4)
    points_v['label'] = kmeans_v.fit_predict(points_v[['time']])

    kmeans_b = KMeans(n_clusters=4)
    points_b['label'] = kmeans_b.fit_predict(points_b[['time']])
    
    #points_v = points_v.sort_values(by='label')
    #points_b = points_b.sort_values(by='label')
    
    #### Checked for the middle of the clusters, that was easier
    #### Between wave forms, time differs, and so does the speed, not strange
    #### So, I will take the maximum values of each cluster, probably speed will be more similar
    #### Hopefully :)
    
    centroids_v = kmeans_v.cluster_centers_.ravel()
    centroids_v = np.sort(centroids_v)
    centroids_b = kmeans_b.cluster_centers_.ravel()
    centroids_b = np.sort(centroids_b)
    
    time_difference = centroids_b - centroids_v
    
    
    plt.ion()
    
    fig_peaks = plt.figure()
    ax_peaks = fig_peaks.subplots()
    ax_peaks.plot(t_t, s_d[0,:], color = 'k')
    ax_peaks.plot(t_t, s_d[1,:], color = 'm')
    ax_peaks.plot(t_t, s_d[4,:], color = 'b')
    ax_peaks.scatter(peak_pos_c, heights_c, color='y', marker = 'o', label = 'maxima Cadzand')
    ax_peaks.scatter(peak_pos_v, heights_v, color='g', marker = 'o', label = 'maxima Vlissingen')
    ax_peaks.scatter(peak_pos_b, heights_b, color='r', marker = 'o', label = 'maxima Bath')
    plt.title("Peaks")
    ax_peaks.legend()
    
    plt.show()
