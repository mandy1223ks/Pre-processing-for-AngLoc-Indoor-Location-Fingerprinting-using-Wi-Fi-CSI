import sqlite3
import numpy as np
import struct
import io
import os
from collections import deque
import cmath
import matplotlib.pyplot as plt
import math
from scipy.optimize import curve_fit
from mpl_toolkits.mplot3d import Axes3D

#initial parameter setup
nr = 2
nt = 2
num_antenna = nr*nt
subcarrier = 56
feature_number = nr * nt * subcarrier
location = [0]*4
num_location = 4


#load the file
conn = sqlite3.connect('/Users/yishan/desktop/project/20210525_xunwei_newmeeting/20210525_xunwei_newmeeting/20210525_xunwei_newmeeting.sqlite',detect_types=sqlite3.PARSE_DECLTYPES)

def adapt_array(arr):
    out = io.BytesIO()
    np.save(out, arr)
    out.seek(0)
    return sqlite3.Binary(out.read())

def convert_array(text):
    out = io.BytesIO(text)
    out.seek(0)
    return np.load(out)

sqlite3.register_adapter(np.ndarray, adapt_array)
sqlite3.register_converter("array", convert_array)
c = conn.cursor()
c.execute("SELECT * FROM PD WHERE perfomed_at")


#load the data X_train,Y_train
def createTrainFromSQLite(c, timeStep=1):
    x_temp, y_temp = deque(), deque()
    count = 0
    for row in c:
        x_temp.append(np.transpose(row[0]))
        y_temp.append(row[1])
        count = count + 1
    return np.array(x_temp), np.array(y_temp) ,count

X, Y,num_data = createTrainFromSQLite(c, 1) #CFR
#%%
'''
def shuffle(X,Y):
  np.random.seed(10)
  randomList = np.arange(X.shape[0])
  np.random.shuffle(randomList)
  return X[randomList], Y[randomList]

X_train_sh, Y_train_sh = shuffle(X_train,Y_train)
'''
#calculate the number of packets for each location
for i in range(num_data):
    for j in range(num_location):
        if Y[i] == j:
            location[j]=location[j]+1
            break


#calculate the amplitude and phase
X_amp = abs(X)
X_phase = np.unwrap(np.angle(X),axis=-1)
#%%
#def the plot function for amplitude (one packet)
def amp_plot(x,text,text1):
    if text1 == 's' :
        text1 = 'Subcarrier'
    elif text1 == 't' :
        text1 = 'Tap'
    for i in range(nr):
        for j in range(nt):
            plt.title('{}'.format(text))
            plt.xlabel('{} Index'.format(text1))
            plt.ylabel('Amplitude')
            plt.legend(title='RX{} TX{}'.format(i+1,j+1))
            plt.plot(x[50][i*2+j],'b-.')
            plt.show()

#def the plot function for unwrapped phase (one packet)            
def phase_plot(x,text,text1):
    if text1 == 's' :
        text1 = 'Subcarrier'
    elif text1 == 't' :
        text1 = 'Tap'
    for i in range(nr):
        for j in range(nt):
            plt.title('{}'.format(text))
            plt.xlabel('{} Index'.format(text1))
            plt.ylabel('Unwrapped Phase')
            plt.legend(title='RX{} TX{}'.format(i+1,j+1))
            plt.plot(x[50][i*2+j],'b-.')
            plt.show()   


#plot original CFR Amplitude
amp_plot(X_amp,'Origianl CFR','s')

#plot original CFR Unwrapped Phase
phase_plot(X_phase,'Original CFR','s')

#convert CFR into CIR
X_CIR = np.fft.ifft(X)
X_CIR_amp = abs(X_CIR)
X_CIR_phase = np.unwrap(np.angle(X_CIR),axis=-1)

#plot original CIR Amplitude
amp_plot(X_CIR_amp,'Origianl CIR','t')

#plot original CIR Unwrapped Phase
phase_plot(X_CIR_phase,'Original CIR','t')

#%%
#calculate the average channel power of each tap
X_CIR_power = pow(abs(X_CIR),2)

'''
#calculate the cumulative contribution rate
sum_channel_power = np.zeros((num_data,num_location))
sum_channel_power = np.sum(X_train_CIR_power,axis=2)

ccr = np.zeros((num_data, num_location,subcarrier))
for i in range(num_data):
    for j in range(num_location):
        temp=0
        for k in range(subcarrier):
            ccr[i][j][k]=temp+X_train_CIR_power[i][j][k]/sum_channel_power[i][j]
            temp=ccr[i][j][k]

#find the length T of the rectangular window
len_T=0
for i in range(num_data):
    for j in range(num_location):
        for k in range(subcarrier):
            if ccr[i][j][k] >= 0.55:
                break
        len_T=len_T+k
len_T=round(len_T/num_data/num_location)

#truncte the tap after T
X_train_CIR_filtered=X_train_CIR
for i in range(num_data):
    for j in range(num_location):
        for k in range(50,subcarrier):
            X_train_CIR_filtered[i][j][k]=0


#convert filered CIR to filtered CFR
X_train_filtered=np.fft.fft(X_train_CIR_filtered)

for i in range(50):
    plt.plot(abs(X_train_filtered[i+500][3]),'-.')
plt.show()
'''
#%%
#design myself_tap_filtering_1

'''
#calculate the contribution rate
sum_channel_power = np.sum(X_CIR_power,axis=2)
cr = np.zeros((num_data, num_location,subcarrier))
for i in range(num_data):
    for j in range(num_antenna):
        cr[i][j]=X_CIR_power[i][j]/sum_channel_power[i][j]

temp=0
for i in range(round(subcarrier/2)):
    temp = temp + cr[0][1][i] + cr[0][1][55-i]
    if (temp>=0.99):
        break;

#remove the tap which contribution rate is under 0.1%
for i in range(num_data):
    for j in range(num_antenna):
        for k in range(subcarrier):
            if cr[i][j][k] <= 0.001:
                X_CIR_filtered[i][j][k]=0
'''
#design myself_tap_filtering_2
X_CIR_filtered=X_CIR[:]

#calculate the contribution rate
sum_channel_power = np.sum(X_CIR_power,axis=2)
cr = np.zeros((num_data, num_location,subcarrier))
for i in range(num_data):
    for j in range(num_antenna):
        cr[i][j]=X_CIR_power[i][j]/sum_channel_power[i][j]

# do the tap filtering for index form 10 to 47
for i in range(num_data):
    for j in range(num_antenna):
        X_CIR_filtered[i][j][10:47]=0
 
#calculate the amplitude and phase of CIR after filtering        
X_CIR_filtered_amp = abs(X_CIR_filtered)
X_CIR_filtered_phase = np.unwrap(np.angle(X_CIR_filtered),axis=-1)

#plot filtered CIR Amplitude
amp_plot(X_CIR_filtered_amp,'Filtered CIR','t')

#plot filtered CIR Unwrapped Phase
phase_plot(X_CIR_filtered_phase,'Filtered CIR','t')



#convert filtered CIR to filtered CFR
X_filtered=np.fft.fft(X_CIR_filtered)
#calculate the amplitude and phase of CFR after filtering
X_filtered_amp = abs(X_filtered)
X_filtered_phase = np.unwrap(np.angle(X_filtered),axis=-1)

#plot filtered CFR Amplitude
amp_plot(X_filtered_amp,'Filtered CFR','s')

#plot filtered CFR Unwrapped Phase
phase_plot(X_filtered_phase,'Filtered CFR','s')


#%%
#find SFO
f=312500  #312.5k (HZ),subcarrier spacing
#define curve fitting variables
w=np.linspace(-3,-0,500) 
o=np.linspace(-0.000003,0.000003,500) 
sum_linear=np.zeros((len(w),len(o)))
W, O = np.meshgrid(w,o)


X_CIR_filtered_phase_T=np.transpose(X_CIR_filtered_phase,(0,2,1))

def linear_func1(W,O,k1,n1):
    temp = 2*np.pi*f*(k1+1)*O+W
    temp1 = np.zeros((nr*nt,500,500))
    for i in range(nr*nt):
        temp1[i] = X_CIR_filtered_phase_T[n1][k1][i]+temp
    return np.power(temp1,2)

def o_argmin(num_data):
    ind=[[0]*2]*56
    argmin_o=np.zeros(subcarrier)
    for k in range(subcarrier):
        sum_linear = sum(linear_func1(W,O,k,num_data))
        ind[k]=np.unravel_index(np.argmin(sum_linear),sum_linear.shape)
        argmin_o[k]=o[ind[k][1]]*(k+1)
    return argmin_o

X_sfo_phase = np.zeros((num_data, num_location,subcarrier))  

#calculte the CFR phase after sfo removal
for i in range(location[0]):
    X_sfo_phase[i] = X_filtered_phase[i] - o_argmin(i) * (2*np.pi*f)  
X_sfo_phase = np.unwrap(X_sfo_phase,axis=-1)   

#calculate the CFR after sfo removal    
X_sfo = X[:]
for i in range(location[0]):
    for j in range(nr*nt):
        for k in range(subcarrier):
            X_sfo[i][j][k] = cmath.rect(X_filtered_amp[i][j][k],X_sfo_phase[i][j][k])

#calculate the CFR amplitude after sfo removal
X_sfo_amp = abs(X_sfo)

#plot sfo removal CFR Amplitude
amp_plot(X_sfo_amp,'sfo removel CFR','s')

#plot sfo removla CFR Unwrapped Phase
phase_plot(X_sfo_phase,'sfo removal CFR','s')

#convert sfo removal CFR to sfo removal CIR and calculate their amplitude, phase
X_CIR_sfo = np.fft.ifft(X_sfo)
X_CIR_sfo_amp = abs(X_CIR_sfo)
X_CIR_sfo_phase = np.unwrap(np.angle(X_CIR_sfo),axis=-1)  

#plot sfo removal CIR Amplitude
amp_plot(X_CIR_sfo_amp,'sfo removal CIR','t')

#plot sfo removal CIR Unwrapped Phase
phase_plot(X_CIR_sfo_phase,'sfo removal CIR','t')

#%%

'''
W, O = np.meshgrid(w,o)
fig = plt.figure()
ax = plt.axes(projection='3d')
ax.contour3D(w,o,linear_func(W,O))
plt.show()
'''

#%%
#find STO
temp_sto = np.zeros((num_data,nr*nt))

#calculate estimate sto
def func3(n):
    return np.argmax(pow(abs(X_CIR_sfo[n]),2),axis=1)
 
for i in range(num_data):
    temp_sto[i]=func3(i)  
t=np.zeros(subcarrier)
for i in range(56):
    t[i]=i+1
        
sto = np.zeros((num_data,nr*nt,subcarrier))
for i in range(num_data):  
    for j in range(nr*nt):
        sto[i][j] = -2*np.pi*t/subcarrier*temp_sto[i][j]

#calculte the CFR phase after sto removal
X_sto_phase = X_sfo_phase - sto
X_sto_phase = np.unwrap(X_sto_phase,axis=-1)

#plot sto removal CFR Unwrapped Phase
phase_plot(X_sto_phase,'sto removal CFR','s')
X_sto = X[:]

#calculate the CFR after sto removal
for i in range(num_data):
    for j in range(nr*nt):
        for k in range(subcarrier):
            X_sto[i][j][k] = cmath.rect(X_sfo_amp[i][j][k],X_sto_phase[i][j][k])
X_sto_amp = abs(X_sto)

#plot sto removal CFR Amplitude
amp_plot(X_sto_amp,'sto removal CFR','s')

#convert sto removal CFR to sto removal CIR and calculate the amplitude, phase
X_CIR_sto = np.fft.ifft(X_sto)
X_CIR_sto_amp = abs(X_CIR_sto)
X_CIR_sto_phase = np.unwrap(np.angle(X_CIR_sto),axis=-1)

#plot sto removal CIR Amplitude
amp_plot(X_CIR_sto_amp,'sto removal CIR','t')

#plot sto removal CFR Unwrapped Phase
phase_plot(X_CIR_sto_phase,'sto removal CIR','t')


#%%
#find CFO

#56-dimentional H by conducting element-wise multiplication for 30 packets
H = pow(X_sto[50],1/30)
        
for i in range(51,80):
    H = H * pow(X_sto[i],1/30)

H_amp = abs(H)
H_phase = np.unwrap(np.angle(H),axis=-1)


#plot cfo removal CFR Amplitude
for i in range(nr):
    for j in range(nt):
        plt.title('cfo removal CFR')
        plt.xlabel('Subcarrier Index')
        plt.ylabel('Amplitude')
        plt.legend(title='RX{} TX{}'.format(i+1,j+1))
        plt.plot(H_amp[i*2+j],'b-.')
        plt.show()

#plot cfo removal CFR Unwrapped phase        
for i in range(nr):
    for j in range(nt):
        plt.title('cfo removal CFR')
        plt.xlabel('Subcarrier Index')
        plt.ylabel('Unwrapped Phase')
        plt.legend(title='RX{} TX{}'.format(i+1,j+1))
        plt.plot(H_phase[i*2+j],'b-.')
        plt.show()

#convert cfo removal CFR to cfo removal CIR and calculate the amplitude, phase
H_CIR = np.fft.ifft(H)
H_CIR_amp = abs(H_CIR)
H_CIR_phase = np.unwrap(np.angle(H_CIR),axis=-1)    


#plot cfo removal CIR Amplitude
for i in range(nr):
    for j in range(nt):
        plt.title('cfo removal CIR')
        plt.xlabel('Tap Index')
        plt.ylabel('Amplitude')
        plt.legend(title='RX{} TX{}'.format(i+1,j+1))
        plt.plot(H_CIR_amp[i*2+j],'b-.')
        plt.show()
 
#plot cfo removal CIR Unwrapped phase  
for i in range(nr):
    for j in range(nt):
        plt.title('cfo removal CIR')
        plt.xlabel('Tap Index')
        plt.ylabel('Unwrapped Phase')
        plt.legend(title='RX{} TX{}'.format(i+1,j+1))
        plt.plot(H_CIR_phase[i*2+j],'b-.')
        plt.show()


#%%    

'''
t=[0]*56
for i in range(56):
    t[i]=i+1
plt.xlabel('Time Taps')
plt.ylabel('filtered CIR Amplitude')
plt.stem(t,abs(X_CIR)[10][0])

plt.stem(t,X_CIR_amp[0][1])
plt.stem(t,X_CIR_amp[0][2])
plt.stem(t,X_CIR_amp[100][0])
'''

#%%
def plot_amp(text,text1,x):
    if text1 == 's' :
        text1 = 'Subcarrier'
    elif text1 == 't' :
        text1 = 'Tap'
    for i in range(nr):
        for j in range(nt):
            plt.xlabel('{} Index'.format(text1))
            plt.ylabel('{} Amplitude'.format(text))
            plt.legend(title='RX{} TX{}'.format(i+1,j+1))
            for k in range(100):
                plt.plot(x[k+500][i*2+j],'b-.')
            plt.show()

def plot_phase(text,text1,x):
    if text1 == 's' :
        text1 = 'Subcarrier'
    elif text1 == 't' :
        text1 = 'Tap'
    for i in range(nr):
        for j in range(nt):
            plt.xlabel('{} Index'.format(text1))
            plt.ylabel('{} Unwrapped Phase'.format(text))
            plt.legend(title='RX{} TX{}'.format(i+1,j+1))
            for k in range(100):
                plt.plot(x[k+500][i*2+j],'b-.')
            plt.show()
            
#%%
'''
#plot all the figure    

#plot original CFR Amplitude
amp_plot(X_amp,'Origianl CFR','s')

#plot original CFR Unwrapped Phase
phase_plot(X_phase,'Original CFR','s')

#plot original CIR Amplitude
amp_plot(X_CIR_amp,'Original CIR','t')

#plot original CIR Unwarpped Phase
phase_plot(X_CIR_phase,'Original CIR','t')

#plot filtered CFR Amplitude
amp_plot(X_filtered_amp,'Origianl CFR','s')

#plot filtered CFR Unwrapped Phase
phase_plot(X_filtered_phase,'Original CFR','s')

#plot filtered CIR Amplitude
amp_plot(X_CIR_filtered_amp,'Original CIR','t')

#plot filtered CIR Unwarpped Phase
phase_plot(X_CIR_filtered_phase,'Original CIR','t')

#plot sfo removal CFR Amplitude
amp_plot(X_sfo_amp,'sfo removal CFR','s')

#plot sfo removal CFR Unwrapped Phase
phase_plot(X_sfo_phase,'sfo removal CFR','s')

#plot sfo removal CIR Amplitude
amp_plot(X_CIR_sfo_amp,'sfo removal CIR','t')

#plot sfo removal CIR Unwarpped Phase
phase_plot(X_CIR_sfo_phase,'sfo removal CIR','t')

#plot sto removal CFR Amplitude
amp_plot(X_sto_amp,'sto removal CFR','s')

#plot sto removal CFR Unwrapped Phase
phase_plot(X_sto_phase,'sto removal CFR','s')

#plot sto removal CIR Amplitude
amp_plot(X_CIR_sto_amp,'sto removal CIR','t')

#plot sto removal CIR Unwarpped Phase
phase_plot(X_CIR_sto_phase,'sto removal CIR','t')

#plot cfo removal CFR Amplitude
amp_plot(H_amp,'cfo removal CFR','s')

#plot cfo removal CFR Unwrapped Phase
phase_plot(H_phase,'cfo removal','s')

#plot cfo removal CIR Amplitude
amp_plot(H_CIR_amp,'cfo removal','t')

#plot cfo removal CIR Unwarpped Phase
phase_plot(H_CIR_phase,'cfo removal','t')
'''
