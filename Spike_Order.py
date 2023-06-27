import matplotlib.pyplot as plt
import pyspike as spk
from numpy import *
from pylab import *
from pyspike.isi_lengths import default_thresh
from pyspike import SpikeTrain
from matplotlib import pyplot
from pyspike import spike_directionality
def f_all_trains(spikes):
    num_trains = len(spikes)
    num_spikes_per_train = [len(train) for train in spikes]
    dummy = [0] + num_spikes_per_train
    all_indy = [0] * sum(num_spikes_per_train)
    
    for trc in range(num_trains):
        start_idx = sum(dummy[0:trc+1])
        end_idx = start_idx + num_spikes_per_train[trc]
        all_indy[start_idx:end_idx] = [trc+1] * num_spikes_per_train[trc]
    
    sp_flat = [spike for train in spikes for spike in train]
    sp_indy = sorted(range(len(sp_flat)), key=lambda i: sp_flat[i])
    all_trains = [all_indy[idx] for idx in sp_indy]
    
    return all_trains
tmax=1
tmin = 0
spike_trains = spk.load_spike_trains_from_txt("./examples/PySpike_testdata5.txt", edges=(tmin, tmax))
print(f_all_trains(spike_trains))

all_trains = f_all_trains(spike_trains)
def spike_order_profile(spike_trains, all_trains):
    num_trains = len(spike_trains)
    st = [len(train) for train in spike_trains]
    ss = np.zeros((num_trains, len(all_trains)))
    x =0
    for i in range(num_trains):
        for j in range(i+1, num_trains):
            train1 = spike_trains[i]
            train2 = spike_trains[j]
            for z in range(len(all_trains)):
                if all_trains[z] == i+1 or all_trains[z] == j+1:
                    ss[x, z] = 1
                    
            x = x+1            
        
    return ss

print(spike_order_profile(spike_trains, all_trains))
def find_coincidences(spike_train1, spike_train2):
    C = []
    for i in range(len(spike_train1)):
        if i == 0:
            to_i = (spike_train1[i+1] - spike_train1[i]) / 2
        elif i == len(spike_train1) - 1:
            to_i = (spike_train1[i] - spike_train1[i-1]) / 2
        else:
            to_i = min((spike_train1[i+1] - spike_train1[i]), (spike_train1[i] - spike_train1[i-1])) / 2
        A = []
        for j in range(len(spike_train2)):
            if j == 0:
                to_j = (spike_train2[j+1] - spike_train2[j]) / 2
            elif j == len(spike_train2) - 1:
                to_j = (spike_train2[j] - spike_train2[j-1]) / 2
            else: 
                to_j = min((spike_train2[j+1] - spike_train2[j]), (spike_train2[j] - spike_train2[j-1])) / 2
            to_ij = min(to_i, to_j)
            if abs(spike_train1[i] - spike_train2[j]) < to_ij:
                A.append(1)
        if not A:
            C.append(0)
        else:
            C.append(1)
    return C

spike_trains = spk.load_spike_trains_from_txt("./examples/PySpike_testdata5.txt", edges=(0, 10))
spike_train1 = spike_trains[2]
spike_train2 = spike_trains[1]
print(find_coincidences(spike_train1, spike_train2))

def ss_plot(spike_trains,all_trains):
    all_trains = f_all_trains(spike_trains)
    num_trains = len(spike_trains)
    st = [len(train) for train in spike_trains]
    ss = spike_order_profile(spike_trains, all_trains)
    x = 0 
    y = 0
    a = 0
    while a < num_trains : 
        for i in range(num_trains):
            for j in range(i+1,num_trains):
                spike_train1 = spike_trains[i]
                spike_train2 = spike_trains[j]
                mat1 = find_coincidences(spike_train1, spike_train2)
                mat2 = find_coincidences(spike_train2, spike_train1)
                for z in range(len(all_trains)):
                    if all_trains[z] == i+1 and ss[a, z] == 1:
                        ss[a, z] = mat1[x]
                        x= x+1
                    elif all_trains[z] == j+1 and ss[a, z] == 1:
                        ss[a, z] = mat2[y]
                        y = y+1
                    else:
                        ss[a, z] = 0
                a = a+1
                x = 0
                y = 0
    return ss
print(ss_plot(spike_trains, all_trains))  
ss = ss_plot(spike_trains, all_trains)
def so_matrix(matrix):
    transformed_matrix = []
    for row in matrix:
        transformed_row = []
        count = 0
        for element in row:
            if element == 1:
                count += 1
                if count % 2 == 0:
                    transformed_row.append(-1)
                else:
                    transformed_row.append(element)
            else:
                transformed_row.append(element)
        
        transformed_matrix.append(transformed_row)
    
    return transformed_matrix




print(so_matrix(ss))
