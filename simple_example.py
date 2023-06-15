import matplotlib.pyplot as plt
import pyspike as spk
from numpy import *
from pylab import *
from pyspike.isi_lengths import default_thresh
from pyspike import SpikeTrain
from matplotlib import pyplot
from pyspike import spike_directionality

measures   =  16   # +1:ISI,+2:SPIKE,+4:RI-SPIKE,+8:SPIKE-Synchro,+16:SpikeOrder
a_measures =  0   # +1:A-ISI,+32:A-SPIKE,+64:A-RI-SPIKE,+128:A-SPIKE-Synchro
showing =      15   # +1:spike trains,+2:distances,+4:profiles,+8:matrices
plotting =     15   # +1:spike trains,+2:distances,+4:profiles,+8:matrices
sorting =       1   # 0: Unsorted order; 1: Sorted order

tmin=0;
tmax=1000;
threshold=1000;

dataset=5;     # choose one of the four main examples used so far, the parameter is just the different number of spike trains of this examples (2, 4, 6, 40)
print("\n\n\ndataset: %3i" % (dataset))

if dataset == 2:
    tmax=100;
    spike_trains = []
    spike_trains.append(spk.SpikeTrain([12, 16, 28, 32, 44, 48, 60, 64, 76, 80], [tmin, tmax]))
    spike_trains.append(spk.SpikeTrain([8, 20, 24, 36, 40, 52, 56, 68, 72, 84], [tmin, tmax]))
elif dataset == 3:
    tmax=10
    spike_trains = []
    spike_trains.append(spk.SpikeTrain([0, 1, 5, 8, 10], [tmin, tmax]))
    spike_trains.append(spk.SpikeTrain([0, 3, 7, 10], [tmin, tmax]))
else:
    if dataset == 4:
        tmax=1000;
        spike_trains = spk.load_spike_trains_from_txt("./examples/PySpike_testdata4.txt", edges=(tmin, tmax))
    else:
        if dataset == 5:
            tmax=1
            spike_trains = spk.load_spike_trains_from_txt("./examples/PySpike_testdata5.txt", edges=(tmin, tmax))
        else:
            if dataset == 6:
                tmax=1000;
                spike_trains = spk.load_spike_trains_from_txt("./examples/PySpike_testdata6.txt", edges=(tmin, tmax))
                spike_trains[3]=spk.SpikeTrain([], [spike_trains[0].t_start, spike_trains[0].t_end])
                spike_trains[4]=spk.SpikeTrain([], [spike_trains[0].t_start, spike_trains[0].t_end])
            else:
                if dataset == 40:
                    tmax=4000;
                    spike_trains = spk.load_spike_trains_from_txt("./examples/PySpike_testdata.txt", edges=(tmin, tmax))
    
num_trains = len (spike_trains);
print("\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\nnum_trains: %3i" % (num_trains))



if showing % 2 > 0:                                                    # Spike trains
    for i in arange (num_trains):
        print("\nSpike Train %3i:" % (i+1))
        for j in arange ( len(spike_trains[i]) ):
            print("%i %.8f" % (j+1, spike_trains[i][j]))
    print("\n")

if plotting % 2 > 0:
    plt.figure(figsize=(17, 10), dpi=80)
    plt.title("Rasterplot", color='k', fontsize=24)
    plt.xlabel('Time', color='k', fontsize=18)
    plt.ylabel('Spike Trains', color='k', fontsize=18)
    plt.axis([tmin-0.05*(tmax-tmin), tmax+0.05*(tmax-tmin), 0, num_trains+1])
    plt.xticks(arange(tmin,tmax+1,1000), fontsize=14)
    plt.yticks(arange(num_trains)+1, fontsize=14)
    plt.plot((tmin, tmax), (0.5, 0.5), ':', color='k', linewidth=1)
    plt.plot((tmin, tmax), (num_trains+0.5, num_trains+0.5), ':', color='k', linewidth=1)
    plt.plot((tmin, tmin), (0.5, num_trains+0.5), ':', color='k', linewidth=1)
    plt.plot((tmax, tmax), (0.5, num_trains+0.5), ':', color='k', linewidth=1)
    for i in arange (num_trains):
        for j in arange (len (spike_trains[i])):
            plt.plot((spike_trains[i][j], spike_trains[i][j]), (i+0.5, i+1.5), '-', color='k', linewidth=1)



if measures % 2 > 0:                                                    # ISI-distance
    if showing % 16 > 1 or plotting % 16 > 1:
        isi_distance = spk.isi_distance(spike_trains)
        if showing % 4 > 1:
           print("\nISI-Distance: %.8f\n" % isi_distance)

    if showing % 8 > 3 or plotting % 8 > 3:
        isi_profile = spk.isi_profile(spike_trains)
        x, y = isi_profile.get_plottable_data()

        if showing % 8 > 3:
            num_xy = len (x);
            print("\nISI-Distance-Profile:\n")
            print("x            y\n")
            for i in arange (num_xy):
                print("%.8f   %.8f\n" % (x[i], y[i]), end = "")
            print("\n")

        if plotting % 8 > 3:
            plt.figure(figsize=(17, 10), dpi=80)
            plt.plot(x, y, '-k')
            plt.axis([tmin-0.05*(tmax-tmin), tmax+0.05*(tmax-tmin), -0.05, 1.05])
            plt.plot((tmin, tmax), (0, 0), ':', color='k', linewidth=1)
            plt.plot((tmin, tmax), (1, 1), ':', color='k', linewidth=1)
            plt.plot((tmin, tmin), (0, 1), ':', color='k', linewidth=1)
            plt.plot((tmax, tmax), (0, 1), ':', color='k', linewidth=1)
            plt.plot((tmin, tmax), (isi_distance, isi_distance), '--', color='k', linewidth=1)
            plt.xticks(fontsize=12)
            plt.yticks(fontsize=12) # rotation=90
            plt.title("ISI-distance-profile   (ISI-distance = %.8f)" % (isi_distance), color='k', fontsize=24)
            plt.xlabel('Time', color='k', fontsize=18)
            plt.ylabel('ISI', color='k', fontsize=18)

    if showing % 16 > 7 or plotting % 16 > 7:
        isi_distance_mat = spk.isi_distance_matrix(spike_trains)
        if showing % 16 > 7:
            print("\nISI-Distance-Matrix:")
            for i in arange (num_trains):
                print("\n%i     " % (i+1), end = "")
                for j in arange (num_trains):
                    print("%.8f " % (isi_distance_mat[i][j]), end = "")
            print("\n")

        if plotting % 16 > 7:
            plt.figure(figsize=(17, 10), dpi=80)
            plt.imshow(isi_distance_mat, interpolation='none')
            plt.title("ISI-distance-matrix   (ISI-distance = %.8f)" % (isi_distance), color='k', fontsize=24)
            plt.xlabel('Spike Trains', color='k', fontsize=18)
            plt.ylabel('Spike Trains', color='k', fontsize=18)
            plt.xticks(arange(num_trains),arange(num_trains)+1, fontsize=14)
            plt.yticks(arange(num_trains),arange(num_trains)+1, fontsize=14)
            plt.jet()
            plt.colorbar()



if measures % 4 > 1:                                                    # SPIKE-distance
    if showing % 16 > 1 or plotting % 16 > 1:
        spike_distance = spk.spike_distance(spike_trains)
        if showing % 4 > 1:
            print("\nSPIKE-Distance: %.8f\n" % spike_distance)

    if showing % 8 > 3 or plotting % 8 > 3:
        spike_profile = spk.spike_profile(spike_trains)
        x, y = spike_profile.get_plottable_data()

        if showing % 8 > 3:
            num_xy = len (x);
            print("\nSPIKE-Distance-Profile:\n")
            print("x            y\n")
            for i in arange (num_xy):
                print("%.8f   %.8f\n" % (x[i], y[i]), end = "")
            print("\n")

        if plotting % 8 > 3:
            plt.figure(figsize=(17, 10), dpi=80)
            plt.plot(x, y, '-k')
            plt.axis([tmin-0.05*(tmax-tmin), tmax+0.05*(tmax-tmin), -0.05, 1.05])
            plt.plot((tmin, tmax), (0, 0), ':', color='k', linewidth=1)
            plt.plot((tmin, tmax), (1, 1), ':', color='k', linewidth=1)
            plt.plot((tmin, tmin), (0, 1), ':', color='k', linewidth=1)
            plt.plot((tmax, tmax), (0, 1), ':', color='k', linewidth=1)
            plt.plot((tmin, tmax), (spike_distance, spike_distance), '--', color='k', linewidth=1)
            plt.xticks(fontsize=12)
            plt.yticks(fontsize=12) # rotation=90
            plt.title("SPIKE-distance-profile   (SPIKE-distance = %.8f)" % (spike_distance), color='k', fontsize=24)
            plt.xlabel('Time', color='k', fontsize=18)
            plt.ylabel('SPIKE', color='k', fontsize=18)

    if showing % 16 > 7 or plotting % 16 > 7:
        spike_distance_mat = spk.spike_distance_matrix(spike_trains)
        if showing % 16 > 7:
            print("\nSPIKE-Distance-Matrix:")
            for i in arange (num_trains):
                print("\n%i     " % (i+1), end = "")
                for j in arange (num_trains):
                    print("%.8f " % (spike_distance_mat[i][j]), end = "")
            print("\n")

        if plotting % 16 > 7:
            plt.figure(figsize=(17, 10), dpi=80)
            plt.imshow(spike_distance_mat, interpolation='none')
            plt.title("SPIKE-distance-matrix   (SPIKE-distance = %.8f)" % (spike_distance), color='k', fontsize=24)
            plt.xlabel('Spike Trains', color='k', fontsize=18)
            plt.ylabel('Spike Trains', color='k', fontsize=18)
            plt.xticks(arange(num_trains),arange(num_trains)+1, fontsize=14)
            plt.yticks(arange(num_trains),arange(num_trains)+1, fontsize=14)
            plt.jet()
            plt.colorbar()



if measures % 8 > 3:                                                    # RI-SPIKE-distance
    if showing % 16 > 1 or plotting % 16 > 1:
        ri_spike_distance = spk.spike_distance(spike_trains, RI=True)
        if showing % 4 > 1:
            print("\nRI-SPIKE-Distance: %.8f\n" % ri_spike_distance)
      
    if showing % 8 > 3 or plotting % 8 > 3:
        ri_spike_profile = spk.spike_profile(spike_trains, RI=True)
        x, y = ri_spike_profile.get_plottable_data()

        if showing % 8 > 3:
            num_xy = len (x);
            print("\nRI-SPIKE-Distance-Profile:\n")
            print("x            y\n")
            for i in arange (num_xy):
                print("%.8f   %.8f\n" % (x[i], y[i]), end = "")
            print("\n")

        if plotting % 8 > 3:
            plt.figure(figsize=(17, 10), dpi=80)
            plt.plot(x, y, '-k')
            plt.axis([tmin-0.05*(tmax-tmin), tmax+0.05*(tmax-tmin), -0.05, 1.05])
            plt.plot((tmin, tmax), (0, 0), ':', color='k', linewidth=1)
            plt.plot((tmin, tmax), (1, 1), ':', color='k', linewidth=1)
            plt.plot((tmin, tmin), (0, 1), ':', color='k', linewidth=1)
            plt.plot((tmax, tmax), (0, 1), ':', color='k', linewidth=1)
            plt.plot((tmin, tmax), (ri_spike_distance, ri_spike_distance), '--', color='k', linewidth=1)
            plt.xticks(fontsize=12)
            plt.yticks(fontsize=12) # rotation=90
            plt.title("RI-SPIKE-distance-profile   (RI-SPIKE-distance = %.8f)" % (ri_spike_distance), color='k', fontsize=24)
            plt.xlabel('Time', color='k', fontsize=18)
            plt.ylabel('RI-SPIKE', color='k', fontsize=18)

    if showing % 16 > 7 or plotting % 16 > 7:
        ri_spike_distance_mat = spk.spike_distance_matrix(spike_trains, RI=True)
        if showing % 16 > 7:
            print("\nRI-SPIKE-Distance-Matrix:")
            for i in arange (num_trains):
                print("\n%i     " % (i+1), end = "")
                for j in arange (num_trains):
                    print("%.8f " % (ri_spike_distance_mat[i][j]), end = "")
            print("\n")

        if plotting % 16 > 7:
            plt.figure(figsize=(17, 10), dpi=80)
            plt.imshow(ri_spike_distance_mat, interpolation='none')
            plt.title("RI-SPIKE-distance-matrix   (RI-SPIKE-distance = %.8f)" % (ri_spike_distance), color='k', fontsize=24)
            plt.xlabel('Spike Trains', color='k', fontsize=18)
            plt.ylabel('Spike Trains', color='k', fontsize=18)
            plt.xticks(arange(num_trains),arange(num_trains)+1, fontsize=14)
            plt.yticks(arange(num_trains),arange(num_trains)+1, fontsize=14)
            plt.jet()
            plt.colorbar()



if measures % 16 > 7:                                                    # SPIKE-synchro
    if showing % 16 > 1 or plotting % 16 > 1:
        spike_synchro = spk.spike_sync(spike_trains)
        if showing % 4 > 1:
            print("\nSPIKE-Synchro: %.8f\n" % spike_synchro)

    if showing % 8 > 3 or plotting % 8 > 3:
        spike_sync_profile = spk.spike_sync_profile(spike_trains)
        x, y = spike_sync_profile.get_plottable_data()

        if showing % 8 > 3:
            num_xy = len (x);
            print("\nSPIKE-Synchro-Profile:\n")
            print("x            y\n")
            for i in arange (num_xy):
                print("%.8f   %.8f\n" % (x[i], y[i]), end = "")
            print("\n")

        if plotting % 8 > 3:
            plt.figure(figsize=(17, 10), dpi=80)
            plt.plot(x, y, '-k')
            plt.axis([tmin-0.05*(tmax-tmin), tmax+0.05*(tmax-tmin), -0.05, 1.05])
            plt.plot((tmin, tmax), (0, 0), ':', color='k', linewidth=1)
            plt.plot((tmin, tmax), (1, 1), ':', color='k', linewidth=1)
            plt.plot((tmin, tmin), (0, 1), ':', color='k', linewidth=1)
            plt.plot((tmax, tmax), (0, 1), ':', color='k', linewidth=1)
            plt.plot((tmin, tmax), (spike_synchro, spike_synchro), '--', color='k', linewidth=1)
            plt.xticks(fontsize=12)
            plt.yticks(fontsize=12) # rotation=90
            plt.title("SPIKE-Synchro-profile   (SPIKE-Synchro = %.8f)" % (spike_synchro), color='k', fontsize=24)
            plt.xlabel('Time', color='k', fontsize=18)
            plt.ylabel('SPIKE-Synchro', color='k', fontsize=18)

    if showing % 16 > 7 or plotting % 16 > 7:
        spike_synchro_mat = spk.spike_sync_matrix(spike_trains)
        if showing % 8 > 3:
            print("\nSPIKE-Synchro-Matrix:")
            for i in arange (num_trains):
                print("\n%i     " % (i+1), end = "")
                for j in arange (num_trains):
                    print("%.8f " % (spike_synchro_mat[i][j]), end = "")
            print("\n")

        if plotting % 16 > 7:
            plt.figure(figsize=(17, 10), dpi=80)
            plt.imshow(spike_synchro_mat, interpolation='none')
            plt.title("SPIKE-Synchro-matrix   (SPIKE-Synchro = %.8f)" % (spike_synchro), color='k', fontsize=24)
            plt.xlabel('Spike Trains', color='k', fontsize=18)
            plt.ylabel('Spike Trains', color='k', fontsize=18)
            plt.xticks(arange(num_trains),arange(num_trains)+1, fontsize=14)
            plt.yticks(arange(num_trains),arange(num_trains)+1, fontsize=14)
            plt.jet()
            plt.colorbar()

if  measures % 32 > 15:                                                   #SPIKE ORDER
     if showing % 16 > 1 or plotting % 16 > 1:
        spike_order = spk.spike_train_order(spike_trains)
        if showing % 4 > 1:
            print("\nSPIKE-Order: %.8f\n" % spike_order)

     if showing % 8 > 3 or plotting % 8 > 3:
        spike_order_profile = spk.spike_train_order_profile(spike_trains)
        x, y = spike_order_profile.get_plottable_data()

        if showing % 8 > 3:
            num_xy = len (x);
            print("\nSPIKE-Order-Profile:\n")
            print("x            y\n")
            for i in arange (num_xy):
                print("%.8f   %.8f\n" % (x[i], y[i]), end = "")
            print("\n")

        if plotting % 8 > 3:
            plt.figure(figsize=(17, 10), dpi=80)
            plt.plot(x, y, '-k')
            plt.axis([tmin-0.05*(tmax-tmin), tmax+0.05*(tmax-tmin), -1.05, 1.05])
            plt.plot((tmin, tmax), (0, 0), ':', color='k', linewidth=1)
            plt.plot((tmin, tmax), (1, 1), ':', color='k', linewidth=1)
            plt.plot((tmin, tmin), (0, 1), ':', color='k', linewidth=1)
            plt.plot((tmax, tmax), (0, 1), ':', color='k', linewidth=1)
            plt.plot((tmin, tmax), (spike_order, spike_order), '--', color='k', linewidth=1)
            plt.xticks(fontsize=12)
            plt.yticks(fontsize=12) # rotation=90
            plt.title("SPIKE-Order-profile   (SPIKE-Order = %.8f)" % (spike_order), color='k', fontsize=24)
            plt.xlabel('Time', color='k', fontsize=18)
            plt.ylabel('SPIKE-Order', color='k', fontsize=18)

     if showing % 16 > 7 or plotting % 16 > 7:
        spike_order_mat = spk.spike_directionality_matrix(spike_trains)
        if showing % 8 > 3:
            print("\nSPIKE-Order-Matrix:")
            for i in arange (num_trains):
                print("\n%i     " % (i+1), end = "")
                for j in arange (num_trains):
                    print("%.8f " % (spike_order_mat[i][j]), end = "")
            print("\n")

        if plotting % 16 > 7:
            plt.figure(figsize=(17, 10), dpi=80)
            plt.imshow(spike_order_mat, interpolation='none')
            plt.title("SPIKE-Order-matrix   (SPIKE-Order = %.8f)" % (spike_order), color='k', fontsize=24)
            plt.xlabel('Spike Trains', color='k', fontsize=18)
            plt.ylabel('Spike Trains', color='k', fontsize=18)
            plt.xticks(arange(num_trains),arange(num_trains)+1, fontsize=14)
            plt.yticks(arange(num_trains),arange(num_trains)+1, fontsize=14)
            plt.jet()
            plt.colorbar()

if sorting % 2 > 0:                                                       #Sorting
    D_init = spk.spike_directionality_matrix(spike_trains)
    phi, _ = spk.optimal_spike_train_sorting(spike_trains)
    F_opt = spk.spike_train_order(spike_trains, indices=phi)
    print("Synfire Indicator of optimized spike train sorting:", F_opt)

    D_opt = spk.permutate_matrix(D_init, phi)

    plt.figure()
    plt.imshow(D_init)
    plt.title("Initial Directionality Matrix")

    plt.figure()
    plt.imshow(D_opt)
    plt.title("Optimized Directionality Matrix")

    plt.show()

    
if a_measures % 2 > 0:                                                    # A-ISI-distance
    if showing % 16 > 1 or plotting % 16 > 1:
        #adaptive_isi_distance_thr_0 = spk.isi_distance(spike_trains, MRTS=0)        # falls back to original ISI-distance
        #print("\nA-ISI-Distance_thr_0: %.8f\n" % adaptive_isi_distance_thr_0)
        adaptive_isi_distance_thr = spk.isi_distance(spike_trains, MRTS=threshold)
        adaptive_isi_distance_auto = spk.isi_distance(spike_trains, MRTS='auto')        
        spike_train_list = []
        for j in arange (num_trains):
            spike_train_list.append(spike_trains[j])
        auto_threshold = default_thresh(spike_train_list)
        if showing % 4 > 1:
            print("\nA-ISI-Distance-Thr: %.8f\n" % adaptive_isi_distance_thr)
            print("\nA-ISI-Distance-Auto: %.8f\n" % adaptive_isi_distance_auto)
            print('\nAuto-Threshold: %.8f\n' % auto_threshold)

        #adaptive_isi_distance_auto_extr_thr = spk.isi_distance(spike_trains, MRTS=Thresh)
        #print("A-ISI-Distance-Auto_extr_thr: %.8f\n" % adaptive_isi_distance_auto_extr_thr)

        #num_spike_train_list = len (spike_train_list);
        #print("\n\nnum_spike_train_list: %3i\n" % (num_spike_train_list))

        #for j in arange (num_spike_train_list):
        #    for i in arange (len(spike_train_list[j])):
        #        print("%i %i %.8f" % (i+1, j+1, spike_train_list[j][i]))
        #print("\n")

    if showing % 8 > 3 or plotting % 8 > 3:
        adaptive_isi_profile = spk.isi_profile(spike_trains, MRTS=threshold)
        x, y = adaptive_isi_profile.get_plottable_data()

        if showing % 8 > 3:
            num_xy = len (x);
            print("\nA-ISI-Distance-Profile:\n")
            print("x            y\n")
            for i in arange (num_xy):
                print("%.8f   %.8f\n" % (x[i], y[i]), end = "")
            print("\n")

        if plotting % 8 > 3:
            plt.figure(figsize=(17, 10), dpi=80)
            plt.plot(x, y, '-k')
            plt.axis([tmin-0.05*(tmax-tmin), tmax+0.05*(tmax-tmin), -0.05, 1.05])
            plt.plot((tmin, tmax), (0, 0), ':', color='k', linewidth=1)
            plt.plot((tmin, tmax), (1, 1), ':', color='k', linewidth=1)
            plt.plot((tmin, tmin), (0, 1), ':', color='k', linewidth=1)
            plt.plot((tmax, tmax), (0, 1), ':', color='k', linewidth=1)
            plt.plot((tmin, tmax), (adaptive_isi_distance_thr, adaptive_isi_distance_thr), '--', color='k', linewidth=1)
            plt.xticks(fontsize=12)
            plt.yticks(fontsize=12) # rotation=90
            plt.title("A-ISI-distance-profile   (A-ISI-distance = %.8f)" % (adaptive_isi_distance_thr), color='k', fontsize=24)
            plt.xlabel('Time', color='k', fontsize=18)
            plt.ylabel('A-ISI', color='k', fontsize=18)

    if showing % 16 > 7 or plotting % 16 > 7:
        adaptive_isi_distance_mat = spk.isi_distance_matrix(spike_trains, MRTS=threshold)
        if showing % 16 > 7:
            print("\nA-ISI-Distance-Matrix:")
            for i in arange (num_trains):
                print("\n%i     " % (i+1), end = "")
                for j in arange (num_trains):
                    print("%.8f " % (adaptive_isi_distance_mat[i][j]), end = "")
            print("\n")

        if plotting % 16 > 7:
            plt.figure(figsize=(17, 10), dpi=80)
            plt.imshow(adaptive_isi_distance_mat, interpolation='none')
            plt.title("A-ISI-distance-matrix   (A-ISI-distance = %.8f)" % (adaptive_isi_distance_thr), color='k', fontsize=24)
            plt.xlabel('Spike Trains', color='k', fontsize=18)
            plt.ylabel('Spike Trains', color='k', fontsize=18)
            plt.xticks(arange(num_trains),arange(num_trains)+1, fontsize=14)
            plt.yticks(arange(num_trains),arange(num_trains)+1, fontsize=14)
            plt.jet()
            plt.colorbar()




if a_measures % 4 > 1:                                                    # A-SPIKE-distance
    if showing % 16 > 1 or plotting % 16 > 1:
        adaptive_spike_distance_thr = spk.spike_distance(spike_trains, MRTS=threshold)
        adaptive_spike_distance_auto = spk.spike_distance(spike_trains, MRTS='auto')        
        if showing % 4 > 1:
            #print('\nThreshold: %.8f\n'%threshold)
            print("\nA-SPIKE-Distance-Thr: %.8f\n" % adaptive_spike_distance_thr)
            print("\nA-SPIKE-Distance-Auto: %.8f\n" % adaptive_spike_distance_auto)

    if showing % 8 > 3 or plotting % 8 > 3:
        adaptive_spike_profile = spk.spike_profile(spike_trains, MRTS=threshold)
        x, y = adaptive_spike_profile.get_plottable_data()

        if showing % 8 > 3:
            num_xy = len (x);
            print("\nA-SPIKE-Distance-Profile:\n")
            print("x            y\n")
            for i in arange (num_xy):
                print("%.8f   %.8f\n" % (x[i], y[i]), end = "")
            print("\n")

        if plotting % 8 > 3:
            plt.figure(figsize=(17, 10), dpi=80)
            plt.plot(x, y, '-k')
            plt.axis([tmin-0.05*(tmax-tmin), tmax+0.05*(tmax-tmin), -0.05, 1.05])
            plt.plot((tmin, tmax), (0, 0), ':', color='k', linewidth=1)
            plt.plot((tmin, tmax), (1, 1), ':', color='k', linewidth=1)
            plt.plot((tmin, tmin), (0, 1), ':', color='k', linewidth=1)
            plt.plot((tmax, tmax), (0, 1), ':', color='k', linewidth=1)
            plt.plot((tmin, tmax), (adaptive_spike_distance_thr, adaptive_spike_distance_thr), '--', color='k', linewidth=1)
            plt.xticks(fontsize=12)
            plt.yticks(fontsize=12) # rotation=90
            plt.title("A-SPIKE-distance-profile   (A-SPIKE-distance = %.8f)" % (adaptive_spike_distance_thr), color='k', fontsize=24)
            plt.xlabel('Time', color='k', fontsize=18)
            plt.ylabel('SPIKE', color='k', fontsize=18)

    if showing % 16 > 7 or plotting % 16 > 7:
        adaptive_spike_distance_mat = spk.spike_distance_matrix(spike_trains, MRTS=threshold)
        if showing % 16 > 7:
            print("\nA-SPIKE-Distance-Matrix:")
            for i in arange (num_trains):
                print("\n%i     " % (i+1), end = "")
                for j in arange (num_trains):
                    print("%.8f " % (adaptive_spike_distance_mat[i][j]), end = "")
            print("\n")

        if plotting % 16 > 7:
            plt.figure(figsize=(17, 10), dpi=80)
            plt.imshow(adaptive_spike_distance_mat, interpolation='none')
            plt.title("A-SPIKE-distance-matrix   (A-SPIKE-distance = %.8f)" % (adaptive_spike_distance_thr), color='k', fontsize=24)
            plt.xlabel('Spike Trains', color='k', fontsize=18)
            plt.ylabel('Spike Trains', color='k', fontsize=18)
            plt.xticks(arange(num_trains),arange(num_trains)+1, fontsize=14)
            plt.yticks(arange(num_trains),arange(num_trains)+1, fontsize=14)
            plt.jet()
            plt.colorbar()




if a_measures % 8 > 3:                                                    # A-RI-SPIKE-distance
    if showing % 16 > 1 or plotting % 16 > 1:
        adaptive_ri_spike_distance_thr = spk.spike_distance(spike_trains, MRTS=threshold, RI=True)
        adaptive_ri_spike_distance_auto = spk.spike_distance(spike_trains, MRTS='auto', RI=True)        
        if showing % 4 > 1:
            print("\nA-RI-SPIKE-Distance-Thr: %.8f\n" % adaptive_ri_spike_distance_thr)
            print("\nA-RI-SPIKE-Distance-Auto: %.8f\n" % adaptive_ri_spike_distance_auto)
      
    if showing % 8 > 3 or plotting % 8 > 3:
        adaptive_ri_spike_profile = spk.spike_profile(spike_trains, MRTS=threshold, RI=True)
        x, y = adaptive_ri_spike_profile.get_plottable_data()

        if showing % 8 > 3:
            num_xy = len (x);
            print("\nA-RI-SPIKE-Distance-Profile:\n")
            print("x            y\n")
            for i in arange (num_xy):
                print("%.8f   %.8f\n" % (x[i], y[i]), end = "")
            print("\n")

        if plotting % 8 > 3:
            plt.figure(figsize=(17, 10), dpi=80)
            plt.plot(x, y, '-k')
            plt.axis([tmin-0.05*(tmax-tmin), tmax+0.05*(tmax-tmin), -0.05, 1.05])
            plt.plot((tmin, tmax), (0, 0), ':', color='k', linewidth=1)
            plt.plot((tmin, tmax), (1, 1), ':', color='k', linewidth=1)
            plt.plot((tmin, tmin), (0, 1), ':', color='k', linewidth=1)
            plt.plot((tmax, tmax), (0, 1), ':', color='k', linewidth=1)
            plt.plot((tmin, tmax), (adaptive_ri_spike_distance_thr, adaptive_ri_spike_distance_thr), '--', color='k', linewidth=1)
            plt.xticks(fontsize=12)
            plt.yticks(fontsize=12) # rotation=90
            plt.title("A-RI-SPIKE-distance-profile   (A-RI-SPIKE-distance = %.8f)" % (adaptive_ri_spike_distance_thr), color='k', fontsize=24)
            plt.xlabel('Time', color='k', fontsize=18)
            plt.ylabel('A-RI-SPIKE', color='k', fontsize=18)

    if showing % 16 > 7 or plotting % 16 > 7:
        adaptive_ri_spike_distance_mat = spk.spike_distance_matrix(spike_trains, MRTS=threshold, RI=True)
        if showing % 16 > 7:
            print("\nA-RI-SPIKE-Distance-Matrix:")
            for i in arange (num_trains):
                print("\n%i     " % (i+1), end = "")
                for j in arange (num_trains):
                    print("%.8f " % (adaptive_ri_spike_distance_mat[i][j]), end = "")
            print("\n")

        if plotting % 16 > 7:
            plt.figure(figsize=(17, 10), dpi=80)
            plt.imshow(adaptive_ri_spike_distance_mat, interpolation='none')
            plt.title("A-RI-SPIKE-distance-matrix   (A-RI-SPIKE-distance = %.8f)" % (adaptive_ri_spike_distance_thr), color='k', fontsize=24)
            plt.xlabel('Spike Trains', color='k', fontsize=18)
            plt.ylabel('Spike Trains', color='k', fontsize=18)
            plt.xticks(arange(num_trains),arange(num_trains)+1, fontsize=14)
            plt.yticks(arange(num_trains),arange(num_trains)+1, fontsize=14)
            plt.jet()
            plt.colorbar()



if a_measures % 16 > 7:                                                    # A-SPIKE-synchro
    if showing % 16 > 1 or plotting % 16 > 1:
        adaptive_spike_synchro_thr = spk.spike_sync(spike_trains, MRTS=threshold)
        adaptive_spike_synchro_auto = spk.spike_sync(spike_trains, MRTS='auto')
        if showing % 4 > 1:
            print("\nA-SPIKE-Synchro-Thr: %.8f\n" % adaptive_spike_synchro_thr)
            print("\nA-SPIKE-Synchro-Auto: %.8f\n" % adaptive_spike_synchro_auto)

    if showing % 8 > 3 or plotting % 8 > 3:
        adaptive_spike_sync_profile = spk.spike_sync_profile(spike_trains, MRTS=threshold)
        x, y = adaptive_spike_sync_profile.get_plottable_data()

        if showing % 8 > 3:
            num_xy = len (x);
            print("\nA-SPIKE-Synchro-Profile:\n")
            print("x            y\n")
            for i in arange (num_xy):
                print("%.8f   %.8f\n" % (x[i], y[i]), end = "")
            print("\n")

        if plotting % 8 > 3:
            plt.figure(figsize=(17, 10), dpi=80)
            plt.plot(x, y, '-k')
            plt.axis([tmin-0.05*(tmax-tmin), tmax+0.05*(tmax-tmin), -0.05, 1.05])
            plt.plot((tmin, tmax), (0, 0), ':', color='k', linewidth=1)
            plt.plot((tmin, tmax), (1, 1), ':', color='k', linewidth=1)
            plt.plot((tmin, tmin), (0, 1), ':', color='k', linewidth=1)
            plt.plot((tmax, tmax), (0, 1), ':', color='k', linewidth=1)
            plt.plot((tmin, tmax), (adaptive_spike_synchro_thr, adaptive_spike_synchro_thr), '--', color='k', linewidth=1)
            plt.xticks(fontsize=12)
            plt.yticks(fontsize=12) # rotation=90
            plt.title("A-SPIKE-Synchro-profile   (A-SPIKE-Synchro = %.8f)" % (adaptive_spike_synchro_thr), color='k', fontsize=24)
            plt.xlabel('Time', color='k', fontsize=18)
            plt.ylabel('A-SPIKE-Synchro', color='k', fontsize=18)

    if showing % 16 > 7 or plotting % 16 > 7:
        adaptive_spike_synchro_mat = spk.spike_sync_matrix(spike_trains, MRTS=threshold)
        if showing % 16 > 7:
            print("\nA-SPIKE-Synchro-Matrix:")
            for i in arange (num_trains):
                print("\n%i     " % (i+1), end = "")
                for j in arange (num_trains):
                    print("%.8f " % (adaptive_spike_synchro_mat[i][j]), end = "")
            print("\n")

        if plotting % 16 > 7:
            plt.figure(figsize=(17, 10), dpi=80)
            plt.imshow(adaptive_spike_synchro_mat, interpolation='none')
            plt.title("A-SPIKE-Synchro-matrix   (A-SPIKE-Synchro = %.8f)" % (adaptive_spike_synchro_thr), color='k', fontsize=24)
            plt.xlabel('Spike Trains', color='k', fontsize=18)
            plt.ylabel('Spike Trains', color='k', fontsize=18)
            plt.xticks(arange(num_trains),arange(num_trains)+1, fontsize=14)
            plt.yticks(arange(num_trains),arange(num_trains)+1, fontsize=14)
            plt.jet()
            plt.colorbar()


if plotting > 0:
    plt.show()





