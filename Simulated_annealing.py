from __future__ import absolute_import

from numpy import *
import matplotlib.pyplot as plt
import pyspike
from pyspike.spikes import reconcile_spike_trains, reconcile_spike_trains_bi
from pyspike.generic import _generic_profile_multi, resolve_keywords
from functools import partial
from pyspike import DiscreteFunc
from pyspike.cython.python_backend import get_tau

import matplotlib.pyplot as plt
import pyspike as spk
from numpy import *
from pylab import *
from pyspike.isi_lengths import default_thresh
from pyspike import SpikeTrain
from matplotlib import pyplot
from pyspike import spike_directionality
############################################################
# spike_order_profile
############################################################
def f_spike_order_profile(*args, **kwargs):
    """ Computes the spike directionality profile :math:`D(t)` of the given
    spike trains. Returns the profile as a DiscreteFunction object.

    Valid call structures::

      spike_order_profile(st1, st2)       # returns the bi-variate profile
      spike_order_profile(st1, st2, st3)  # multi-variate profile of 3
                                                # spike trains

      spike_trains = [st1, st2, st3, st4]       # list of spike trains
      spike_order_profile(spike_trains)   # profile of the list of spike trains
      spike_order_profile(spike_trains, indices=[0, 1])  # use only the spike trains
                                                               # given by the indices

    Additonal arguments: 
    :param max_tau: Upper bound for coincidence window, `default=None`.
    :param indices: list of indices defining which spike trains to use,
                    if None all given spike trains are used (default=None)

    :returns: The spike train order profile :math:`E(t)`
    :rtype: :class:`.DiscreteFunction`
    """
    if len(args) == 1:
        return f_spike_order_profile_multi(args[0], **kwargs)
    elif len(args) == 2:
        return f_spike_order_profile_bi(args[0], args[1], **kwargs)
    else:
        return f_spike_order_profile_multi(args, **kwargs)



############################################################
# spike_order_profile_bi
############################################################
def f_spike_order_profile_bi(spike_train1, spike_train2, 
                                 max_tau=None, **kwargs):
    """ Computes the spike directionality profile D(t) of the two given
    spike trains. Returns the profile as a DiscreteFunction object.

    :param spike_train1: First spike train.
    :type spike_train1: :class:`pyspike.SpikeTrain`
    :param spike_train2: Second spike train.
    :type spike_train2: :class:`pyspike.SpikeTrain`
    :param max_tau: Maximum coincidence window size. If 0 or `None`, the
                    coincidence window has no upper bound.
    :returns: The spike directionality profile :math:`D(t)`.
    :rtype: :class:`pyspike.function.DiscreteFunction`
    """
    if kwargs.get('Reconcile', True):
        spike_train1, spike_train2 = reconcile_spike_trains_bi(spike_train1, spike_train2)
    MRTS, RI = resolve_keywords(**kwargs)
    if isinstance(MRTS, str):
        MRTS = default_thresh([spike_train1, spike_train2])

    # check whether the spike trains are defined for the same interval
    assert spike_train1.t_start == spike_train2.t_start, \
        "Given spike trains are not defined on the same interval!"
    assert spike_train1.t_end == spike_train2.t_end, \
        "Given spike trains are not defined on the same interval!"

    # cython implementation
    try:
        from pyspike.cython.cython_directionality import \
            spike_order_profile_cython as \
            spike_order_profile_impl
    except ImportError:
        # raise NotImplementedError()
        pyspike.NoCythonWarn()

        # use python backend
        #from pyspike.cython.directionality_python_backend import \
        #    spike_order_profile_python2 as spike_order_profile_impl

    if max_tau is None:
        max_tau = 0.0

    times, coincidences, multiplicity \
        = f_spike_order_profile_python2(spike_train1.spikes,    # impl
                                         spike_train2.spikes,
                                         spike_train1.t_start,
                                         spike_train1.t_end,
                                         max_tau, MRTS)

    return DiscreteFunc(times, coincidences, multiplicity)



############################################################
# spike_order_profile_multi
############################################################
def f_spike_order_profile_multi(spike_trains, indices=None,
                                    max_tau=None, **kwargs):
    """ Computes the multi-variate spike directionality profile for a set of
    spike trains. For each spike in the set of spike trains, the multi-variate
    profile is defined as the sum of asymmetry values divided by the number of
    spike trains pairs involving the spike train of containing this spike,
    which is the number of spike trains minus one (N-1).

    :param spike_trains: list of :class:`pyspike.SpikeTrain`
    :param indices: list of indices defining which spike trains to use,
                    if None all given spike trains are used (default=None)
    :type indices: list or None
    :param max_tau: Maximum coincidence window size. If 0 or `None`, the
                    coincidence window has no upper bound.
    :returns: The multi-variate directionality profile :math:`<D>(t)`
    :rtype: :class:`pyspike.function.DiscreteFunction`
    """
    prof_func = partial(f_spike_order_profile_bi, max_tau=max_tau)
    average_prof, M = _generic_profile_multi(spike_trains, prof_func,
                                             indices, **kwargs)
    return average_prof



############################################################
# spike_order
############################################################
def f_spike_order(*args, **kwargs):
    """ Computes the spike directionality of the given spike trains.

    Valid call structures::

      spike_order(st1, st2, normalize=True)  # normalized bi-variate
                                                    # spike train order
      spike_order(st1, st2, st3)  # multi-variate result of 3 spike trains

      spike_trains = [st1, st2, st3, st4]       # list of spike trains
      spike_order(spike_trains)   # result for the list of spike trains
      spike_order(spike_trains, indices=[0, 1])  # use only the spike trains
                                                       # given by the indices

    Additonal arguments: 
     - `max_tau` Upper bound for coincidence window, `default=None`.
     - `normalize` Flag indicating if the reslut should be normalized by the
       number of spikes , default=`False`


    :returns: The spike train order value (Synfire Indicator)
    """
    if len(args) == 1:
        return f_spike_order_multi(args[0], **kwargs)
    elif len(args) == 2:
        return f_spike_order_bi(args[0], args[1], **kwargs)
    else:
        return f_spike_order_multi(args, **kwargs)


############################################################
# spike_order_bi
############################################################
def f_spike_order_bi(spike_train1, spike_train2, normalize=True,
                         interval=None, max_tau=None, **kwargs):
    """ Computes the overall spike directionality for two spike trains.

    :param spike_train1: First spike train.
    :type spike_train1: :class:`pyspike.SpikeTrain`
    :param spike_train2: Second spike train.
    :type spike_train2: :class:`pyspike.SpikeTrain`
    :param normalize: Normalize by the number of spikes (multiplicity).
    :param max_tau: Maximum coincidence window size. If 0 or `None`, the
                    coincidence window has no upper bound.
    :returns: The spike train order value (Synfire Indicator)
    """
    c, mp = _spike_order_impl(spike_train1, spike_train2, interval, max_tau, **kwargs)
    if normalize:
        return 1.0*c/mp
    else:
        return c

############################################################
# f_spike_order_multi
############################################################
def f_spike_order_multi(spike_trains, indices=None, normalize=True,
                            interval=None, max_tau=None, **kwargs):
    """ Computes the overall spike directionality for many spike trains.

    :param spike_trains: list of :class:`.SpikeTrain`
    :param indices: list of indices defining which spike trains to use,
                    if None all given spike trains are used (default=None)
    :param normalize: Normalize by the number of spike (multiplicity).
    :param interval: averaging interval given as a pair of floats, if None
                     the average over the whole function is computed.
    :type interval: Pair of floats or None.
    :param max_tau: Maximum coincidence window size. If 0 or `None`, the
                    coincidence window has no upper bound.
    :returns: Spike directionality D for the given spike trains.
    :rtype: double
    """
    MRTS, RI = resolve_keywords(**kwargs)
    if isinstance(MRTS, str):
        MRTS = default_thresh(spike_trains)
    if indices is None:
        indices = arange(len(spike_trains))
    indices = array(indices)
    # check validity of indices
    assert (indices < len(spike_trains)).all() and (indices >= 0).all(), \
        "Invalid index list."
    # generate a list of possible index pairs
    pairs = [(indices[i], j) for i in range(len(indices))
             for j in indices[i+1:]]

    d_total = 0.0
    m_total = 0.0
    for (i, j) in pairs:
        d, m = _f_spike_order_impl(spike_trains[i], spike_trains[j],
                                       interval, max_tau, MRTS=MRTS, RI=RI)
        d_total += d
        m_total += m

    if m == 0.0:
        return 1.0
    else:
        return d_total/m_total



############################################################
# _spike_order_impl
############################################################
def _f_spike_order_impl(spike_train1, spike_train2,
                            interval=None, max_tau=None, **kwargs):
    """ Implementation of bi-variatae spike directionality value

    :param spike_train1: First spike train.
    :type spike_train1: :class:`pyspike.SpikeTrain`
    :param spike_train2: Second spike train.
    :type spike_train2: :class:`pyspike.SpikeTrain`
    :param max_tau: Maximum coincidence window size. If 0 or `None`, the
                    coincidence window has no upper bound.
    :returns: The spike train order value (Synfire Indicator)
    """
    MRTS, RI = resolve_keywords(**kwargs)
    if isinstance(MRTS, str):
        MRTS = default_thresh([spike_train1, spike_train2])
    if interval is None:
        # distance over the whole interval is requested: use specific function
        # for optimal performance
        try:
            from spk.cython.cython_directionality import \
                spike_order_cython as spike_order_func
            if max_tau is None:
                max_tau = 0.0
            c, mp = spike_order_func(spike_train1.spikes,
                                           spike_train2.spikes,
                                           spike_train1.t_start,
                                           spike_train1.t_end,
                                           max_tau, MRTS)
        except ImportError:
            # Cython backend not available: fall back to profile averaging
            c, mp = f_spike_order_profile(spike_train1, spike_train2,
                                              max_tau=max_tau,
                                              MRTS=MRTS).integral(interval)
        return c, mp
    else:
        # some specific interval is provided: not yet implemented
        raise NotImplementedError("Parameter `interval` not supported.")



############################################################      # to do: add also the cython version #########
# spike_order_profile_python2
############################################################
def f_spike_order_profile_python2(spikes1, spikes2, t_start, t_end,
                                     max_tau, MRTS=0.):
    true_max = t_end - t_start
    if max_tau > 0:
        true_max = min(true_max, 2*max_tau)

    N1 = len(spikes1)
    N2 = len(spikes2)
    i = -1
    j = -1
    n = 0
    st = zeros(N1 + N2 + 2)  # spike times
    d = zeros(N1 + N2 + 2)   # coincidences
    mp = ones(N1 + N2 + 2)   # multiplicity
    while i + j < N1 + N2 - 2:
        if (i < N1-1) and (j == N2-1 or spikes1[i+1] < spikes2[j+1]):
            i += 1
            n += 1
            tau = get_tau(spikes1, spikes2, i, j, true_max, MRTS)
            st[n] = spikes1[i]
            if j > -1 and spikes1[i]-spikes2[j] < tau:
                # current spike gets marked with -1 and previous spike with 1
                d[n] = -1
                d[n-1] = 1
        elif (j < N2-1) and (i == N1-1 or spikes1[i+1] > spikes2[j+1]):
            j += 1
            n += 1
            tau = get_tau(spikes1, spikes2, i, j, true_max, MRTS)
            st[n] = spikes2[j]
            if i > -1 and spikes2[j]-spikes1[i] < tau:
                # current spike gets marked with -1 and previous spike with 1
                d[n] = -1
                d[n-1] = 1
        else:   # spikes1[i+1] = spikes2[j+1]
            # advance in both spike trains
            j += 1
            i += 1
            n += 1
            # add only one event with zero asymmetry value and multiplicity 2
            st[n] = spikes1[i]
            d[n] = 0
            mp[n] = 2

    st = st[:n+2]
    d = d[:n+2]
    mp = mp[:n+2]

    st[0] = t_start
    st[len(st)-1] = t_end
    if N1 + N2 > 0:
        d[0] = d[1]
        d[len(d)-1] = d[len(d)-2]
        mp[0] = mp[1]
        mp[len(mp)-1] = mp[len(mp)-2]
    else:
        d[0] = 1
        d[1] = 1

    return st, d, mp

    

############################################################
# spike_train_order_matrix
############################################################
def f_spike_train_order_matrix(spike_trains, normalize=True, indices=None,
                                interval=None, max_tau=None, **kwargs):
    """ Computes the spike_train_order matrix for the given spike trains.

    :param spike_trains: List of spike trains.
    :type spike_trains: List of :class:`pyspike.SpikeTrain`
    :param normalize: Normalize by the number of spikes (multiplicity).
    :param indices: list of indices defining which spike trains to use,
                    if None all given spike trains are used (default=None)
    :type indices: list or None
    :param max_tau: Maximum coincidence window size. If 0 or `None`, the
                    coincidence window has no upper bound.
    :returns: The spike_train_order values.
    """
    if kwargs.get('Reconcile', True):
        spike_trains = reconcile_spike_trains(spike_trains)
    MRTS, RI = resolve_keywords(**kwargs)
    if isinstance(MRTS, str):
        MRTS = default_thresh(spike_trains)
    if indices is None:
        indices = arange(len(spike_trains))
    indices = array(indices)
    # check validity of indices
    assert (indices < len(spike_trains)).all() and (indices >= 0).all(), \
        "Invalid index list."
    # generate a list of possible index pairs
    pairs = [(indices[i], j) for i in range(len(indices))
             for j in indices[i+1:]]

    distance_matrix = zeros((len(indices), len(indices)))
    for i, j in pairs:
        d = pyspike.spike_train_order(spike_trains[i], spike_trains[j])
        #d = spike_train_order(spike_trains[i], spike_trains[j], normalize,
        #                         interval, max_tau=max_tau, 
        #                         MRTS=MRTS, RI=RI, Reconcile=False)
        distance_matrix[i, j] = d
        distance_matrix[j, i] = d
    return distance_matrix



############################################################
# all_trains
############################################################
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
    pooled = [sp_flat[idx] for idx in sp_indy]
    
    return all_trains, pooled



############################################################
# get_profiles_mat
############################################################
def f_get_profiles_mat(spikes,name):
    [all_trains,pooled] = f_all_trains(spikes)
    num_all_spikes = len(all_trains)        
    num_trains = len(spikes)
    num_pairs=int(num_trains*(num_trains-1)/2);
    bi_profiles_mat = zeros((num_pairs,num_all_spikes))
    pc=0;
    for i in arange (num_trains):
        for j in arange (i+1,num_trains):
            match name:
                case "SPIKE-Synchro":
                    bi_profile = pyspike.spike_sync_profile(spikes[i],spikes[j])
                case "SPIKE-Order":
                    bi_profile = f_spike_order_profile(spikes[i],spikes[j])                
                case "Spike Train Order":
                    bi_profile = pyspike.spike_train_order_profile(spikes[i],spikes[j]) 
            x, y = bi_profile.get_plottable_data()

            yc=0;
            for k in arange (num_all_spikes):
                if all_trains[k]-1==i or all_trains[k]-1==j:
                    bi_profiles_mat[pc][k]=y[yc+1]
                    yc=yc+1;
            pc=pc+1;
    return bi_profiles_mat



############################################################
# print_spike_trains
############################################################
def f_print_spike_trains(spikes,name=None,phi=None):
    if name!=None:
        print("\n%s:" % name)
    for i in arange (len(spikes)):
        if phi==None:
            print("\n%3i:" % (i+1))
        else:
            print("\n%3i:" % (phi[i]+1))                
        for j in arange ( len(spikes[i]) ):
            if phi==None:
                print("%i %.5f" % (j+1, spikes[i][j]))
            else:
                print("%i %.5f" % (j+1, spikes[phi[i]][j]))
    print("\n")
    


############################################################
# print_profile
############################################################
def f_print_profile(x_prof,y_prof,name):
    print("\n%s:\n" % name)
    print("x            y\n")
    for i in arange (len (x_prof)):
        print("%.5f   %.5f\n" % (x_prof[i], y_prof[i]), end = "")
    print("\n")


    
############################################################
# print_matrix
############################################################
def f_print_matrix(matrix,name):
    num_trains = len(matrix)
    print("\n%s:" % name)
    for i in arange (num_trains):
        print("\n%i     " % (i+1), end = "")
        for j in arange (num_trains):
            print("%.5f " % (matrix[i][j]), end = "")
    print("\n")



############################################################
# print_profiles
############################################################
def f_print_profiles(bi_profiles_mat,name):
    print("\n%s:\n" % name)
    for i in arange (len(bi_profiles_mat)):
        print("\n%i     " % (i+1), end = "")
        for j in arange (len(bi_profiles_mat[0])):
            print("%.5f " % (bi_profiles_mat[i][j]), end = "")
    print("\n")
    


############################################################
# plot_spike_trains
############################################################
def f_plot_spike_trains(spikes,tmin,tmax,name,phi=None):
    num_trains = len(spikes)
    plt.figure(figsize=(17, 10), dpi=80)
    plt.title("%s" % name, color='k', fontsize=24)
    plt.xlabel('Time', color='k', fontsize=18)
    plt.ylabel('Spike Trains', color='k', fontsize=18)
    plt.axis([tmin-0.05*(tmax-tmin), tmax+0.05*(tmax-tmin), 0, num_trains+1])
    plt.xticks(fontsize=14)
    if phi==None:
        plt.yticks(arange(num_trains)+1, arange(num_trains,0,-1), fontsize=14)
    else:
        plt.yticks(arange(num_trains)+1, reversed([x+1 for x in phi]), fontsize=14)
    plt.plot((tmin, tmax), (0.5, 0.5), ':', color='k', linewidth=1)
    plt.plot((tmin, tmax), (num_trains+0.5, num_trains+0.5), ':', color='k', linewidth=1)
    plt.plot((tmin, tmin), (0.5, num_trains+0.5), ':', color='k', linewidth=1)
    plt.plot((tmax, tmax), (0.5, num_trains+0.5), ':', color='k', linewidth=1)
    for i in arange (num_trains):
        for j in arange (len (spikes[i])):
            if phi==None:
                plt.plot((spikes[i][j], spikes[i][j]), (num_trains-i+0.5, num_trains-i-.5), '-', color='k', linewidth=1)
            else:
                plt.plot((spikes[phi[i]][j], spikes[phi[i]][j]), (num_trains-i+0.5, num_trains-i-.5), '-', color='k', linewidth=1)



############################################################
# plot_profile
############################################################
def f_plot_profile(x_prof,y_prof,tmin,tmax,ymin,ymax,name,name2,value=None,name3=None):
    plt.figure(figsize=(17, 10), dpi=80)
    plt.plot(x_prof, y_prof, '-k', marker="*", markersize=15)
    plt.axis([tmin-0.05*(tmax-tmin), tmax+0.05*(tmax-tmin), ymin-0.05, ymax+0.05])
    plt.plot((tmin, tmax), (ymin, ymin), ':', color='k', linewidth=1)
    plt.plot((tmin, tmax), (ymax, ymax), ':', color='k', linewidth=1)
    plt.plot((tmin, tmin), (ymin, ymax), ':', color='k', linewidth=1)
    plt.plot((tmax, tmax), (ymin, ymax), ':', color='k', linewidth=1)
    if ymin==-1:
        plt.plot((tmin, tmax), (0, 0), ':', color='k', linewidth=1)    
    plt.plot((tmin, tmax), (value, value), '--', color='k', linewidth=1)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14) # rotation=90
    if name2==None or value==None:
        plt.title("%s" % name, color='k', fontsize=24)
    else:
        plt.title("%s   (%s = %.5f)" % (name,name3,value), color='k', fontsize=24)
    plt.xlabel('Time', color='k', fontsize=18)
    plt.ylabel('%s' % name2, color='k', fontsize=18)




############################################################
# plot_bivariate_profiles
############################################################
def f_plot_bivariate_profiles(spikes,tmin,tmax,ymin,ymax,name,name2):
    cols='kbrgmc'
    symbs='*vo+^dx<hs>'
    num_trains = len(spikes)
    plt.figure(figsize=(17, 10), dpi=80)
    pc=0
    for i in arange (num_trains):
        for j in arange (i+1,num_trains):
            match name2:
                case "SPIKE-Synchro":
                    bi_profile = pyspike.spike_sync_profile(spikes[i],spikes[j])
                case "SPIKE-Order":
                    bi_profile = f_spike_order_profile(spikes[i],spikes[j])                
                case "Spike Train Order":
                    bi_profile = pyspike.spike_train_order_profile(spikes[i],spikes[j])                
            x, y = bi_profile.get_plottable_data()
            cindy=pc % len(cols)
            plt.plot(x, y, '-', color=cols[cindy], marker=symbs[cindy], markersize=12, label="%i-%i" % (i+1,j+1))
            pc=pc+1
    plt.axis([tmin-0.05*(tmax-tmin), tmax+0.05*(tmax-tmin), ymin-0.05, ymax+0.05])
    plt.plot((tmin, tmax), (ymin, ymin), ':', color='k', linewidth=1)
    plt.plot((tmin, tmax), (ymax, ymax), ':', color='k', linewidth=1)
    plt.plot((tmin, tmin), (ymin, ymax), ':', color='k', linewidth=1)
    plt.plot((tmax, tmax), (ymin, ymax), ':', color='k', linewidth=1)
    plt.legend(loc=2,bbox_to_anchor=(1.01, 1))
    if ymin==-1:
        plt.plot((tmin, tmax), (0, 0), ':', color='k', linewidth=1)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14) # rotation=90
    plt.title("%s" % name, color='k', fontsize=24)
    plt.xlabel('Time', color='k', fontsize=18)
    plt.ylabel("%s" % name2, color='k', fontsize=18)
            

    
############################################################
# plot_matrix
############################################################
def f_plot_matrix(matrix,ymin,ymax,name,value=None,name2=None):
    num_trains = len(matrix)
    plt.figure(figsize=(17, 10), dpi=80)
    plt.imshow(matrix, interpolation='none', vmin=ymin, vmax=ymax)
    if name2==None or value==None:
        plt.title("%s" % name, color='k', fontsize=24)
    else:
        plt.title("%s   (%s = %.5f)" % (name,name2,value), color='k', fontsize=24)        
    plt.xlabel("Spike Trains", color='k', fontsize=18)
    plt.ylabel("Spike Trains", color='k', fontsize=18)
    plt.xticks(arange(num_trains),arange(num_trains)+1, fontsize=14)
    plt.yticks(arange(num_trains),arange(num_trains)+1, fontsize=14)
    plt.jet()
    plt.colorbar()

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
    pooled = [sp_flat[idx] for idx in sp_indy]
    
    return all_trains, pooled


def f_plot_spike_trains(spikes,tmin,tmax,name,phi=None):
    num_trains = len(spikes)
    plt.figure(figsize=(17, 10), dpi=80)
    plt.title("%s" % name, color='k', fontsize=24)
    plt.xlabel('Time', color='k', fontsize=18)
    plt.ylabel('Spike Trains', color='k', fontsize=18)
    plt.axis([tmin-0.05*(tmax-tmin), tmax+0.05*(tmax-tmin), 0, num_trains+1])
    plt.xticks(fontsize=14)
    if phi==None:
        plt.yticks(arange(num_trains)+1, arange(num_trains,0,-1), fontsize=14)
    else:
        plt.yticks(arange(num_trains)+1, reversed([x+1 for x in phi]), fontsize=14)
    plt.plot((tmin, tmax), (0.5, 0.5), ':', color='k', linewidth=1)
    plt.plot((tmin, tmax), (num_trains+0.5, num_trains+0.5), ':', color='k', linewidth=1)
    plt.plot((tmin, tmin), (0.5, num_trains+0.5), ':', color='k', linewidth=1)
    plt.plot((tmax, tmax), (0.5, num_trains+0.5), ':', color='k', linewidth=1)
    for i in arange (num_trains):
        for j in arange (len (spikes[i])):
            if phi==None:
                plt.plot((spikes[i][j], spikes[i][j]), (num_trains-i+0.5, num_trains-i-.5), '-', color='k', linewidth=1)
            else:
                plt.plot((spikes[phi[i]][j], spikes[phi[i]][j]), (num_trains-i+0.5, num_trains-i-.5), '-', color='k', linewidth=1)


def generate_list(x, y, k, z):
    result_list = [x]
    while x < z:
        x += k * y
        result_list.append(x)
    return result_list

def f_create_synfire(tmin, tmax, num_trains, num_events, overlap,order):
   
    distance_btw_events = (tmax-tmin)/overlap/((num_events-1)/overlap+1)
    event_duration = distance_btw_events*overlap
    spike_time_diff = event_duration/(num_trains-1)
    spike_trains = []
    
    for i in range(num_trains):
        if order == 0:
            spike_trains.append(spk.SpikeTrain(generate_list(tmin+i*spike_time_diff,distance_btw_events,1,(tmax-distance_btw_events)+i*spike_time_diff), [tmin, tmax]))

        elif order == 1:
            spike_trains.append(spk.SpikeTrain(generate_list((tmin+distance_btw_events)-i*spike_time_diff,distance_btw_events,1,tmax-i*spike_time_diff), [tmin, tmax]))

    return spike_trains, spike_time_diff


spikes,_ = f_create_synfire(0,1,5,5,0.25,0)

f_plot_spike_trains(spikes=spikes,tmin=0,tmax= 1,name="Rasterplot")
_,pooled = f_all_trains(spikes)
all_trains,_ = f_all_trains(spikes)
def pairs(spikes):
    num_trains = len(spikes)
    time_matrix = []
    all_trains,_ = f_all_trains(spikes)
    
    _,pooled = f_all_trains(spikes)
    
    so = f_get_profiles_mat(spikes,"SPIKE-Order")
    ss = np.abs(so)
    count = 0
    pairs = [[] for i in range(len(all_trains)-1)]
    for i in range(len(ss)):
        
        for j in range(len(all_trains)):
            if ss[i][j] == 1:
                pairs[i].append(j)
                
    return pairs



def time_matrix(spikes):
    num_trains = len(spikes)
    all_trains, _ = f_all_trains(spikes)
    _, pooled = f_all_trains(spikes)
    so = f_get_profiles_mat(spikes, "SPIKE-Order")
    ss = np.abs(so)
    
    
    time_matrix = []
    
    for i in range(len(ss)):
        row = []
        j = 0
        while j <(len(pair[i]) - 1):
            
            row.append(pooled[pair[i][j + 1]] - pooled[pair[i][j]])
            j = j+2
        time_matrix.append(row)
            
    return time_matrix


def stdm_matrix(spikes):
    time_laps = time_matrix(spikes)
    average = []
    for i in range(len(time_laps)):
        total = sum(time_laps[i])
        average.append(total/len(time_laps[i]))
    
    
    n_average = len(average)
    #n_matrix = np.sqrt(2*len(average))
   
    n_matrix = math.ceil(math.sqrt(2*n_average))
    matrix = np.zeros((n_matrix, n_matrix))
    
    c = 0
    
    for i in range(n_matrix):
        for j in range(i+1, n_matrix):
            
            matrix[i,j] = average[c]
            c = c+1
           
    matrix_t=-matrix.T
    matrix = matrix_t+matrix
            
        
                

    return matrix, average


pair = pairs(spikes)
#STDM = stdm_matrix(spikes)
#m=np.min(STDM)
#M = np.max(STDM)
#min=m 
#max = M 
#f_plot_matrix(matrix=STDM,value=None,ymin=min,ymax=max,name='Spike Time Difference Matrix')
        

TM= time_matrix(spikes)




sto = f_get_profiles_mat(spikes,"Spike Train Order")


def mean_without_numpy(arr):
    if len(arr) == 0:
        raise ValueError("Input array must not be empty")
    
    total_sum = 0
    for value in arr:
        total_sum += value
    
    return total_sum / len(arr)
def signs(sto):
    
    stoo =  np.array([row[row != 0] for row in sto])
    sign = np.array([row[::2] for row in stoo])
    return sign
signe = signs(sto)        

import matplotlib.pyplot as plt
import numpy as np
from itertools import combinations
def f_latency_correction_simulated_annealing(start_spike_diffs, start_tim, som, som_matches, all_trains):
    num_pairs = len(start_spike_diffs)
    num_trains = int(0.5 + np.sqrt(0.25 + 2 * num_pairs))

    sa_temp_fact = 10000
    iter_unit = 2000000

    start_cost = np.mean(np.abs(start_spike_diffs))

    old_tim = start_tim.copy()
    old_spike_diffs = start_spike_diffs.copy()
    old_cost = start_cost

    pairs = np.array(list(combinations(range(num_trains), 2)))
    indies = np.zeros((num_trains, num_trains - 1), dtype=int)
    for trc in range( num_trains ):
        indies[trc, :], _ = np.where(pairs == trc)
    indies = np.sort(indies, axis=1)

    costs = np.full(iter_unit, np.nan)
    min_cost = start_cost
    min_tim = start_tim.copy()
    T = 1
    T_end = T / sa_temp_fact
    alpha = 0.9
    min_iter = 0
    total_iter = 1
    sum_condi = 0
    
    while T > T_end:
        iterations = 0
        succ_iter = 0

        while iterations < 100 * num_trains and succ_iter < 10 * num_trains:
            new_tim = old_tim.copy()
            #print('tim',new_tim)
            train = np.random.randint(0, num_trains )
            displacement = np.random.randn(1) * old_cost
            #print('disp',displacement)
            new_tim[all_trains == train] = old_tim[all_trains == train] + displacement[0]
            #print('tim',new_tim,old_tim)
            new_spike_diffs = old_spike_diffs.copy()
            #print(new_spike_diffs)
            for pac in indies[train - 1, :]:
                if som_matches[pac] > 0:
                    
                    signs = signe
                   
                    #print(len(signs))

                    new_tim_pos = new_tim_neg = []
                    #print((new_tim))
                    #print(som)
                    new_tim_neg_indices = [i for i in range(len(new_tim)) if som[pac, i] == -1]
                    new_tim_neg = np.array([new_tim[i] for i in new_tim_neg_indices])
                    new_tim_pos_indices = [i for i in range(len(new_tim)) if som[pac, i] == 1]
                    new_tim_pos = np.array([new_tim[i] for i in new_tim_pos_indices])
                    
                    tim_pos = np.mean(new_tim_pos)
                    tim_neg = np.mean(new_tim_neg)

                        
                    

                    new_spike_diffs[pac] = np.mean((tim_neg - tim_pos) * signs[pac]).astype(np.float64)
                    #print('nspd',new_spike_diffs[pac],(tim_neg - tim_pos))
                    #print('sign',signs[pac],pac)
            #print(new_spike_diffs)
            new_cost = np.mean(np.abs(new_spike_diffs))

            delta_cost = new_cost - old_cost
            condi = delta_cost < 0 or np.exp(-delta_cost / T) > np.random.rand(1)
            sum_condi += condi

            if condi > 0:
                old_tim = new_tim
                old_spike_diffs = new_spike_diffs
                old_cost = new_cost
                succ_iter += 1
                if new_cost < min_cost:
                    min_iter = total_iter + iterations
                    min_cost = new_cost
                    min_tim = old_tim

            iterations += 1
            costs[total_iter + iterations - 1] = old_cost

        total_iter += iterations
        if total_iter % iter_unit > iter_unit - 100 * num_trains:
            costs[(total_iter // iter_unit + 1) * iter_unit:] = np.nan
            if num_trains > 10:
                print("Iteration:", total_iter)
                print("Cost:", old_cost)

        T *= alpha

        if succ_iter == 0:
            break

    last_cost = old_cost
    costs = costs[0:total_iter]
    return min_tim,start_cost,costs,total_iter, last_cost, min_cost

som  = f_get_profiles_mat(spikes, "SPIKE-Order")
pair = pairs(spikes)
som_matches = []
for i in range(len(signe)):
    som_matches.append(len(signe[i]))

    
#som_matches = len(pairs(spikes))/2
_,average = stdm_matrix(spikes)
new_tim,start_cost,costs,iterations,last_cost, min_cost = f_latency_correction_simulated_annealing(average, pooled, som, som_matches, all_trains)
print(new_tim,start_cost,last_cost,min_cost)
iterations = np.arange(1, len(costs) + 1) * iterations
plt.plot(iterations,costs)
plt.xlabel('Iterations')
plt.ylabel('Cost')
plt.title('Cost Function')
plt.show()




num_trains = len(spikes)
print(num_trains, all_trains)
new_spikes = np.zeros((num_trains,len(new_tim)//num_trains))
for i in range(num_trains):
    for j in range(len(new_tim)//num_trains):
        new_spikes[i][j]=new_tim[i+(num_trains)*j]
        
f_plot_spike_trains(spikes=new_spikes,tmin=0,tmax= 1,name="Rasterplot")
