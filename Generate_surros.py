from __future__ import absolute_import

from numpy import *
import matplotlib.pyplot as plt
import pyspike
from pyspike.spikes import reconcile_spike_trains, reconcile_spike_trains_bi
from pyspike.generic import _generic_profile_multi, resolve_keywords
from functools import partial
from pyspike import DiscreteFunc
from pyspike.cython.python_backend import get_tau
from cython_simulated_annealing import sim_ann_cython as sim_ann

from pyspike.isi_lengths import default_thresh


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
# optimal_spike_train_sorting_from_matrix
############################################################
def optimal_spike_train_sorting_from_matrix(D, full_output=False):
    """ Finds the best sorting via simulated annealing.
    Returns the optimal permutation p and A value.
    Not for direct use, call :func:`.optimal_spike_train_sorting` instead.

    :param D: The directionality (Spike-ORDER) matrix.
    :param full_output: If true, then function will additionally return the
                        number of performed iterations (default=False)
    :return: (p, F) - tuple with the optimal permutation and synfire indicator.
             if `full_output=True` , (p, F, iter) is returned.
    """
    N = len(D)
    A = np.sum(np.triu(D, 0))

    p = np.arange(N)

    T_start = 2*np.max(D)    # starting temperature
    T_end = 1E-5 * T_start   # final temperature
    alpha = 0.9              # cooling factor

    try:
        from cython_simulated_annealing import sim_ann_cython as sim_ann
    except ImportError:
        raise NotImplementedError("PySpike with Cython required for computing spike train"
                                  " sorting!")

    p, A, total_iter = sim_ann(D, T_start, T_end, alpha)

    if full_output:
        return p, A, total_iter
    else:
        return p

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


def generate_surros_np(so_profs, sto_profs, all_trains, num_surros):
    num_pairs = len(sto_profs)
    num_trains = int((1 + np.sqrt(1+8*num_pairs))/2)
    upper_triangular_matrix = np.triu(np.ones((num_trains, num_trains)), k=1)
    
    firsts, seconds = np.where(upper_triangular_matrix)
    
    pair, leader_pos = np.where(so_profs == 1)
    dummy, follower_pos = np.where(so_profs == -1)
    ddd = sto_profs
    xxx = ddd[so_profs==1]
    num_coins = len(dummy)
    num_spikes = len(all_trains)
    indies = np.array([pair, firsts[pair] * (xxx == 1) + seconds[pair] * (xxx == -1), seconds[pair] * (xxx == 1) + firsts[pair] * (xxx == -1),
    leader_pos, follower_pos])
    num_swaps = num_spikes
    print(f'First surrogate with long transient: {num_swaps} swaps --- All others without transient: {np.round(num_swaps / 2)} swaps')
    synf = np.zeros(num_surros)
    for suc in range(num_surros):  # Python ranges exclude the end value, so we add 1
        if suc == 2:
            num_swaps = round(num_swaps / 2)
        sc = 1
        error_count = 0
        while sc <= num_swaps:
            brk = 0
            dummy = indies.copy()
        
            coin = np.random.randint(0, num_coins + 1)  # Python randint range is inclusive
        
            train1 = indies[1, coin - 1]  # Python uses 0-based indexing
            train2 = indies[2, coin - 1]
            pos1 = indies[3, coin - 1]
            pos2 = indies[4, coin - 1]
            fi11 = np.where(indies[3, :] == pos1)[0]
            fi21 = np.where(indies[4, :] == pos1)[0]
            fi12 = np.where(indies[3, :] == pos2)[0]
            fi22 = np.where(indies[4, :] == pos2)[0]
            fiu = np.unique(np.concatenate([fi11, fi21, fi12, fi22]))
            indies[1, fi11] = train2
            indies[2, fi21] = train2
            indies[1, fi12] = train1
            indies[2, fi22] = train1
            for fc in fiu:
                new_trains = np.sort(indies[0:2, fc], axis=0)  # switch train numbers
                indices = np.where((firsts == new_trains[0]) & (seconds == new_trains[1]))[0]
                if np.any(indices):
                    indies[0, fc] = indices[0]

            # indies
            for fc in fiu:
                sed = np.setdiff1d(np.where(indies[0] == indies[0, fc])[0], fc)  # all other coincidences from that pair of spike trains
                for sedc in range(sed.size):
                    if len(np.intersect1d(indies[3:5, sed[sedc]], indies[3:5, fc])):
                        error_count += 1
                        #print(777777777777)
                        indies = dummy
                        brk = 1
                        break
                if brk == 1:
                    break
            if brk == 1:
                if error_count <= 2 * num_coins:
                    continue
                else:
                    sc = num_swaps
            sc += 1
        surro_sto_profs = np.zeros(sto_profs.shape)

        for cc in range(num_coins):
            train_index = indies[0, cc-1]
            pos_indices = indies[2:4, cc-1]
            surro_sto_profs[train_index, pos_indices] = (indies[1, cc-1] < indies[2, cc-1]).astype(int) ^ (indies[1, cc-1] > indies[2, cc-1]).astype(int)
            

        
        surro_mat_entries = np.sum(surro_sto_profs, axis=1) / 2
        lower_tri_indices = np.tril_indices(num_trains, k=-1)
        surro_mat = np.zeros((num_trains, num_trains))
        surro_mat[lower_tri_indices] = surro_mat_entries
        
        surro_mat = surro_mat - surro_mat.T
        #print(surro_mat)
        phi = optimal_spike_train_sorting_from_matrix(surro_mat)
        
        surro_mat_opt = spk.permutate_matrix(surro_mat, phi)
        #print(surro_mat_opt)
        upper_triangle = np.triu(surro_mat_opt, k=1)
        total = np.sum(upper_triangle)
        #print(len(all_trains))
        synf[suc] = total*2/((num_trains-1)*len(all_trains))
        total_iter = 0  # Placeholder value for total_iter
        st_indy_simann = np.zeros((num_trains, num_trains), dtype=int)
    return synf





import matplotlib.pyplot as plt
import pyspike as spk
from numpy import *
from pylab import *
from pyspike.isi_lengths import default_thresh
from pyspike import SpikeTrain


measures   =   0    # +1:ISI,+2:SPIKE,+4:RI-SPIKE,+8:SPIKE-Synchro,+16:SPIKE-order,+32:Spike Train Order
showing =      2     # +1:spike trains,+2:distances,+4:profiles,+8:matrices,+16:pairwise profiles
plotting =     0    # +1:spike trains,+2:distances,+4:profiles,+8:matrices,+16:pairwise profiles
sort_spike_trains=1    # 0-no,1-yes

tmin=0
tmax=1000
threshold=1000

dataset=4     # choose one of the four main examples used so far, the parameter is just the different number of spike trains of this examples (2, 4, 6, 40)

match dataset:
    case 1:
        tmax=10
        spikes = []
        spikes.append(spk.SpikeTrain([  1.9,    3.9,     7,       ], [tmin, tmax]))
        spikes.append(spk.SpikeTrain([  2,               7.1,    9], [tmin, tmax]))
        spikes.append(spk.SpikeTrain([  2.1,    4.1,     6.9,     ], [tmin, tmax]))
    case 2:
        tmax=30
        spikes = []
        spikes.append(spk.SpikeTrain([12, 16, 28, 32, 44, 48, 60, 64, 76, 80], [tmin, tmax]))
        spikes.append(spk.SpikeTrain([8, 20, 24, 36, 40, 52, 56, 68, 72, 84], [tmin, tmax]))
    case 3:
        tmax=1
        spikes = []
        spikes.append(spk.SpikeTrain([0.0001, 0.7142], [tmin, tmax]))
        spikes.append(spk.SpikeTrain([0.2858, 0.9999], [tmin, tmax]))   # I switched the second and the third
        spikes.append(spk.SpikeTrain([0.1429, 0.8571], [tmin, tmax]))
    case 4:
        tmax=1000
        spikes = spk.load_spike_trains_from_txt("./examples/PySpike_testdata4.txt", edges=(tmin, tmax))
    case 5:
        tmax=1
        spikes = []
        spikes.append(spk.SpikeTrain([0.0001, 0.2942, 0.5882, 0.8823], [tmin, tmax]))
        spikes.append(spk.SpikeTrain([0.0589, 0.3530, 0.6470, 0.9411], [tmin, tmax]))   # I switched the second and the third as well as the fourth and the fifth spike train
        spikes.append(spk.SpikeTrain([0.0295, 0.3236, 0.6176, 0.9117], [tmin, tmax]))
        spikes.append(spk.SpikeTrain([0.1177, 0.4118, 0.7058, 0.9999], [tmin, tmax]))
        spikes.append(spk.SpikeTrain([0.0883, 0.3824, 0.6764, 0.9705], [tmin, tmax]))
    case 6:
        tmax=1000;
        spikes = spk.load_spike_trains_from_txt("./examples/PySpike_testdata6.txt", edges=(tmin, tmax))
        spikes[3]=spk.SpikeTrain([], [spikes[0].t_start, spikes[0].t_end])
        spikes[4]=spk.SpikeTrain([], [spikes[0].t_start, spikes[0].t_end])
    case 40:
        showing=8
        tmax=4000
        spikes = spk.load_spike_trains_from_txt("./examples/PySpike_testdata.txt", edges=(tmin, tmax))
        
num_trains = len (spikes);
print("\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n Dataset: %i,  Num_trains: %3i" % (dataset,num_trains))



if showing % 2 > 0 and sort_spike_trains==0:                                                                    # Spike trains
    f_print_spike_trains(spikes)
if plotting % 2 > 0 and sort_spike_trains==0:
    f_plot_spike_trains(spikes=spikes,tmin=tmin,tmax=tmax,name="Rasterplot")



if measures % 2 > 0:                                                                                             # ISI-distance
    if showing % 16 > 1 or plotting % 16 > 1:
        isi_distance = spk.isi_distance(spikes)
        if showing % 4 > 1:
           print("\nISI-Distance: %.5f\n" % isi_distance)

    if showing % 8 > 3 or plotting % 8 > 3:
        isi_profile = spk.isi_profile(spikes)
        x, y = isi_profile.get_plottable_data()
        if showing % 8 > 3:
            f_print_profile(x,y,"ISI-Distance-Profile")
        if plotting % 8 > 3:
            f_plot_profile(x_prof=x,y_prof=y,tmin=tmin,tmax=tmax,ymin=0,ymax=1,name="ISI-distance-profile",name2='ISI',value=isi_distance,name3="ISI-distance")

    if showing % 16 > 7 or plotting % 16 > 7:
        isi_distance_mat = spk.isi_distance_matrix(spikes)
        if showing % 16 > 7:
            f_print_matrix(isi_distance_mat,"ISI-Distance-Matrix")
        if plotting % 16 > 7:
            f_plot_matrix(isi_distance_mat,0,1,'ISI-distance-Matrix',isi_distance,'ISI-distance')



if measures % 4 > 1:                                                                                              # SPIKE-distance
    if showing % 16 > 1 or plotting % 16 > 1:
        spike_distance = spk.spike_distance(spikes)
        if showing % 4 > 1:
            print("\nSPIKE-Distance: %.5f\n" % spike_distance)

    if showing % 8 > 3 or plotting % 8 > 3:
        spike_profile = spk.spike_profile(spikes)
        x, y = spike_profile.get_plottable_data()
        if showing % 8 > 3:
            f_print_profile(x,y,"SPIKE-Distance-Profile")
        if plotting % 8 > 3:
            f_plot_profile(x_prof=x,y_prof=y,tmin=tmin,tmax=tmax,ymin=0,ymax=1,name="SPIKE-distance-profile",name2='SPIKE',value=spike_distance,name3="SPIKE-distance")

    if showing % 16 > 7 or plotting % 16 > 7:
        spike_distance_mat = spk.spike_distance_matrix(spikes)
        if showing % 16 > 7:
            f_print_matrix(spike_distance_mat,"SPIKE-Distance-Matrix")
        if plotting % 16 > 7:
            f_plot_matrix(spike_distance_mat,0,1,'SPIKE-distance-Matrix',spike_distance,'SPIKE-distance')



if measures % 8 > 3:                                                                                             # RI-SPIKE-distance
    if showing % 16 > 1 or plotting % 16 > 1:
        ri_spike_distance = spk.spike_distance(spikes, RIA=True)
        if showing % 4 > 1:
            print("\nRI-SPIKE-Distance: %.5f\n" % ri_spike_distance)
      
    if showing % 8 > 3 or plotting % 8 > 3:
        ri_spike_profile = spk.spike_profile(spikes, RIA=True)
        x, y = ri_spike_profile.get_plottable_data()
        if showing % 8 > 3:
            f_print_profile(x,y,"RI-SPIKE-Distance-Profile")
        if plotting % 8 > 3:
            f_plot_profile(x_prof=x,y_prof=y,tmin=tmin,tmax=tmax,ymin=0,ymax=1,name="RI-SPIKE-distance-profile",name2='RI-SPIKE',value=ri_spike_distance,name3="RI-SPIKE-distance")

    if showing % 16 > 7 or plotting % 16 > 7:
        ri_spike_distance_mat = spk.spike_distance_matrix(spikes, RIA=True)
        if showing % 16 > 7:
            f_print_matrix(ri_spike_distance_mat,"RI-SPIKE-Distance-Matrix")
        if plotting % 16 > 7:
            f_plot_matrix(ri_spike_distance_mat,0,1,'RI-SPIKE-distance-Matrix',ri_spike_distance,'RI-SPIKE-distance')



if measures % 16 > 7:                                                                                            # SPIKE-synchro
    if showing % 16 > 1 or plotting % 16 > 1:
        spike_synchro = spk.spike_sync(spikes)
        if showing % 4 > 1:
            print("\nSPIKE-Synchro: %.5f\n" % spike_synchro)

    if showing % 8 > 3 or plotting % 8 > 3:
        spike_sync_profile = spk.spike_sync_profile(spikes)
        x, y = spike_sync_profile.get_plottable_data()
        if showing % 8 > 3:
            f_print_profile(x,y,"SPIKE-Synchro-Profile")
        if plotting % 8 > 3:
            f_plot_profile(x_prof=x,y_prof=y,tmin=tmin,tmax=tmax,ymin=0,ymax=1,name="SPIKE-Synchro-profile",name2='SPIKE-Synchro',value=spike_synchro,name3="SPIKE-Synchro")

    if showing % 16 > 7 or plotting % 16 > 7:
        spike_synchro_mat = spk.spike_sync_matrix(spikes)
        if showing % 16 > 7:
            f_print_matrix(spike_synchro_mat,"SPIKE-Synchro-Matrix")
        if plotting % 16 > 7:
            f_plot_matrix(spike_synchro_mat,0,1,'SPIKE-Synchro-Matrix',spike_synchro,'SPIKE-Synchro')

    if showing % 32 > 15 or plotting % 32 > 15:
        spike_synchro_bi_profiles_mat=f_get_profiles_mat(spikes,"SPIKE-Synchro")
        if showing % 32 > 15:
            f_print_profiles(spike_synchro_bi_profiles_mat,"SPIKE-Synchro - Bivariate-Profiles")
        if plotting % 32 > 15:
            f_plot_bivariate_profiles(spikes=spikes,tmin=tmin,tmax=tmax,ymin=0,ymax=1,name="SPIKE-Synchro-profiles",name2="SPIKE-Synchro")



if measures % 32 > 15:                                                                                               # SPIKE-order
    if showing % 16 > 1 or plotting % 16 > 1:
        spike_order=f_spike_order(spikes)
        if showing % 4 > 1:
            print("\nSPIKE-Order: %.5f\n" % spike_order)
            
    if showing % 8 > 3 or plotting % 8 > 3:
        spike_order_profile = f_spike_order_profile(spikes)
        x, y = spike_order_profile.get_plottable_data()
        if showing % 8 > 3:
            f_print_profile(x,y,"SPIKE-Order-Profile")
        if plotting % 8 > 3:
            f_plot_profile(x_prof=x,y_prof=y,tmin=tmin,tmax=tmax,ymin=-1,ymax=1,name="SPIKE-Order-profile",name2='SPIKE-Order',value=spike_order,name3="SPIKE-Order")

    if showing % 16 > 7 or plotting % 16 > 7:
        spike_order_mat = spk.spike_directionality_matrix(spikes)
        if showing % 16 > 7:
            f_print_matrix(spike_order_mat,"SPIKE-Order-Matrix")
        if plotting % 16 > 7:
            f_plot_matrix(spike_order_mat,-1,1,'SPIKE-Order-Matrix',spike_order,'SPIKE-Order')

    if showing % 32 > 15 or plotting % 32 > 15:
        spike_order_bi_profiles_mat=f_get_profiles_mat(spikes,"SPIKE-Order")
        if showing % 32 > 15:
            f_print_profiles(spike_order_bi_profiles_mat,"SPIKE-Order - Bivariate-Profiles")
        if plotting % 32 > 15:
            f_plot_bivariate_profiles(spikes=spikes,tmin=tmin,tmax=tmax,ymin=-1,ymax=1,name="SPIKE-Order-profiles",name2="SPIKE-Order")


            

if measures % 64 > 31:                                                                                                  # Spike Train Order
    if showing % 16 > 1 or plotting % 16 > 1:
        spike_train_order = spk.spike_train_order(spikes)
        if showing % 4 > 1:
            print("\nSpike Train Order: %.8f\n" % spike_train_order)

    if showing % 8 > 3 or plotting % 8 > 3:
        spike_train_order_profile = spk.spike_train_order_profile(spikes)
        x, y = spike_train_order_profile.get_plottable_data()
        if showing % 8 > 3:
            f_print_profile(x,y,"Spike Train Order-Profile")
        if plotting % 8 > 3:
            f_plot_profile(x_prof=x,y_prof=y,tmin=tmin,tmax=tmax,ymin=-1,ymax=1,name="Spike Train Order profile",name2='Spike Train Order',value=spike_train_order,name3="Synfire Indicator")

    if showing % 16 > 7 or plotting % 16 > 7:
        spike_train_order_mat = f_spike_train_order_matrix(spikes)
        if showing % 16 > 7:
            f_print_matrix(spike_train_order_mat,"Spike Train Order - Matrix")
        if plotting % 16 > 7:
            f_plot_matrix(spike_train_order_mat,-1,1,'Spike Train Order - Matrix',spike_train_order,'Synfire Indicator')
            
    if showing % 32 > 15 or plotting % 32 > 15:        
        spike_train_order_bi_profiles_mat=f_get_profiles_mat(spikes,"Spike Train Order")
        if showing % 32 > 15:
            f_print_profiles(spike_train_order_bi_profiles_mat,"Spike Train Order - Bivariate-Profiles")
        if plotting % 32 > 15:
            f_plot_bivariate_profiles(spikes=spikes,tmin=tmin,tmax=tmax,ymin=-1,ymax=1,name="Spike Train Order - profiles",name2="Spike Train Order")




if sort_spike_trains==1:                                                                                                # Spike Train Sorting
    F_init = spk.spike_train_order(spikes)
    phi, _ = spk.optimal_spike_train_sorting(spikes)
    F_opt = spk.spike_train_order(spikes, indices=phi)

    if showing % 2 > 0:                                                                     # Spike trains
        f_print_spike_trains(spikes,"Initial Spike Trains before spike train sorting")
        f_print_spike_trains(spikes,"Optimized Spike Trains after spike train sorting",phi)
    if plotting % 2 > 0:
        f_plot_spike_trains(spikes=spikes,tmin=tmin,tmax=tmax,name="Rasterplot before spike train sorting")
        f_plot_spike_trains(spikes=spikes,tmin=tmin,tmax=tmax,name="Rasterplot after spike train sorting",phi=phi)        

    if showing % 4 > 1:                                                                     # Synfire Indicator
        print("Initial Synfire Indicator before spike train sorting:", F_init)
        print("Optimized Synfire Indicator after spike train sorting:", F_opt)

    if showing % 8 > 3 or plotting % 8 > 3:                                                 # Spike Train Order Profile
        spike_train_order_profile_init = spk.spike_train_order_profile(spikes)
        x_init, y_init = spike_train_order_profile_init.get_plottable_data()
        spike_train_order_profile_opt = spk.spike_train_order_profile(spikes, indices=phi)
        x_opt, y_opt = spike_train_order_profile_opt.get_plottable_data()
        if showing % 8 > 3:
            f_print_profile(x_init,y_init,"Initial Spike Train Order - Profile before spike train sorting")
            f_print_profile(x_opt,y_opt,"Optimized Spike Train Order - Profile after spike train sorting")
        if plotting % 8 > 3:
            f_plot_profile(x_prof=x_init,y_prof=y_init,tmin=tmin,tmax=tmax,ymin=-1,ymax=1, \
                name="Spike Train Order profile before spike train sorting",name2='Spike Train Order',value=F_init,name3="Initial Synfire Indicator")
            f_plot_profile(x_prof=x_opt,y_prof=y_opt,tmin=tmin,tmax=tmax,ymin=-1,ymax=1, \
                name="Spike Train Order profile after spike train sorting",name2='Spike Train Order',value=F_opt,name3="Optimized Synfire Indicator")

    if showing % 16 > 7 or plotting % 16 > 7:                                               # SPIKE-Order Matrix
        D_init = spk.spike_directionality_matrix(spikes)
        D_opt = spk.permutate_matrix(D_init, phi)       
        if showing % 16 > 7:
            f_print_matrix(D_init,"Initial SPIKE-Order Matrix before spike train sorting")
            f_print_matrix(D_opt,"Optimized SPIKE-Order Matrix after spike train sorting")
        if plotting % 16 > 7:
            f_plot_matrix(matrix=D_init,value=F_init,ymin=-1,ymax=1,name='Initial SPIKE-Order Matrix before spike train sorting',name2='Synfire Indicator')
            f_plot_matrix(matrix=D_opt,value=F_opt,ymin=-1,ymax=1,name='Optimized SPIKE-Order Matrix after spike train sorting',name2='Synfire Indicator')

        

if plotting > 0:
    plt.show()

so = f_get_profiles_mat(spikes,"SPIKE-Order")
sto = f_get_profiles_mat(spikes,"Spike Train Order")


print("-----------------")
all_trains,_ = f_all_trains(spikes)
print("all train:",all_trains)
surros_mat = generate_surros_np(so, sto, all_trains,19)
print(surros_mat)
highlight_value = F_opt
values = surros_mat.flatten()

mean_value = np.mean(values)
std_value = np.std(values)
min_value = float(mean_value - (std_value / 2))
max_value = float(mean_value + (std_value / 2))
print(min_value, max_value, mean_value,std_value, (min_value + max_value) / 2)
plt.axvline(x=mean_value, color='green', linestyle='--')
plt.axhline(10, xmin=min_value/1.1, xmax=max_value/1.1, color='green', linewidth=2)
plt.xlim(0, 1.1)
plt.hist(values, color='black', width=0.01)
plt.hist(highlight_value, color='red', width=0.01)
plt.xlabel('Value')
plt.ylabel('Frequency')
plt.title('Histogram')    
plt.show()



