
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
        print(surro_mat)
        phi = optimal_spike_train_sorting_from_matrix(surro_mat)
        print(phi)
        #surro_mat_opt = spk.permutate_matrix(surro_mat, phi)
        #print(surro_mat_opt)
        #upper_triangle = np.triu(surro_mat_opt, k=1)
        #total = np.sum(upper_triangle)
        synf[suc] = phi*2/(num_trains-1)/len(all_trains)
        total_iter = 0  # Placeholder value for total_iter
        st_indy_simann = np.zeros((num_trains, num_trains), dtype=int)
    return synf

