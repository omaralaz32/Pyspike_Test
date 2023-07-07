import numpy as np
import matplotlib.pyplot as plt

def f_lc_shift_diagonals(spike_diff_mat, stop_diagonal, diagonals_test):
    num_trains = spike_diff_mat.shape[0]

    shifts = np.zeros(num_trains)  

    for i in range(1, num_trains):
        shift_row = min(i, stop_diagonal)

        for j in range(1, shift_row + 1):
            shifts[i] += shifts[i - j] + spike_diff_mat[i - j, i]

        shifts[i] /= shift_row  

    if diagonals_test == 1: 
        shifts2 = np.zeros(num_trains)  
        spike_diff_mat2 = spike_diff_mat.T

        for i in range(1, num_trains):
            shift_row = min(i, stop_diagonal)

            for j in range(1, shift_row + 1):
                shifts2[i] += shifts2[i - j] + spike_diff_mat2[i - j, i]

            shifts2[i] /= shift_row  

        shifts = (shifts + np.flipud(shifts2) - shifts2[-1]) / 2

    return shifts


def f_latency_correct_shift_rows_avg3(shifts_mat, stop):
    num_trains = shifts_mat.shape[0]
    mask = np.triu(np.ones((num_trains, num_trains)), k=-stop) - np.triu(np.ones((num_trains, num_trains)), k=stop+1)
    shifts_mat = np.triu(shifts_mat, k=-stop) - np.triu(shifts_mat, k=stop+1)

    for trc in range(1, num_trains):
        shifts_mat[trc, :] = (shifts_mat[trc, :] + shifts_mat[trc - 1, trc]) * mask[trc, :]  # Cumulative (always uses first spike train as reference)

    shifts = np.sum(shifts_mat, axis=0) / np.sum(mask, axis=0)

    return shifts





def f_latency_correct_shift_lambda(spike_time_diff_matrix, stop_diagonal):
    num_trains = spike_time_diff_matrix.shape[0]

    for d in range(stop_diagonal + 1, num_trains):
        mask = np.triu(np.ones((num_trains, num_trains)), k=-d+1) - np.triu(np.ones((num_trains, num_trains)), k=d)
        for i in range(num_trains - d):
            mask_vec = mask[i, :] * mask[i + d, :]
            spike_time_diff_matrix[i, i + d] = np.sum((spike_time_diff_matrix[i, :] + spike_time_diff_matrix[i + d, :]) * mask_vec) / np.sum(mask_vec)

    spike_time_diff_matrix = spike_time_diff_matrix - np.tril(spike_time_diff_matrix, -stop_diagonal-1) - np.triu(spike_time_diff_matrix, stop_diagonal+1).T
    shifts = np.mean(spike_time_diff_matrix, axis=0)

    return shifts



def f_lc_shift_rows(spike_diff_mat, stop):
    num_trains = spike_diff_mat.shape[0]

    shifts = np.zeros(num_trains)
    prev_shift = 0
    rc = 1

    while rc < num_trains:
        next_overlap = rc + min(stop, num_trains - rc)

        shifts[rc:next_overlap] = spike_diff_mat[rc-1, rc:next_overlap] + prev_shift

        prev_shift = shifts[next_overlap-1]

        rc = next_overlap

    return shifts


def f_first_diagonal(spike_diff_mat):
    shifts = np.zeros((1, spike_diff_mat.shape[0]))
    print(shifts)
    for i in range(1, spike_diff_mat.shape[0]):
        shifts[0, i] = shifts[0, i-1] + spike_diff_mat[i-1][ i]
        print(shifts, shifts[0, i-1], spike_diff_mat[i-1][ i])
    return shifts
                   




num_trains = 5
max_pos = 3
include_maxdiag = 0
row = 1
spike_diffs_mat = np.zeros((num_trains, num_trains))
for d in range(1, max_pos + 1):
    spike_diffs_mat += np.triu(np.ones((num_trains, num_trains)), d)
for d in range(max_pos + 1, num_trains):
    spike_diffs_mat -= np.triu(np.ones((num_trains, num_trains)), d)

spike_diffs_mat = spike_diffs_mat + spike_diffs_mat.T
spike_diffs_mat = spike_diffs_mat - np.min(np.min(spike_diffs_mat)) + 1
spike_diffs_mat = spike_diffs_mat - np.diag(np.diag(spike_diffs_mat))

diag_mean = np.zeros(num_trains - 1)
for d in range(1, num_trains):
    diag_mean[d - 1] = np.mean(np.diag(spike_diffs_mat, d))

maxi = np.argmax(diag_mean)

if include_maxdiag == 1:
    cost_mat = np.triu(spike_diffs_mat, 1) - np.triu(spike_diffs_mat, maxi + 1)
    cost = np.sum(cost_mat) / (np.sum(np.arange(num_trains - 1, num_trains - maxi, -1)) + 1e-10)
else:
    cost_mat = np.triu(spike_diffs_mat, 1) - np.triu(spike_diffs_mat, maxi)
    cost = np.sum(cost_mat) / (np.sum(np.arange(num_trains - 1, num_trains - maxi + 1, -1)) + 1e-10)

# Plotting
plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.imshow(spike_diffs_mat, cmap='jet')
plt.colorbar()
plt.title(f'Maximum = {max_pos} --- Include Maximum Diagonal = {include_maxdiag}', fontsize=14)

plt.subplot(1, 2, 2)
plt.plot(diag_mean, 'r', linewidth=2)
plt.plot(maxi, diag_mean[maxi], 'bo', markersize=8)
if include_maxdiag == 1:
    plt.plot(np.arange(1, maxi + 1), diag_mean[:maxi], 'kx-', linewidth=2, markersize=8)
else:
    plt.plot(np.arange(1, maxi), diag_mean[:maxi - 1], 'kx-', linewidth=2, markersize=8)
plt.ylim([0, max(diag_mean)])
plt.xlabel('Diagonal', fontsize=12)
plt.ylabel('Mean Value', fontsize=12)
plt.title('Diagonal Mean Values', fontsize=14)

plt.tight_layout()
plt.show()
stop_diagonal = 3
print(spike_diffs_mat)
print(f_latency_correct_shift_lambda(spike_diffs_mat, stop_diagonal))
print(f_lc_shift_rows(spike_diffs_mat, stop_diagonal))
print(f_lc_shift_diagonals(spike_diffs_mat, stop_diagonal, include_maxdiag))
print(f_latency_correct_shift_rows_avg3(spike_diffs_mat, stop_diagonal))
print(f_first_diagonal(spike_diffs_mat))
print(spike_diffs_mat[row])
