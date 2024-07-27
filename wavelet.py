import numpy as np
import math
import pywt

def get_disc_grid(disc_pts):
    x_rec  = np.arange(0, 1, 1 / disc_pts)
    arrays = [x_rec, x_rec]
    la = len(arrays)
    dtype = np.result_type(*arrays)
    arr = np.empty([len(a) for a in arrays] + [la], dtype=dtype)
    for i, a in enumerate(np.ix_(*arrays)):
        arr[...,i] = a
    return arr.reshape(-1, la)

class WaveletBasis:
    def __init__(self):
        self.wavelet_family = "db2"
        self.wavelet = pywt.Wavelet(self.wavelet_family)
        self.A = 0.0105 # db2: ~0.0105
        self.B = self.wavelet.rec_len - 2

        self.basis_func = None

    def _find_nearest(self, array, value):
        idx = np.searchsorted(array, value, side="left")
        if idx > 0 and (idx == len(array) or math.fabs(value - array[idx-1]) < math.fabs(value - array[idx])):
            return idx-1
        return idx

    def _get_disc_idxs(self, xi, ks, inv_s, x_wav):
        disc_xs = inv_s * xi - ks + self.B
        idx_tmp = np.searchsorted(x_wav, disc_xs)
        idxs = np.clip(idx_tmp - 1, 0, len(x_wav)-1)
        return idxs

    def _get_raw_basis_decomp(self, f):
        # ---- Wavelet decomposition parameters
        scale = int(np.log2(f.shape[0]))
        k_s   = 1. / math.sqrt(pow(2, scale))
        fL, fH, x_wav = self.wavelet.wavefun(level=scale+2)

        # ---- Wavelet decomposition logic 
        coeffs = pywt.dwt2(f, self.wavelet_family)
        nb_levels = len(coeffs) - 1

        LL, (LH, HL, HH) = coeffs
        coeffs = np.array([LL, LH, HL, HH]).reshape(-1)
        coeffs = np.expand_dims(np.expand_dims(coeffs, axis=-1), axis=-1)
        if self.basis_func is not None: # assume the flow is raw basis -> filter after which only use filter 
            return coeffs, None

        self.x_grid = get_disc_grid(f.shape[0])
        x_grid_offset = np.expand_dims(np.expand_dims(self.x_grid - self.A, 1), 1)

        inv_s = pow(2, scale - nb_levels)
        ks    = np.array([[(k_1, k_2) for k_2 in range(LL.shape[1])] for k_1 in range(LL.shape[0])])
        idxs  = self._get_disc_idxs(x_grid_offset, ks, inv_s, x_wav)

        final_shape_tuple = tuple(f.shape) + (-1,)

        fL_x, fL_y = fL[idxs[...,0]].reshape(final_shape_tuple), fL[idxs[...,1]].reshape(final_shape_tuple)
        fH_x, fH_y = fH[idxs[...,0]].reshape(final_shape_tuple), fH[idxs[...,1]].reshape(final_shape_tuple)

        disc_fLL = (fL_x * fL_y).reshape(final_shape_tuple)
        disc_fLH = (fH_x * fL_y).reshape(final_shape_tuple)
        disc_fHL = (fL_x * fH_y).reshape(final_shape_tuple)
        disc_fHH = (fH_x * fH_y).reshape(final_shape_tuple)
        disc_f   = (k_s * math.sqrt(inv_s) / math.sqrt(2)) * np.concatenate([disc_fLL, disc_fLH, disc_fHL, disc_fHH], axis=-1)
        disc_f   = np.transpose(disc_f, axes=(2, 0, 1))
        return coeffs, disc_f

    def _get_redundant_basis_idx(self, raw_basis):
        full_rank = raw_basis.shape[-1]
        rows = list(raw_basis)

        removed = []
        num_removed = 0
        to_remove = 0

        while len(rows) != full_rank:
            print(f"Removing: {to_remove} -- Len: {len(rows)}")

            cur_rows = rows[:to_remove] + rows[to_remove+1:]
            cur_matrix = np.array(cur_rows)
            new_rank = np.linalg.matrix_rank(cur_matrix)
            if new_rank == full_rank:
                removed.append(num_removed + to_remove)
                num_removed += 1
                rows = cur_rows
            else:
                to_remove += 1
        return rows, removed

    def _filter_redundant_basis(self, raw_coeffs, raw_basis):
        raw_basis_idx = set(range(len(raw_coeffs)))
        to_remove = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 51, 52, 68, 69, 85, 86, 102, 103, 119, 120, 136, 137, 153, 154, 170, 171, 187, 188, 204, 205, 221, 222, 238, 239, 255, 256, 272, 273, 289, 290, 306, 307, 323, 324, 340, 341, 357, 358, 374, 375, 391, 392, 408, 409, 425, 426, 442, 443, 459, 460, 476, 477, 493, 494, 510, 511, 527, 528, 544, 545, 561, 562, 578, 579, 580, 581, 582, 583, 584, 585, 586, 587, 588, 589, 590, 591, 592, 593, 594, 595, 596, 597, 598, 599, 600, 601, 602, 603, 604, 605, 606, 607, 608, 609, 610, 611]
        remove_idx_filter = np.array(to_remove)
        basis_idx_filter  = np.array(sorted(list(raw_basis_idx - set(to_remove))))
        
        if self.basis_func is None:
            flat_raw_basis = raw_basis.reshape(raw_basis.shape[0], -1)
            self.basis_func = flat_raw_basis[basis_idx_filter].T
            removed_basis_els = flat_raw_basis[remove_idx_filter].T
            self.removed_weights = np.linalg.solve(self.basis_func, removed_basis_els)

        filtered_coeff  = raw_coeffs[basis_idx_filter].flatten()
        removed_coeffs  = raw_coeffs[remove_idx_filter].flatten()
        combined_coeffs = filtered_coeff + self.removed_weights @ removed_coeffs
        return combined_coeffs

    def get_decomp(self, f):
        raw_coeffs, raw_basis = self._get_raw_basis_decomp(f)
        return self._filter_redundant_basis(raw_coeffs, raw_basis)