"""
dont put anything in init that takes time

1) class for data collection, converting to the correct mapping should be done seperately
2) normalization can be class, but no expensive calculations in it
3) get_exact accuracy: different object. All calculations for determining performance separate: overlap, accuracy
4) compression pretty much the same, but no accuracy calc. --> class needed?
5) check functions: time, save file, check if exists.
6) comments, documentation
"""
import matplotlib.pyplot as plt
import scipy
import skimage
import tqdm
from numba import jit
from scipy.stats.mstats import gmean

import feature_maps
from data_loader import load_data
from useful_funcs import *


class DataPreprocessing:
    """
    Performs mean pooling on every image. self.data contains the images as np.ndarray: (D,N,WxH) if even_distribution.
    Otherwise, (list(D), N_D, WxH)
    D: number of labels = 10
    N: number of images = 4506
    H: Length of Image in pixels
    W: Width of image in pixels

    """

    def __init__(self, dataset, even_distribution=False, max_nb_images=None, name='train', force_run=False, sort=False):
        self.max_nb_images = max_nb_images
        self.original_images, self.labels = dataset

        self.data = self.mean_pooling(self.original_images, 2)
        if sort:
            self.data = save_data_to_pickle(self.get_digits, (self.data, self.labels),
                                            f'results/preprocessing/{name}.p', force_run=force_run)

        if even_distribution:
            self.data = self.create_even_distribution(self.data)

    @staticmethod
    def flat_to_2d(data):
        H = W = int(np.sqrt(data.shape[-1]))
        return data.reshape((-1, H, W))

    @staticmethod
    def square_to_flat(data):
        H = data.shape[-1]
        return data.reshape((-1, H ** 2))

    def mean_pooling(self, images_, filter_size):
        images_2d = self.flat_to_2d(images_)
        pooled_image = skimage.measure.block_reduce(images_2d, (1, filter_size, filter_size), np.mean)
        result = self.square_to_flat(pooled_image)
        result = self.scale(result, domain=(0, 1))
        return result

    @staticmethod
    def scale(images_, domain=(0, 255), range_=(0, 1)):
        scaled = images_ / (domain[1] - domain[0]) * (range_[1] - range_[0]) + range_[0]
        return scaled

    @staticmethod
    def get_digits(mat, labels_):
        sorted_digits_ = []
        for label in np.unique(labels_):
            label_slice = (labels_ == label)
            sorted_digits_.append(mat[label_slice])

        return sorted_digits_

    def create_even_distribution(self, batch):
        if self.max_nb_images is None:
            smallest_batch_size = np.min([len(dbatch) for dbatch in batch])
            even_batch = np.array([dbatch[:smallest_batch_size] for dbatch in batch]).copy()
            return even_batch
        else:
            even_batch = np.array([dbatch[:self.max_nb_images] for dbatch in batch]).copy()
            return even_batch

    def show_pic(self):
        pixels = self.data.shape[-1]
        h = int(pixels ** (1 / 2))

        for i in range(4):
            plt.figure()
            image = self.data[i, i, :].reshape((h, h))
            plt.imshow(image)


def overlap_same_dims(psi1, psi2):
    @jit(nopython=True)
    def contract_physical(node_idx):
        return np.ascontiguousarray(psi1[:, node_idx, :]) @ np.ascontiguousarray(
            psi2[:, node_idx, :].T.conj())

    product = contract_physical(0)
    for i in range(1, 196):
        product *= contract_physical(i)
    norm = np.sum(product)
    norm = np.abs(norm)
    return norm


def _get_norm_label(label_data):
    """
    @param label_data: (N, WxH, 2)
    N: number of images per label
    W: width of image
    H: height of image
    --> WxH = number of nodes (Matrices in MPS)
    2: physical legs
    @return:
    """

    @jit(nopython=True)
    def contract_physical(node_idx):
        return np.ascontiguousarray(label_data[:, node_idx, :]) @ np.ascontiguousarray(
            label_data[:, node_idx, :].T.conj())

    product = contract_physical(0)
    for i in range(1, 196):
        product *= contract_physical(i)
    norm = np.sum(product)
    norm = np.abs(norm)
    return norm


def get_norms(wf_full, printout=False):
    norms = []
    for label_data in tqdm.tqdm(wf_full):
        norm = _get_norm_label(label_data)
        norms.append(norm)
        if printout:
            print(norm)
    return norms


def normalize(wf_full, basis=None, printout=False, force_run=False):
    data = wf_full.copy()
    norms = np.array(save_data_to_pickle(get_norms, (data, printout), f'results/{basis}/norms.p', force_run=force_run))
    for i in range(len(data)):
        data[i] /= (norms[i] ** (1 / (2 * data[i].shape[1])))
    if printout:
        print(norms)
    return data


def get_truncation_overlap(wf_compressed, wf_full, average=False):
    """
    wf_compressed: (list(N),list(D),L,P,R)
    wf_full: (D,N,HxW, P)
    average: if True: return the geometric mean of the overlaps squared
    @return:
    """
    overlaps = []
    for cmps, emps in tqdm.tqdm(zip(wf_compressed, wf_full)):
        contract = np.ones((1, 1))
        for i in range(len(cmps)):
            """
            emps: (N,HxW,P)
            cmps[i]: (L,P,R)
            """
            physical = np.tensordot(emps[:, i, :], cmps[i].conj(), (1, 1))
            """"
            (N,L,R)
            """
            contract = np.ascontiguousarray(contract) @ np.ascontiguousarray(physical)

        overlaps.append(np.linalg.norm(np.sum(contract)))
    if average:
        overlaps = gmean(np.abs(overlaps) ** 2)
    return overlaps


def _get_prediction(wf_compressed, image):
    physical_dim = wf_compressed[0][0].shape[2]
    braket = np.ones((len(wf_compressed[0]), 1, 1))
    for ket_i, bra_i in zip(wf_compressed, image):
        physical_contract = np.sum([ket_i[:, :, i, :] * bra_i[i].conj() for i in range(physical_dim)], axis=0)
        # physical_contract = ket_i[:, :, 0, :] * bra_i[0].conj() + ket_i[:, :, 1, :] * bra_i[1].conj()

        braket = braket @ physical_contract

    prediction = np.argmax(np.absolute(braket[:, 0, 0]))
    return prediction


def get_accuracy(wf_compressed, test_images, test_labels, nb_tests_cap=None):
    if nb_tests_cap is None:
        nb_tests_cap = len(test_images)

    wf_compressed_transpose = transpose_wf_compressed(wf_compressed)

    preds = []
    for image in tqdm.tqdm(test_images[:nb_tests_cap]):
        preds.append(_get_prediction(wf_compressed_transpose, image))
    preds = np.array(preds)
    accuracy = np.sum(test_labels[:nb_tests_cap] == preds) / nb_tests_cap
    return accuracy


class CompressedWFS:
    def __init__(self, chi_max, wf_full, feature_map):
        """
        
        @param chi_max: 
        @param wf_full: uncompressed MPS: (D,N,HxW, P)
        @param feature_map: 
        """
        self.sweep_number = 0
        self.feature_map = feature_map
        self.wf_full = wf_full

        self.chi_max = chi_max
        self.wf_compressed = self.create_random_compressed_wfs(wf_full.shape[2], chi_max, wf_full.shape[-1])

        self.path = None
        self.file_name = None
        self.update_path()

    def sweep(self, force_run=False):
        def sweep_inner(self):
            wf_compressed = []
            print('Sweeping through MPS')
            for cmps, emps in tqdm.tqdm(zip(self.wf_compressed, self.wf_full)):
                self.perform_svd_full_mps(cmps)
                self.sweep_mps(cmps, emps)
                wf_compressed.append(cmps)
            return wf_compressed

        self.sweep_number += 1
        self.update_path()
        self.wf_compressed = save_data_to_pickle(sweep_inner, (self,),
                                                 self.path,
                                                 force_run=force_run)

    def update_path(self):
        self.file_name = 'cwfs_chi{}_s{}.p'.format(self.chi_max, self.sweep_number)
        self.path = os.path.join('results', self.feature_map, 'cwfs', self.file_name)

    @staticmethod
    def sweep_mps(cmps, emps):
        """

        :param cmps: list: 196 entries with shape [chi_left, physical, chi_right]
        :param emps: np.array: [4506, 196, 2]
        :return:
        """
        nb_basis_elements = emps.shape[-1]

        def contract_single_bond(i):
            physical_contract_ = np.tensordot(emps[:, i, :], cmps[i].conj(), (1, 1))
            return physical_contract_

        right_contr_list = [np.ones((1, 1, len(emps)))]

        for m_idx in reversed(range(len(cmps) - 1)):
            virtual = np.array([right_contr_list[-1] * emps[:, m_idx + 1, basis_element] for basis_element in
                                range(nb_basis_elements)])
            # virtual = np.array([right_contr_list[-1] * emps[:,m_idx+1,0], right_contr_list[-1] * emps[:,m_idx+1,1]])
            physical = np.tensordot(cmps[m_idx + 1].conj(), virtual, ((1, 2), (0, 1)))
            # physical_contract = contract_single_bond(m_idx + 1)
            # right_contract = np.ascontiguousarray(physical_contract) @ np.ascontiguousarray(right_contr_list[-1])
            right_contr_list.append(physical)
        right_contr_list.reverse()

        left_contr_list = [np.ones((1, 1, len(emps)))]
        for m_idx in range(len(cmps) - 1):
            right_contract = right_contr_list[m_idx]
            updated_m_left = np.array(
                [left_contr_list[-1] * emps[:, m_idx, basis_element] for basis_element in range(nb_basis_elements)])
            # updated_m_left = np.array([left_contr_list[-1] * emps[:, m_idx, 0], left_contr_list[-1] * emps[:, m_idx, 1]])
            updated_m = np.tensordot(updated_m_left, right_contract, (-1, -1))

            updated_m_sliced = updated_m[:, :, 0, :, 0].transpose((1, 0, 2))
            legs_together = updated_m_sliced.reshape((-1, updated_m_sliced.shape[-1]))
            A, S, V = scipy.linalg.svd(legs_together, full_matrices=False, lapack_driver='gesdd')

            updated_m_left_canonical = A.reshape(updated_m_sliced.shape[0], nb_basis_elements, -1)
            cmps[m_idx] = updated_m_left_canonical

            virtual = np.array(
                [left_contr_list[-1] * emps[:, m_idx, basis_element] for basis_element in range(nb_basis_elements)])
            # virtual = np.array([left_contr_list[-1] * emps[:,m_idx,0], left_contr_list[-1] * emps[:,m_idx,1]])
            physical = np.tensordot(cmps[m_idx].conj(), virtual, ((1, 0), (0, 1)))

            # physical_contract = contract_single_bond(m_idx)
            # contract_left = np.ascontiguousarray(left_contr_list[-1]) @ np.ascontiguousarray(physical_contract)
            left_contr_list.append(physical)

        right_contr_list = [np.ones((1, 1, len(emps)))]
        for m_idx in reversed(range(len(cmps))):
            left_contract = left_contr_list[m_idx]
            updated_m_right = np.array(
                [right_contr_list[-1] * emps[:, m_idx, basis_element] for basis_element in range(nb_basis_elements)])
            # updated_m_right = np.stack([right_contr_list[-1] * emps[:, m_idx, 0], right_contr_list[-1] * emps[:, m_idx, 1]])

            updated_m = np.tensordot(left_contract, updated_m_right, (-1, -1))

            updated_m_sliced = updated_m[:, 0, :, :, 0]
            legs_together = updated_m_sliced.reshape((updated_m_sliced.shape[0], -1))
            A, S, V = scipy.linalg.svd(legs_together, full_matrices=False, lapack_driver='gesdd')
            updated_m_right_canonical = V.reshape(-1, nb_basis_elements, updated_m_sliced.shape[-1])
            cmps[m_idx] = updated_m_right_canonical

            virtual = np.array(
                [right_contr_list[-1] * emps[:, m_idx, basis_element] for basis_element in range(nb_basis_elements)])
            # virtual = np.array([right_contr_list[-1] * emps[:,m_idx,0], right_contr_list[-1] * emps[:,m_idx,1]])
            physical = np.tensordot(cmps[m_idx].conj(), virtual, ((1, 2), (0, 1)))
            right_contr_list.append(physical)

    @staticmethod
    def create_random_compressed_wfs(mps_length, chi_max, physical_dim):
        def create_mps():
            mps = [np.random.random((1, physical_dim, chi_max))]
            if mps_length - 2:
                tile = np.zeros((chi_max, physical_dim, chi_max))
                for i in range(physical_dim):
                    tile[:, i, :] = np.diag(np.random.random(chi_max))
                    # tile[:, 1, :] = np.diag(np.random.random(max_bond_dim))
                mps += [tile] * (mps_length - 2)
            mps.append(np.random.random((chi_max, physical_dim, 1)))
            return mps

        compressed_wfs = []
        for i in range(10):
            compressed_wfs.append(create_mps())

        return compressed_wfs

    @staticmethod
    def perform_svd_full_mps(mps):

        """
        @param mps: ndarray/list of MPS: (nb_matrices, left virtual, physical, right virtual)
        @return: mps in canonical form
        """
        for m_i in reversed(range(len(mps))):
            matrix = np.array(mps[m_i])
            if m_i == 0:
                mps[m_i] /= np.sqrt(np.tensordot(matrix, matrix.conj(), axes=([0, 1, 2], [0, 1, 2])))
                continue
            else:
                pass

            legs_together = matrix.reshape((matrix.shape[0], -1))
            A, S, V = scipy.linalg.svd(legs_together, full_matrices=False, lapack_driver='gesdd')
            V_legs_apart = V.reshape(-1, *matrix.shape[1:])

            mps[m_i] = V_legs_apart

            mps[m_i - 1] = mps[m_i - 1] @ A * S

    def __repr__(self):
        output = \
            f"""--------------
Compressed MPS: 
𝜒: {self.chi_max}
map: {self.feature_map}
sweeps: {self.sweep_number}
path: {self.path}      
            """
        return output


if __name__ == '__main__':
    mnist_data_path = os.path.join('data', 'mnist.pkl.gz')

    data = load_data(mnist_data_path)
    train_data, val_data, test_data = data

    train_o = DataPreprocessing(train_data, even_distribution=True, name='train', force_run=True, sort=True)
    val_o = DataPreprocessing(val_data, name='val', force_run=True)
    test_o = DataPreprocessing(test_data, name='test', force_run=True)

    basis = 'origin'
    nb_sweeps = 2

    train_not_normalized_data = feature_maps.origin(train_o.data)
    train = normalize(train_not_normalized_data, basis)
    val = feature_maps.origin(val_o.data)

    for chi in [2, 10, 20, 30, 50]:
        print()
        print('-'*100)
        print(f'𝜒: {chi}')
        c = CompressedWFS(chi, train, basis)
        for i in range(nb_sweeps):
            c.sweep()
        # overlap = get_truncation_overlap(c.wf_compressed, train, average=True)
        acc = get_accuracy(c.wf_compressed, val, val_o.labels, 10000)
        print(acc)
