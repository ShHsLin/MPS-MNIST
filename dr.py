import os
import sys
import time
from contextlib import contextmanager

import matplotlib.pyplot as plt
import pandas as pd
from cycler import cycler
from numba import jit
from tqdm import notebook

from data_loader import *
import skimage.measure
fontsize = 12
plt.rcParams['axes.labelsize'] = fontsize
plt.rcParams['axes.titlesize'] = fontsize
plt.rcParams['axes.prop_cycle'] = cycler('color', ['#4C72B0', '#55A868', '#C44E52', '#8172B2', '#CCB974', '#64B5CD'])

plt.rcParams['legend.fontsize'] = fontsize
plt.rcParams['legend.title_fontsize'] = fontsize
plt.rcParams['xtick.labelsize'] = fontsize
plt.rcParams['ytick.labelsize'] = fontsize
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['mathtext.fontset'] = 'cm'
plt.rcParams['mathtext.default'] = 'regular'
plt.rcParams['text.usetex'] = 'false'
# plt.rcParams['axes.grid'] = 'True'
plt.rcParams['figure.figsize'] = 5, 5


@contextmanager
def suppress_stdout():
    with open(os.devnull, "w") as devnull:
        old_stdout = sys.stdout
        sys.stdout = devnull
        try:
            yield
        finally:
            sys.stdout = old_stdout


def timeit(func):
    def inner(*args, **kwargs):
        t0 = time.time()
        result = func(*args, **kwargs)
        print('duration of "{}": {}'.format(func.__name__, time.time() - t0))
        return result

    return inner


def save_data_to_pickle(func, arguments, file_path, run_always=False):
    if os.path.exists(file_path) and not run_always:

        with open(file_path, 'rb') as f:
            print('Data already exists, loading data...')
            result = pickle.load(f)

    else:
        if not os.path.exists(os.path.dirname(file_path)):
            print(os.path.dirname(file_path))
            os.makedirs(os.path.dirname(file_path))
        result = func(*arguments)
        with open(file_path, 'wb') as f:
            pickle.dump(result, f)

    return result


class DataProcessing:
    def __init__(self, dataset, mapping_basis, even_distribution=False, max_nb_images=None):
        assert mapping_basis in ['orthogonal', 'original', 'sin_cos', 'sines',
                                 'sines2'], 'mapping_basis should be one of ["orthogonal", "original", "sin_cos"]'
        self.mapping_basis = mapping_basis
        self.max_nb_images = max_nb_images
        self.original_images, self.labels = dataset

        self.images = self.transform(self.original_images)

        if even_distribution:
            self.sorted_images = self.get_digits(self.images, self.labels)
            self.sorted_images = self.create_even_distribution(self.sorted_images)

    @staticmethod
    def flat_to_2d(batch):
        H = W = int(np.sqrt(batch.shape[-1]))
        return batch.reshape((-1, H, W))

    @staticmethod
    def square_to_flat(batch):
        H = batch.shape[-1]
        return batch.reshape((-1, H ** 2))

    def mean_pooling(self, images_, filter_size):
        images_2d = self.flat_to_2d(images_)
        pooled_image = skimage.measure.block_reduce(images_2d, (1, filter_size, filter_size), np.mean)
        result = self.square_to_flat(pooled_image)
        return result

    @staticmethod
    def scale(images_, domain=(0, 255), range_=(0, 1)):
        scaled = images_ / (domain[1] - domain[0]) * (range_[1] - range_[0]) + range_[0]
        return scaled

    def mapping(self, mat):
        if self.mapping_basis == 'orthogonal':
            qstate = np.array([np.exp(mat * 3.j * np.pi / 2) * np.cos(mat * np.pi / 2),
                               np.exp(-mat * 3.j * np.pi / 2) * np.sin(mat * np.pi / 2)])
        elif self.mapping_basis == 'original':
            qstate = np.array([np.cos(mat * np.pi / 2), np.sin(mat * np.pi / 2)])
        elif self.mapping_basis == 'sin_cos':
            qstate = np.array([np.sin(np.pi * 1 / 2 * mat),
                               np.cos(np.pi * 1 / 2 * mat),
                               np.sin(np.pi * 3 / 2 * mat),
                               np.cos(np.pi * 3 / 2 * mat)])

        elif self.mapping_basis == 'sines':
            qstate = np.array([np.sin(np.pi * 1 / 2 * mat),
                               np.sin(np.pi * 3 / 2 * mat),
                               np.sin(np.pi * 5 / 2 * mat),
                               np.sin(np.pi * 7 / 2 * mat)])
            qstate /= np.linalg.norm(qstate, axis=0)

        elif self.mapping_basis == 'sines2':
            qstate = np.array([np.sin(np.pi * 1 * mat),
                               np.sin(np.pi * 2 * mat),
                               np.sin(np.pi * 3 * mat),
                               np.sin(np.pi * 4 * mat)])

            # print(np.linalg.norm(qstate, axis=0).shape)
            qstate /= (np.linalg.norm(qstate, axis=0))
        else:
            print('Mapping basis not valid')
            return
        return qstate.transpose((1, 2, 0))

    @staticmethod
    def get_digits(mat, labels_):
        sorted_digits_ = []
        for label in np.unique(labels_):
            label_slice = (labels_ == label)
            sorted_digits_.append(mat[label_slice])

        return sorted_digits_

    def transform(self, images_):
        course_grain = self.mean_pooling(images_, 2)

        if self.mapping_basis == 'sines':
            scaled = self.scale(course_grain, domain=(0, 1), range_=(0.1, 1))
        elif self.mapping_basis == 'sines2':
            scaled = self.scale(course_grain, domain=(0, 1), range_=(0.1, 0.9))
        else:
            scaled = course_grain
        mapped = self.mapping(scaled)
        return mapped

    def create_even_distribution(self, batch):
        if self.max_nb_images is None:
            smallest_batch_size = np.min([len(dbatch) for dbatch in batch])
            even_batch = np.array([dbatch[:smallest_batch_size] for dbatch in batch]).copy()
            return even_batch
        else:
            even_batch = np.array([dbatch[:self.max_nb_images] for dbatch in batch]).copy()
            return even_batch

    def show_pic(self):
        pixels = self.sorted_images.shape[-2]
        h = int(pixels ** (1 / 2))

        for i in range(4):
            plt.figure()
            image = self.sorted_images[0, 0, :, i].reshape((h, h))
            plt.imshow(image)


class Normalization:
    def __init__(self, batch, mapping_basis):
        nb_images = batch[0].shape[0]
        norms_save_path = os.path.join('results', mapping_basis, 'norms.p')
        self.norms = save_data_to_pickle(self.get_norms_all_digits, (batch,), norms_save_path)
        self.wfs = self.normalize_all_digits(batch, self.norms)

    @staticmethod
    def normalize_all_digits(batch, norms):
        for i in range(len(batch)):
            batch[i] /= (norms[i] ** (1 / (2 * batch[i].shape[1])))
        return batch

    def get_norms_all_digits(self, batch):
        norms = []
        for digit_batch in notebook.tqdm_notebook(batch):
            norm = self.get_norm_single_digit(digit_batch)
            norms.append(norm)
        return np.array(norms)

    @staticmethod
    def get_norm_single_digit(digit_batch):
        @jit(nopython=True)
        def contract_physical(node_idx):
            return np.ascontiguousarray(digit_batch[:, node_idx, :]) @ np.ascontiguousarray(
                digit_batch[:, node_idx, :].T.conj())

        product = contract_physical(0)
        for i in range(1, 196):
            product *= contract_physical(i)
        norm = np.sum(product)
        norm = np.abs(norm)
        print(norm)
        return norm

    @staticmethod
    def get_overlaps_single_image(image_mps, dfunc):
        physical_contract = np.sum([image_mps[:, i].conj() * dfunc[:, :, :, i] for i in range(image_mps.shape[-1])],
                                   axis=0)
        virtual_contract = np.prod(physical_contract, 2)
        overlap = np.sum(virtual_contract, 1)
        overlap = np.abs(overlap)
        return overlap

    @timeit
    def get_accuracy_exact(self, test_images, test_labels, N=None):
        if N is None:
            N = len(test_images)
        preds = []
        for image in notebook.tqdm_notebook(test_images[:N]):
            pred = np.argmax(self.get_overlaps_single_image(image, self.wfs))
            preds.append(pred)
        preds = np.array(preds)
        accuracy = np.sum(test_labels[:N] == preds) / N

        return accuracy



class CompressedWFS:
    def __init__(self, ewfs, compression_path):
        self.ewfs = ewfs

        self.nb_basis_elements = self.ewfs[0].shape[-1]

        with open(compression_path, 'rb') as f:
            self.cwfs = list(pickle.load(f).values())
        self.cwfs_reshaped = self.reshape_cwfs()

    def get_prediction(self, image):
        braket = np.ones((10, 1, 1))
        for ket_i, bra_i in zip(self.cwfs_reshaped, image):
            physical_contract = np.sum([ket_i[:, :, i, :] * bra_i[i] for i in range(self.nb_basis_elements)], axis=0)
            # physical_contract = ket_i[:, :, 0, :] * bra_i[0].conj() + ket_i[:, :, 1, :] * bra_i[1].conj()
            braket = braket @ physical_contract

        prediction = np.argmax(np.absolute(braket[:, 0, 0]))
        return prediction

    def get_accuracy(self, test_images, test_labels, N=None):
        if N is None:
            N = len(test_images)

        preds = []
        for image in notebook.tqdm_notebook(test_images):
            preds.append(self.get_prediction(image))
        preds = np.array(preds)
        accuracy = np.sum(test_labels[:N] == preds) / N
        return accuracy

    def get_truncation_overlap(self):
        overlaps = []
        for cmps, emps in notebook.tqdm_notebook(zip(self.cwfs, self.ewfs)):
            contract = np.ones((1, 1))
            for i in range(len(cmps)):
                physical = np.tensordot(emps[:, i, :], cmps[i].conj(), (1, 1))
                contract = np.ascontiguousarray(contract) @ np.ascontiguousarray(physical)
            overlaps.append(np.linalg.norm(np.sum(contract)))

        return overlaps

    def reshape_cwfs(self):
        cwfs_reshaped = []
        for m_i in range(len(self.cwfs[0])):
            single_m = []
            for mps in self.cwfs:
                single_m.append(mps[m_i])
                print(mps[m_i].shape)
            single_m = np.array(single_m)
            cwfs_reshaped.append(single_m)
        return cwfs_reshaped


class Analysis:
    def __init__(self, ewfs_o, chimaxs, nb_sweeps, mapping_basis, val):
        self.val = val
        self.ewfs_o = ewfs_o
        self.chimaxs = chimaxs
        self.nb_sweeps = nb_sweeps
        self.mapping_basis = mapping_basis

        self.data_root = os.path.join('results', mapping_basis)

        self.compression_path = os.path.join(self.data_root, 'results.csv')
        self.df = self.load_df(self.compression_path,
                               ['chi_max', 'nb_sweeps', 'compression_duration', 'accuracy', 'truncation_overlap',
                                'nb_test_images'])
        self.df = self.df.reset_index(drop=True)

        print('Basis: ', mapping_basis)
        self.get_performance_vs_chimax()
        self.get_performance_vs_nb_images()

    def get_performance_vs_nb_images(self, nb_of_images=(4506)):
        nb_images_df_path = os.path.join(self.data_root, 'acc_vs_nb_path.csv')
        nb_images_df = self.load_df(nb_images_df_path, ['nb_images_exact', 'accuracy', 'truncation_overlap'])
        nb_images_df = nb_images_df.reset_index(drop=True)

        for nb_images in [4506, ]:
            print('Running: # of images: ', nb_images)
            if nb_images in nb_images_df['nb_images_exact'].astype(int).values:
                print('Already performed calc. skipping...')
                continue

            train_small = DataProcessing(train_data, even_distribution=True, max_nb_images=nb_images,
                                         mapping_basis=self.mapping_basis)
            ewfs_o_small = Normalization(train_small.sorted_images, self.mapping_basis)
            overlaps = self.get_overlap_empss_new(self.ewfs_o.wfs, ewfs_o_small.wfs)
            accuracy = ewfs_o_small.get_accuracy_exact(self.val.images, self.val.labels)
            # accuracy = 0.9209

            nb_images_df.loc[len(nb_images_df)] = [nb_images, accuracy, overlaps.tolist()]
            nb_images_df.to_csv(nb_images_df_path, sep=',')

    def get_overlap_empss_new(self, emps_full, emps_small):
        overlaps = []
        for digit in notebook.tqdm_notebook(range(len(emps_small))):
            overlap = self.get_overlap_single_digit(digit, emps_full, emps_small)
            overlaps.append(overlap)
        return np.array(overlaps)

    @staticmethod
    def get_overlap_single_digit(digit, emps_full, emps_small):
        @jit(nopython=True)
        def contract_physical(node_idx):
            return np.ascontiguousarray(emps_full[digit, :, node_idx, :]) @ np.ascontiguousarray(
                emps_small[digit, :, node_idx, :].T.conj())

        product = contract_physical(0)
        for i in range(1, 196):
            product *= contract_physical(i)
        overlap = np.sum(product)
        overlap = np.abs(overlap)
        return overlap

    @staticmethod
    def get_overlap_empss(emps_full, emps_small):

        nb_basis_elements = emps_full.shape[-1]
        overlaps = []
        for digit in notebook.tqdm_notebook(range(len(emps_small))):
            overlap = np.ones((emps_small.shape[1], emps_full.shape[1]))
            for m_i in range(emps_small.shape[2]):
                m_full = emps_full[digit, :, m_i, :]
                m_small = emps_small[digit, :, m_i, :]

                # virtual_contract_full = np.array([overlap * m_full[:,0], overlap * m_full[:,1]])
                virtual_contract_full = np.array([overlap * m_full[:, i] for i in range(nb_basis_elements)])
                # virtual_contract_small = np.array([virtual_contract_full * m_small[:,0][..., np.newaxis].conj(), virtual_contract_full * m_small[:,1][..., np.newaxis].conj()])
                virtual_contract_small = np.array(
                    [virtual_contract_full * m_small[:, i][..., np.newaxis].conj() for i in range(nb_basis_elements)])

                overlap = np.sum([virtual_contract_small[i, i, :] for i in range(nb_basis_elements)], axis=0)

                # overlap = virtual_contract_small[0,0,:] + virtual_contract_small[1,1,:]

            overlap = np.linalg.norm(np.sum(overlap))
            overlaps.append(overlap)
        return overlaps

    @staticmethod
    def load_df(path, columns):
        if not os.path.exists(path):
            df = pd.DataFrame(columns=columns)
        else:
            df = pd.read_csv(path, delimiter=',', index_col=0)
        return df

    def get_performance_vs_chimax(self):

        for chimax in self.chimaxs:
            print('Running chimax: ', chimax)
            all_sweeps_exists = self.check_if_all_sweeps_already_performed(chimax, self.nb_sweeps)
            # if all_sweeps_exists:
            #     print('Chimax already fully calculated')
            #     continue

            cwfs = CompressedWFS(chimax, ewfs=ewfs_o.wfs, mapping_basis=self.mapping_basis)
            for sweep in range(1, self.nb_sweeps + 1):
                print('Sweep: ', sweep)
                t0 = time.time()
                cwfs.sweep(sweep_number=sweep)
                sweep_duration = time.time() - t0
                print('Compression done.')
                exists = self.check_if_already_performed(chimax, sweep)
                if exists:
                    found_index, cwfs_path = exists
                    continue
                else:
                    print('Getting accuracy..')
                    accuracy = cwfs.get_accuracy(val.images, val.labels)
                    print('Getting truncation overlap..')
                    truncation_overlap = cwfs.get_truncation_overlap()
                    results = [chimax, sweep, sweep_duration, accuracy, truncation_overlap, len(val.labels)]
                    self.df.loc[len(self.df.index)] = results
                    self.df.to_csv(self.compression_path, sep=',')

    def check_if_all_sweeps_already_performed(self, chimax, max_sweep):

        for sweep in range(1, max_sweep + 1):
            df_found = self.df[(self.df['chi_max'] == chimax) & (self.df['nb_sweeps'] == sweep)]
            if not len(df_found):
                return False

            cwfs_file_name = 'cwfs_chi{}_s{}.p'.format(chimax, sweep)
            cwfs_path = os.path.join(self.data_root, 'cwfs', cwfs_file_name)
            if not os.path.exists(cwfs_path):
                return False

        return True

    def check_if_already_performed(self, chimax, sweep):
        results_exists = False
        cwfs_exists = False

        df_found = self.df[(self.df['chi_max'] == chimax) & (self.df['nb_sweeps'] == sweep)]
        if len(df_found) > 0:
            results_exists = True

        cwfs_file_name = 'cwfs_chi{}_s{}.p'.format(chimax, sweep)
        cwfs_path = os.path.join(self.data_root, 'cwfs', cwfs_file_name)
        if os.path.exists(cwfs_path):
            cwfs_exists = True

        if cwfs_exists & results_exists:
            return df_found.index, cwfs_path


if __name__ == '__main__':
    mnist_data_path = os.path.join('data', 'mnist.pkl.gz')

    data = load_data(mnist_data_path)
    train_data, val_data, test_data = data
