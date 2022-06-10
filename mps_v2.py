"""
dont put anything in init that takes time

1) class for data collection, converting to the correct mapping should be done seperately
2) normalization can be class, but no expensive calculations in it
3) get_exact accuracy: different object. All calculations for determining performance separate: overlap, accuracy
4) compression pretty much the same, but no accuracy calc. --> class needed?
5) check functions: time, save file, check if exists.
6) comments, documentation
"""
from useful_funcs import *
import numpy as np
import skimage
import matplotlib.pyplot as plt
from data_loader import load_data
from numba import jit
import feature_maps
import tqdm

class DataPreprocessing:
    """
    Performs mean pooling on every image. self.data contains the images as np.ndarray: (D,N,WxH):
    D: number of labels = 10
    N: number of images = 4506
    H: Length of Image in pixels
    W: Width of image in pixels
    """
    def __init__(self, dataset, even_distribution=False, max_nb_images=None, name='train', force_run=False):
        self.max_nb_images = max_nb_images
        self.original_images, self.labels = dataset

        images_course = self.mean_pooling(self.original_images, 2)
        self.data = save_data_to_pickle(self.get_digits, (images_course, self.labels), f'results/preprocessing/{name}.p', force_run=force_run)

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

    # def transform(self, images_):
    #     course_grain = self.mean_pooling(images_, 2)
    #     return course_grain

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


def overlap_diag():
    pass


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


def get_norm_label(label_data):
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

def get_norms(data, printout=False):
    norms = []
    for label_data in tqdm.tqdm(data):
        norm = get_norm_label(label_data)
        norms.append(norm)
        if printout:
            print(norm)
    return norms

def normalize(data_, basis=None, printout=False, force_run=False):
    data = data_.copy()
    norms = np.array(save_data_to_pickle(get_norms, (data,printout), f'results/{basis}/norm.p', force_run=force_run))
    for i in range(len(data)):
        data[i] /= (norms[i] ** (1 / (2 * data[i].shape[1])))
    if printout:
        print(norms)
    return data


def get_accuracy(test_images, test_labels, nb_tests_cap=None):
    if nb_tests_cap is None:
        nb_tests_cap = len(test_images)

    preds = []
    for image in tqdm.tqdm(test_images):
        preds.append(get_prediction(image))
    preds = np.array(preds)
    accuracy = np.sum(test_labels[:nb_tests_cap] == preds) / nb_tests_cap
    return accuracy

def get_truncation_overlap(data, data_exact):
    """
    cwfs: (list(N),list(D),L,P,R)
    ewfs: (D,N,HxW, P)
    @return:
    """
    overlaps = []
    for cmps, emps in tqdm.tqdm(zip(data, data_exact)):
        contract = np.ones((1, 1))
        for i in range(len(cmps)):
            """
            emps: (N,HxW,P)
            cmps[i]: (L,P,R)
            """
            physical = np.tensordot(emps[:, i, :], cmps[i].conj(), (1,1))
            """"
            (N,L,R)
            """
            contract = np.ascontiguousarray(contract) @ np.ascontiguousarray(physical)


        overlaps.append(np.linalg.norm(np.sum(contract)))

    return overlaps


if __name__ == '__main__':
    mnist_data_path = os.path.join('data', 'mnist.pkl.gz')

    data = load_data(mnist_data_path)
    train_data, val_data, test_data = data
    # train_data, val_data, test_data = 0,0,0

    train = DataPreprocessing(train_data, even_distribution=True, name='train')
    val = DataPreprocessing(val_data, name='val')
    test = DataPreprocessing(test_data, name='test')

    basis = 'ortho'
    train_orth = feature_maps.ortho(train.data)
    train_normalized = normalize(train_orth, basis)


