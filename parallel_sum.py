import numpy as np
import matplotlib.pyplot as plt

import data_loader
import mps_func


class DataProcessing:
    def __init__(self, dataset, even_distribution=False, max_nb_images=None):
        self.max_nb_images = max_nb_images
        self.original_images, self.labels = dataset

        self.images = self.transform(self.original_images)

        # if even_distribution:
        #     self.sorted_images = self.get_digits(self.images, self.labels)
        #     self.sorted_images = self.create_even_distribution(self.sorted_images)

        self.images_dict = self.separate_digits(self.images, self.labels)

    @staticmethod
    def flat_to_2d(batch):
        H = W = int(np.sqrt(batch.shape[-1]))
        assert H * W == batch.shape[-1]
        return batch.reshape((-1, H, W))

    @staticmethod
    def square_to_flat(batch):
        H = batch.shape[-1]
        return batch.reshape((-1, H ** 2))

    def mean_pooling(self, images_, filter_size):
        images_2d = self.flat_to_2d(images_)

        import skimage.measure
        pooled_image = skimage.measure.block_reduce(images_2d, (1, filter_size, filter_size), np.mean)

        return self.square_to_flat(pooled_image)

    def transform(self, images_):
        course_grain_data = self.mean_pooling(images_, 2)
        return course_grain_data
        # scaled = self.scale(course_grain, domain=(0, 1))
        # mapped = self.mapping(scaled)
        # return mapped

    @staticmethod
    def separate_digits(images, labels):
        images_dict = {}
        for label in range(10):
            images_dict[label] = images[labels==[label]]

        return images_dict


    # @staticmethod
    # def scale(images_, domain=(0, 255), range_=(0, 1)):
    #     scaled = images_ / (domain[1] - domain[0]) * (range_[1] - range_[0])
    #     return scaled

    # @staticmethod
    # def mapping(mat):
    #     qstate = np.array([np.exp(mat * 3.j * np.pi / 2) * np.cos(mat * np.pi / 2), np.exp(-mat * 3.j * np.pi / 2) * np.sin(mat * np.pi / 2)])
    #     # qstate = np.array([np.cos(mat * np.pi / 2), np.sin(mat * np.pi / 2)])
    #     return qstate.transpose(1, 2, 0)

    # @staticmethod
    # def get_digits(mat, labels_):
    #     sorted_digits_ = []
    #     for label in np.unique(labels_):
    #         label_slice = (labels_ == label)
    #         sorted_digits_.append(mat[label_slice])

    #     return sorted_digits_

    # def create_even_distribution(self, batch):
    #     if self.max_nb_images is None:
    #         smallest_batch_size = np.min([len(dbatch) for dbatch in batch])
    #         even_batch = np.array([dbatch[:smallest_batch_size] for dbatch in batch]).copy()
    #         return even_batch
    #     else:
    #         even_batch = np.array([dbatch[:self.max_nb_images] for dbatch in batch]).copy()
    #         return even_batch



class MPS_model:
    def __init__(self, images_dict, chi):
        self.images_dict = images_dict
        self.chi = chi
        self.mps_dict = self.create_mps_dict_iter(chi)
        # Each mps from mps_dict[digit] is the full mps of that digits

        return

    def create_mps_dict(self):
        '''
        This function create the exact MPS of the whole dataset.
        This is inefficient.
        '''
        mps_dict = {}
        for label in self.images_dict.keys():
            images = self.images_dict[label]  # (num_data, num_pixels)
            num_data, num_pixels = images.shape
            phi_x = self.local_feature_map(images).transpose([1, 2, 0])  # [num_pixels, phys_dim, num_data]
            num_pixels, phys_dim, num_data = phi_x.shape

            expanded = np.zeros(phi_x.shape + phi_x.shape[-1:], dtype=phi_x.dtype)
            # [num_pixels, phys_dim, num_data, num_data]
            diagonals = np.diagonal(expanded, axis1=-2, axis2=-1)
            diagonals.setflags(write=True)
            diagonals[:] = phi_x

            mps_phi_x = [tensor for tensor in (expanded.transpose([0, 2, 1, 3]))[:]]
            # [num_data, phys_dim, num_data]
            one_vec = np.ones([num_data, 1])
            mps_phi_x[0] = np.tensordot(one_vec, mps_phi_x[0], [[0], [0]])
            mps_phi_x[-1] = np.tensordot(mps_phi_x[-1], one_vec, [[2], [0]])

            mps_dict[label] = mps_phi_x

        return mps_dict


    def create_mps_dict_iter(self, chi):
        mps_dict = {}
        for label in self.images_dict.keys():
            images = self.images_dict[label]  # (num_data, num_pixels)
            num_data, num_pixels = images.shape

            images_iter_list = [images[idx*chi : (idx+1)*chi] for idx in range(np.ceil(num_data/chi).astype(int))]
            mps_iter_list = []
            for batch_images in images_iter_list:
                phi_x = self.local_feature_map(batch_images).transpose([1, 2, 0])  # [num_pixels, phys_dim, num_data]
                num_pixels, phys_dim, num_batch_data = phi_x.shape

                expanded = np.zeros(phi_x.shape + phi_x.shape[-1:], dtype=phi_x.dtype)
                # [num_pixels, phys_dim, num_batch_data, num_batch_data]
                diagonals = np.diagonal(expanded, axis1=-2, axis2=-1)
                diagonals.setflags(write=True)
                diagonals[:] = phi_x

                mps_phi_x = [tensor for tensor in (expanded.transpose([0, 2, 1, 3]))[:]]
                # [num_data, phys_dim, num_batch_data]
                one_vec = np.ones([num_batch_data, 1])
                mps_phi_x[0] = np.tensordot(one_vec, mps_phi_x[0], [[0], [0]])
                mps_phi_x[-1] = np.tensordot(mps_phi_x[-1], one_vec, [[2], [0]])
                mps_phi_x, trunc_err = mps_func.right_canonicalize(mps_func.lpr_2_plr(mps_phi_x))
                mps_phi_x = mps_func.plr_2_lpr(mps_phi_x)
                mps_iter_list.append(mps_phi_x)

            while(len(mps_iter_list)>1):
                print("new iter")
                for mm in mps_iter_list:
                    print(mm[50].shape)

                new_mps_iter_list = []

                for iter_idx in range(len(mps_iter_list)//2):
                    mps1 = mps_iter_list[iter_idx*2 + 0]
                    mps2 = mps_iter_list[iter_idx*2 + 1]

                    new_mps = mps_func.addition_MPS_compression_variational([t.copy() for t in mps1], mps1, mps2, verbose=0)
                    new_mps_iter_list.append(new_mps)

                if len(mps_iter_list) % 2 != 0:
                    mps_iter_list[-1][-1] = mps_iter_list[-1][-1] * 0.5
                    new_mps_iter_list.append(mps_iter_list[-1])

                mps_iter_list = new_mps_iter_list

            assert len(mps_iter_list) == 1
            mps_dict[label] = mps_iter_list[0]

        return mps_dict

    @staticmethod
    def local_feature_map(X):
        return np.stack([np.cos(X * np.pi / 2), np.sin(X * np.pi / 2)], axis=-1)



training_data, val_data, test_data = data_loader.load_data('data/mnist.pkl.gz')

print("loaded")

pooled_training_data = DataProcessing(training_data)
pooled_val_data = DataProcessing(val_data)
pooled_test_data = DataProcessing(test_data)

print("pooled")

for chi in [30, 40, 50]:
    model = MPS_model(pooled_training_data.images_dict, chi=chi)

    import pickle; pickle.dump(model.mps_dict, open('/tuph/t30/space/ga63zuh/qTEBD/sampling_mps/training_chi%d.pkl' % chi, 'wb'))

# print("MPS created")
# print(mps_func.MPS_dot(model.mps_dict[0], model.mps_dict[0]))


