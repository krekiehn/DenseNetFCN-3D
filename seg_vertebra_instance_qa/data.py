# data generator for training

import os
import random as rd

import pandas as pd
import numpy as np
import tensorflow as tf
import skimage.measure
from scipy import ndimage
from SimpelNet import FCN_model

# source: https://stanford.edu/~shervine/blog/keras-how-to-generate-data-on-the-fly

# # default
#
# dataframe_file_path = r'S:\Data\VerSe_full\nnunet\output\281'
# dataframe_file = r'seg_instance_masks_dataframe_manuel.csv'
# data_path = r'S:\Data\VerSe_full\nnunet\output\281\segmentation_instance_masks'
from seg_vertebra_instance_qa import dataframe_file_path, dataframe_file, data_path

rd.seed = 1


class DataGenerator(tf.keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, df, batch_size=3, n_channels=1,
                 n_classes=3, shuffle=True, data_path=data_path, partition='train', aug_max_shift=0.1, min_shape_size = 12):
        'Initialization'
        #self.dim = dim
        self.batch_size = batch_size
        assert self.batch_size % 3 == 0, 'batch_size have to be a multiple of three.'
        #self.labels = labels
        #self.list_IDs = list_IDs
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.shuffle = shuffle
        self.min_shape_size = min_shape_size

        self.partition = partition
        self.df = df.loc[df.partition == self.partition]
        df_neg, df_pos_frac, df_pos_no_frac, df_implants = self.split_cases_in_classes(self.df)

        self.list_dfs = [df_neg, df_pos_frac, df_pos_no_frac, df_implants]
        self.list_df_indices = [[]]*4
        self.dict_df_2_list_index = {'df_neg': 0, 'df_pos_frac': 1, 'df_pos_no_frac': 2, 'df_implants': 3}
        self.data_path = data_path

        self.aug_max_shift = aug_max_shift

        self.on_epoch_end()
        self.len = self.__len__()

    def __len__(self):
        'Denotes the number of batches per epoch'
        lens = []
        for v in self.dict_df_2_list_index.values():
            lens.append(int(np.floor(len(self.list_df_indices[v]) / self.batch_size)))
        return np.max(lens)

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        # indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]
        indices_4_batch_list = []
        for i in range(self.batch_size // 3):
            if len(self.list_df_indices[0]) == 0:
                self.re_init_indices_df(0)
            indices_4_batch_list.append(self.list_df_indices[0].pop())

            pos_random_source_index = rd.randint(1, 2)
            if len(self.list_df_indices[pos_random_source_index]) == 0:
                self.re_init_indices_df(pos_random_source_index)
            indices_4_batch_list.append(self.list_df_indices[pos_random_source_index].pop())

            if len(self.list_df_indices[3]) == 0:
                self.re_init_indices_df(3)
            indices_4_batch_list.append(self.list_df_indices[3].pop())

        rd.shuffle(indices_4_batch_list)
        # Generate data
        X, y = self.__data_generation(indices_4_batch_list)

        assert np.isnan(X).sum() == 0, f"NaNs: {np.isnan(X).sum()}, X {indices_4_batch_list} includes NaN. {X.shape}"
        assert np.isnan(y).sum() == 0, f"y {indices_4_batch_list} includes NaN. {y}"
        return X, y

    def re_init_indices_df(self, index):
        # source: df_neg, df_pos_frac, df_pos_no_frac, df_implants
        # assert source in ['df_neg', 'df_pos_frac', 'df_pos_no_frac', 'df_implants']
        # index = self.dict_df_2_list_index[source_df]
        self.list_df_indices[index] = list(self.list_dfs[index].index.values)
        if self.shuffle:
            rd.shuffle(self.list_df_indices[index])

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        for v in self.dict_df_2_list_index.values():
            self.re_init_indices_df(v)
        # self.indexes = np.arange(len(self.list_IDs))
        # if self.shuffle == True:
        #     np.random.shuffle(self.indexes)

    def __data_generation(self, list_IDs_temp):
        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
        # Initialization
        X_list = []
        y_list = []

        for ID in list_IDs_temp:
            file_name = self.df.loc[ID].rel_file
            # Store sample
            arr = np.load(os.path.join(self.data_path, file_name))
            # normalize
            arr_norm = arr/arr.max()
            arr_norm = arr_norm*2 - 1
            # arr_norm = (arr_norm - arr_norm.mean())/arr.std()
            X_list.append(arr_norm)
            # ToDo: add some kind of augmentation here. e.g. Rotation, Zoom, noise for segmentation mask???
            # add noise: values between arr_norm.min and arr_norm.max() - do this by model architecture
            if self.partition == 'train':
                # rot90 0 to 3 times for 0 to 3 axis
                # how many axis?
                num_axis = rd.randint(0, 3)
                axes = [0, 1, 2]
                rd.shuffle(axes)
                for i in range(num_axis):
                    axis = axes.pop()
                    axes_4_rot = [0, 1, 2]
                    axes_4_rot.remove(axis)  # remaining axes build up the prependicular plane to the rotation axis
                    rd.shuffle(axes_4_rot)  # direction of rotation. clockwise or counter clockwise is randomly used
                    times = rd.randint(0, 3)  # rot 0 to 3 times. 4 would like 0
                    arr_norm = np.rot90(arr_norm, times, axes=axes_4_rot)
                # flip for 0 to 1 time for 0 to 3 axis
                # how many axis?
                num_axis = rd.randint(0, 3)
                axes = [0, 1, 2]
                rd.shuffle(axes)
                for i in range(num_axis):
                    axis = axes.pop()
                    times = rd.randint(0, 1)  # rot 0 to 1 times. flip or not flip
                    if times:
                        arr_norm = np.flip(arr_norm, axis=axis)

                # shift by 0 to xx % of pixel-length a long axis for 0 to 3 axis # make sure, there are positive values left in the array
                # how many axis?
                num_axis = rd.randint(0, 3)
                axes = [0, 1, 2]
                rd.shuffle(axes)
                # shift by 0 to max 10? % of side length? make max shift a func argument
                for i in range(num_axis):
                    axis = axes.pop()
                    max_length = arr_norm.shape[axis]*self.aug_max_shift
                    factor = 1 - rd.random()*2

                    arr_norm_tmp = np.zeros(arr_norm.shape) - 1
                    shift_index = int(np.round(max_length*abs(factor)))
                    index_list = [slice(None), slice(None), slice(None)]
                    index_list_tmp = [slice(None), slice(None), slice(None)]
                    index_list_tmp[axis] = slice(shift_index, None)
                    index_list[axis] = slice(None, arr_norm.shape[axis] - shift_index)
                    if factor > 0:  # shift to the "right" side along axis
                        arr_norm_tmp[index_list_tmp] = arr_norm[index_list]
                    elif factor < 0:
                        arr_norm_tmp[index_list] = arr_norm[index_list_tmp]
                    else:
                        continue
                    # check if at least one pixel positive value
                    if np.any(arr == 1):
                        arr_norm = arr_norm_tmp
                # zoom in or out by xxx % of what?

            # Store class
            # y_class = -1
            if self.df.loc[ID].fracture_status == 'x':
                y_class = 2
            elif self.df.loc[ID].status == 0:
                y_class = 0
            elif self.df.loc[ID].status == 1:
                y_class = 1
            else:
                raise ValueError(f'The data belonging to ID {ID} are either an implant nor the status is 0 or 1. status: {self.df.loc[ID].status}')
            y_list.append(y_class)

        # get the max image shape
        max_shape = list(max(image.shape[x] for image in X_list) for x in range(3))
        for index, scalar in enumerate(max_shape):
            if scalar < self.min_shape_size:
                max_shape[index] = self.min_shape_size
        max_shape = tuple(max_shape)
        
        self.dim = max_shape
        print(self.dim)
        # Initialization
        # X = np.empty((self.batch_size, *self.dim, self.n_channels))
        X = np.zeros((self.batch_size, *self.dim, self.n_channels)) - 1  # empty values are signed by vbaölue "-1"
        y = np.empty((self.batch_size), dtype=int)

        for image_index, image in enumerate(X_list):
            # Store sample
            offset_list = [self.dim[i] - image.shape[i] for i in range(len(self.dim))]
            offset_random_list = [rd.randint(0, offset_list[i]) for i in range(len(self.dim))]
            X[image_index,
            offset_random_list[0]:offset_random_list[0]+image.shape[0],
            offset_random_list[1]:offset_random_list[1]+image.shape[1],
            offset_random_list[2]:offset_random_list[2]+image.shape[2],
            0] = image
            # Store class
            y[image_index] = y_list[image_index]

        return X, tf.keras.utils.to_categorical(y, num_classes=self.n_classes)
    
    def split_cases_in_classes(self, df):
        # file_instance = r'S:\Data\VerSe_full\nnunet\output\281\seg_instance_masks_dataframe.csv'
        # df = pd.read_csv(file_instance, dtype={"Ids": str})

        # df.columns
        # Index(['index', 'Ids', 'label', 'status', 'file', 'peeling', 'gt_labels',
        #        'fracture_status'],
        #       dtype='object')
        df_neg = df.loc[(df.status == 1) & (df.gt_labels != 1)]
        print(f'Number of negative cases: {len(df_neg)}')
        # df_fracture = df.dropna()
        df_fracture = df.loc[df.fracture_status.dropna().index]
        df_pos_no_frac = df_fracture.loc[(df_fracture.fracture_status == '0') | (df_fracture.fracture_status == '0.0')]
        print(f'Number of positive cases without fracture: {len(df_pos_no_frac)}')
        df_pos_frac = df_fracture.loc[(df_fracture.fracture_status != '0') & (df_fracture.fracture_status != '0.0')]
        print(f'Number of positive cases with fracture: {len(df_pos_frac)}')
        df_implants = df.loc[df.fracture_status == 'x']
        print(f'Number of cases with implants: {len(df_implants)}')
        return df_neg, df_pos_frac, df_pos_no_frac, df_implants
    


class DataGenerator_performance(tf.keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, df, batch_size=1, n_channels=1,
                 n_classes=3, data_path=data_path, partition='train', min_shape_size=12):
        'Initialization'
        self.batch_size = batch_size
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.min_shape_size = min_shape_size
        
        self.partition = partition
        self.df = df.loc[df.partition == self.partition]
        
        self.data_path = data_path
        self.ids = []
        
        self.on_epoch_end()
        self.len = self.__len__()
        self.step_counter = 0

    def __len__(self):
        'Denotes the number of batches per epoch'
        lens = int(np.floor(self.df.partition.count() / self.batch_size))
        return lens

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        # indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]
        print(f'step in epoch: {self.step_counter} / {self.len}')
        self.step_counter += 1
        indices_4_batch_list = []
        for i in range(self.batch_size):
            if not self.ids:
                self.on_epoch_end()
            indices_4_batch_list.append(self.ids.pop(0))

        # Generate data
        X, y = self.__data_generation(indices_4_batch_list)

        print(f"Shape of X: {X.shape}")
        assert np.isnan(X).sum() == 0, f"NaNs: {np.isnan(X).sum()}, X {indices_4_batch_list} includes NaN. {X.shape}"
        assert np.isnan(y).sum() == 0, f"y {indices_4_batch_list} includes NaN. {y}"
        return X, y

    def re_init_indices_df(self, index):
        # source: df_neg, df_pos_frac, df_pos_no_frac, df_implants
        # assert source in ['df_neg', 'df_pos_frac', 'df_pos_no_frac', 'df_implants']
        # index = self.dict_df_2_list_index[source_df]
        self.list_df_indices[index] = list(self.list_dfs[index].index.values)
        if self.shuffle:
            rd.shuffle(self.list_df_indices[index])

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.ids = list(self.df.index)
        self.step_counter = 0

    def __data_generation(self, list_IDs_temp):
        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
        # Initialization
        X_list = []
        y_list = []

        for ID in list_IDs_temp:
            file_name = self.df.loc[ID].rel_file
            # Store sample
            arr = np.load(os.path.join(self.data_path, file_name))
            # normalize
            arr_norm = arr/arr.max()
            arr_norm = arr_norm*2 - 1
            # arr_norm = (arr_norm - arr_norm.mean())/arr.std()
            X_list.append(arr_norm)
            # ToDo: add some kind of augmentation here. e.g. Rotation, Zoom, noise for segmentation mask???
            # add noise: values between arr_norm.min and arr_norm.max() - do this by model architecture

            # Store class
            # y_class = -1
            if self.df.loc[ID].fracture_status == 'x':
                y_class = 2
            elif self.df.loc[ID].status == 0:
                y_class = 0
            elif self.df.loc[ID].status == 1:
                y_class = 1
            else:
                raise ValueError(f'The data belonging to ID {ID} are either an implant nor the status is 0 or 1. status: {self.df.loc[ID].status}')
            y_list.append(y_class)

        # get the max image shape
        max_shape = list(max(image.shape[x] for image in X_list) for x in range(3))
        for index, scalar in enumerate(max_shape):
            if scalar < self.min_shape_size:
                max_shape[index] = self.min_shape_size
        max_shape = tuple(max_shape)
        
        self.dim = max_shape
        print(self.dim)
        # Initialization
        # X = np.empty((self.batch_size, *self.dim, self.n_channels))
        X = np.zeros((self.batch_size, *self.dim, self.n_channels)) - 1  # empty values are signed by vbaölue "-1"
        y = np.empty((self.batch_size), dtype=int)

        for image_index, image in enumerate(X_list):
            # Store sample
            offset_list = [self.dim[i] - image.shape[i] for i in range(len(self.dim))]
            offset_random_list = [rd.randint(0, offset_list[i]) for i in range(len(self.dim))]
            X[image_index,
            offset_random_list[0]:offset_random_list[0]+image.shape[0],
            offset_random_list[1]:offset_random_list[1]+image.shape[1],
            offset_random_list[2]:offset_random_list[2]+image.shape[2],
            0] = image
            # Store class
            y[image_index] = y_list[image_index]

        return X, tf.keras.utils.to_categorical(y, num_classes=self.n_classes)



class DataGenerator_4_classes(tf.keras.utils.Sequence):
    'Generates data for Keras'

    def __init__(self, df, batch_size=4, n_channels=2,
                 n_classes=4, shuffle=True, data_path=data_path, partition='train', aug_max_shift=0.1,
                 min_shape_size=16, mode=None, down_sampling=False, caching=True, data_source_mode='indirect'):
        'Initialization'
        self.batch_size = batch_size
        assert self.batch_size % 4 == 0, 'batch_size have to be a multiple of four.'
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.shuffle = shuffle
        self.min_shape_size = min_shape_size
        self.mode = mode
        self.data_path = data_path
        self.data_source_mode = data_source_mode
        if self.data_source_mode == 'direct':
            self.data_path = os.path.dirname(self.data_path)
        self.down_sampling = down_sampling
        self.caching = caching
        if self.caching:
            self.cache = {}

        self.partition = partition
        self.df = df.loc[df.partition == self.partition]
        self.df_classes_split_list = self.split_cases_in_classes()
        self.list_df_indices = [[] for i in range(self.n_classes)]
        for cla in range(self.n_classes):
            self.re_init_indices_df(cla=cla)

        self.aug_max_shift = aug_max_shift

        self.on_epoch_end()
        self.len = self.__len__()

    def __len__(self):
        'Denotes the number of batches per epoch'
        lens = []
        for cla in range(self.n_classes):
            lens.append(int(np.floor(len(self.list_df_indices[cla]) / self.batch_size)))
        return np.max(lens) - 1

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        # indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]
        indices_4_batch_list = []
        for i in range(self.batch_size // self.n_classes):
            for cla in range(self.n_classes):
                if len(self.list_df_indices[cla]) == 0:
                    self.re_init_indices_df(cla=cla)
                indices_4_batch_list.append(self.list_df_indices[cla].pop())

        rd.shuffle(indices_4_batch_list)
        # Generate data
        X, y = self.__data_generation(indices_4_batch_list)

        if np.isnan(X).sum() != 0:
            for b in range(X.shape[0]):
                if np.isnan(X[b,:,:,:,0]).sum() != 0:
                    print('seg mask', indices_4_batch_list[b])
                if np.isnan(X[b, :, :, :, 1]).sum() != 0:
                    print('raw', indices_4_batch_list[b])
        assert np.isnan(X).sum() == 0, f"NaNs: {np.isnan(X).sum()}, X {indices_4_batch_list} includes NaN. {X.shape}"
        assert np.isnan(y).sum() == 0, f"y {indices_4_batch_list} includes NaN. {y}"
        return X, y

    def re_init_indices_df(self, cla):
        self.list_df_indices[cla] = list(self.df_classes_split_list[cla].index.values)
        if self.shuffle:
            rd.shuffle(self.list_df_indices[cla])

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        for cla in range(self.n_classes):
            self.re_init_indices_df(cla)

    def __load(self, file_name, file_name_bbox_ct_name, ID):
        if self.data_source_mode == 'indirect':
            arr = np.load(os.path.join(self.data_path, file_name))
            arr_ct = np.load(os.path.join(self.data_path, 'raw', file_name_bbox_ct_name))
        elif self.data_source_mode == 'direct':
            arr = np.load(os.path.join(self.data_path, file_name))
            arr_ct = np.load(os.path.join(self.data_path, file_name_bbox_ct_name))
        else:
            assert False, f"Error: unkown data_source_mode {self.data_source_mode}!"
        # set all values in seg mask to 1 (mask) or 0 (background)
        label = self.df.instance_id.loc[ID]
        arr[arr != label] = 0
        arr[arr == label] = 1

        if self.down_sampling:
            # do it only for case where all dim are higher than self.min_shape_size
            if all(np.array(arr.shape) > self.min_shape_size):
                # find short axis, which should not be down sampeld
                short_axis = np.array(arr.shape).argmin()
                # check if long axes = not short axis longer as 24:
                axes = [0, 1, 2]
                axes.remove(short_axis)
                if (arr.shape[axes[0]] >= 24) and (arr.shape[axes[1]] >= 24):
                    # define pooling kernel which ignore short axis
                    pooling_kernel = [2, 2, 2]
                    pooling_kernel[short_axis] = 1
                    pooling_kernel = tuple(pooling_kernel)

                    # pooling

                    def mean_round(x, axis=None):
                        return int(np.round(np.mean(x, axis=axis)))

                    arr = skimage.measure.block_reduce(arr, pooling_kernel, np.max)
                    arr_ct = skimage.measure.block_reduce(arr_ct, pooling_kernel, np.mean)

        if self.caching:
            self.cache[ID] = {'mask': arr, 'ct': arr_ct}
        return arr, arr_ct


    def __data_generation(self, list_IDs_temp):
        'Generates data containing batch_size samples'
        # X : (n_samples, *dim, n_channels)
        # Initialization
        X_list = []
        y_list = []
        #print(list_IDs_temp)
        for ID in list_IDs_temp:
            # todo: load bbox from original CT and do the SAME augmentation on it
            if self.data_source_mode == 'indirect':
                file_name = self.df.loc[ID].instance_file
                file_name_bbox_ct_name = file_name.split('.')[0] + '_raw.' + file_name.split('.')[1]
            elif self.data_source_mode == 'direct':
                file_name = self.df.rel_path_seg_file.loc[ID]
                file_name_bbox_ct_name = self.df.rel_path_raw_file.loc[ID]
            else:
                raise Exception(f"data_source_mode does not support: {self.data_source_mode}")
            # Store sample
            if self.caching:
                if ID in self.cache.keys():
                    arr = self.cache[ID]['mask']
                    arr_ct = self.cache[ID]['ct']
                else:
                    arr, arr_ct = self.__load(file_name, file_name_bbox_ct_name, ID)
                    # if self.data_source_mode == 'indirect':
                    #     arr = np.load(os.path.join(self.data_path, file_name))
                    #     arr_ct = np.load(os.path.join(self.data_path, 'raw', file_name_bbox_ct_name))
                    # elif self.data_source_mode == 'direct':
                    #     arr = np.load(os.path.join(self.data_path, file_name))
                    #     arr_ct = np.load(os.path.join(self.data_path, file_name_bbox_ct_name))
                    # # set all values in seg mask to 1 (mask) or 0 (background)
                    # label = self.df.instance_id.loc[ID]
                    # arr[arr != label] = 0
                    # arr[arr == label] = 1
                    # self.cache[ID] = {'mask': arr, 'ct': arr_ct}
            else:
                arr, arr_ct = self.__load(file_name, file_name_bbox_ct_name, ID)
                # arr = np.load(os.path.join(self.data_path, file_name))
                # arr_ct = np.load(os.path.join(self.data_path, 'raw', file_name_bbox_ct_name))
                # # set all values in seg mask to 1 (mask) or 0 (background)
                # label = self.df.instance_id.loc[ID]
                # arr[arr != label] = 0
                # arr[arr == label] = 1
            ## down_sample if options is set:
            # if self.down_sampling:
            #     #do it only for case where all dim are higher than self.min_shape_size
            #     if all(np.array(arr.shape) > self.min_shape_size):
            #         # find short axis, which should not be down sampeld
            #         short_axis = np.array(arr.shape).argmin()
            #         # check if long axes = not short axis longer as 24:
            #         axes = [0, 1, 2]
            #         axes.remove(short_axis)
            #         if (arr.shape[axes[0]] >= 24) and (arr.shape[axes[1]] >= 24):
            #             # define pooling kernel which ignore short axis
            #             pooling_kernel = [2, 2, 2]
            #             pooling_kernel[short_axis] = 1
            #             pooling_kernel = tuple(pooling_kernel)
            #             # pooling
            #             arr = skimage.measure.block_reduce(arr, pooling_kernel, np.mean)
            #             arr_ct = skimage.measure.block_reduce(arr_ct, pooling_kernel, np.mean)

            arr_norm = arr
            arr_ct_norm = arr_ct

            # ToDo: add some kind of augmentation here. e.g. Rotation, Zoom, noise for segmentation mask???
            # add noise: values between arr_norm.min and arr_norm.max() - do this by model architecture
            # if self.partition == 'train':
            if self.partition == 'false':
                # ToDo: add elastic deformity to seg mask, source: https://dltk.github.io/DLTK/api/dltk.io.html#dltk.io.augmentation.elastic_transform
                # rot90 0 to 3 times for 0 to 3 axis
                # how many axis?
                num_axis = rd.randint(0, 3)
                axes = [0, 1, 2]
                rd.shuffle(axes)
                for i in range(num_axis):
                    axis = axes.pop()
                    axes_4_rot = [0, 1, 2]
                    axes_4_rot.remove(axis)  # remaining axes build up the prependicular plane to the rotation axis
                    rd.shuffle(axes_4_rot)  # direction of rotation. clockwise or counter clockwise is randomly used
                    times = rd.randint(0, 3)  # rot 0 to 3 times. 4 would like 0
                    arr_norm = np.rot90(arr_norm, times, axes=axes_4_rot)
                    arr_ct_norm = np.rot90(arr_ct_norm, times, axes=axes_4_rot)
                # flip for 0 to 1 time for 0 to 3 axis
                # how many axis?
                num_axis = rd.randint(0, 3)
                axes = [0, 1, 2]
                rd.shuffle(axes)
                for i in range(num_axis):
                    axis = axes.pop()
                    times = rd.randint(0, 1)  # flip 0 to 1 times. flip or not flip
                    if times:
                        arr_norm = np.flip(arr_norm, axis=axis)
                        arr_ct_norm = np.flip(arr_ct_norm, axis=axis)

                # shift by 0 to xx % of pixel-length a long axis for 0 to 3 axis # make sure, there are positive values left in the array
                # how many axis?
                num_axis = rd.randint(0, 3)
                axes = [0, 1, 2]
                rd.shuffle(axes)
                # shift by 0 to max 10? % of side length? make max shift a func argument
                for i in range(num_axis):
                    axis = axes.pop()
                    max_length = arr_norm.shape[axis] * self.aug_max_shift
                    factor = 1 - rd.random() * 2

                    arr_norm_tmp = np.zeros(arr_norm.shape) - 1
                    arr_ct_norm_tmp = np.zeros(arr_ct_norm.shape) - 1
                    shift_index = int(np.round(max_length * abs(factor)))
                    index_list = [slice(None), slice(None), slice(None)]
                    index_list_tmp = [slice(None), slice(None), slice(None)]
                    index_list_tmp[axis] = slice(shift_index, None)
                    index_list[axis] = slice(None, arr_norm.shape[axis] - shift_index)
                    index_list = tuple(index_list)
                    index_list_tmp = tuple(index_list_tmp)
                    if factor > 0:  # shift to the "right" side along axis
                        arr_norm_tmp[index_list_tmp] = arr_norm[index_list]
                        arr_ct_norm_tmp[index_list_tmp] = arr_ct_norm[index_list]
                    elif factor < 0:
                        arr_norm_tmp[index_list] = arr_norm[index_list_tmp]
                        arr_ct_norm_tmp[index_list] = arr_ct_norm[index_list_tmp]
                    else:
                        continue
                    # check if at least one pixel positive value
                    if np.any(arr_norm_tmp == 1):
                        arr_norm = arr_norm_tmp
                        arr_ct_norm = arr_ct_norm_tmp
                # zoom in or out by xxx % of what?

                #Rotation by free degree:
                if np.array(arr_ct.shape).min() >= self.min_shape_size:
                    # define axes
                    axes = [0,1,2]
                    axis = rd.choice(axes)
                    axes.remove(axis)
                    axes_rot = tuple(axes)
                    # define some rotation angles
                    angle = rd.randint(-45, 45)
                    arr_norm = self.augmentation_rotate_volume(axes=axes_rot, angle=angle, volume=arr_norm, mode='int')
                    arr_ct_norm = self.augmentation_rotate_volume(axes=axes_rot, angle=angle, volume=arr_ct_norm, mode='float')

            
            # normalize to [0,1]
            # arr_norm = (arr - arr.min()) / (arr - arr.min()).max()
            # assert (arr_ct_norm - arr_ct_norm.min()).max() > 0, f"all values zero for ct with index {ID}."
            # arr_ct_norm = (arr_ct_norm - arr_ct_norm.min()) / (arr_ct_norm - arr_ct_norm.min()).max()
            # normilaize to houndsfield unit range bone
            limit_bottom = -1000
            arr_ct_norm[arr_ct_norm < limit_bottom] = limit_bottom
            limit_up = 1000
            arr_ct_norm[arr_ct_norm > limit_up] = limit_up
            arr_ct_norm = (arr_ct_norm - limit_bottom) / (limit_up - limit_bottom)
            # normalize to [-1,1]
            arr_norm = arr_norm * 2 - 1
            arr_ct_norm = arr_ct_norm * 2 - 1
            # arr_norm = (arr_norm - arr_norm.mean())/arr.std()

            X_list.append(np.concatenate((arr_norm[...,None],arr_ct_norm[...,None]), axis=3))
            
            # Store class
            y_list.append(self.df.label.loc[ID])
            #print(y_list[-1])

        # get the max image shape
        max_shape = list(max(image.shape[x] for image in X_list) for x in range(3))
        for index, scalar in enumerate(max_shape):
            if scalar < self.min_shape_size:
                max_shape[index] = self.min_shape_size
        max_shape = tuple(max_shape)

        self.dim = max_shape
        print(self.dim)
        # Initialization
        # X = np.empty((self.batch_size, *self.dim, self.n_channels))
        X = np.zeros((self.batch_size, *self.dim, self.n_channels)) - 1  # empty values are signed by value "-1"
        y = np.empty((self.batch_size), dtype=int)

        for image_index, image in enumerate(X_list):
            # Store sample
            offset_list = [self.dim[i] - image.shape[i] for i in range(len(self.dim))]
            offset_random_list = [rd.randint(0, offset_list[i]) for i in range(len(self.dim))]
            X[image_index,
            offset_random_list[0]:offset_random_list[0] + image.shape[0],
            offset_random_list[1]:offset_random_list[1] + image.shape[1],
            offset_random_list[2]:offset_random_list[2] + image.shape[2],
            :] = image
            # Store class
            y[image_index] = y_list[image_index]

        return X, tf.keras.utils.to_categorical(y, num_classes=self.n_classes)

    def split_cases_in_classes(self):
        df_classes_split_list = []
        for cla in range(self.n_classes):
            _ = self.df.loc[(self.df.label == cla)]
            df_classes_split_list.append(_)
            print(f'Number of classes {cla} cases: {len(_)}')
        return df_classes_split_list

    def augmentation_rotate_volume(self, axes, angle, volume, mode='float'):
        # define axes
        # axes = [0,1,2]
        # axis = random.choice(axes)
        # axes.remove(axis)
        axes_rot = tuple(axes)
        # define some rotation angles
        # angle = random.randint(-20, 20)
        if angle == 0:
            return volume
        # rotate volume
        if mode=='float':
            volume = ndimage.rotate(volume, angle, reshape=False, axes=axes_rot)
            # volume[volume < 0] = 0
            # volume[volume > 1] = 1
        elif mode=='int':
            volume_ont_hot = tf.one_hot(volume, self.n_classes)
            # do rot
            volume_ont_hot = ndimage.rotate(volume_ont_hot, angle, reshape=False, axes=axes_rot)
            volume_ont_hot_softmax = tf.nn.softmax(volume_ont_hot)
            volume = np.argmax(volume_ont_hot_softmax, axis=-1)
        return volume


class DataGenerator_4_classes_performance(tf.keras.utils.Sequence):
    'Generates data for Keras'

    def __init__(self, df, batch_size=1, n_channels=2,
                 n_classes=4, shuffle=True, data_path=data_path, partition='vali', aug_max_shift=0.1,
                 min_shape_size=12, mode=None, down_sampling=False, caching=True, data_source_mode='indirect'):
        'Initialization'
        self.batch_size = batch_size
        # assert self.batch_size % 4 == 0, 'batch_size have to be a multiple of four.'
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.shuffle = shuffle
        self.min_shape_size = min_shape_size
        self.mode = mode
        self.data_path = data_path
        self.data_source_mode = data_source_mode
        if self.data_source_mode == 'direct':
            self.data_path = os.path.dirname(self.data_path)
        self.down_sampling = down_sampling
        self.caching = caching
        if self.caching:
            self.cache = {}

        self.partition = partition
        self.df = df.loc[(df.partition == self.partition) & (df.peeling == 1)]
        self.df_classes_split_list = self.split_cases_in_classes()
        self.list_df_indices = [[] for i in range(self.n_classes)]
        for cla in range(self.n_classes):
            self.re_init_indices_df(cla=cla)
        self.list_indices = list(self.df.index.values)

        self.aug_max_shift = aug_max_shift

        self.on_epoch_end()
        self.len = self.__len__()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.list_indices)  / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        # indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]
        indices_4_batch_list = []
        for i in range(self.batch_size):
            indices_4_batch_list.append(self.list_indices.pop())

        # Generate data
        X, y = self.__data_generation(indices_4_batch_list)

        if np.isnan(X).sum() != 0:
            for b in range(X.shape[0]):
                if np.isnan(X[b, :, :, :, 0]).sum() != 0:
                    print('seg mask', indices_4_batch_list[b])
                if np.isnan(X[b, :, :, :, 1]).sum() != 0:
                    print('raw', indices_4_batch_list[b])
        assert np.isnan(X).sum() == 0, f"NaNs: {np.isnan(X).sum()}, X {indices_4_batch_list} includes NaN. {X.shape}"
        assert np.isnan(y).sum() == 0, f"y {indices_4_batch_list} includes NaN. {y}"
        return X, y

    def re_init_indices_df(self, cla):
        self.list_df_indices[cla] = list(self.df_classes_split_list[cla].index.values)
        if self.shuffle:
            rd.shuffle(self.list_df_indices[cla])

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.list_indices = list(self.df.index.values)

    def __load(self, file_name, file_name_bbox_ct_name, ID):
        if self.data_source_mode == 'indirect':
            arr = np.load(os.path.join(self.data_path, file_name))
            arr_ct = np.load(os.path.join(self.data_path, 'raw', file_name_bbox_ct_name))
        elif self.data_source_mode == 'direct':
            arr = np.load(os.path.join(self.data_path, file_name))
            arr_ct = np.load(os.path.join(self.data_path, file_name_bbox_ct_name))
        else:
            assert False, f"Error: unkown data_source_mode {self.data_source_mode}!"
        # set all values in seg mask to 1 (mask) or 0 (background)
        label = self.df.instance_id.loc[ID]
        arr[arr != label] = 0
        arr[arr == label] = 1

        if self.down_sampling:
            # do it only for case where all dim are higher than self.min_shape_size
            if all(np.array(arr.shape) > self.min_shape_size):
                # find short axis, which should not be down sampeld
                short_axis = np.array(arr.shape).argmin()
                # check if long axes = not short axis longer as 24:
                axes = [0, 1, 2]
                axes.remove(short_axis)
                if (arr.shape[axes[0]] >= 24) and (arr.shape[axes[1]] >= 24):
                    # define pooling kernel which ignore short axis
                    pooling_kernel = [2, 2, 2]
                    pooling_kernel[short_axis] = 1
                    pooling_kernel = tuple(pooling_kernel)

                    # pooling

                    def mean_round(x, axis=None):
                        return int(np.round(np.mean(x, axis=axis)))

                    arr = skimage.measure.block_reduce(arr, pooling_kernel, np.max)
                    arr_ct = skimage.measure.block_reduce(arr_ct, pooling_kernel, np.mean)

        if self.caching:
            self.cache[ID] = {'mask': arr, 'ct': arr_ct}
        return arr, arr_ct

    def __data_generation(self, list_IDs_temp):
        'Generates data containing batch_size samples'
        # X : (n_samples, *dim, n_channels)
        # Initialization
        X_list = []
        y_list = []
        # print(list_IDs_temp)
        for ID in list_IDs_temp:
            # todo: load bbox from original CT and do the SAME augmentation on it
            if self.data_source_mode == 'indirect':
                file_name = self.df.loc[ID].instance_file
                file_name_bbox_ct_name = file_name.split('.')[0] + '_raw.' + file_name.split('.')[1]
            elif self.data_source_mode == 'direct':
                file_name = self.df.rel_path_seg_file.loc[ID]
                file_name_bbox_ct_name = self.df.rel_path_raw_file.loc[ID]
            else:
                raise Exception(f"data_source_mode does not support: {self.data_source_mode}")
            # Store sample
            if self.caching:
                if ID in self.cache.keys():
                    arr = self.cache[ID]['mask']
                    arr_ct = self.cache[ID]['ct']
                else:
                    arr, arr_ct = self.__load(file_name, file_name_bbox_ct_name, ID)
            else:
                arr, arr_ct = self.__load(file_name, file_name_bbox_ct_name, ID)

            arr_norm = arr
            arr_ct_norm = arr_ct

            # normalize to [0,1]
            # arr_norm = (arr - arr.min()) / (arr - arr.min()).max()
            # assert (arr_ct_norm - arr_ct_norm.min()).max() > 0, f"all values zero for ct with index {ID}."
            # arr_ct_norm = (arr_ct_norm - arr_ct_norm.min()) / (arr_ct_norm - arr_ct_norm.min()).max()
            # normilaize to houndsfield unit range bone
            limit_bottom = -1000
            arr_ct_norm[arr_ct_norm < limit_bottom] = limit_bottom
            limit_up = 1000
            arr_ct_norm[arr_ct_norm > limit_up] = limit_up
            arr_ct_norm = (arr_ct_norm - limit_bottom) / (limit_up - limit_bottom)
            # normalize to [-1,1]
            arr_norm = arr_norm * 2 - 1
            arr_ct_norm = arr_ct_norm * 2 - 1
            # arr_norm = (arr_norm - arr_norm.mean())/arr.std()

            X_list.append(np.concatenate((arr_norm[..., None], arr_ct_norm[..., None]), axis=3))

            # Store class
            y_list.append(self.df.label.loc[ID])
            # print(y_list[-1])

        # get the max image shape
        max_shape = list(max(image.shape[x] for image in X_list) for x in range(3))
        for index, scalar in enumerate(max_shape):
            if scalar < self.min_shape_size:
                max_shape[index] = self.min_shape_size
        max_shape = tuple(max_shape)

        self.dim = max_shape
        print(self.dim)
        # Initialization
        # X = np.empty((self.batch_size, *self.dim, self.n_channels))
        X = np.zeros((self.batch_size, *self.dim, self.n_channels)) - 1  # empty values are signed by value "-1"
        y = np.empty((self.batch_size), dtype=int)

        for image_index, image in enumerate(X_list):
            # Store sample
            offset_list = [self.dim[i] - image.shape[i] for i in range(len(self.dim))]
            offset_random_list = [rd.randint(0, offset_list[i]) for i in range(len(self.dim))]
            X[image_index,
            offset_random_list[0]:offset_random_list[0] + image.shape[0],
            offset_random_list[1]:offset_random_list[1] + image.shape[1],
            offset_random_list[2]:offset_random_list[2] + image.shape[2],
            :] = image
            # Store class
            y[image_index] = y_list[image_index]

        return X, tf.keras.utils.to_categorical(y, num_classes=self.n_classes)

    def split_cases_in_classes(self):
        df_classes_split_list = []
        for cla in range(self.n_classes):
            _ = self.df.loc[(self.df.label == cla)]
            df_classes_split_list.append(_)
            print(f'Number of classes {cla} cases: {len(_)}')
        return df_classes_split_list


if __name__ == '__main__':
    # load data csv file:
    # df_data = pd.read_csv(os.path.join(dataframe_file_path, dataframe_file), index_col='index', dtype={"Ids": str, "status": int})
    # 
    # df_neg, df_pos_frac, df_pos_no_frac, df_implants = split_cases_in_classes(df_data)
    # 
    # test = DataGenerator(df_data)
    # 
    # X, y = test[0]
    # 
    # model_fcn = FCN_model(len_classes=3)
    # 
    # print(model_fcn.predict(X))
    # print(y)
    pass