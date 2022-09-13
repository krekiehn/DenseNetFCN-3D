# data generator for training

import os
import random as rd

import pandas as pd
import numpy as np
import tensorflow as tf

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
                 n_classes=3, shuffle=True, data_path=data_path, partition='train'):
        'Initialization'
        #self.dim = dim
        self.batch_size = batch_size
        assert self.batch_size % 3 == 0, 'batch_size have to be a multiple of three.'
        #self.labels = labels
        #self.list_IDs = list_IDs
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.shuffle = shuffle

        self.partition = partition
        self.df = df.loc[df.partition == self.partition]
        df_neg, df_pos_frac, df_pos_no_frac, df_implants = split_cases_in_classes(self.df)

        self.list_dfs = [df_neg, df_pos_frac, df_pos_no_frac, df_implants]
        self.list_df_indices = [[]]*4
        self.dict_df_2_list_index = {'df_neg': 0, 'df_pos_frac': 1, 'df_pos_no_frac': 2, 'df_implants': 3}
        self.data_path = data_path

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
            # arr_norm = (arr_norm - arr_norm.mean())/arr.std()
            X_list.append(arr_norm)
            # ToDo: add some kind of augmentation here. e.g. Rotation, Zoom, noise for segmentation mask???
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
        max_shape = tuple(max(image.shape[x] for image in X_list) for x in range(3))
        self.dim = max_shape
        print(self.dim)
        # Initialization
        X = np.empty((self.batch_size, *self.dim, self.n_channels))
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


def split_cases_in_classes(df):
    # file_instance = r'S:\Data\VerSe_full\nnunet\output\281\seg_instance_masks_dataframe.csv'
    # df = pd.read_csv(file_instance, dtype={"Ids": str})

    # df.columns
    # Index(['index', 'Ids', 'label', 'status', 'file', 'peeling', 'gt_labels',
    #        'fracture_status'],
    #       dtype='object')
    df_neg = df.loc[(df.status == 1) & (df.gt_labels != 1)]
    print(f'Number of negative cases: {len(df_neg)}')
    #df_fracture = df.dropna()
    df_fracture = df.loc[df.fracture_status.dropna().index]
    df_pos_no_frac = df_fracture.loc[(df_fracture.fracture_status == '0') | (df_fracture.fracture_status == '0.0')]
    print(f'Number of positive cases without fracture: {len(df_pos_no_frac)}')
    df_pos_frac = df_fracture.loc[(df_fracture.fracture_status != '0') & (df_fracture.fracture_status != '0.0')]
    print(f'Number of positive cases with fracture: {len(df_pos_frac)}')
    df_implants = df.loc[df.fracture_status == 'x']
    print(f'Number of cases with implants: {len(df_implants)}')
    return df_neg, df_pos_frac, df_pos_no_frac, df_implants


if __name__ == '__main__':
    # load data csv file:
    df_data = pd.read_csv(os.path.join(dataframe_file_path, dataframe_file), index_col='index', dtype={"Ids": str, "status": int})

    df_neg, df_pos_frac, df_pos_no_frac, df_implants = split_cases_in_classes(df_data)

    test = DataGenerator(df_data)

    X, y = test[0]

    model_fcn = FCN_model(len_classes=3)

    print(model_fcn.predict(X))
    print(y)