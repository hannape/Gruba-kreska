# -*- coding: utf-8 -*-
"""
Created on Thu Apr 22 13:54:13 2021

@author: Hania
"""

from sklearn.metrics import average_precision_score
from keras import backend as K
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten #Activation, 
from keras.layers import Conv2D, MaxPooling2D
from  sklearn.model_selection import PredefinedSplit
import keras

def av_precision(y_true, y_pred, normalize = False):
    #return tf.py_function(average_precision_score, (y_true, y_pred), tf.float64)   
    #return tf.convert_to_tensor(average_precision_score(y_true, y_pred), dtype=tf.float32)

#def sk_pr_auc(y_true, y_pred):
    return tf.py_function(average_precision_score, (y_true, y_pred), tf.float64)

# =============================================================================
# def av_precision3(y_true, y_pred, normalize = False):
# 
#     y_true = y_true.reshape((-1))
#     y_pred = y_pred.reshape((-1))
#     return tf.py_function(average_precision_score, (y_true, y_pred), tf.float64)
# 
# =============================================================================
def av_precision2(y_true, y_pred, normalize = False):
    return average_precision_score(y_true, y_pred)

def make_cnn_model1(input_shape = (63, 148, 1),  dense_layer_sizes=128, filters=10, kernel_size=(3,3),
                pool_size=(2,2), drop_out=0.5):
  
# =============================================================================
#     dense_layer_sizes=128
#     filters=10
#     kernel_size=(3,3) #(3,3)
#     pool_size= (2,2) 
#     lr=0.0001 
#     drop_out = 0.5
# =============================================================================
    #import tensorflow
    #from tensorflow import keras
    from keras.models import Sequential
    from keras.layers import Dense, Dropout, Flatten #Activation, 
    from keras.layers import Conv2D, MaxPooling2D
    model = Sequential()
    model.add(Conv2D(filters, kernel_size,input_shape=input_shape, activation='relu'))
    model.add(MaxPooling2D(pool_size=pool_size))
    
    model.add(Conv2D(filters, kernel_size, activation='relu'))
    model.add(MaxPooling2D(pool_size=pool_size))
        
    
    model.add(Flatten())
    model.add(Dense(dense_layer_sizes, activation='relu'))
    model.add(Dropout(drop_out))
    model.add(Dense(32, activation='relu'))
    model.add(Dropout(drop_out))
    model.add(Dense(1, activation='sigmoid'))
    
    model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=[tf.keras.metrics.AUC(curve='PR')])#  [av_precision]

    return model


def make_cnn_grid(input_shape = (63, 148, 1),  lr=0.001, filters = 10, drop_out = 0.5, layers = 3,
                  dense_layer_sizes=256, activation_function = 'relu' ):
  
# =============================================================================
#     'lr': [0.0005, 0.001, 0.01, 0.1],
#     'decay': [0, 0.01],
#     'filters': [10, 16, 20],
#     #kernel_size = [(3,3), (5,5)]
#     #pool_size = [(2,2), (3,3)]
#     'dropout': [0, 0.1, 0.35, 0.5],
#     'batch_size': [32, 64, 128],
#     'epochs': [10, 20, 30, 50],
#     'dense_layer_sizes': [128, 256],
#     'activation_function': ['LeakyReLU', 'ReLU']
# 
# =============================================================================
    kernel_size = (3,3)
    pool_size = (2,2)
    
    from keras.models import Sequential
    from keras.layers import Dense, Dropout, Flatten #Activation, 
    from keras.layers import Conv2D, MaxPooling2D
    model = Sequential()
    model.add(Conv2D(filters, kernel_size,input_shape=input_shape, activation=activation_function))
    model.add(MaxPooling2D(pool_size=pool_size))
    
    for i in range(layers-1):
        model.add(Conv2D(filters, kernel_size, activation=activation_function))
        model.add(MaxPooling2D(pool_size=pool_size))
       
    model.add(Flatten())
    model.add(Dense(dense_layer_sizes, activation=activation_function))
    model.add(Dropout(drop_out))
    model.add(Dense(32, activation=activation_function))
    model.add(Dropout(drop_out))
    model.add(Dense(1, activation='sigmoid'))
    
    model.compile(loss='binary_crossentropy',
                  optimizer=keras.optimizers.Adam(learning_rate = lr),
                  metrics=[tf.keras.metrics.AUC(curve='PR')])#  [av_precision]

    return model

def make_cnn_grid_corr(input_shape = (63, 148, 1),  lr=0.001, filters = 10, drop_out = 0.5, layers = 3,
                  dense_layer_sizes=256, activation_function = 'relu', if_dropout = 0 ): # , if_scheduler = 0
  
# =============================================================================
#     'lr': [0.0005, 0.001, 0.01, 0.1],
#     'decay': [0, 0.01],
#     'filters': [10, 16, 20],
#     #kernel_size = [(3,3), (5,5)]
#     #pool_size = [(2,2), (3,3)]
#     'dropout': [0, 0.1, 0.35, 0.5],
#     'batch_size': [32, 64, 128],
#     'epochs': [10, 20, 30, 50],
#     'dense_layer_sizes': [128, 256],
#     'activation_function': ['LeakyReLU', 'ReLU']
# 
# =============================================================================
    kernel_size = (3,3)
    pool_size = (2,2)
    
    from keras.models import Sequential
    from keras.layers import Dense, Dropout, Flatten #Activation, 
    from keras.layers import Conv2D, MaxPooling2D, SpatialDropout2D
    model = Sequential()
    model.add(Conv2D(filters, kernel_size,input_shape=input_shape, activation=activation_function))
    model.add(MaxPooling2D(pool_size=pool_size))
    model.add(SpatialDropout2D(if_dropout))
    
    for i in range(layers-1):
        model.add(Conv2D(filters, kernel_size, activation=activation_function))
        model.add(MaxPooling2D(pool_size=pool_size))
        model.add(SpatialDropout2D(if_dropout))
       
    model.add(Flatten())
    model.add(Dense(dense_layer_sizes, activation=activation_function))
    model.add(Dropout(drop_out))
    model.add(Dense(32, activation=activation_function))
    model.add(Dropout(drop_out))
    model.add(Dense(1, activation='sigmoid'))
    
# =============================================================================
#     if if_scheduler==1:
#         step = tf.Variable(0, trainable=False)
#         boundaries = [160000]
#         values = [lr, lr*0.1]
#         learning_rate_fn = keras.optimizers.schedules.PiecewiseConstantDecay(
#             boundaries, values)
#         
#         # Later, whenever we perform an optimization step, we pass in the step.
#         #learning_rate = learning_rate_fn(step)
#         optimizer=keras.optimizers.Adam(learning_rate = learning_rate_fn)
#     else:
#         optimizer=keras.optimizers.Adam(learning_rate = lr)
#     
# =============================================================================
    
    model.compile(loss='binary_crossentropy',
                  optimizer=keras.optimizers.Adam(learning_rate = lr),
                  metrics=[tf.keras.metrics.AUC(curve='PR')])#  [av_precision]

    return model



def make_cnn_bulbul(input_shape = (63, 148, 1),  drop_out=0.5):
  
 # dense_layer_sizes=128, filters=10, kernel_size=(3,3), pool_size=(2,2),   
# =============================================================================
#     dense_layer_sizes=128
#     filters=10
#     kernel_size=(3,3) #(3,3)
#     pool_size= (2,2) 
#     lr=0.0001 
#     drop_out = 0.5
# =============================================================================
    #import tensorflow
    #from tensorflow import keras

    model = Sequential()
    
    model.add(Conv2D(16, kernel_size=(3,3),input_shape=input_shape, activation='relu'))
    model.add(MaxPooling2D(pool_size=(3,3)))
    
    model.add(Conv2D(16, kernel_size=(3,3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(3,3)))
    
    model.add(Conv2D(16, kernel_size=(1,3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(1,3)))
    
    #model.add(Conv2D(16, kernel_size=(1,3), activation='relu'))
    #model.add(MaxPooling2D(pool_size=(1,3)))
        
    
    model.add(Flatten())
    model.add(Dropout(drop_out))
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(drop_out))
    model.add(Dense(32, activation='relu'))
    model.add(Dropout(drop_out))
    model.add(Dense(1, activation='sigmoid'))
    
    model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=[tf.keras.metrics.AUC(curve='PR')])#  [av_precision] daje nany w validzie

    return model


def make_cnn_github2019(input_shape = (63, 148, 1), dense_layer_sizes=128, filters=10, kernel_size=(3,3),
                pool_size=(2,2), drop_out=0.5):
    # https://github.com/hannape/CNN-second/blob/master/CNN_for_new_data.ipynb
    import functools
    from functools import partial, update_wrapper
    dense_layer_sizes=128
    filters=10
    kernel_size=(3,3) #(3,3)
    pool_size= (2,2) 
    #lr=0.0001 
    drop_out = 0.5
    
# ========= nie działa, jakie stare wersje kerasów, tfów i pythonów =================================
    def wrapped_partial(func, *args, **kwargs):
     	partial_func = partial(func, *args, **kwargs)
     	update_wrapper(partial_func, func)
     	return partial_func

    def binary_crossentropy_weigted(y_true, y_pred, class_weights):
     	y_pred = K.clip(y_pred, K.epsilon(), 1.0 - K.epsilon())
     	loss = K.mean(class_weights*(-y_true * K.log(y_pred) - (1.0 - y_true) * K.log(1.0 - y_pred)),axis=-1)
     	return loss
    
    custom_loss = wrapped_partial(binary_crossentropy_weigted, class_weights=np.array([0.02, 0.98])) ## scoring for model.compile
    
# =====https://datascience.stackexchange.com/questions/58735/weighted-binary-cross-entropy-loss-keras-implementation
           
    def weighted_bce(y_true, y_pred):
        weights = (y_true * 49) + 1
        bce = K.binary_crossentropy(y_true, y_pred)
        weighted_bce = K.mean(bce * weights)
        return weighted_bce
        
    
    model = Sequential()
    model.add(Conv2D(filters, kernel_size,input_shape=input_shape, activation='relu'))
   # model.add(Conv2D(16, (3, 3), input_shape=input_shape ))

    model.add(MaxPooling2D(pool_size=pool_size))
    model.add(Conv2D(filters, kernel_size, activation='relu'))
    model.add(MaxPooling2D(pool_size=pool_size))
    model.add(Conv2D(filters, kernel_size, activation='relu'))
    model.add(MaxPooling2D(pool_size=pool_size))
    
    
    model.add(Flatten())
    model.add(Dense(dense_layer_sizes, activation='relu'))
    model.add(Dropout(drop_out))
    model.add(Dense(32, activation='relu'))
    model.add(Dropout(drop_out))
    model.add(Dense(1, activation='sigmoid'))
    
    model.compile(loss= 'binary_crossentropy', #weighted_bce, #'binary_crossentropy',
                  optimizer='adam',#keras.optimizers.Adam(lr),#'adam',
                  metrics=['accuracy', tf.keras.metrics.AUC(curve='PR'), av_precision])

    return model




def wczyt_dane_moje(chosen_repr, config, set_type):
    import pandas as pd
    import os
    from useful_stuff import  make_test
    
    if set_type == 'wlasc':
        print("dane wlasciwe")
        df1 = pd.read_hdf(os.path.join('..','..','jupyter','data','df_train_'+ str(chosen_repr) +'_norm.h5'),'repr' + \
                          str((config.representation_2d + [''] + config.representation_1d).index(chosen_repr)+1))
    
        df2 = pd.read_hdf(os.path.join('..','..','jupyter','data','df_valid_norm.h5'),'df')
        all_repr = config.representation_1d + config.representation_2d
        all_repr.remove(chosen_repr)
        df2 = df2.drop(all_repr, axis=1)
        cv_split, dX, dy, di, cv = make_cv_split_cnn(df1, df2, chosen_repr)    
        print("Własciwy train+walid wczytany")
        
        valid = df2.loc[:,['rec_name', 'chunk_ids', chosen_repr, 'has_bird']].values
        train = df1.loc[:,['rec_name', 'chunk_ids', chosen_repr, 'has_bird']].values
        #valid_identifiers = valid[:, 0:2]
        valid_X = np.stack(valid[:, 2])
        valid_y = valid[:, 3]
        train_X = np.stack(train[:, 2])
        train_y = train[:, 3]
        
    
    if set_type == 'testy': 
        print("dane mini, do testow")
        df = pd.read_hdf(os.path.join('..','..','jupyter','data','df_val300_norm.h5'),'val')
        all_repr = config.representation_1d + config.representation_2d
        all_repr.remove(chosen_repr)
        df = df.drop(all_repr, axis=1)
        df1 = df[50:300]#df.sample(n=250, random_state = 667)
        df2 = df[0:50]#df.sample(n=50, random_state = 667)
        cv_split, dX, dy, di, cv = make_cv_split_cnn(df1, df2, chosen_repr) 
        print("Testowy train+walid wczytany")
        valid_X = []
        valid_y = []
        
    df3 = pd.read_hdf(os.path.join('..','..','jupyter','data','df_test_3rec_'+ str(chosen_repr) +'_norm.h5'),'df')
    dX_test, dy_test, di_test = make_test(df3, chosen_repr)
    #print("Test wczytany")
    
    return cv_split, dX, dy, di, dX_test, dy_test, di_test, cv, valid_X, valid_y, train_X, train_y


def make_cv_split_cnn(train, valid, chosen_repr, classifier=False):   
    
    # 
    train = train.loc[:,['rec_name', 'chunk_ids', chosen_repr, 'has_bird']].values
    train_identifiers = train[:, 0:2]
    train_X = np.stack(train[:, 2])
    train_y = train[:, 3]
    
    valid = valid.loc[:,['rec_name', 'chunk_ids', chosen_repr, 'has_bird']].values
    valid_identifiers = valid[:, 0:2]
    valid_X = np.stack(valid[:, 2])
    valid_y = valid[:, 3]

    dX = np.vstack((train_X, valid_X))
    print('min train+walid: ', np.amin(dX))
    dy = np.hstack((train_y, valid_y))
    di = np.vstack((train_identifiers, valid_identifiers))
    
    train_indices = np.array(range(0, train_X.shape[0]))
    val_indices = np.array(range(train_X.shape[0], dX.shape[0]))
    
    cv_split = [(train_indices, val_indices)] # ,(train_indices, val_indices)
    
    dX = np.reshape(dX, newshape=(dX.shape[0],-1))
    dy = dy.astype(int)
   
    # http://www.wellformedness.com/blog/using-a-fixed-training-development-test-split-in-sklearn/
    test_fold = np.concatenate([
    # The training data.
    np.full( train_X.shape[0],-1, dtype=np.int8),
    # The development data.
    np.zeros(valid_X.shape[0], dtype=np.int8)
    ])
    cv = PredefinedSplit(test_fold)

    print(test_fold)
    return cv_split, dX, dy, di, cv


def wczytaj_testy(chosen_repr):
    # najpierw wczyt podziału całosci
    from useful_stuff import make_cv_split, make_test  
    from useful_stuff import read_best_models_8_classic, read_best_models_8_classic_plus_MIR
    import pandas as pd
    import os
    import config
    from sklearn.metrics import roc_auc_score, average_precision_score
    from wczytanie_danych import funkcja_wczytanie_danych
    from reprezentacje import funkcja_reprezentacja
    import numpy as np
    import pandas as pd
    import json   
    import glob
    import time 
    
    start_time = time.time()
    print(config.path_test1618_txt)
    path_main = os.path.join('C:\\','Users','szaro','Desktop','jupyter')
    print(path_main)
    path_test1618_txt = os.path.join(path_main, config.path_test1618_txt)
    path_train161718_txt = os.path.join(path_main, config.path_train161718_txt)
    path_test1618_wav = os.path.join(path_main, config.path_test1618_wav)
    path_train161718_wav = os.path.join(path_main, config.path_train161718_wav)
    
    test_new_only = 1
    _,_,_, data_test_new = funkcja_wczytanie_danych(path_test1618_txt, path_train161718_txt, path_test1618_wav,\
                                                                                     path_train161718_wav, config.balance_types, config.balance_ratios, config.chunk_length_ms,\
                                                                                     config.chunk_overlap, config.calls_0, config.calls_1, config.calls_unknown, config.tolerance, config.valid_set,\
                                                                                     config.test_rec_to_cut, config.columns_dataframe, test_new_only)
    print("--- Funkcja wczytanie danych: %s sekund ---" % (time.time() - start_time))
    #  Wczytanie testowych reprezentacji
    import numpy as np
    
    start_time = time.time()
    np.warnings.filterwarnings('ignore', category=np.VisibleDeprecationWarning)
    
    
    ## !!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    if_scaler = True 
    #chosen_repr = '8_classic'
    nr_nagr_pocz = 0
    wielkosc_batcha = 18
    ## !!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    
    
    ## repr_full3, 
    file_names, indices, info_chunksy, repr_full_scaled = funkcja_reprezentacja(path_test1618_wav, data_test_new, chosen_repr, if_scaler, \
                                                                   config.repr_1d_summary, config.summary_1d, config.sr, config.chunk_length_ms, \
                                                                   config.n_fft, config.win_length, config.hop_length, config.window, config.f_min, config.f_max,\
                                                                   config.n_mels, config.N, config.step, config.Np, config.K, config.tm, \
                                                                   config.flock, config.tlock, nr_nagr_pocz, wielkosc_batcha)
    #file_names, indices, info_chunksy, representation_type, repr_full = funkcja_reprezentacje(path_train161718_wav, data_train, '8_classic', repr_1d_summary, summary_1d, sr, chunk_length_ms, \
    #                          n_fft, win_length, hop_length, window, f_min, f_max, n_mels, N, step, Np, K, tm, flock, tlock)
    
    print("--- Funkcja reprezentacja danych: %s minut ---" % ((time.time() - start_time)/60))
    
    
    file_names_list, chunk_ids_list, has_bird_list, representation_list = [],[],[],[]
    
    
    for num,i in enumerate(indices[nr_nagr_pocz:nr_nagr_pocz+wielkosc_batcha]):
    
      file_names_list.extend([file_names[num] for i in range(len(i))])
      chunk_ids_list.extend((info_chunksy[num][0]))  #.tolist()
      has_bird_list.extend((info_chunksy[num][3]))
      representation_list.extend([repr_full_scaled[num][i] for i in range(len(i))]) # cała reprezentacja1
      
      
    testing_features = np.array(representation_list) #pd.DataFrame(data = representation_list, columns =str(chosen_repr))
    testing_target = has_bird_list
    
    testing_features = np.reshape(testing_features, newshape=(testing_features.shape[0],-1))
    
    print("Test wczytany 18 nagran")

    return testing_features, testing_target




"""This file is part of the TPOT library.
TPOT was primarily developed at the University of Pennsylvania by:
    - Randal S. Olson (rso@randalolson.com)
    - Weixuan Fu (weixuanf@upenn.edu)
    - Daniel Angell (dpa34@drexel.edu)
    - and many more generous open source contributors
TPOT is free software: you can redistribute it and/or modify
it under the terms of the GNU Lesser General Public License as
published by the Free Software Foundation, either version 3 of
the License, or (at your option) any later version.
TPOT is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
GNU Lesser General Public License for more details.
You should have received a copy of the GNU Lesser General Public
License along with TPOT. If not, see <http://www.gnu.org/licenses/>.
"""

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils import check_array


class ZeroCount(BaseEstimator, TransformerMixin):
    """Adds the count of zeros and count of non-zeros per sample as features."""

    def fit(self, X, y=None):
        """Dummy function to fit in with the sklearn API."""
        return self

    def transform(self, X, y=None):
        """Transform data by adding two virtual features.
        Parameters
        ----------
        X: numpy ndarray, {n_samples, n_components}
            New data, where n_samples is the number of samples and n_components
            is the number of components.
        y: None
            Unused
        Returns
        -------
        X_transformed: array-like, shape (n_samples, n_features)
            The transformed feature set
        """
        X = check_array(X)
        n_features = X.shape[1]

        X_transformed = np.copy(X)

        non_zero_vector = np.count_nonzero(X_transformed, axis=1)
        non_zero = np.reshape(non_zero_vector, (-1, 1))
        zero_col = np.reshape(n_features - non_zero_vector, (-1, 1))

        X_transformed = np.hstack((non_zero, X_transformed))
        X_transformed = np.hstack((zero_col, X_transformed))

        return X_transformed