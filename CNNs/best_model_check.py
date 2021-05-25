# -*- coding: utf-8 -*-
"""
Created on Tue May 25 12:59:30 2021

@author: Hania
 z best_model_from_random.py 
 
do sprawdzenia modelu wybranego z random searcha. Fit na trainie, early stopping dzięki walidowi. 

"""

import config
import numpy as np
from keras import backend as K
from keras.wrappers.scikit_learn import KerasClassifier
K.set_image_data_format('channels_last')
from sklearn.metrics import average_precision_score, make_scorer
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import roc_curve
import matplotlib.pyplot as plt
#from config_agi import parse_model_config

from saving_utils import save_predictions, save_as_json, LoggerWrapper
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from useful_func_cnn import wczyt_dane_moje, make_cnn_model1, av_precision2, av_precision, make_cnn_bulbul, make_cnn_github2019, make_cnn_grid_corr, wczytaj_testy
#chosen_repr = 'spektrogram'
#chosen_cnn = 'bulbul' 
import os
import pickle
import time
import sys
import tensorflow as tf
import json

#import tensorflow as tf
from tensorflow.compat.v1.keras.backend import set_session
config2 = tf.compat.v1.ConfigProto()
config2.gpu_options.per_process_gpu_memory_fraction = 0.5
set_session(tf.compat.v1.Session(config=config2))

# python best_model_from_random.py reprezentacja link_najlepsze_par ile_epok
# python best_model_check.py spektrogram ..\results\grid\spektrogram\2021-05-13-00-05\best_params.json 50
# python best_model_check.py spektrogram ..\results\grid\spektrogram\2021-05-19-23-05\best_params.json 100

if __name__=='__main__':

    # Load configs
    chosen_repr = sys.argv[1] # 'spektrogram' #
    with open(sys.argv[2], 'r') as fp:
        best_params = json.load(fp)
    epochs = int(sys.argv[3] )
    exp_dir = os.path.join("..","results","PR", f"{chosen_repr}", time.strftime('%Y-%m-%d-%H-%M'))
    try:
        os.makedirs(exp_dir)
    except FileExistsError:
        pass
       
    
    start_time = time.time()
    logger_wrapper = LoggerWrapper(exp_dir)
    sys.stderr.write = logger_wrapper.log_errors
    logger_wrapper.logger.info(f'Running {sys.argv}')
        
    cv_split, dX, dy, di, dX_test, dy_test, di_test, cv, X_val, y_val, train_X, train_y = wczyt_dane_moje(chosen_repr, config, 'wlasc') 
    
    print(chosen_repr)
    if chosen_repr=="spektrogram":
        in_shape = (63, 148, 1)
    if chosen_repr=="mel-spektrogram":
        in_shape = (60, 148, 1)    
        
    
    dX_resh = dX.reshape(dX.shape[0], in_shape[0], in_shape[1], 1).astype('float32')     # trzeba wrócić z liniowego, które miało isć do podst. klasyf -> do obrazka 2d
    input_shape = (np.shape(dX_resh)[1], np.shape(dX_resh)[2], np.shape(dX_resh)[3]) # wielkoc obrazka, 1 kanał, na końcu podawany
    
    
    from tensorflow.python.client import device_lib
    print(device_lib.list_local_devices())

    
    print("Fit model on training data")
    y_val = y_val.astype(int)
    train_y = train_y.astype(int)
    
    X_val_resh = X_val.reshape(X_val.shape[0],in_shape[0], in_shape[1], 1).astype('float32') 
    train_X_resh = train_X.reshape(train_X.shape[0], in_shape[0], in_shape[1], 1).astype('float32')
    
    model = make_cnn_grid_corr(input_shape = best_params['input_shape'],
                                lr = best_params['lr'],
                                filters = best_params['filters'],
                                drop_out = best_params['drop_out'],
                                layers = best_params['layers'],
                                if_dropout = best_params['if_dropout'],
                                dense_layer_sizes = best_params['dense_layer_sizes']
                               # activation_function = best_params['activation_function']
                                ) #input_shape = best_params['input_shape'])
    
    my_callbacks = [
    tf.keras.callbacks.EarlyStopping(monitor="val_auc", mode="max", patience=3, restore_best_weights = True),
    ]

    
    result = model.fit(train_X_resh, train_y, validation_data = (X_val_resh,y_val), epochs=best_params['epochs'],
                    batch_size=best_params['batch_size'],callbacks = my_callbacks, verbose = 1)
    
    
# %%    
    
    # wczytanie testu
    testing_features, testing_target = wczytaj_testy(chosen_repr)    
    testing_features_resh = testing_features.reshape(testing_features.shape[0], in_shape[0], in_shape[1], 1).astype('float32')
    
    # zapis predykcji
    save_predictions(dX_resh, dy, cv_split, testing_features_resh, testing_target, model, exp_dir)
    y_preds = model.predict_proba(testing_features_resh) 
    # %%  
    # serialize weights to HDF5
    model.save(os.path.join(exp_dir,"my-model.h5"))
  
    print("Saved model to disk")
    
    all_scores = {}
    all_scores['test_AUC_PR'] = average_precision_score(testing_target, y_preds)  #scorer(model, dX_test, dy_test)
    all_scores['test_AUC_ROC'] = roc_auc_score(testing_target, y_preds)
    
    
    print('Z GRIDA: ',all_scores)
    
    
    logger_wrapper.logger.info('z grida')
    logger_wrapper.logger.info(all_scores)
    
       
    # zapisanie historii, by dało się wyrysować krzywe uczenia w przyszłosci , loss i pr_auc
    with open(os.path.join(exp_dir,'history.pckl'), 'wb') as file_pi:
        pickle.dump(result.history, file_pi)
    # jak wczytać:   
    #f = open('history.pckl', 'rb')
    # = pickle.load(f)
    #f.close()
   
    model_json = (model.to_json(indent=0))
    with open(os.path.join(exp_dir,"architecture-only-refit.json"), "w") as json_file:
        json_file.write(model_json)

    # wyrys krzywych roc auc i pr auc 
    prec_nb, recall_nb, _ = precision_recall_curve(testing_target, y_preds)
    fpr_nb, tpr_nb, _ = roc_curve(testing_target, y_preds)
    fig = plt.figure(figsize=(10,10))
    plt.rcParams.update({'font.size': 22})
    plt.title('AUC PR')
    plt.plot(recall_nb, prec_nb,  'b', label = 'CNN AUC = %0.2f%% ' % (100*average_precision_score(testing_target, y_preds)))
    plt.legend(loc = 'upper right')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.ylabel('Precyzja (Precision)')
    plt.xlabel('Czułość (Recall)')
    #plt.show()
    fig.savefig(os.path.join(exp_dir,'AUC-PR-refit'+ time.strftime('%Y-%m-%d-%H-%M')))
    
    fig = plt.figure(figsize=(10,10))
    plt.rcParams.update({'font.size': 22})
    plt.title('AUC ROC')
    plt.plot(fpr_nb, tpr_nb, 'b', label = 'CNN AUC = %0.2f%%' % (100*roc_auc_score(testing_target, y_preds)))
    plt.legend(loc = 'lower right')
    plt.plot([0, 1], [0, 1],'r--')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.ylabel('Wskaźnik czułości (True Positive Rate)')
    plt.xlabel('Odsetek fałszywie pozytywnych alarmów (False Positive Rate)')
    #plt.show()
    fig.savefig(os.path.join(exp_dir,'AUC-ROC-refit'+ time.strftime('%Y-%m-%d-%H-%M')))

    # wyrys krzywych uczenia dla loss i pr auc
    # summarize history for loss
    fig = plt.figure(figsize=(10,10))
    plt.plot(result.history['loss'])
    plt.plot(result.history['val_loss'])
    plt.title('Model loss')
    plt.ylabel('Koszt (loss)')
    plt.xlabel('Liczba epok')
    plt.legend(['trening','walidacja'], loc='upper right')
   # plt.show()
    fig.savefig(os.path.join(exp_dir,'uczenie-loss-refit'+ time.strftime('%Y-%m-%d-%H-%M')))
    
    #auc_nr = 'auc_'+str(int(n_iter) + 1)
    
    fig = plt.figure(figsize=(10,10))
    plt.plot(result.history['auc'])
    plt.plot(result.history['val_auc'])
    plt.title('Model PR AUC')
    plt.ylabel('Pole pod krzywą PR')
    plt.xlabel('Liczba epok')
    plt.legend(['trening','walidacja'], loc='lower right')
    #plt.show()
    fig.savefig(os.path.join(exp_dir,'uczenie-PR-AUC-refit' + time.strftime('%Y-%m-%d-%H-%M')))
    
    