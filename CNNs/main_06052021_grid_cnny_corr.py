# -*- coding: utf-8 -*-
"""
Created on Tue May 18 11:55:04 2021

@author: Hania
"""

# -*- coding: utf-8 -*-
"""
Created on Thu May  6 15:58:03 2021

@author: Hania

Sprawdzenie grida, zmiany po rozmowie z Agą 13.05.21

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
from useful_func_cnn import wczyt_dane_moje, make_cnn_model1, av_precision2, av_precision, make_cnn_grid_corr, wczytaj_testy
#chosen_repr = 'spektrogram'
#chosen_cnn = 'bulbul' 
import os
import pickle
import time
import sys
import tensorflow as tf
import json
import pandas as pd

# cos żeby się GPU nie dławiło: 
# =============================================================================
from tensorflow.compat.v1.keras.backend import set_session
config2 = tf.compat.v1.ConfigProto()
config2.gpu_options.per_process_gpu_memory_fraction = 0.3 #0.5
set_session(tf.compat.v1.Session(config=config2))

# =============================================================================
# =============================================================================
# config3 = tf.compat.v1.ConfigProto()
# config3.gpu_options.allow_growth=True
# set_session(tf.compat.v1.Session(config=config3))
# =============================================================================


#%%


# python main_04052021_cnny.py spektrogram bulbul batch_size epochs
# python main_06052021_grid_cnny_corr.py spektrogram 100

if __name__=='__main__':

    # Load configs
    chosen_repr = sys.argv[1] # 'spektrogram' #
    n_iter = sys.argv[2]


    exp_dir = os.path.join("..","results","grid", f"{chosen_repr}", time.strftime('%Y-%m-%d-%H-%M'))
    try:
        os.makedirs(exp_dir)
    except FileExistsError:
        pass
       
    
    start_time = time.time()
    logger_wrapper = LoggerWrapper(exp_dir)
    sys.stderr.write = logger_wrapper.log_errors
    logger_wrapper.logger.info(f'Running {sys.argv}')
    
    # cv - podział do randomizedsearcha, by nie robił mi CV. dX - train + valid. Oprócz tego są też osobno jako X_val i train_X
    cv_split, dX, dy, di, dX_test, dy_test, di_test, cv, X_val, y_val, train_X, train_y = wczyt_dane_moje(chosen_repr, config, 'wlasc') 
    
    print(chosen_repr)
    if chosen_repr=="spektrogram":
        in_shape = (63, 148, 1)
    if chosen_repr=="mel-spektrogram":
        in_shape = (60, 148, 1)    
        
    
    dX_resh = dX.reshape(dX.shape[0], in_shape[0], in_shape[1], 1).astype('float32')     # trzeba wrócić z liniowego, które miało isć do podst. klasyf -> do obrazka 2d
    input_shape = (np.shape(dX_resh)[1], np.shape(dX_resh)[2], np.shape(dX_resh)[3]) # wielkoc obrazka, 1 kanał, na końcu podawany
    X_val_resh = X_val.reshape(X_val.shape[0], in_shape[0], in_shape[1], 1).astype('float32') 
    
    # upewnienie sie że działam na GPU
    #from tensorflow.python.client import device_lib
    #print(device_lib.list_local_devices())
   
    
    params = {
    'lr': [0.0001, 0.0005, 0.001, 0.005], # początkowy LR do Adama 0.01 za duży jest
    'input_shape': [in_shape],
    'filters': [10, 25, 50, 100, 150],
    'layers': [3, 4],  # ilosć warstw [konwolucja +pooling] 3, 4, 5,
    'drop_out': [0, 0.1, 0.2, 0.3, 0.4, 0.5],  # dropout po dwóch warstwach gęstych
    'if_dropout': [0, 0.1, 0.2], # parametry do spatialDropout2D. dropout po poolingach.
    # Internety mówią że rzadziej się daje dropout po conv,a jesli już to małe wartoci
    'batch_size': [16, 32, 64, 128],
    'epochs': [100],#[5, 8, 10, 15, 20, 30, 50],
    #'padding': ["same"],  # "valid",  "same" - nie ma sensu, bo to pooling najbardziej zmniejsza reprezentację, nie conv i padding.
    # Trzeba by zmienić archotekturę i wrzucać więcj conv, anie na przemian conv i pool
    'dense_layer_sizes': [128, 256], # rozmiar pierwszej warstwy gęstej
    #'if_scheduler': [0, 1]
    }
    
       
     
    model = KerasClassifier(build_fn = make_cnn_grid_corr, verbose=1)
    logger_wrapper.logger.info('macierz badanych parametrów')
    logger_wrapper.logger.info(params)
    
    with open(os.path.join(exp_dir,"matrix_params.json"), "w") as json_file:
        json.dump(params, json_file)  # json_file.write(params.to_json(indent=2))    
    
    my_callbacks = [
             tf.keras.callbacks.EarlyStopping(monitor="val_loss",patience=5, restore_best_weights = True),
         ]


    grid = RandomizedSearchCV(estimator=model, param_distributions=params, scoring = make_scorer(average_precision_score),
                        cv = cv, n_jobs=1, n_iter=int(n_iter), verbose = 3, random_state =667, refit = False) #
     # nie refitujemy, bo wtedy valid będzie i w treningu i w walidzie, więć early stopping nie będzie miał sensu
    grid_result = grid.fit(dX_resh, dy, validation_data=(X_val_resh, y_val.astype(np.float32)), callbacks = my_callbacks)
   
    
    print(grid_result.best_params_)
# =============================================================================
#   Tak było wczesniej, bez early stoppingu.
  
#     grid = RandomizedSearchCV(estimator=model, param_distributions=params, scoring = make_scorer(average_precision_score),
#                         cv = cv, n_jobs=1, n_iter=int(n_iter), verbose = 3, random_state =667) #
#     grid_result = grid.fit(dX_resh, dy)
#     
# =============================================================================
    
    logger_wrapper.logger.info('grid search trwał (s): ')
    logger_wrapper.logger.info(time.time() - start_time)
    
    best_params = grid_result.best_params_
    
    with open(os.path.join(exp_dir,"best_params.json"), "w") as json_file:
        json.dump(best_params, json_file, indent=1)    
    
        
    pd.DataFrame(grid_result.cv_results_['params']).to_pickle(os.path.join(exp_dir,"all_models.pkl")) 
     
    pd.DataFrame(grid_result.cv_results_['split0_test_score']).to_pickle(os.path.join(exp_dir,"all_models_scores.pkl") )
    
    
    def display_cv_results(search_results):
        print('Best score = {:.4f} using {}'.format(search_results.best_score_, search_results.best_params_))
        means = search_results.cv_results_['mean_test_score']
        stds = search_results.cv_results_['std_test_score']
        params = search_results.cv_results_['params']
        for mean, stdev, param in zip(means, stds, params):
            print('mean test accuracy +/- std = {:.4f} +/- {:.4f} with: {}'.format(mean, stdev, param))    
    
    display_cv_results(grid_result)   
    
    

        
    # odczyt:  pd.read_pickle("best20.pkl") 
    logger_wrapper.logger.info('All params:')
    logger_wrapper.logger.info(params)
    logger_wrapper.logger.info('Best params:', best_params)
    logger_wrapper.logger.info( best_params)
    logger_wrapper.logger.info(grid_result.cv_results_)
    
    
    # trzeba zrefitować na najlepszym zestawie, bo 
    #" This RandomizedSearchCV instance was initialized with refit=False. predict_proba is available only after refitting on the best parameters.
    # You can refit an estimator manually using the ``best_params_`` attribute" 
    # refit - ale trening dalej na trainie, a nie na całosci train + walid
    train_X_resh = train_X.reshape(train_X.shape[0], in_shape[0], in_shape[1], 1).astype('float32')
    
    model_refit = make_cnn_grid_corr(input_shape = best_params['input_shape'],
                            lr = best_params['lr'],
                            filters = best_params['filters'],
                            drop_out = best_params['drop_out'],
                            layers = best_params['layers'],
                            if_dropout = best_params['if_dropout'],
                            dense_layer_sizes = best_params['dense_layer_sizes']
                            )
    hist = model_refit.fit(train_X_resh, train_y.astype(np.float32), validation_data=(X_val_resh, y_val.astype(np.float32)), callbacks = my_callbacks,
                                   epochs=best_params['epochs'], batch_size=best_params['batch_size'])
    
    
     # zapisz model
    model_refit.save(os.path.join(exp_dir,"my-model.h5"))
    #grid_result.best_estimator_.model.save(os.path.join(exp_dir,"my-model.h5"))    
    print("Saved model to disk")
    
    # wczytanie testu
    testing_features, testing_target = wczytaj_testy(chosen_repr)  
    testing_features_resh = testing_features.reshape(testing_features.shape[0], in_shape[0], in_shape[1], 1).astype('float32')
    
    # zapis predykcji
    save_predictions(dX_resh, dy, cv_split, testing_features_resh, testing_target, model_refit, exp_dir)
    
    y_preds = model_refit.predict_proba(testing_features_resh) 

    # zapisanie historii, by dało się wyrysować krzywe uczenia w przyszłosci , loss i pr_auc
    with open(os.path.join(exp_dir,'history-z-refita.pckl'), 'wb') as file_pi:
        pickle.dump(hist.history, file_pi)
    # jak wczytać:   
    #f = open('history.pckl', 'rb')
    # = pickle.load(f)
    #f.close()
 # %%     
 
    model_json = (model_refit.to_json(indent=2))
    with open(os.path.join(exp_dir,"architecture-only-refit.json"), "w") as json_file:
        json_file.write(model_json)
        
 # %%    
    prec_nb, recall_nb, _ = precision_recall_curve(testing_target, y_preds)
    fpr_nb, tpr_nb, _ = roc_curve(testing_target, y_preds)
    print('AUC PR: ', average_precision_score(testing_target, y_preds))
    
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
    

    
    # summarize history for loss
    fig = plt.figure(figsize=(10,10))
    plt.plot(hist.history['loss'])
    plt.plot(hist.history['val_loss'])
    plt.title('Model loss')
    plt.ylabel('Koszt (loss)')
    plt.xlabel('Liczba epok')
    plt.legend(['trening','walid'], loc='upper right')
   # plt.show()
    fig.savefig(os.path.join(exp_dir,'uczenie-loss-refit'+ time.strftime('%Y-%m-%d-%H-%M')))
    
    auc_nr = 'auc_'+str(int(n_iter) )
    val_auc_nr = 'val_auc_'+str(int(n_iter) )
    
    fig = plt.figure(figsize=(10,10))
    plt.plot(hist.history[auc_nr])
    plt.plot(hist.history[val_auc_nr])
    plt.title('Model PR AUC')
    plt.ylabel('Pole pod krzywą PR')
    plt.xlabel('Liczba epok')
    plt.legend(['trening','walid'], loc='lower right')
    #plt.show()
    fig.savefig(os.path.join(exp_dir,'uczenie-PR-AUC-refit' + time.strftime('%Y-%m-%d-%H-%M')))
    
    