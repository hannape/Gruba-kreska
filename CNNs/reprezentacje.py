# -*- coding: utf-8 -*-
"""
Created on Tue Mar 23 13:29:58 2021

@author: Hania
"""

# IMPORTY
import joblib
import os

import numpy as np
import librosa
import time

def funkcja_reprezentacja(path_wav, data_settype, representation_type, if_scaler, repr_1d_summary, summary_1d, sr, chunk_length_ms, \
                          n_fft, win_length, hop_length, window, f_min, f_max, n_mels, N, step, Np, K, tm, flock, tlock, nr_nagr_pocz, wielkosc_batcha):
   
  '''
  Jest to funkcja zawierająca cały potrzebny kod z pliku 'reprezentacja śmieci/Untitled1', bez zbędnych printów, komentarzy itd.

  Args: 
    te wszystkie parametry wejściowe co powyżej, opisane w miarę w mainie

  Returns:
    file_names_train_set - nazwy nagrań wchodzące w skład zbioru
    indices - indeksy chunksów wybrane z poszczególnych nagrań
    info_chunksy - informacje o wybranych chunksach - te dane co wcześniej w dataframe, ale tylko dla wybranych chunksów 
      ['chunk_ids', 'chunk_start', 'chunk_end', 'has_bird', 'chunks_species', 'call_id', 'has_unknown', 'has_noise']
    representation_type - - to co na wejściu było, jaki jeden z pięciu typów reprezentacji
    repr_full  - wybrana reprezentacja dla każdego z wybranych chunksów z nagrań. Rozmiary:
      spektrogram: 1 nagranie - chunksy x 63 x 148
      mel-spektrogram: 1 nagranie - chunksy x 60 x 148
      multitaper: 1 nagranie - chunksy x 64 x 149
      8_classic: 1 nagranie - chunksy x 1-4 x 8
      8_classic_plus_MIR: 1 nagranie - chunksy x 1-4 x 39

  ''' 

  ## FUNKCJE
 
  def my_spektrogram(y, n_fft, win_length, hop_length, window, sr, f_min, f_max, scaler = None):
    stft = librosa.stft(y, n_fft=n_fft, win_length=win_length, hop_length=hop_length, window= window) # krótkoczasowa transformata fouriera STFT
    stft1 = librosa.amplitude_to_db(np.abs(stft)**2)                                              # kwadrat wartości bezwzględnej plus przerzucenie na skalę decybelową
    freqs = librosa.core.fft_frequencies(n_fft=n_fft, sr=sr)                             # wyznaczenie częstotliwości
    x,  = np.where( freqs >= min(freqs[(freqs >= f_min)]))
    j,  = np.where( freqs <= max(freqs[(freqs <= f_max)]))
    stft1 = stft1[min(x):max(j),]                                                        # stft tylko w wybranym zakresie
    #repr1_spectro = stft1
    if np.shape(stft1)[1]!= 148:
      stft1 = np.pad(stft1, ((0, 0), (0, 148 - np.shape(stft1)[1])), 'constant', constant_values=(-100))
      print("padding do ",np.shape(stft1))
    if (scaler!=None):
      return scaler.transform(stft1)   #repr1_spectro
    else:
      return stft1
 
 
  def my_melspektrogram(y, n_fft, win_length, hop_length, window, sr, f_min, f_max, n_mels, scaler = None):
    stft = librosa.stft(np.array(y), n_fft=n_fft, win_length=win_length, hop_length= hop_length, window=window)
    abs2_stft = (stft.real*stft.real) + (stft.imag*stft.imag)
    melspect = librosa.feature.melspectrogram(y=None, S=abs2_stft, sr=sr, n_mels= n_mels, fmin = f_min, fmax=f_max, hop_length=hop_length, n_fft=n_fft)
    repr2_melspec = 0.5 * librosa.amplitude_to_db(melspect, ref=1.0)
    if np.shape(repr2_melspec)[1]!= 148:
      repr2_melspec = np.pad(repr2_melspec, ((0, 0), (0, 148 - np.shape(repr2_melspec)[1])), 'constant', constant_values=(-50))    
    if (scaler!=None):
      return scaler.transform(repr2_melspec)   #repr1_spectro
    else:
      return repr2_melspec
  '''
  def my_multitaper(y, N, step, Np, K, tm, flock, tlock, f_min, f_max, scaler = None): 
    result5b = libtfr.tfr_spec(y, N = N, step = step, Np = Np, K = 2, tm = tm, flock = flock, tlock = tlock)     
    freqs, ind = libtfr.fgrid(sr, N, fpass=(f_min,f_max)) 
    repr3_multitaper = librosa.amplitude_to_db(result5b[ind,]); # tylko interesujące nas pasmo, w log
    #stft1 = librosa.amplitude_to_db(np.abs(stft)**2) 
    if np.shape(repr3_multitaper)[1]!= 149:
      repr3_multitaper = np.pad(repr3_multitaper, ((0, 0), (0, 149 - np.shape(repr3_multitaper)[1])), 'constant', constant_values=(-100))    
    if (scaler!=None):
      return scaler.transform(repr3_multitaper )   #repr1_spectro
    else:
      return repr3_multitaper
  '''
  def FeatureSpectralFlux(X):  #https://www.audiocontentanalysis.org/code/audio-features/spectral-flux-2/
    # difference spectrum (set first diff to zero)
    X = np.c_[X[:, 0], X]
    afDeltaX = np.diff(X, 1, axis=1)
    # flux
    vsf = np.sqrt((afDeltaX**2).sum(axis=0)) / X.shape[0]
    return  vsf[1:]                 # pozbycie się pierwszego elementu, który zawsze jest zerem , np.squeeze(vsf[1:]) if isSpectrum else

  def classic_base(y, n_fft, win_length, hop_length, window, sr, f_min, f_max, scaler4 = None):
    S1 = np.abs(librosa.stft(np.array(y), n_fft=n_fft, win_length=win_length, hop_length= hop_length, window=window))
    freqs = librosa.core.fft_frequencies(n_fft=n_fft, sr=sr) 
    o,  = np.where( freqs >= min(freqs[(freqs >= f_min)]))
    j,  = np.where( freqs <= max(freqs[(freqs <= f_max)]))
    freqs1 = freqs[min(o):max(j),]
    S = S1[min(o):max(j),] 

    param_0 = np.sum(S, axis=0)             #librosa.feature.spectral_bandwidth(S=S, p = 1)  # moc sygnału. Ale z wartości absolutnej spektorgramu, bez tego kwadratu jeszcz
    param_1 = librosa.feature.spectral_centroid(S=S, freq=freqs1)                            # centroid https://www.mathworks.com/help/audio/ref/spectralcentroid.html
    param_2 = np.power(librosa.feature.spectral_bandwidth(S=S, freq=freqs1, p = 2),2)        # 2 rzędu  
    param_3 = np.power(librosa.feature.spectral_bandwidth(S=S, freq=freqs1, p = 3),3)        # 3 rzędu
    param_4 = np.power(librosa.feature.spectral_bandwidth(S=S, freq=freqs1, p = 4),4)        # 4 rzędu
    skosnosc = param_3[0] / np.power(param_2[0],1.5)                        # https://www.mathworks.com/help/audio/ref/spectralskewness.html #skosnosc2 = skew(S, axis=0)
    kurtoza =  param_4[0]/ np.power(param_2[0],2) - 3                       #kurtoza2 = kurtosis(S, axis=0)
    plaskosc = librosa.feature.spectral_flatness(S=S)                       #gmean(S_squared)/np.mean(S_squared)

    return param_0, param_1, param_2, param_3, param_4, skosnosc, kurtoza, plaskosc, S


  def my_8_classic(y, n_fft, win_length, hop_length, window, sr, f_min, f_max, summary_1d, scaler4 = None): ## TO DO scalers

    param_0, param_1, param_2, param_3, param_4, skosnosc, kurtoza, plaskosc, _ = classic_base(y, n_fft, win_length, hop_length, window, sr, f_min, f_max, scaler4)
    nb_summary = np.sum(summary_1d)
    paramsy = [[[] for _ in range(8)] for _ in range(nb_summary)]
    idx = 0

    for m in range(np.shape(summary_1d)[0]):
      if summary_1d[m]:
        f = getattr(np, repr_1d_summary[m])
        paramsy[idx]=[f(param_0), f(param_1), f(param_2), f(param_3), f(param_4), f(skosnosc), f(kurtoza), f(plaskosc)]
        idx += 1
        
    if (scaler4!=None):
        return scaler4.transform(paramsy)
    else:
        return paramsy

  def my_8_classic_plus_MIR(y, n_fft, win_length, hop_length, window, sr, f_min, f_max, summary_1d, scaler5 = None): ## TO DO scalers
    param_0, param_1, param_2, param_3, param_4, skosnosc, kurtoza, plaskosc, S = classic_base(y, n_fft, win_length, hop_length, window, sr, f_min, f_max, scaler4)
    nb_summary = np.sum(summary_1d)
    stft = librosa.stft(np.array(y), n_fft=n_fft, win_length=win_length, hop_length= hop_length, window=window)
    abs2_stft = (stft.real*stft.real) + (stft.imag*stft.imag)
    melspect = librosa.feature.melspectrogram(y=None, S=abs2_stft, sr=sr, n_mels= n_mels, fmin = f_min, fmax=f_max, hop_length=hop_length, n_fft=n_fft)
    
    mfccs =librosa.feature.mfcc(S=librosa.power_to_db(melspect), n_mfcc=12)
    mfcc_delta = librosa.feature.delta(mfccs)
    zcr = sum(librosa.feature.zero_crossing_rate(y, frame_length=win_length, hop_length= hop_length))  #  ZCR can be interpreted as a measure of the noisiness of a signal. For example, it usually exhibits higher values in the case of noisy signals. It is also known to reflect, in a rather coarse manner, the spectral characteristics of a signal. std to recognize speech vs music
    contrast = librosa.feature.spectral_contrast(S=S, hop_length=hop_length, n_fft=n_fft, n_bands = 2, fmin = f_min)
    rolloff = librosa.feature.spectral_rolloff(S=S, hop_length=hop_length, n_fft=n_fft, roll_percent=0.85)    
    rms = librosa.feature.rms(S=S, hop_length=hop_length, frame_length = 124) 
    spectral_flux = FeatureSpectralFlux(S)
    np.shape(rms)
    paramsy = [[[] for _ in range(39)] for _ in range(nb_summary)]
    idx = 0

    for m in range(np.shape(summary_1d)[0]):
      if summary_1d[m]:
        f = getattr(np, repr_1d_summary[m])  # która statystyka wybrana repr_1d_summary = ['min', 'max', 'mean', 'std']
        paramsy_mir = [f(param_0), f(param_1), f(param_2), f(param_3), f(param_4), f(skosnosc), f(kurtoza), f(plaskosc)]
        paramsy_mir.extend(f(mfccs, axis = 1).tolist())
        paramsy_mir.extend(f(mfcc_delta, axis = 1).tolist())
        paramsy_mir.extend([f(zcr)])
        paramsy_mir.extend(f(contrast, axis = 1).tolist())
        paramsy_mir.extend([f(rolloff), f(rms), f(spectral_flux)])
        paramsy[idx]= paramsy_mir
        idx += 1

    if (scaler5!=None):
        return scaler5.transform(paramsy)
    else:
        return paramsy


  # nie powinno mieć to wszystko train w nazwie, no ale już trudno. Chodzi o dowolny dataset który będzie na wejściu 
  # z funkcji wczytanie:
  file_names_train_set = data_settype[0]
  indices = data_settype[1]
  result_dataframe_train = data_settype[2]  # trzeba powyciągać tylko interesujące nas chunksy
  df_to_np = result_dataframe_train.to_numpy()  # bo nie umiem sobie poradiś jak jest to df, wiele funkcji które chcę użyć nie działa


  repr_full = [[] for _ in range(np.shape(file_names_train_set)[0])]
  #repr_full1, repr_full2, repr_full3 = [[] for _ in range(np.shape(file_names_train_set)[0])],[[] for _ in range(np.shape(file_names_train_set)[0])],[[] for _ in range(np.shape(file_names_train_set)[0])]
  #repr_full4, repr_full5 = [[] for _ in range(np.shape(file_names_train_set)[0])], [[] for _ in range(np.shape(file_names_train_set)[0])]
  info_chunksy = [[[] for _ in range(8)] for _ in range(np.shape(file_names_train_set)[0])]  # 8 kolumn w dataframie było
 
  print(file_names_train_set) 

  # by nie wczytywać scalerów przy każdym chunku albo nagraniu. True - mamy już wyznaczone scalery,  i będziemy wszystko normalizować scalerami z traina. Jeśli false, to bez normalizacji:
  path_main = os.path.join('C:\\','Users','szaro','Desktop','jupyter')

  scaler1 = joblib.load(os.path.join(path_main,'scaler','scaler_spektrogram')) if if_scaler==True else None  
  scaler2 = joblib.load(os.path.join(path_main,'scaler','scaler_mel_spektrogram')) if if_scaler==True else None  
  '''scaler3 = joblib.load('scaler_multitaper') if if_scaler==True else None '''
  scaler4 = joblib.load(os.path.join(path_main,'scaler','scaler_8_classic')) if if_scaler==True else None  
  scaler5 = joblib.load(os.path.join(path_main,'scaler','scaler_8_classic_plus_MIR')) if if_scaler==True else None  

  # k - kolejne nagrania w zbiorze 
  
  for k in range(nr_nagr_pocz, nr_nagr_pocz+ wielkosc_batcha): #np.shape(file_names_train_set)[0]): #np.shape(file_names_train_set)[0]): #np.shape(file_names_train_set)[0]):  # 87 dla zbioru train  len(file_names_train_set)

    ##### pętla do wycięgnięcia info o interesujących chunksach
    empty_list = np.array([[] for _ in range(np.shape(indices[k])[0])]) #np.array([[] for _ in range(np.shape(indices[k])[0])])
    # warunki: bo może się zdarzyć że macierz tablica pustych elementów będzie w tych dwóch 'chunks_species', 'call_id', i wtedy np.take nie działa 
    if not any(df_to_np[k][4]) and not any(df_to_np[k][5]):  
      info_chunksy[k] = [np.take(df_to_np[k][i],indices[k]) for i in [0,1,2,3]] + [empty_list, empty_list] + [np.take(df_to_np[k][i],indices[k]) for i in [6, 7]]

      print(np.shape(info_chunksy[k][5]), 'no calls at all')
    elif not any(df_to_np[k][5]):
      info_chunksy[k] = [np.take(df_to_np[k][i],indices[k]) for i in [0,1,2,3,4]] + [empty_list] + [np.take(df_to_np[k][i],indices[k]) for i in [6, 7]]                  
      print(np.shape(info_chunksy[k][5]), 'no calls of interest')
    else:
      info_chunksy[k] = [np.take(df_to_np[k][i],indices[k]) for i in [0,1,2,3,4,5,6,7]]

    start_time = time.time()
    if representation_type=='spektrogram':
        representation = np.empty([np.shape(indices[k])[0], 63, 148])
    if representation_type=='mel-spektrogram':
        representation = np.empty([np.shape(indices[k])[0], 60, 148])
    '''representation3 = np.empty([np.shape(indices[k])[0], 64, 149])'''
    if representation_type=='8_classic':
        representation = np.empty([np.shape(indices[k])[0], np.sum(summary_1d), 8])
    if representation_type=='8_classic_plus_MIR':
        representation = np.empty([np.shape(indices[k])[0], np.sum(summary_1d), 39])
   
    ##### a to już pętla do reprezentacji, pojedyncze chunksy
    for num, i in enumerate(indices[k]):                        # przykładowo 262 dla pierwszego nagrania w zbiorze train, bo tyle mamy tam chunksów
      
      # wczytanie chunka naszego półsekundowego:
      y, sr = librosa.load(path_wav + '/'+ file_names_train_set[k], sr = 44100, offset = result_dataframe_train['chunk_start'][k][i]/sr, duration = chunk_length_ms/1000)
      if representation_type=='spektrogram':   
      ####### reprezentacja 1  ------- 63 x 148 ------ SPEKTROGRAM
          representation[num] = my_spektrogram(y, n_fft, win_length, hop_length, window, sr, f_min, f_max, scaler1)
      if representation_type=='mel-spektrogram': 
      ####### reprezentacja 2 (3 V3) - mel spektrogram  ------- 60 x 148 ------  
          representation[num] = my_melspektrogram(y, n_fft, win_length, hop_length, window, sr, f_min, f_max, n_mels, scaler2)
        
      ####### reprezentacja 3 (5b) - multitaper o większej rozdzielczości ------- 64 x 149 ------
      # Ale tak naprawdę to nie jest multitaper, ale coś innego, nie wiem w sumie do końca co. Multitaper + Time-frequency reassignement spectrogram
      '''representation3[num] = my_multitaper(y, N, step, Np, K, tm, flock, tlock, f_min, f_max, scaler3) '''
      if representation_type=='8_classic': 
          representation[num] = my_8_classic(y, n_fft, win_length, hop_length, window, sr, f_min, f_max, summary_1d, scaler4)
      if representation_type=='8_classic_plus_MIR': 
          representation[num] = my_8_classic_plus_MIR(y, n_fft, win_length, hop_length, window, sr, f_min, f_max, summary_1d, scaler5)

    repr_full[k] = representation
    print(k,'-', file_names_train_set[k], '- chunks:', np.shape(indices[k])[0], '- time:', time.time()-start_time)

  return [file_names_train_set , indices, info_chunksy, repr_full]     # repr_full3, representation_type
    