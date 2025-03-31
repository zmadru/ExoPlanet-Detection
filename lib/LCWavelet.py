import pandas as pd
import lightkurve as lk
import numpy as np
import pywt
import pickle
import os
import matplotlib.pyplot as plt
from lib.LCWavelet import *
from lightkurve.lightcurve import FoldedLightCurve

class LightCurveWaveletFoldCollection():
  
    def __init__(self,light_curve,wavelets):
        self._light_curve = light_curve
        self._lc_w_collection = wavelets

    def get_detail_coefficent(self,level = None):
        if level != None:
            return self._lc_w_collection[level-1][1]
        return self._lc_w_collection[:][1]

    def get_approximation_coefficent(self,level = None):
        if level != None:
            return self._lc_w_collection[level-1][0]
        return self._lc_w_collection[:][0]
    
    def get_wavelets(self):
        return self._lc_w_collection

    def plot(self):
        wavelet = self._lc_w_collection
        time = self._light_curve.time.value
        data = self._light_curve.flux.value
        plt.figure(figsize=(16, 4))
        plt.plot(time,data)
        ig, axarr = plt.subplots(nrows=len(wavelet), ncols=2, figsize=(16,12))
        for i,lc_w in enumerate(wavelet):
            (data, coeff_d) = lc_w
            axarr[i, 0].plot(data, 'r')
            axarr[i, 1].plot(coeff_d, 'g')
            axarr[i, 0].set_ylabel("Level {}".format(i + 1), fontsize=14, rotation=90)
            axarr[i, 0].set_yticklabels([])
            if i == 0:
                axarr[i, 0].set_title("Approximation coefficients", fontsize=14)
                axarr[i, 1].set_title("Detail coefficients", fontsize=14)
            axarr[i, 1].set_yticklabels([])
        # plt.show()

class LightCurveWaveletCollection():
    def __init__(self,id,headers,lc_par,lc_inpar):
        self.pliegue_par = lc_par
        self.pliegue_inpar = lc_inpar
        self.kepler_id = id
        self.headers = headers

    def save(self, path = ""):
        file_name = path + 'kic '+str(self.kepler_id)+'-'+self.headers['Kepler_name']+'.pickle'
        with open(file_name, "wb") as f:
            pickle.dump(self, f)

    def load(path):
        if path.endswith(".pickle"):
            with open(path, "rb") as f:
                w_loaded = pickle.load(f)
            return w_loaded

    def plot_comparative(self):
        light_curve_p = self.pliegue_par._light_curve
        light_curve_i = self.pliegue_inpar._light_curve
        w_par_Collection = self.pliegue_par
        w_inpar_Collection = self.pliegue_inpar
        wavelet_p=w_par_Collection.get_wavelets()
        wavelet_i=w_inpar_Collection.get_wavelets()
        plt.figure(figsize=(26, 8))
        plt.plot(light_curve_p.time.value,light_curve_p.flux.value,c='blue',label='par')
        plt.plot(light_curve_i.time.value,light_curve_i.flux.value,c='red',label='inpar')
        
        ig, axarr = plt.subplots(nrows=len(wavelet_p), ncols=2, figsize=(26,26))
        for i,zip_curves in enumerate(zip(wavelet_p,wavelet_i)):
            (data_p, coeff_p),(data_i, coeff_i) = zip_curves
            axarr[i, 0].plot(data_p,c='blue',label='par')
            axarr[i, 0].plot(data_i, c='red',label='inpar')
            axarr[i, 1].plot(coeff_p, c='blue',label='par')
            axarr[i, 1].plot(coeff_i, c='red',label='inpar')
            axarr[i, 0].set_ylabel("Level {}".format(i + 1), fontsize=14, rotation=90)
            axarr[i, 0].set_yticklabels([])
            if i == 0:
                axarr[i, 0].set_title("Approximation coefficients", fontsize=14)
                axarr[i, 1].set_title("Detail coefficients", fontsize=14)
            axarr[i, 1].set_yticklabels([])
        plt.show()


class LightCurveGlobalLocalCollection():
    def __init__(self,id, headers,
                 lc_par_global: FoldedLightCurve,
                 lc_impar_global: FoldedLightCurve,
                 lc_par_local: FoldedLightCurve,
                 lc_impar_local: FoldedLightCurve):
        self.pliegue_par_global = lc_par_global
        self.pliegue_impar_global = lc_impar_global
        self.pliegue_par_local = lc_par_local
        self.pliegue_impar_local = lc_impar_local
        self.kepler_id = id
        self.headers = headers

    def save(self, path = ""):
        file_name = path + '/kic '+str(self.kepler_id)+'-'+self.headers['Kepler_name']+'.pickle'
        with open(file_name, "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def from_pickle(cls, path):
        if path.endswith(".pickle"):
            with open(path, "rb") as f:
                w_loaded = pickle.load(f)
            return w_loaded

    def plot(self, **kwargs):

        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(nrows=2, ncols=2)
        self.pliegue_impar_global.plot(**{"ax": ax1, "title": "global impar", **kwargs})
        self.pliegue_par_global.plot(**{"ax": ax2, "title": "global par",  **kwargs})
        self.pliegue_impar_local.plot(**{"ax": ax3, "title": "local impar", **kwargs})
        self.pliegue_par_local.plot(**{"ax": ax4, "title": "local par", **kwargs})
        
    def scatter(self, **kwargs):
        self.pliegue_par_global.scatter(**kwargs)
        self.pliegue_impar_global.scatter(**kwargs)
        self.pliegue_par_local.scatter(**kwargs)
        self.pliegue_impar_local.scatter(**kwargs)


class LightCurveWaveletGlobalLocalCollection():
    def __init__(self,id, headers,
                 lc_par_global: LightCurveWaveletFoldCollection,
                 lc_impar_global: LightCurveWaveletFoldCollection,
                 lc_par_local: LightCurveWaveletFoldCollection,
                 lc_impar_local: LightCurveWaveletFoldCollection,
                 levels):
        self.pliegue_par_global = lc_par_global
        self.pliegue_impar_global = lc_impar_global
        self.pliegue_par_local = lc_par_local
        self.pliegue_impar_local = lc_impar_local
        self.kepler_id = id
        self.headers = headers
        self.levels = levels

    def save(self, path = ""):
        file_name = path + '/kic '+str(self.kepler_id)+'-'+self.headers['Kepler_name']+'.pickle'
        with open(file_name, "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def from_pickle(cls, path):
        if path.endswith(".pickle"):
            with open(path, "rb") as f:
                w_loaded = pickle.load(f)
            return w_loaded

    def plot(self, **kwargs):
        if kwargs.get("figure_paths") is not None:
            figure_paths = kwargs.get("figure_paths")
            self.pliegue_impar_global.plot()
            plt.savefig(figure_paths[0])
            plt.close('all')
            self.pliegue_par_global.plot()
            plt.savefig(figure_paths[1])
            plt.close('all')
            self.pliegue_impar_local.plot()
            plt.savefig(figure_paths[2])
            plt.close('all')
            self.pliegue_par_local.plot()
            plt.savefig(figure_paths[3])
            plt.close('all')
        else:
            self.pliegue_impar_global.plot()
            self.pliegue_par_global.plot()
            self.pliegue_impar_local.plot()
            self.pliegue_par_local.plot()
        
    def scatter(self, **kwargs):
        self.pliegue_par_global.scatter(**kwargs)
        self.pliegue_impar_global.scatter(**kwargs)
        self.pliegue_par_local.scatter(**kwargs)
        self.pliegue_impar_local.scatter(**kwargs)




def fold_curve(light_curve_collection, period, epoch, sigma = 20, sigma_upper = 4):
    """
    Toma la coleccion de la curvas entregadas, las pliega y devuelve una sola con todos los datos.
    
    Parameters
    ----------
    light_curve_collection: LightCurveCollection
        coleccion de curvas de luz.
    period: float
        periodo de la orbita.
    epoch: float
        tiempo de cada transcurso.
    sigma: int
        valor de desviaciones estandas
    sigma_upper: int
        valor maximo de variacion
    Returns
    ----------
    una sola curva de luz
    """
    # lc_collection = lk.LightCurveCollection([lc.remove_outliers(sigma=20, sigma_upper=4) for lc in light_curve_collection])
    
    lc_ro = lc_collection.stitch()
    
    # lc_ro = lc_ro.remove_outliers(sigma=sigma, sigma_upper=sigma_upper)
    
    lc_nonans = lc_ro.remove_nans()
    lc_fold = lc_nonans.fold(period = period,epoch_time = epoch)
    lc_odd=lc_fold[lc_fold.odd_mask]
    lc_even = lc_fold[lc_fold.even_mask]
    return lc_fold,lc_odd,lc_even

def apply_wavelet(light_curve,w_family, levels,cut_border_percent=0.1):
    time = light_curve.time.value
    data = light_curve.flux.value
    lc_wavelet = []
    try:
        for level in range(levels):
            level_w = pywt.dwt(data, w_family)
            lc_wavelet.append(cut_border(level_w,cut_border_percent))
            #lc_wavelet.append(level_w)
            data = level_w[0]
    except TypeError: 
        for level in levels:
            level_w = pywt.dwt(data, w_family)
            lc_wavelet.append(cut_border(level_w,cut_border_percent))
            #lc_wavelet.append(level_w)
            data = level_w[0]
    return LightCurveWaveletFoldCollection(light_curve,lc_wavelet)

def load_light_curve(kepler_id,mission='Kepler'):
    kic = 'KIC '+str(kepler_id)
    lc_search = lk.search_lightcurve(kic, mission=mission)
    lc_collection = lc_search.download_all(download_dir="data/")
    return lc_collection

def cut_wavelet(lightCurve,window):
    time = lightCurve.time
    data = lightCurve.flux
    flux_error = lightCurve.flux_err
    index = np.argmin(np.absolute(time))
    min_w = index - int(window/2)
    max_w = index + int(window/2)+1
    time_selected = time[min_w:max_w]
    data_selected = data[min_w:max_w]
    flux_error_selected = flux_error[min_w:max_w]
    return lk.lightcurve.FoldedLightCurve(time=time_selected,flux=data_selected,flux_err=flux_error_selected)

def cut_border(data_old,cut_percent=0.1):
    data_len_cut = int(len(data_old[0])*(cut_percent/2))
    data_new = [data[data_len_cut:(len(data)-data_len_cut)] for data in data_old ]
    return data_new
    
def process_light_curve(kepler_id,kepler_name,disp,period,epoch,w_family,levels,plot = False, plot_comparative=False,save=False, path="",wavelet_window=None,cut_border_percent=0.2):
    # cargamos la curva de segun su Kepler_ID
    print("descargando curvas de luz...")
    lc_collection=load_light_curve(kepler_id)
    # aplicamos el pliege a las curvas de luz y las separamos en pares e inpares
    print('Aplicando pliegue y separando en pares e inpares....') 
    _,lc_inpar,lc_par = fold_curve(lc_collection,period,epoch)

    if not wavelet_window == None:
      print('Aplicando ventana ...')
      lc_inpar = cut_wavelet(lc_inpar,wavelet_window)
      lc_par = cut_wavelet(lc_par,wavelet_window)
    
    print('Aplicando wavelets...')
    # aplicamos wavelets a curvas par
    lc_w_par = apply_wavelet(lc_par,w_family,levels,cut_border_percent=cut_border_percent)
    # aplicamos wavelets a curvas inpar
    lc_w_inpar = apply_wavelet(lc_inpar,w_family,levels,cut_border_percent=cut_border_percent)
    headers = {
        "period": period,
        "epoch": epoch,
        "class": disp,
        "wavelet_family":w_family,
        "levels":levels,
        "window":wavelet_window,
        "border_cut":cut_border_percent,
        "Kepler_name":kepler_name
    }
    lc_wavelet_collection = LightCurveWaveletCollection(kepler_id,headers,lc_w_par,lc_w_inpar)
    if(plot):
        print('graficando wavelets obtenidas...')
        lc_w_par.plot()
        lc_w_inpar.plot()
    if(plot_comparative):
        print('graficando wavelets obtenidas...')
        lc_wavelet_collection.plot_comparative()
    if(save):
        print('guardando wavelets obtenidas...')
        lc_wavelet_collection.save(path)
    return lc_wavelet_collection

def process_dataset(df_koi,plot = False, plot_comparative = False,repeat_completed=True,completed=None, save_path=None, wavelet_family=None, level=None, save_lc=None, wavelet_windows=None):
    lc_wavelets = dict()
    lc_errors = []
    for i in range (len(df_koi)):

        koi_id,koi_name,disp, period, epoch=df_koi[['kepid','kepoi_name','koi_disposition','koi_period','koi_time0bk']].iloc[i]
        percent = i*100/(len(df_koi))
        print(f'procesando curva de luz KIC {int(koi_id)}-{koi_name}[{disp}] [{percent:.0f}%]')
        if not repeat_completed and (str(koi_id)+"-"+koi_name) in completed:
          print("curva de luz procesada anteriormente")
          continue
        try:
            process_light_curve(int(koi_id),koi_name,disp,period,epoch,wavelet_family,level,plot= plot,plot_comparative=plot_comparative,save = save_lc, path = save_path,wavelet_window=wavelet_windows)
            pass
        except:
            lc_errors.append(koi_id)
            print(f'Error with KIC {koi_id}')
            from IPython import get_ipython
            ipython = get_ipython()
            ipython.magic("tb Verbose")
    f = open (save_path+'errors.txt','w')
    for lc_error in lc_errors:
        text = 'KIC '+str(lc_error)+'\n'
        f.write(text)
    f.close()
    return lc_errors


def plot_results(history):
    # GRÁFICO DE LA PRECISIÓN y PERDIDA CON DATOS DE ENTRENAMIENTO
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']

    epochs = range(len(acc))

    plt.plot(epochs, acc, 'r', label='Presición Entrenamiento')
    plt.plot(epochs, val_acc, 'b', label='Presición Validación')
    plt.title('Presición entrenamiento y test')
    plt.legend(loc=0)
    plt.figure()
    plt.show()

    plt.plot(epochs, loss, 'r',linestyle = 'dashed', label='Pérdida de Entrenamiento')
    plt.plot(epochs, val_loss, 'b',linestyle = 'dashed', label='Perdida de Validación')
    plt.title('Pérdida entrenamiento y test')
    plt.legend(loc=0)
    plt.show()
    
def load_files(path):
  completed = os.listdir(path)
  if "errors.txt" in completed:  
    completed.remove('errors.txt')
  completed_id = []
  for element in completed:
    completed_id.append(path+element)
  return completed_id

def generate_dataset_model_1(path,level=8, progress=True):
  files = load_files(path)
  dataset_par =[]
  dataset_inpar= []
  labels = []
  len_points = None

  for i,file in enumerate(files):
    # output.clear()
    if progress:
        print(f"loading [{i*100/len(files):.0f}%] file:{file}")
    lcwC = LightCurveWaveletCollection.load(file)
    status = lcwC.headers['class']
    curva_par = lcwC.pliegue_par.get_approximation_coefficent(level=level)
    curva_inpar = lcwC.pliegue_inpar.get_approximation_coefficent(level=level)
    #print(i,len(curva_par),len(curva_inpar),['-' for x in range(int(len(curva_par)/10))])
    if len_points == None:
      len_points = len(curva_par)
    if len(curva_par)!= len_points or  len(curva_inpar)!= len_points:
      continue
    dataset_par.append(curva_par)
    #dataset_par=np.append(dataset_par,[curva_par])
    dataset_inpar.append(curva_inpar)
    #dataset_inpar=np.append(dataset_inpar,[curva_inpar])
    labels.append(0 if status == 'FALSE POSITIVE' else 1)
  
  dataset_par = np.array(dataset_par)
  dataset_inpar = np.array(dataset_inpar)
  labels = np.array(labels)
  return dataset_par,dataset_inpar,labels

def generate_dataset_model_2(path,levels=[8],show_loading = True):
  files = load_files(path)
  #print(f"file len:{len(files)}")
  labels = []
  len_points = {}
  curvas = {}

  for level in levels:
    curvas["par_"+str(level)] = []
    curvas["impar_"+str(level)] = []
    len_points[str(level)]=None

  for i,file in enumerate(files):
    skip_label = False
    if show_loading:
      print(f"loading [{i*100/len(files):.0f}%] file:{file}")
    lcwC = LightCurveWaveletCollection.load(file)
    status = lcwC.headers['class']

    for level in levels:
      curva_par = lcwC.pliegue_par.get_approximation_coefficent(level=level)
      curva_inpar = lcwC.pliegue_inpar.get_approximation_coefficent(level=level)

      if len_points[str(level)] == None:
         len_points[str(level)] = len(curva_par)
      if len(curva_par)!= len_points[str(level)] or  len(curva_inpar)!= len_points[str(level)]:
        skip_label = True
        break
      curvas["par_"+str(level)].append(curva_par)
      curvas["impar_"+str(level)].append(curva_inpar)

    if not skip_label:
      labels.append(0 if status == 'FALSE POSITIVE' else 1)
  #dataset_par = np.array(dataset_par)
  #dataset_inpar = np.array(dataset_inpar)
  for level in levels:
    curvas["par_"+str(level)] = np.array(curvas["par_"+str(level)])
    curvas["impar_"+str(level)] = np.array(curvas["impar_"+str(level)])
  labels = np.array(labels)
  #print("len curvas",len(curvas["par_"+str(levels[0])]),  " len labels", len(labels) )
  return curvas,labels


def split_dataset(dataset_p, dataset_i, labels, split=.80):
  split = int(len(labels)*split)
  print(f"before par:{np.shape(dataset_p)} impar:{np.shape(dataset_i)}, labels:{len(labels)}")
  X_p_train = dataset_p[:split]
  X_i_train = dataset_i[:split]
  y_train = labels[:split]

  X_p_test = dataset_p[split:]
  X_i_test = dataset_i[split:]
  y_test = labels[split:]

  X_p_train = np.expand_dims(X_p_train, axis=-1)
  X_i_train = np.expand_dims(X_i_train, axis=-1)
  X_p_test = np.expand_dims(X_p_test, axis=-1)
  X_i_test = np.expand_dims(X_i_test, axis=-1)
  #print(f"par:{np.shape(X_p_test)} impar:{np.shape(X_i_test)}, labels:{len(y_test)}")
  return [X_p_train, X_i_train], [X_p_test, X_i_test], y_train, y_test

def normalize_data(data):
  min = np.min(data)
  max = np.max(data)
  return (data - min)/(max-min) 

def normalize_data_2(data_p,data_i):
  min = np.min(data_p) if np.min(data_p) < np.min(data_i) else np.min(data_i)
  max = np.max(data_p) if np.max(data_p) > np.max(data_i) else np.max(data_i)
  return [(data_p - min)/(max-min) , (data_i - min)/(max-min)]

def normalize_LC(curvas_dic):
    return [ [normalize_data(curvas_dic[ list(curvas_dic.keys())[i]]),normalize_data(curvas_dic[ list(curvas_dic.keys())[i+1]]) ] for i in range(0,len(curvas_dic.keys()),2)   ] 
    # return [ normalize_data_2(curvas_dic[ list(curvas_dic.keys())[i]],curvas_dic[ list(curvas_dic.keys())[i+1]])  for i in range(0,len(curvas_dic.keys()),2)   ] 
def split_data_list(list_data,labels):
  ds_train = []
  ds_test = []
  label_train = []
  label_test = []
  first = True
  for c_par, c_impar in list_data:
    X_train, X_test, y_train, y_test = split_dataset(c_par, c_impar,labels)
    ds_train.append(X_train)
    ds_test.append(X_test)
    if first :
      label_train = y_train
      label_test = y_test
      first = False

  return ds_train,ds_test,label_train,label_test



def evaluate_model(model,dataset,verbose = 0,epochs=1000):
  # Modelado Dataset
  print('normalizando datos...')
  ds_levels = normalize_LC(dataset[0])
  print('dividiendo datos en entrenamioento y test...')
  ds_train,ds_test,label_train,label_test = split_data_list(ds_levels,dataset[1])
  # entrenamiento
  print('generando modelo...')
  model_g = model(ds_train,activation = tf.keras.layers.LeakyReLU())
  print('compilando modelo....')
  model_g.compile(loss = 'binary_crossentropy', optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001), metrics=['accuracy','binary_crossentropy'])
  print('entrenando....')
  #early_stopping_acc = tf.keras.callbacks.EarlyStopping(monitor='accuracy', min_delta=0.0005, patience=30, mode='max', verbose = 1)#EarlyStopping(monitor='accuracy', patience=15, min_delta=0.005, mode='max')
  early_stopping_loss = tf.keras.callbacks.EarlyStopping(monitor='loss', min_delta=0.0005, patience=15, mode='min', verbose = 1)
  history = model_g.fit(ds_train, label_train, epochs=epochs, batch_size=64,validation_split=0.20,shuffle=True,verbose=verbose, callbacks=[early_stopping_loss])
  print('obteniendo resultados..')
  plot_results(history)
  result = model_g.evaluate(ds_test, label_test)
  print(result)
  del model_g, history, result, ds_train,ds_test,label_train, label_test, ds_levels
  
def evaluate_model_1_level(model,dataset,level,verbose = 0):
  print('Cargando dataset...')
  ds_p,ds_i,label = generate_dataset_model_1(dataset_path,level)
  print('normalizando datos...')
  ds = normalize_data_2(ds_p,ds_i)
  print('dividiendo dataset...')
  X_train, X_test, y_train, y_test = split_dataset(ds[0], ds[1], label)
  print('generando modelo....')
  model_g = model(X_train,activation = tf.keras.layers.LeakyReLU())
  model_g.compile(loss = 'binary_crossentropy', optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001), metrics=['accuracy'])
  print('entrenando...')
  early_stopping = tf.keras.callbacks.EarlyStopping(monitor='accuracy', patience=30, min_delta=0.005, mode='max')

  history_2 = model_g.fit(X_train, y_train, epochs=1000, batch_size=64,validation_split=0.15,shuffle=True,verbose = verbose, callbacks=[early_stopping])
  print('Resultados...')
  plot_results(history_2)
  _, accuracy = model_g.evaluate(X_test, y_test)
  print('Accuracy: %.2f' % (accuracy*100))

  y_prediction =[0 if x <= 0.5 else 1 for x in model_g.predict(X_test) ]
  result = confusion_matrix(y_test, y_prediction)
  disp = ConfusionMatrixDisplay(confusion_matrix=result)
  disp.plot()
  plt.show()
  del  X_train, X_test, y_train, y_test, ds_p, ds_i, ds, label, model_g, history_2, accuracy
  
    
