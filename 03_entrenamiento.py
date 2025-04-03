# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.16.2
#   kernelspec:
#     display_name: TFM
#     language: python
#     name: tfm
# ---

# %%
import warnings
warnings.filterwarnings(action="ignore", category=DeprecationWarning)
warnings.filterwarnings(action="ignore", category=FutureWarning)
from lib.LCWavelet import *
from tqdm import tqdm
from collections import defaultdict
from parallelbar import progress_map
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, concatenate,Conv1D, Flatten,Dropout , BatchNormalization, MaxPooling1D
from tensorflow.keras.models import Model, Sequential
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from functools import partial
import datetime

path = "all_data/"
files = [file for file in os.listdir(path) if file.endswith(".pickle")]
lightcurves = []

def load_files(file, path):
    global_local = LightCurveWaveletGlobalLocalCollection.from_pickle(path+file)
    try:
        getattr(global_local, "levels")
    except AttributeError:
        global_local.levels = [1, 2, 3, 4]
    return global_local

func = partial(load_files, path=path)

# lightcurves = progress_map(func, files, n_cpu=128, total=len(files), executor='processes', error_behavior='raise')

for file in tqdm(files):
    lightcurves.append(func(file))

# %%

pliegue_par_global = defaultdict(list)
pliegue_impar_global = defaultdict(list)
pliegue_par_local = defaultdict(list)
pliegue_impar_local = defaultdict(list)

for lc in lightcurves:
    for level in lc.levels:
        pliegue_par_global[level].append(lc.pliegue_par_global.get_approximation_coefficent(level=level))
        pliegue_impar_global[level].append(lc.pliegue_impar_global.get_approximation_coefficent(level=level))
        pliegue_par_local[level].append(lc.pliegue_par_local.get_approximation_coefficent(level=level))
        pliegue_impar_local[level].append(lc.pliegue_impar_local.get_approximation_coefficent(level=level))
        

pliegue_par_global = {k: np.array(v) for k, v in pliegue_par_global.items() if k in (1 ,2)}
pliegue_par_global = {k: v.reshape(list(v.shape)+[1]) for k, v in pliegue_par_global.items()}
pliegue_impar_global = {k: np.array(v) for k, v in pliegue_impar_global.items() if k in (1, 2)}
pliegue_impar_global = {k: v.reshape(list(np.shape(v))+[1]) for k, v in pliegue_impar_global.items()}


pliegue_par_local = {k: np.array(v) for k, v in pliegue_par_local.items() if k in (1, 2)}
pliegue_par_local = {k: v.reshape(list(np.shape(v))+[1]) for k, v in pliegue_par_local.items()}
pliegue_impar_local = {k: np.array(v) for k, v in pliegue_impar_local.items() if k in (1, 2)}
pliegue_impar_local = {k: v.reshape(list(np.shape(v))+[1]) for k, v in pliegue_impar_local.items()}


inputs = (pliegue_par_global, pliegue_impar_global), (pliegue_par_local, pliegue_impar_local)


# %%

# %%
# inputs
# %pdb off
def gen_model_2_levels(inputs, classes, activation = 'relu',summary=False):
    
    (pliegue_par_global, pliegue_impar_global), (pliegue_par_local, pliegue_impar_local) = inputs    

    input_shape_global = [x.shape for x in pliegue_par_global.values()]
    assert input_shape_global == [x.shape for x in pliegue_impar_global.values()]
    
    input_shape_local = [x.shape for x in pliegue_par_local.values()]
    assert input_shape_local == [x.shape for x in pliegue_impar_local.values()]


    net = defaultdict(list)
 
    for (n, data), (n_inv, data_inv) in zip(sorted(pliegue_par_global.items(), key=lambda d: d[0]), sorted(pliegue_par_global.items(), key=lambda d: -d[0])):
        block = Sequential()
        for i in range(n_inv):
            block.add( Conv1D(16*2**n_inv, 5, activation=activation, input_shape=data.shape[1:], ))
            block.add(Conv1D(16*2**n_inv, 5, activation=activation, ))
            block.add(MaxPooling1D(pool_size=3, strides=1, ))

        block.add(Flatten())
        net["global_par"].append(block)
        
    for (n, data), (n_inv, data_inv) in zip(sorted(pliegue_impar_global.items(), key=lambda d: d[0]), sorted(pliegue_impar_global.items(), key=lambda d: -d[0])):
        block = Sequential()
        for i in range(n_inv):
            block.add(Conv1D(16*2**n_inv, 5, activation=activation, input_shape=data.shape[1:], ))
            block.add(Conv1D(16*2**n_inv, 5, activation=activation, ))
            block.add(MaxPooling1D(pool_size=3, strides=1, ))

        block.add(Flatten())
        net["global_impar"].append(block)

    for (n, data), (n_inv, data_inv) in zip(sorted(pliegue_par_local.items(), key=lambda d: d[0]), sorted(pliegue_par_local.items(), key=lambda d: -d[0])):
        block = Sequential()
        for i in range(n_inv):
            block.add(Conv1D(16*2**i, 5, activation=activation, input_shape=data.shape[1:], ))
            block.add(Conv1D(16*2**i, 5, activation=activation, ))
            block.add(MaxPooling1D(pool_size=3, strides=1, ))
        block.add(Flatten())
        net["local_par"].append(block)
        
    for (n, data), (n_inv, data_inv) in zip(sorted(pliegue_impar_local.items(), key=lambda d: d[0]), sorted(pliegue_impar_local.items(), key=lambda d: -d[0])):
        block = Sequential()
        for i in range(n_inv):
            block.add( Conv1D(16*2**i, 5, activation=activation, input_shape=data.shape[1:], ))
            block.add( Conv1D(16*2**i, 5, activation=activation, ))
            block.add(MaxPooling1D(pool_size=3, strides=1, ))
        block.add(Flatten())
        net["local_impar"].append(block)
                             
    model_f = concatenate([m.output for m in net["global_par"]] + [m.output for m in net["global_impar"]] + [m.output for m in net["local_par"]] + [m.output for m in net["local_impar"]], axis=-1)
    model_f = BatchNormalization(axis=-1)(model_f)
    model_f = Dense(512,activation=activation)(model_f)
    model_f = Dense(512,activation=activation)(model_f)
    model_f = Dense(512,activation=activation)(model_f)
    model_f = Dense(512,activation=activation)(model_f)
    model_f = Dense(len(classes),activation='softmax')(model_f)
    
    model_f = Model([[m.input for m in net["global_par"]], [m.input for m in net["global_impar"]]  , [m.input for m in net["local_par"]], [m.input for m in net["local_impar"]]],model_f)
    # model_f = Model([[model_p_7.input,model_i_7.input],[model_p_8.input,model_i_8.input]],model_f)


    if summary:
      model_f.summary()
    return model_f
    
    # model_f = concatenate([model_p_7.output,model_i_7.output,
    #                        model_p_8.output,model_i_8.output], axis=-1)
    # model_f = BatchNormalization(axis=-1)(model_f)
    # model_f = Dense(512,activation=activation)(model_f)
    # model_f = Dense(512,activation=activation)(model_f)
    # model_f = Dense(512,activation=activation)(model_f)
    # model_f = Dense(512,activation=activation)(model_f)
    # model_f = Dense(1,activation='sigmoid')(model_f)
    # model_f = Model([[model_p_7.input,model_i_7.input],[model_p_8.input,model_i_8.input]],model_f)

    # tamaño nivel 7
    input_shape_1 = np.shape(pliegue_par_global[1])[1:]
    # tamaño nivel 8
    input_shape_2 = np.shape(pliegue_par_global[2])[1:]
   
    # rama par level 7
    model_p_7 = tf.keras.Sequential()
    model_p_7.add(Conv1D(16, 5, activation=activation, input_shape=input_shape_1))
    model_p_7.add(Conv1D(16, 5, activation=activation)) 
    model_p_7.add(MaxPooling1D(pool_size=3, strides=1)) 
    model_p_7.add(Conv1D(32,5, activation=activation))
    model_p_7.add(Conv1D(32,5, activation=activation))
    model_p_7.add(MaxPooling1D(pool_size=3, strides=1))
    model_p_7.add(Conv1D(64,5, activation=activation))
    model_p_7.add(Conv1D(64,5, activation=activation))
    model_p_7.add(MaxPooling1D(pool_size=3, strides=1))
    model_p_7.add(Conv1D(128,5, activation=activation))
    model_p_7.add(Conv1D(128,5, activation=activation))
    model_p_7.add(MaxPooling1D(pool_size=3, strides=1))
    model_p_7.add(Flatten())
    
    # rama par level 8
    model_p_8 = tf.keras.Sequential()
    model_p_8.add(Conv1D(16, 5, activation=activation, input_shape=input_shape_2))
    model_p_8.add(Conv1D(16, 5, activation=activation)) 
    model_p_8.add(MaxPooling1D(pool_size=3, strides=1)) 
    model_p_8.add(Conv1D(32,5, activation=activation))
    model_p_8.add(Conv1D(32,5, activation=activation))
    model_p_8.add(MaxPooling1D(pool_size=3, strides=1))
    model_p_8.add(Conv1D(64,5, activation=activation))
    model_p_8.add(Conv1D(64,5, activation=activation))
    model_p_8.add(MaxPooling1D(pool_size=3, strides=1))
    model_p_8.add(Flatten())
    
    # rama impar level 7
    model_i_7 = tf.keras.Sequential()
    model_i_7.add(Conv1D(16, 5, activation=activation, input_shape=input_shape_1))
    model_i_7.add(Conv1D(16, 5, activation=activation)) 
    model_i_7.add(MaxPooling1D(pool_size=3, strides=1)) 
    model_i_7.add(Conv1D(32,5, activation=activation))
    model_i_7.add(Conv1D(32,5, activation=activation))
    model_i_7.add(MaxPooling1D(pool_size=3, strides=1))
    model_i_7.add(Conv1D(64,5, activation=activation))
    model_i_7.add(Conv1D(64,5, activation=activation))
    model_i_7.add(MaxPooling1D(pool_size=3, strides=1))
    model_i_7.add(Conv1D(128,5, activation=activation))
    model_i_7.add(Conv1D(128,5, activation=activation))
    model_i_7.add(MaxPooling1D(pool_size=3, strides=1))
    model_i_7.add(Flatten())
    
    # rama impar level 8
    model_i_8 = tf.keras.Sequential()
    model_i_8.add(Conv1D(16, 5, activation=activation, input_shape=input_shape_2))
    model_i_8.add(Conv1D(16, 5, activation=activation)) 
    model_i_8.add(MaxPooling1D(pool_size=3, strides=1)) 
    model_i_8.add(Conv1D(32,5, activation=activation))
    model_i_8.add(Conv1D(32,5, activation=activation))
    model_i_8.add(MaxPooling1D(pool_size=3, strides=1))
    model_i_8.add(Conv1D(64,5, activation=activation))
    model_i_8.add(Conv1D(64,5, activation=activation))
    model_i_8.add(MaxPooling1D(pool_size=3, strides=1))
    model_i_8.add(Flatten())

    # Red profunda
    model_f = concatenate([model_p_7.output,model_i_7.output,
                           model_p_8.output,model_i_8.output], axis=-1)
    
    import sys; sys.__breakpointhook__()
    model_f = BatchNormalization(axis=-1)(model_f)
    model_f = Dense(512,activation=activation)(model_f)
    model_f = Dense(512,activation=activation)(model_f)
    model_f = Dense(512,activation=activation)(model_f)
    model_f = Dense(512,activation=activation)(model_f)
    model_f = Dense(1,activation='sigmoid')(model_f)
    model_f = Model([[model_p_7.input,model_i_7.input],[model_p_8.input,model_i_8.input]],model_f)
    if summary:
      model_f.summary()
    return model_f



flatten = []
for (n, data) in sorted(pliegue_par_global.items(), key=lambda d: d[0]):
    flatten.append(data)
    
for (n, data) in sorted(pliegue_impar_global.items(), key=lambda d: d[0]):
    flatten.append(data)

for (n, data) in sorted(pliegue_impar_local.items(), key=lambda d: d[0]):
    flatten.append(data)
    
for (n, data) in sorted(pliegue_par_local.items(), key=lambda d: d[0]):
    flatten.append(data)

y = np.array([lc.headers['class'] for lc in lightcurves])
output_classes = np.unique([lc.headers['class'] for lc in lightcurves])
class2num = {label: n for n, label in enumerate(output_classes)}
num2class = {n: label for n, label in enumerate(output_classes)}

y = to_categorical([class2num[x] for x in y], num_classes=3)

res = train_test_split(*(flatten+[y]), test_size=0.3, shuffle=False)
*X_train, y_train = [r for n, r in enumerate(res) if n % 2 == 0 ]
*X_test, y_test = [r for n, r in enumerate(res) if n % 2 == 1 ]

model_1 = gen_model_2_levels(inputs, output_classes)
# model_1.compile(loss = 'binary_crossentropy', optimizer=tf.keras.optimizers.Adam(), metrics=['accuracy','binary_crossentropy'])
model_1.compile(optimizer=tf.keras.optimizers.Adam(), loss=tf.keras.losses.BinaryCrossentropy(), metrics=['accuracy', 'binary_crossentropy'])

log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=log_dir,
                                                 save_weights_only=True,
                                                 verbose=1)

history_1 = model_1.fit(X_train, y_train, epochs=10, batch_size=64, validation_data=(X_test, y_test),
                        callbacks=[tensorboard_callback, cp_callback])

# %%
