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
# %pdb off
import pandas as pd
import lightkurve as lk
import numpy as np
import pywt
import pickle
import os
import matplotlib.pyplot as plt
from LCWavelet import *
from binning import bin_and_aggregate
from tqdm import tqdm

df_path = 'cumulative_2022.09.30_09.06.43.csv'
df = pd.read_csv(df_path ,skiprows=144)


def process_light_curve(row, mission="Kepler", download_dir="data2/",
                        binning_parameters=None,
                        sigma=20, sigma_upper=4,
                        wavelet_window=None,
                        wavelet_family=None, levels=None, cut_border_percent=0.1,
                        plot = False, plot_comparative=False,save=False, path=""):

    kic = f'KIC {row.kepid}'
    lc_search = lk.search_lightcurve(kic, mission=mission)
    light_curve_collection = lc_search.download_all(download_dir=download_dir)

    # lc_collection = lk.LightCurveCollection([lc.remove_outliers(sigma=sigma, sigma_upper=sigma_upper) for lc in light_curve_collection])
    lc_collection = lk.LightCurveCollection([lc for lc in light_curve_collection])

    lc_ro = lc_collection.stitch()

    lc_nonans = lc_ro.remove_nans()
    
    lc_fold = lc_nonans.fold(period = row.koi_period,epoch_time = row.koi_time0bk)
    lc_odd = lc_fold[lc_fold.odd_mask]
    lc_even = lc_fold[lc_fold.even_mask]
    
    if binning_parameters is not None and "lambda" in binning_parameters.keys() and "delta" in binning_parameters.keys():
        lmb = binning_parameters["lambda"]
        delta = binning_parameters["delta"]
        lc_odd =  bin_and_aggregate(np.arange(lc_odd), lc_odd)
        lc_odd =  bin_and_aggregate(np.arange(lc_odd), lc_odd)
        pass

    if wavelet_window is not None:
        print('Aplicando ventana ...')
        lc_impar = cut_wavelet(lc_odd, wavelet_window)
        lc_par = cut_wavelet(lc_even, wavelet_window)
    else:
        lc_impar = lc_odd
        lc_par = lc_even

    # para quitar oscilaciones en los bordes (quizás mejor no guardar los datos con esto quitado)
    lc_w_par = apply_wavelet(lc_par, wavelet_family, levels, cut_border_percent=cut_border_percent)
    lc_w_impar = apply_wavelet(lc_impar, wavelet_family, levels, cut_border_percent=cut_border_percent)

    headers = {
        "period": row.koi_period,
        "koi_period_err1": row.koi_period_err1,
        "koi_period_err2": row.koi_period_err2,
        "depth": row.koi_depth,
        "depth_err1": row.koi_depth_err1,
        "depth_err2": row.koi_depth_err2,
        "duration": row.koi_duration,
        "duration_err1": row.koi_duration_err1,
        "duration_err2": row.koi_duration_err2,
        "steff": row.koi_steff,
        "steff_err1": row.koi_steff_err1,
        "steff_err2": row.koi_steff_err2,
        "impact": row.koi_impact,
        "impact_err1": row.koi_impact_err1,
        "impact_err2": row.koi_impact_err2,
        "class": row.koi_disposition,
        "wavelet_family":wavelet_family,
        "levels":levels,
        "window":wavelet_window,
        "border_cut":cut_border_percent,
        "Kepler_name":row.kepoi_name
    }
    lc_wavelet_collection = LightCurveWaveletCollection(row.kepid, headers, lc_w_par, lc_w_impar)

    if(plot):
        print('graficando wavelets obtenidas...')
        lc_w_par.plot()
        lc_w_inpar.plot()
    if(plot_comparative):
        print('graficando wavelets obtenidas...')
        lc_wavelet_collection.plot_comparative()
    if(save):
        # print('guardando wavelets obtenidas...')
        lc_wavelet_collection.save(path)
    return lc_wavelet_collection

# for _, row in tqdm(df.iterrows(), total=len(df)):
#     process_light_curve(row, levels=[1, 2, 3, 4], wavelet_family="sym5", plot=False, plot_comparative=False, save=True, path="all_data_2024-06-01/"
#                        binning_parameters={"delta": , "lambda": })

# process_light_curve


# %%
np.arange(len([1, 2, 3, 4]))

# %%
import os
kep_id = list('KIC ' + df.kepid.astype(str))
import traceback
def download_one(kic):
    try:
        lc_search = lk.search_lightcurve(kic, mission="Kepler")
        light_curve_collection = lc_search.download_all(download_dir="data3/")
    except Exception:
        print(traceback.format_exc())
    return None


# for row in tqdm(kep_id):
#     download_one(row)
 

# from prpl import prpl
# from atpbar import flushing
# prpl(target_list=kep_id, target_function=download_one, list_sep=16, timer=True)
from concurrent import futures
from parallelbar import progress_map
# with flushing(), futures.ThreadPoolExecutor(max_workers=32) as executor:
#     res = executor.map(download_one, kep_id)
progress_map(download_one, kep_id, n_cpu=16)

# pd.concat([lc.table.to_pandas() for lc in lc_search]).to_csv("table.csv", index=None)
# lc_search[0].table.to_pandas()
# light_curve_collection

# %%
