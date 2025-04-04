import warnings
warnings.filterwarnings(action="ignore", category=DeprecationWarning)
warnings.filterwarnings(action="ignore", category=FutureWarning)
import pandas as pd
import lightkurve as lk
import numpy as np
import pywt
import pickle
import time
import os
import matplotlib.pyplot as plt
from lib.LCWavelet import *
from lib.binning import global_view, local_view
from parallelbar import progress_map
from tqdm import tqdm
from functools import partial
import logging
import gc


def process_light_curve(row, mission="Kepler", download_dir="data3/",
                        sigma=20, sigma_upper=4,
                        wavelet_window=None,
                        wavelet_family=None, levels=None, cut_border_percent=0.1,
                        plot = False, plot_comparative=False,save=False, path="", plot_folder=None) -> LightCurveWaveletGlobalLocalCollection:
    """

    Args:
        row Fila del csv a procesar: 
        mission (): misión de la que descargarse los  
        download_dir (): directorio de descarga de lightkurve  (si ya existe un archivo en esta ruta no vuelve a bajarlo )
        sigma (): 
        sigma_upper (): 
        wavelet_window (): 
        wavelet_family (): 
        levels (): 
        cut_border_percent (): 
        plot (): 
        plot_comparative (): 
        save (): si es True, guarda los datos procesados en el directorio path
        path (): directorio donde guardar los datos
        plot_folder (): Por defecto None. Si no es None, entonces en vez de enseñar el gráfico lo guarda en {plot_folder}/plot/

    Returns: LightCurveWaveletGlobalLocalCollection

    """


    FORMAT = '%(asctime)s [%(levelname)s] :%(name)s:%(message)s'
    logger = logging.getLogger(f"process_light_curve[{os.getpid()}]")
    if logger.hasHandlers():
        logger.handlers.clear()
    formatter =  logging.Formatter(FORMAT)
    
    logger.setLevel("INFO")
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    if plot_folder is not None:
        os.makedirs(os.path.dirname(f"{plot_folder}/plot/"), exist_ok=True)
    
    # 1. Bajarse los datos con lightkurve o cargarlos si ya están bajados
    logger.info(f"Bajando datos para {mission} {row.kepid}...");
    kic = f'KIC {row.kepid}'
    lc_search = lk.search_lightcurve(kic, mission=mission)
    light_curve_collection = lc_search.download_all(download_dir=download_dir)
    # return light_curve_collection


def process_func_continue(row):
    try:
        return process_func(row)
    except Exception as e:
        print(f"Exception on {row.kepid}")
        import traceback; print("".join(traceback.format_stack()))
        return e

    
if __name__ == "__main__":
    
    df_path = 'csv/cumulative_2022.09.30_09.06.43.csv'
    df = pd.read_csv(df_path ,skiprows=144)
    
    # results = []
    # for _, row in tqdm(df.iterrows(), total=len(df)): # iterative downloading
    #     results.append(process_func_continue(row))
    
    # concurrent downloading
    path = "all_data/"
    download_dir="data3/"
    process_func =  partial(process_light_curve, levels=[1, 2, 3, 4], wavelet_family="sym5", plot=False, plot_comparative=False,
                            save=True, path=path, download_dir=download_dir, plot_folder="all_data/")
    results = progress_map(process_func, [row for _, row in df.iterrows()], n_cpu=12, total=len(df), error_behavior='coerce')

    # failures_idx = [n for n, x in enumerate(results) if not isinstance(x, lk.LightCurveCollection)]
    # # failures_idx = [n for n, x in enumerate(results) if x is not True]
    # failures = [x for x in results if not isinstance(x, lk.LightCurveCollection)]
    # # failures = [x for x in results if x is not True]
    
    # now = int(time.time())
    # df_fail = df.loc[failures_idx].copy()
    # df_fail['exception'] = failures
    # df_fail.to_csv(path+f"/failure_download_{now}.csv", index=False)
