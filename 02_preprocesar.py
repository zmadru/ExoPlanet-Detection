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


def open_light_curve(row, mission="Kepler", download_dir="data3/"):
    """Abrir la curva de luz descargada, juntarlos todos y quitarles Nan

    Args:
        row: Fila del csv a procesar
        mission (str, optional): Mission a buscar en el directorio. Defaults to "Kepler".
        download_dir (str, optional): directorio de descarga. Defaults to "data3/".

    Returns:
       LightCurveCollection: colección de curvas de luz
    """
    kepid = row['kepid']
    kepid = str(kepid).zfill(9)
    files = os.listdir(os.path.join(download_dir, 'mastDownload', mission))
    kepid_dir = [f for f in files if kepid in f][0] if len([f for f in files if kepid in f]) > 0 else None
    if kepid_dir is None:
        return None
    kepid_dir = os.path.join(download_dir, 'mastDownload', mission, kepid_dir)
    kepid_files = [f for f in os.listdir(kepid_dir) if f.endswith('.fits')]
    light_curve_collection = [lk.read(os.path.join(kepid_dir, f)) for f in kepid_files]
    
    return lk.LightCurveCollection([lc for lc in light_curve_collection]) if len(light_curve_collection) > 0 else None


def get_global_local_shape(lc, row):
    """Plegar en fase y dividir en pares e impares

    Args:
        lc (lk.LightCurve): Curva de luz a procesar
        row (pd.Series): Fila del csv a procesar

    Returns:
        tuple: Curvas de luz divididas en pares e impares, global y local
    """
    lc_odd = lc[lc.odd_mask]
    lc_even = lc[lc.even_mask]
    lc_odd.sort("time")
    lc_even.sort("time")
    
    lc_odd_global_flux =  global_view(lc_odd.time.to_value("jd"), lc_odd.flux.to_value(), row.koi_period, normalize=True)
    lc_even_global_flux =  global_view(lc_even.time.to_value("jd"), lc_even.flux.to_value(), row.koi_period, normalize=True)
    lc_odd_local_flux =  local_view(lc_odd.time.to_value("jd"), lc_odd.flux.to_value(), row.koi_period, row.koi_duration, normalize=True)
    lc_even_local_flux =  local_view(lc_even.time.to_value("jd"), lc_even.flux.to_value(), row.koi_period, row.koi_duration, normalize=True)
    
    lc_odd_global = lk.lightcurve.FoldedLightCurve(time=np.arange(len(lc_odd_global_flux)), flux=lc_odd_global_flux)
    lc_even_global = lk.lightcurve.FoldedLightCurve(time=np.arange(len(lc_even_global_flux)), flux=lc_even_global_flux)
    lc_odd_local = lk.lightcurve.FoldedLightCurve(time=np.arange(len(lc_odd_local_flux)), flux=lc_odd_local_flux,)
    lc_even_local = lk.lightcurve.FoldedLightCurve(time=np.arange(len(lc_even_local_flux)), flux=lc_even_local_flux)
    
    return lc_odd_global, lc_even_global, lc_odd_local, lc_even_local


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
    
    # 1. Abrir la curva de luz descargada, juntarlos todos y quitarles Nan
    logger.info(f"Abriendo archivos de {mission} {row.kepid} y Juntando colleción de curvas...")
    lc_collection = open_light_curve(row, mission=mission, download_dir=download_dir)
    if lc_collection is None:
        logger.warning(f"No se han encontrado archivos para {mission} {row.kepid}")
        return None
    lc_collection = lc_collection.stitch()
    lc_collection = lc_collection.remove_nans()

    # 2. Plegar en fase y dividir en pares e impares
    logger.info("Plegando en fase pares/impares...")
    lc_fold = lc_collection.fold(period = row.koi_period,epoch_time = row.koi_time0bk)
    if plot:
        logger.info('graficando series plegadas en fase...')
        lc_fold.plot()
        if plot_folder is not None:
            plt.savefig(f"{plot_folder}/plot/kic_{row.kepid}_01_plegado.png")
            plt.close('all')
        else:
            plt.show()

    # 3. Aplicar bineado en local y global y normalizar
    logger.info("Bineando en vista global y vista local...")
    lc_odd_global, lc_even_global, lc_odd_local, lc_even_local = get_global_local_shape(lc_fold, row)
   
    if plot:
        logger.info('graficando series bineadas...')
        LightCurveGlobalLocalCollection(row.kepid, row, lc_odd_global, lc_even_global, lc_even_local, lc_odd_local).plot()
        if plot_folder is not None:
            plt.savefig(f"{plot_folder}/plot/kic_{row.kepid}_02_bineado.png")
            plt.close('all')
        else:
            plt.show()

    # para quitar oscilaciones en los bordes (quizás mejor no guardar los datos con esto quitado)
    if wavelet_window is not None:
        logger.info("Quitando oscilaciones en los bordes en la ventana seleccionada...")
        lc_odd_global = cut_wavelet(lc_odd_global, wavelet_window)
        lc_even_global = cut_wavelet(lc_even_global, wavelet_window)
        lc_odd_local = cut_wavelet(lc_odd_local, wavelet_window)
        lc_even_local = cut_wavelet(lc_even_local, wavelet_window)
    
    logger.info("Calculando wavelets...")
    lc_even_global = apply_wavelet(lc_even_global, wavelet_family, levels, cut_border_percent=cut_border_percent)
    lc_odd_global = apply_wavelet(lc_odd_global, wavelet_family, levels, cut_border_percent=cut_border_percent)
    lc_even_local = apply_wavelet(lc_even_local, wavelet_family, levels, cut_border_percent=cut_border_percent)
    lc_odd_local = apply_wavelet(lc_odd_local, wavelet_family, levels, cut_border_percent=cut_border_percent)

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
    lc_wavelet_collection = LightCurveWaveletGlobalLocalCollection(row.kepid, headers,
                                                                   lc_even_global,
                                                                   lc_odd_global,
                                                                   lc_even_local,
                                                                   lc_odd_local,
                                                                   levels)

    if(plot):
        logger.info('graficando wavelets obtenidas...')
        if plot_folder is not None:
            figure_paths = (f"{plot_folder}/plot/kic_{row.kepid}_03_wavelet_gi.png",
             f"{plot_folder}/plot/kic_{row.kepid}_03_wavelet_gp.png",
             f"{plot_folder}/plot/kic_{row.kepid}_03_wavelet_li.png",
             f"{plot_folder}/plot/kic_{row.kepid}_03_wavelet_lp.png",
             )
            lc_wavelet_collection.plot(figure_paths=figure_paths)
        else:
            lc_wavelet_collection.plot()
            plt.show()
    if(plot_comparative):
        logger.info('graficando wavelets obtenidas...')
        lc_wavelet_collection.plot_comparative()
    if(save):
        logger.info(f'guardando wavelets obtenidas en {path}...')
        lc_wavelet_collection.save(path)
    return lc_wavelet_collection


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
    #     results.append(process_light_curve(row, levels=[1, 2, 3, 4], wavelet_family="sym5", plot=False, plot_comparative=False,
    #                                             save=False, path="", download_dir="data3/"))
    #     break
    
    # concurrent downloading
    path = "all_data/"
    download_dir="data3/"
    process_func =  partial(process_light_curve, levels=[1, 2, 3, 4], wavelet_family="sym5", plot=False, plot_comparative=False,
                            save=True, path=path, download_dir=download_dir, plot_folder="all_data/")
    # results = progress_map(process_func, [row for _, row in df.iterrows()], n_cpu=8, total=len(df), error_behavior='coerce')
    # lanzar el proceso en paralelo con 12 CPUs, pero fraccionando el ds cada 500 elementos
    results = []
    for i in range(0, len(df), 500):
        res = progress_map(process_func, [row for _, row in df.iloc[i:i+500].iterrows()], n_cpu=8, total=len(df.iloc[i:i+500]), error_behavior='coerce')
        results.extend(res)
        gc.collect()

    failures_idx = [n for n, x in enumerate(results) if not isinstance(x, LightCurveWaveletGlobalLocalCollection)]
    # failures_idx = [n for n, x in enumerate(results) if x is not True]
    failures = [x for x in results if not isinstance(x, LightCurveWaveletGlobalLocalCollection)]
    # failures = [x for x in results if x is not True]
    
    now = int(time.time())
    df_fail = df.loc[failures_idx].copy()
    df_fail['exception'] = failures
    df_fail.to_csv(path+f"/failure_{now}.csv", index=False)
