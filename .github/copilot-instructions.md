<!-- Instrucciones concisas para agentes de codificación asistida (Copilot/GitHub AI) -->

# Instrucciones rápidas para contribuir en ExoPlanet-Detection

Estas notas están pensadas para un agente de codificación automático que debe ser productivo inmediatamente en este repositorio.

- **Lenguaje / entorno**: Python 3 (usar el `requirements.txt` para dependencias). Recomendado Python 3.8+.
- **Instalación rápida**:
  - `pip install -r requirements.txt`

- **Estructura importante**:
  - `lib/LCWavelet.py`: núcleo de utilidades de wavelets/starlet y clases serializables. Prefiere usar `LightCurveWavelet*` y las funciones `apply_starlet` / `apply_wavelet` antes que reescribir transformadas.
  - `all_data*/`: carpetas con objetos serializados (`.pickle`). Los objetos suelen ser instancias de `LightCurveWaveletGlobalLocalCollection` o estructuras compatibles.
  - `models/`: pesos entrenados (`*.pth`) usados por `ExoNet.py` y notebooks.
  - `test/` y `old/`: pruebas y ejemplos históricos. `test/pruebas.ipynb` contiene ejemplos de carga y visualización.
  - Notebooks: `Shallue_*.ipynb`, `Wavelet_model.ipynb` muestran pipelines de ejemplo y validación.

- **APIs y convenciones detectadas**:
  - Para cargar pickles: `LightCurveWaveletGlobalLocalCollection.from_pickle(path)`.
  - Los objetos llevan un dict `headers`; la etiqueta de clase suele estar en `headers['class']` (ej.: `'CONFIRMED'`). Haz comparaciones case-insensitive.
  - Para extraer coeficientes:
    - `obj.pliegue_par_global.get_approximation_coefficent(level=K)` o fallback a `obj.pliegue_par_global.get_wavelets()[K-1][0]` si la API difiere.
  - Normalización visual recomendada: z-score NaN-safe: `(x - np.nanmean(x)) / (np.nanstd(x) + 1e-8)`.

- **Flujos de trabajo comunes**:
  - Descargar datos: `python 01_descarga.py` (ver notebook equivalente). Comprueba que `data3/mastDownload` está poblado.
  - Preprocesado: `python 02_preprocesar.py` prepara series y genera pickles en `all_data*`.
  - Entrenamiento: `python 03_entrenamiento.py` o ejecutar `ExoNet.py` según configuración.
  - Notebooks: abrir en JupyterLab / VS Code y ejecutar celdas; muchos notebooks asumen rutas relativas al root (ej.: `..\\all_data_2025_B3` desde `test/`).

- **Pruebas y depuración**:
  - Hay pruebas sencillas en `test/` (ej. `binning_test.py`). Ejecutar: `python -m pytest test`.
  - Para problemas de dependencias, usar entornos virtuales y verificar `requirements.txt`.

- **Patrones y decisiones de diseño**:
  - El proyecto centraliza transformadas en `lib/LCWavelet.py` — reutiliza estas funciones para mantener coherencia en formatos serializados.
  - Se usan objetos serializados (`.pickle`) que contienen tanto coeficientes globales como locales y metadatos; no asumir un único formato: implementar fallbacks defensivos.
  - Notebooks contienen experimentos reproducibles; las modificaciones ideales son añadir celdas que llamen a las utilidades de `lib/` en lugar de duplicar lógica.

- **Integraciones externas**:
  - Dependencias principales: `lightkurve`, `PyWavelets`, `tensorflow`, `torch` y `astropy` (ver `requirements.txt`).
  - Los modelos guardados en `models/` son PyTorch (`.pth`) — inspeccionar `ExoNet.py` para el código de carga.

- **Sugerencias para el agente**:
  - Antes de cambiar formato serializado, busca usos en `test/`, notebooks y `lib/` para evitar romper compatibilidad.
  - Para visualizaciones comparativas normaliza por serie (z-score) y manejar longitudes distintas (resample/interpolate o plot independiente en eje x).
  - Cuando modifiques notebooks, preserva output si se solicita; preferible añadir nuevas celdas con ejemplos reproducibles.

Si falta algo crítico para ser productivo (p. ej. comandos específicos del entorno, versiones exactas de CUDA/torch), indícalo y lo añado aquí. ¿Quieres que incluya ejemplos de comandos exactos para ejecutar los notebooks desde la línea de comandos (nbconvert/nbclient)?

*** Fin de instrucciones para agentes ***
