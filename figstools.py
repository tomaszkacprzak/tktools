import logging, sys

loaded_tables = {}

default_logger = logging.getLogger("figstools") 
default_logger.setLevel(logging.INFO)  
log_formatter = logging.Formatter("%(asctime)s %(name)s %(levelname)s   %(message)s ","%Y-%m-%d %H:%M:%S")
stream_handler = logging.StreamHandler(sys.stdout)
stream_handler.setFormatter(log_formatter)
default_logger.addHandler(stream_handler)
default_logger.propagate = False

def get_colorscale(n_colors, cmap_name='jet'):

    import numpy as np 
    import matplotlib.cm as cm 
    cmap = cm.get_cmap(cmap_name, n_colors) 
    return cmap(np.arange(n_colors)) 