import logging, sys
import pylab as pl
import numpy as np

default_log = logging.getLogger("plotstools") 
default_log.setLevel(logging.INFO)  
log_formatter = logging.Formatter("%(asctime)s  %(name)s  %(levelname)s  %(message)s ")
stream_handler = logging.StreamHandler(sys.stdout)
stream_handler.setFormatter(log_formatter)
default_log.addHandler(stream_handler)

def adjust_limits(x_offset=0.1,y_offset=0.1):

    pl.axis('tight')
    
    xlim = pl.xlim()
    add_xlim = x_offset*np.abs(max(xlim) - min(xlim))
    pl.xlim( [min(xlim) - add_xlim, max(xlim) + add_xlim] ) 

    ylim = pl.ylim()
    add_ylim = y_offset*np.abs(max(ylim) - min(ylim))
    pl.ylim( [min(ylim) - add_ylim, max(ylim) + add_ylim] ) 

def get_colorscale(n_colors):

    import colorsys
    HSV_tuples = [(x*1.0/n_colors, 0.75, 0.75) for x in range(n_colors)]
    RGB_tuples = map(lambda x: colorsys.hsv_to_rgb(*x), HSV_tuples)

    return RGB_tuples
