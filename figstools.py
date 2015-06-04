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

# def get_scatter_density(x1,x2,points1,points2,xp1,xp2,scatter_point_size=20):

# 	xs=np.concatenate([xp1[:,None],xp2[:,None]],axis=1)
#     P1,P2 = np.meshgrid(points1,points2)
#     points = np.concatenate([P1.flatten()[:,None],P2.flatten()[:,None]],axis=1)
#     points_centers1 = (points1[1:]+points1[:-2])/2.
#     points_centers2 = (points1[1:]+points1[:-2])/2.
#     P1,P2 = np.meshgrid(points_centers1 , points_centers2)
#     points_centers = np.concatenate([P1.flatten()[:,None],P2.flatten()[:,None]],axis=1)
#     ha,_,_ = np.histogram2d(x1, x2, bins = [points1,points2])
#     import scipy.interpolate
#     z = scipy.interpolate.griddata(points_centers,ha.T.flatten(),xs,fill_value=0)
#     return z
