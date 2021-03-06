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

def scatter_density(points1,points2,n_bins=50,lim1=None,lim2=None,**kwargs):

    import numpy as np
    import pylab as pl
    if lim1==None:
        min1 = np.min(points1)
        max1 = np.max(points1)
    else:
        min1 = lim1[0]
        max1 = lim1[1]
    if lim1==None:
        min2 = np.min(points2)
        max2 = np.max(points2)
    else:
        min2 = lim2[0]
        max2 = lim2[1]

    bins_edges1=np.linspace(min1,max1,n_bins)
    bins_edges2=np.linspace(min2,max2,n_bins)

    hv,bv,_ = np.histogram2d(points1,points2,bins=[bins_edges1,bins_edges2])

    bins_centers1 = (bins_edges1 - (bins_edges1[1]-bins_edges1[0])/2)[1:]
    bins_centers2 = (bins_edges2 - (bins_edges2[1]-bins_edges2[0])/2)[1:]

    from scipy.interpolate import griddata

    x1,x2 = np.meshgrid(bins_centers1,bins_centers2)
    points = np.concatenate([x1.flatten()[:,np.newaxis],x2.flatten()[:,np.newaxis]],axis=1)
    xi = np.concatenate([points1[:,np.newaxis],points2[:,np.newaxis]],axis=1)
    c = griddata(points,hv.flatten(),xi,method='linear',rescale=True)

    pl.scatter(points1,points2,c=c,**kwargs)





def planck_cmap(ncolors=256):
    from matplotlib.colors import LinearSegmentedColormap as cm
    """
        Returns a color map similar to the one used for the "Planck CMB Map".
        Parameters
        ----------
        ncolors : int, *optional*
            Number of color segments (default: 256).
        Returns
        -------
        cmap : matplotlib.colors.LinearSegmentedColormap instance
            Linear segmented color map.
    """
    segmentdata = {"red":   [(0.0, 0.00, 0.00), (0.1, 0.00, 0.00),
                             (0.2, 0.00, 0.00), (0.3, 0.00, 0.00),
                             (0.4, 0.00, 0.00), (0.5, 1.00, 1.00),
                             (0.6, 1.00, 1.00), (0.7, 1.00, 1.00),
                             (0.8, 0.83, 0.83), (0.9, 0.67, 0.67),
                             (1.0, 0.50, 0.50)],
                   "green": [(0.0, 0.00, 0.00), (0.1, 0.00, 0.00),
                             (0.2, 0.00, 0.00), (0.3, 0.30, 0.30),
                             (0.4, 0.70, 0.70), (0.5, 1.00, 1.00),
                             (0.6, 0.70, 0.70), (0.7, 0.30, 0.30),
                             (0.8, 0.00, 0.00), (0.9, 0.00, 0.00),
                             (1.0, 0.00, 0.00)],
                   "blue":  [(0.0, 0.50, 0.50), (0.1, 0.67, 0.67),
                             (0.2, 0.83, 0.83), (0.3, 1.00, 1.00),
                             (0.4, 1.00, 1.00), (0.5, 1.00, 1.00),
                             (0.6, 0.00, 0.00), (0.7, 0.00, 0.00),
                             (0.8, 0.00, 0.00), (0.9, 0.00, 0.00),
                             (1.0, 0.00, 0.00)]}

    return cm("Planck-like", segmentdata, N=int(ncolors), gamma=1.0)
