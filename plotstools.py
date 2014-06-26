import logging, sys, mathstools
# import pylab as pl
import matplotlib.pyplot as pl
import numpy as np

default_log = logging.getLogger("plotstools") 
default_log.setLevel(logging.INFO)  
log_formatter = logging.Formatter("%(asctime)s %(name)s %(levelname)s   %(message)s ","%Y-%m-%d %H:%M:%S")
stream_handler = logging.StreamHandler(sys.stdout)
stream_handler.setFormatter(log_formatter)
default_log.addHandler(stream_handler)
default_log.propagate = False
log = default_log

def plot_radec(ra,dec,lat0=0,lon0=0,**kwargs):

    from mpl_toolkits.basemap import Basemap
    import numpy as np
    bmap = Basemap(projection='ortho',lat_0=lat0,lon_0=lon0,resolution='c')
    bmap.drawparallels(np.arange(-90.,120.,30.))
    bmap.drawmeridians(np.arange(0.,420.,60.))
    bmap.drawmapboundary()
    x1,y1 = bmap(ra,dec)
    bmap.scatter(x1,y1,**kwargs) 

    return x1,y1
 

def imshow_grid( grid_x, grid_y, values_c , nx=None , ny=None):
    """
    @brief Create an image from values on a grid.
    @param values_c values of the image, will be turned into intensity of image. 
                    It has to be in a Nx1 vector, where N=nx*ny, 
                    where nx and ny are number of grid points in x and y direction.
    @param grid_x - grid of points
    """


    if nx==None :
        if len(grid_x) == len(values_c):
            nx = len(np.unique(grid_x))
            # raise Exception('this is broken')
        else:
            nx = len(grid_x)

    if ny==None :
        if len(grid_y) == len(values_c):
            ny = len(np.unique(grid_y))
        else:
            ny = len(grid_y)

    if len(grid_x) == len(values_c):
        values_c_reshaped = np.reshape(values_c,[ny,nx],order='F')
    else:
        values_c_reshaped = np.reshape(values_c,[nx,ny],order='F')

    # print nx , ny , nx*ny , len(values_c)

    if nx*ny != len(values_c):
        raise ValueError('length of values_c %d , calculated grid size %d ,  same as length of grid_x and grid_y. ' , len(grid_c) , len(nx*ny) )



    #     # raise ValueError('length of values_c %d , same as length of grid_x and grid_y. Supply the size of original grid before meshing by using nx and ny arguments.' % len(grid_x))

    # if (len(grid_y) == len(values_c)) and ny==None :
    #     ny = len(np.unique(grid_y))
    #     # raise ValueError('length of values_c %d , same as length of grid_x and grid_y. Supply the size of original grid before meshing by using nx and ny arguments.' % len(grid_x))
    # if nx==None:
    # if ny==None:
    #     ny = len(grid_y)

    pl.imshow( values_c_reshaped , extent=[min(grid_x)  , max(grid_x) , min(grid_y), max(grid_y)], origin='low' , aspect='auto')
    pl.xlim([min(grid_x)  , max(grid_x)])
    pl.ylim([min(grid_y)  , max(grid_y)])




def get_bins_centers(bins_edges,constant_spacing=True):

    # ensure np array
    bins_edges=np.array(bins_edges)
    bins_centers=np.zeros(len(bins_edges)-1)

    # get distance
    if constant_spacing:
        dx = bins_edges[1] - bins_edges[0]
        # print len(bins_edges)
        bins_centers = bins_edges[:-1] + dx/2.
    else:

        for be in range(len(bins_edges)-1):
            bins_centers[be] = np.mean([bins_edges[be],bins_edges[be+1]])      

    # print len(bins_centers)
    return bins_centers

def get_bins_edges(bins_centers,constant_spacing=True):

    
    if constant_spacing:
        bins_centers = np.array(bins_centers)
        dx = bins_centers[1] - bins_centers[0]
        # print len(bins_centers)
        bins_edges = bins_centers.copy() - dx/2.
        last = bins_edges[-1]
        bins_edges = np.append(bins_edges , last+dx)

    else:
        bins_centers=np.array(bins_centers)
        bins_edges=np.zeros(len(bins_centers)+1)

        # fill in first and last using spline
        bins_size = np.diff(bins_centers)
        # import pdb; pdb.set_trace()
        import scipy.interpolate
        spline_rep = scipy.interpolate.splrep( range(len(bins_size)), bins_size, xb=-2, xe=len(bins_size)+2)

        bins_edges[0] =  (bins_centers[1] + bins_centers[0])/2.  - scipy.interpolate.splev(-1,spline_rep)
        bins_edges[-1] =  (bins_centers[-2] + bins_centers[-1])/2. + scipy.interpolate.splev(len(bins_size)+1,spline_rep)
        
        # pl.figure()
        # pl.plot(range(len(bins_size)), bins_size , 'x-')
        # pl.plot(range(-1,len(bins_size)), scipy.interpolate.splev(range(-1,len(bins_size)),spline_rep) , 'o-')

        # dx_first=bins_centers[1]-bins_centers[0]
        # bins_edges[0] = bins_centers[0]-dx_first/2.
        # dx_last=bins_centers[-1]-bins_centers[-2]
        # bins_edges[-1] = bins_centers[-1]+dx_last/2.


        for be in range(1,len(bins_centers)):
            # size of first element is 
            bins_edges[be] = np.mean([bins_centers[be-1],bins_centers[be]])      

        # pl.figure()
        # pl.plot(bins_edges,  np.ones_like(bins_edges),'x-',label='edges')
        # pl.plot(bins_centers, np.zeros_like(bins_centers),'o-',label='centers')
        # pl.ylim([-2,3])
        # pl.xscale('log')
        # pl.legend()
        # pl.show()


    bins_centers = np.array(bins_centers)

    dx = bins_centers[1] - bins_centers[0]
    # print len(bins_centers)
    bins_edges = bins_centers.copy() - dx/2.
    last = bins_edges[-1]
    bins_edges = np.append(bins_edges , last+dx)

    # print len(new_bins_hist)
    return bins_edges


def adjust_limits(x_offset_min=0.1,x_offset_max=0.1,y_offset_min=0.1,y_offset_max=0.1):

    pl.axis('tight')

    xlim = pl.xlim()
    add_xlim_min = x_offset_min*np.abs(max(xlim) - min(xlim))
    add_xlim_max = x_offset_max*np.abs(max(xlim) - min(xlim))
    pl.xlim( [min(xlim) - add_xlim_min, max(xlim) + add_xlim_max] ) 

    ylim = pl.ylim()
    add_ylim_min = y_offset_min*np.abs(max(ylim) - min(ylim))
    add_ylim_max = y_offset_max*np.abs(max(ylim) - min(ylim))
    pl.ylim( [min(ylim) - add_ylim_min, max(ylim) + add_ylim_max] ) 

def get_colorscale(n_colors, cmap_name='jet'):

    # import colorsys
    # HSV_tuples = [(x*1.0/n_colors, 0.75, 0.75) for x in range(n_colors)]
    # RGB_tuples = map(lambda x: colorsys.hsv_to_rgb(*x), HSV_tuples)

    import numpy as np 
    import matplotlib.cm as cm 
    cmap = cm.get_cmap(cmap_name, n_colors) 
    return cmap(np.arange(n_colors)) 

    # return RGB_tuples

def get_contours_corrected(like, x, y, n, xmin, xmax, ymin, ymax, contour1, contour2):
  
    N = len(x)
    x_axis = np.linspace(xmin, xmax, n+1)
    y_axis = np.linspace(ymin, ymax, n+1)
    histogram, _, _ = np.histogram2d(x, y, bins=[x_axis, y_axis])

    def objective(limit, target):
            w = np.where(like>limit)
            count = histogram[w]
            return count.sum() - target
    target1 = N*(1-contour1)
    target2 = N*(1-contour2)
    level1 = scipy.optimize.bisect(objective, like.min(), like.max(), args=(target1,), xtol=1./N)
    level2 = scipy.optimize.bisect(objective, like.min(), like.max(), args=(target2,), xtol=1./N)
    return level1, level2, like.sum()



class multi_dim_dist():

    def __init__(self):
    
        self.n_contours = 20
        self.y_offset_min = 0
        self.x_offset_max = 0
        self.x_offset_min = 0
        self.color = 'b'
        self.contour = True
        self.contourf = False
        self.colormesh = True
        self.scatter = False
        self.labels='def'

    def get_grids(self,x,y,bins_x,bins_y):

        if isinstance(bins_x,int):
            grid_x=np.linspace(x.min(),x.max(),bins_x)
            n_grid_x = bins_x
        elif isinstance(bins_x,list) or isinstance(bins_x,np.ndarray):
            grid_x=bins_x
            n_grid_x = len(bins_x)

        if isinstance(bins_y,int):
            grid_y=np.linspace(y.min(),y.max(),bins_y)
            n_grid_y = bins_y
        elif isinstance(bins_y,list) or isinstance(bins_y,np.ndarray):
            grid_y=bins_y
            n_grid_y = len(bins_y)

        log.debug( 'grid_x' )
        log.debug( str(grid_x) )
        log.debug( 'grid_y' )
        log.debug( str(grid_y) )

        return grid_x,grid_y,n_grid_x,n_grid_y

    def kde_grid(self,x,y,bins_x,bins_y):

        from scipy.stats import kde

        data = np.concatenate([x[:,None],y[:,None]],axis=1)
        # k = kde.gaussian_kde(data.T,bw_method='silverman') 
        k = kde.gaussian_kde(data.T) 

        grid_x,grid_y,n_grid_x,n_grid_y = self.get_grids(x,y,bins_x,bins_y)    

        xi, yi = np.meshgrid( grid_x , grid_y )
        z = k(np.vstack([xi.flatten(), yi.flatten()]))
        zi = z.reshape(xi.shape)

        # normalise
        zi /= sum(zi.flatten())

        return xi,yi,zi


    def kde_plot(self,x,y,bins_x,bins_y):
        """
        @param x samples first dimension
        @param y samples second dimension
        @param bins_x bins for first dimension, as for hist function
        @param bins_y bins for first dimension, as for hist function
        """

        # pl.hist2d(x,y,bins=[bins_x,bins_y])
        # pl.imshow(like)
        
        # kde = KDE([x,y])
        # (x_axis,y_axis), like = kde.grid_evaluate(npix, [(min(x),max(x)),(min(y),max(y))] )
        # pl.imshow(like,extent=(min(x),max(x_axis),min(y_axis),max(y_axis)),interpolation='nearest')
        # pl.axis('tight')

        xi,yi,zi=self.kde_grid(x,y,bins_x,bins_y)
        grid_x,grid_y,n_grid_x,n_grid_y = self.get_grids(x,y,bins_x,bins_y)    

        n_contours = min([n_grid_y,n_grid_x]) / 3
        log.debug('n_contours = %d' % n_contours)
        n_contours = self.n_contours

        contour_levels , contour_sigmas = mathstools.get_sigma_contours_levels(zi)
      
        if self.colormesh:
            pl.pcolormesh(xi, yi, zi)
            # pl.pcolormesh(xi, yi, zi , cmap=pl.cm.YlOrBr)

        if self.contour:
            cp = pl.contour(xi, yi, zi,levels=contour_levels,colors=self.color)
            # cp = pl.contour(xi, yi, zi,n_contours,cmap=pl.cm.Blues)
            # plt.clabel(cp, inline=1, fontsize=10)


        if self.contourf:
            cp = pl.contourf(xi, yi, zi,levels=contour_levels, cmap=pl.cm.Blues)

        if self.scatter:
            pl.scatter(x,y,0.1)

    def plot_dist(self,X,bins='def',labels='def'):


        n_points, n_dims = X.shape

        if labels=='def':
           labels = ['param %d ' % ind for ind in range(n_dims)]

        if bins=='def':
            bins = [100]*n_dims
        
        iall=0
        for ip in range(n_dims):

            isub = ip*n_dims + ip + 1
            iall += 1
            log.debug( 'panel %d ip %d ic %d isub %d' % (iall,ip,ip,isub) )
            pl.subplot(n_dims,n_dims,isub)       
            hist_levels, hist_bins , _ = pl.hist(X[:,ip],bins=bins[ip],histtype='step',normed=True,color=self.color)
            pl.ylim([0,  max(hist_levels)*1.1 ])
           
            xticks=list(pl.xticks()[0]); del(xticks[0]); del(xticks[-1])
            yticks=list(pl.yticks()[0]); del(yticks[0]); del(yticks[-1])
            pl.xticks(xticks) ; pl.yticks(yticks)
            if ip != (n_dims-1):
                pl.xticks(xticks,[])
            if iall==1:
                pl.ylabel(labels[ip])
            
            # panels in the middle
            # if (( isub % n_dims) != 1) and (isub <= n_dims*(n_dims-1) ):
            #     pl.yticks=[]
            #     pl.xticks=[]
            #     log.info('no xticks')
            #     log.info('no yticks')

            pl.gca().yaxis.tick_right()

            # adjust_limits(y_offset_min=self.y_offset_min,x_offset_max=self.x_offset_max,x_offset_min=self.x_offset_min)

            if isub==n_dims**2:
                pl.xlabel(labels[ip])

            for ic in range(ip+1,n_dims):
                isub = ic*n_dims + ip +1
                iall += 1
                log.debug( 'panel %d ip %d ic %d isub %d' % (iall,ip,ic,isub) )
                pl.subplot(n_dims,n_dims,isub)
                self.kde_plot(X[:,ip],X[:,ic],bins[ip],bins[ic])
                xticks=list(pl.xticks()[0]); del(xticks[0]); del(xticks[-1])
                yticks=list(pl.yticks()[0]); del(yticks[0]); del(yticks[-1])
                pl.xticks(xticks) ; pl.yticks(yticks)

                # if on the left edge
                if ( isub % n_dims) == 1:
                    pl.ylabel(labels[ic])
                    log.debug('ylabel isub %d %s' % (isub,labels[ic]) )

                    # if not on the bottom
                    if (isub <= n_dims*(n_dims-1) ):
                        pl.xticks([])
                        log.debug('no xticks')
                else:
                    pl.yticks([])

                # if on the bottom
                if (isub > n_dims*(n_dims-1) ):
                    pl.xlabel(labels[ip])
                    log.debug('xlabel isub %d %s' % (isub,labels[ip]) )

                    # if not on the right side
                    if ( isub % n_dims) != 1:
                        pl.yticks([])
                        log.debug('no yticks')
                else:
                    pl.xticks([])

        pl.subplots_adjust(wspace=0, hspace=0)


    def plot_dist_grid(self,X,y,labels='def'):

        n_points, n_dims = X.shape
        
        list_prob_marg, list_params_marg = mathstools.get_marginals(X,y)
        list_prob_marg_2d, list_params_marg_2d = mathstools.get_marginals_2d(X,y)   

        if labels=='def':
            labels = ['param %d ' % ind for ind in range(n_dims)]

        iall=0
        for ip in range(n_dims):

            isub = ip*n_dims + ip + 1
            iall += 1
            log.debug( 'panel %d ip %d ic %d isub %d' % (iall,ip,ip,isub) )
            pl.subplot(n_dims,n_dims,isub)       

            # pl.hist(X[:,ip],bins=bins[ip],histtype='step',normed=True,color=self.color_step)
            # bar_data_x,bar_data_y = get_plot_bar_data(list_params_marg[ip] , list_prob_marg[ip] )
            # pl.plot(bar_data_x,bar_data_y)           
            pl.plot(list_params_marg[ip] , list_prob_marg[ip] , 'x-')
            xticks=list(pl.xticks()[0]); del(xticks[0]); del(xticks[-1])
            yticks=list(pl.yticks()[0]); del(yticks[0]); del(yticks[-1])
            pl.xticks(xticks) ; pl.yticks(yticks)
            if ip != (n_dims-1):
                pl.xticks(xticks,[])
            if iall==1:
                pl.ylabel(labels[ip])

            # panels in the middle
            # if (( isub % n_dims) != 1) and (isub <= n_dims*(n_dims-1) ):
            #     pl.yticks=[]
            #     pl.xticks=[]
            #     log.info('no xticks')
            #     log.info('no yticks')

            # dx=list_params_marg[ip][1]-list_params_marg[ip][0]
            # pl.xlim([ min(list_params_marg[ip])-dx/2. , max(list_params_marg[ip])+dx/2. ])
            pl.gca().yaxis.tick_right()

            # adjust_limits(y_offset_min=0,x_offset_max=0,x_offset_min=0)

            if isub==n_dims**2:
                pl.xlabel(labels[ip])

            for ic in range(ip+1,n_dims):
                isub = ic*n_dims + ip +1
                iall += 1
                log.debug( 'panel %d ip %d ic %d isub %d' % (iall,ip,ic,isub) )
                pl.subplot(n_dims,n_dims,isub)


                xi = list_params_marg_2d[ip][ic][0]
                yi = list_params_marg_2d[ip][ic][1]
                Xi, Yi = np.meshgrid(xi, yi)
                Zi = list_prob_marg_2d[ip][ic].T
                
                # pl.pcolormesh(Xi, Yi, Zi)
                # dx=list_params_marg[ip][ic][0][1]-list_params_marg[ip][ic][0][0]
                # dy=list_params_marg[ip][ic][1][1]-list_params_marg[ip][ic][1][0]
                # pl.imshow(list_prob_marg_2d[ip][ic].T,extent=[ min(list_params_marg_2d[ip][ic][0])-dx/2. , 
                #                                                 max(list_params_marg_2d[ip][ic][0])+dx/2. , 
                #                                                 min(list_params_marg_2d[ip][ic][1])+dy/2. , 
                #                                                 max(list_params_marg_2d[ip][ic][1])+dy/2. ] , 
                #                                                 aspect='auto')
                pl.contourf(Xi, Yi, Zi , self.n_contours) ; #cmap=pl.cm.Blues 

                # elif self.arr_plot_method == 'contourf':

                # pl.xlim([ min(list_params_marg[ip])-dx/2. , max(list_params_marg[ip])+dx/2. ])
                # pl.ylim([ min(list_params_marg[ip])-dx/2. , max(list_params_marg[ip])+dx/2. ])

                # X, Y = np.meshgrid(x, y)

                # self.kde_plot(X[:,ip],X[:,ic],bins[ip],bins[ic])

                xticks=list(pl.xticks()[0]); del(xticks[0]); del(xticks[-1])
                yticks=list(pl.yticks()[0]); del(yticks[0]); del(yticks[-1])
                pl.xticks(xticks) ; pl.yticks(yticks)

                # if on the left edge
                if ( isub % n_dims) == 1:
                    pl.ylabel(labels[ic])
                    log.debug('ylabel isub %d %s' % (isub,labels[ic]) )

                    # if not on the bottom
                    if (isub <= n_dims*(n_dims-1) ):
                        pl.xticks([])
                        log.debug('no xticks')
                else:
                    pl.yticks([])

                # if on the bottom
                if (isub > n_dims*(n_dims-1) ):
                    pl.xlabel(labels[ip])
                    log.debug('xlabel isub %d %s' % (isub,labels[ip]) )

                    # if not on the right side
                    if ( isub % n_dims) != 1:
                        pl.yticks([])
                        log.debug('no yticks')
                else:
                    pl.xticks([])

        pl.subplots_adjust(wspace=0, hspace=0)


                # adjust_limits(y_offset_min=self.y_offset_min,x_offset_max=self.x_offset_max,x_offset_min=self.x_offset_min)

    def plot_dist_meshgrid(self,X,y):

        n_dims = len(X)
        shape_list = np.array([x for x in X[0].shape])
        if self.labels=='def':  self.labels = ['param %d ' % ind for ind in range(n_dims)]

        list_prob_marg_2d = mathstools.empty_lists(n_dims,n_dims)
        list_params_marg_2d = mathstools.empty_lists(n_dims,n_dims)
        list_prob_marg = [None]*n_dims
        list_params_marg = [None]*n_dims
           
        for dim in range(n_dims):
            sum_axis = range(n_dims)
            sum_axis.remove(dim)
            prob_marg = np.sum(y,axis=tuple(sum_axis))
            params_marg = np.sum(X[dim],axis=tuple(sum_axis)) / np.prod(shape_list[sum_axis])
            list_prob_marg[dim] = prob_marg
            list_params_marg[dim] = params_marg


        for dim1 in range(n_dims):
            for dim2 in range(n_dims):
                if dim1==dim2: continue
                n_grid1 = y.shape[dim1]
                n_grid2 = y.shape[dim2]
                sum_axis = range(n_dims)
                sum_axis.remove(dim1)
                sum_axis.remove(dim2)
                prob_marg = np.sum(y,axis=tuple(sum_axis))
                params_marg1 = np.sum(X[dim1],axis=tuple(sum_axis))/ np.prod(shape_list[sum_axis])
                params_marg2 = np.sum(X[dim2],axis=tuple(sum_axis))/ np.prod(shape_list[sum_axis])
                list_prob_marg_2d[dim1][dim2] = prob_marg
                list_params_marg_2d[dim1][dim2] = (params_marg1,params_marg2) 

        iall=0
        for ip in range(n_dims):

            isub = ip*n_dims + ip + 1
            iall += 1
            log.debug( 'panel %d ip %d ic %d isub %d' % (iall,ip,ip,isub) )
            pl.subplot(n_dims,n_dims,isub)       

            # pl.hist(X[:,ip],bins=bins[ip],histtype='step',normed=True,color=self.color_step)
            # bar_data_x,bar_data_y = get_plot_bar_data(list_params_marg[ip] , list_prob_marg[ip] )
            # pl.plot(bar_data_x,bar_data_y)           
            x=list_params_marg[ip]
            y=list_prob_marg[ip]
            pl.plot( x, y , 'x-')
            xticks=list(pl.xticks()[0]); del(xticks[0]); del(xticks[-1])
            yticks=list(pl.yticks()[0]); del(yticks[0]); del(yticks[-1])
            pl.xticks(xticks) ; pl.yticks(yticks)
            if ip != (n_dims-1):
                pl.xticks(xticks,[])
            if iall==1:
                pl.ylabel(self.labels[ip])

            # panels in the middle
            # if (( isub % n_dims) != 1) and (isub <= n_dims*(n_dims-1) ):
            #     pl.yticks=[]
            #     pl.xticks=[]
            #     log.info('no xticks')
            #     log.info('no yticks')

            # dx=list_params_marg[ip][1]-list_params_marg[ip][0]
            # pl.xlim([ min(list_params_marg[ip])-dx/2. , max(list_params_marg[ip])+dx/2. ])
            pl.gca().yaxis.tick_right()

            # adjust_limits(y_offset_min=0,x_offset_max=0,x_offset_min=0)

            if isub==n_dims**2:
                pl.xlabel(self.labels[ip])

            for ic in range(ip+1,n_dims):
                isub = ic*n_dims + ip +1
                iall += 1
                log.debug( 'panel %d ip %d ic %d isub %d' % (iall,ip,ic,isub) )
                pl.subplot(n_dims,n_dims,isub)


                Xi = list_params_marg_2d[ip][ic][0]
                Yi = list_params_marg_2d[ip][ic][1]
                Zi = list_prob_marg_2d[ip][ic]
                    
                if self.contourf:
                    pl.contourf(Xi, Yi, Zi , self.n_contours) ; #cmap=pl.cm.Blues 
                if self.colormesh:
                    pl.pcolormesh(Xi, Yi, Zi)
                if self.contour:
                    contour_levels , contour_sigmas = mathstools.get_sigma_contours_levels(Zi)
                    cp = pl.contour(Xi , Yi, Zi,levels=contour_levels,colors='m')
                pl.axis('tight')    



                xticks=list(pl.xticks()[0]); del(xticks[0]); del(xticks[-1])
                yticks=list(pl.yticks()[0]); del(yticks[0]); del(yticks[-1])
                pl.xticks(xticks) ; pl.yticks(yticks)

                # if on the left edge
                if ( isub % n_dims) == 1:
                    pl.ylabel(self.labels[ic])
                    log.debug('ylabel isub %d %s' % (isub,self.labels[ic]) )

                    # if not on the bottom
                    if (isub <= n_dims*(n_dims-1) ):
                        pl.xticks([])
                        log.debug('no xticks')
                else:
                    pl.yticks([])

                # if on the bottom
                if (isub > n_dims*(n_dims-1) ):
                    pl.xlabel(self.labels[ip])
                    log.debug('xlabel isub %d %s' % (isub,self.labels[ip]) )

                    # if not on the right side
                    if ( isub % n_dims) != 1:
                        pl.yticks([])
                        log.debug('no yticks')
                else:
                    pl.xticks([])

        pl.subplots_adjust(wspace=0, hspace=0)


                # adjust_limits(y_offset_min=self.y_offset_min,x_offset_max=self.x_offset_max,x_offset_min=self.x_offset_min)

    # def get_upsampled_marginal(self,X,log_prob,sum_axis):

    #     n_dim = len(log_prob.shape)
    #     n_dim_sum = len(sum_axis)
    #     if n_dim - n_dim_sum != 2:
    #         raise Exception('number of dimensions to remain after marginalisation shoud be two, is: n_dim=%d -n_dim_sum=%d',(n_dim,n_dim_sum))


    #     # mega slow loops
    #     all_dim_sizes = range(n_dim)
    #     axis_todo = np.zeros(n_dim)
    #     leave_axis = range(n_dim)
    #     for sa in sum_axis:
    #         leave_axis.remove(sa)
    #     n_ax1 = log_prob.shape[leave_axis[0]]
    #     n_ax2 = log_prob.shape[leave_axis[1]]
    #     axis_todo[leave_axis[0]] = 1
    #     axis_todo[leave_axis[1]] = 2

    #     X1 = X[leave_axis[0]]
    #     X2 = X[leave_axis[1]]
    #     select='['
    #     for ia in range(n_dim):
    #         if axis_todo[ia] >0:
    #             select += ':'
    #         else:
    #             select += '0'
    #         if ia != n_dim-1:
    #             select+=','
    #     select+=']'
    #     X1f = eval( 'X1'+ select)
    #     X2f = eval( 'X2'+ select)


    #     v1f = np.linspace(X1f.min(),X1f.max(),X1.shape[0])
    #     v2f = np.linspace(X2f.min(),X2f.max(),X1.shape[1])

    #     v1ff = np.linspace(X1f.min(),X1f.max(),self.n_upsample*X1.shape[0])
    #     v2ff = np.linspace(X2f.min(),X2f.max(),self.n_upsample*X1.shape[1])

    #     X1ff , X2ff = np.meshgrid(v1f,v2f ,ordering='ij')
    #     X1ff = X1ff.T
    #     X2ff = X2ff.T

    #     for i1 in range(n_ax1):
    #         for i2 in range(n_ax2):
    #             select = 'log_prob['
    #             for ia in range(n_dim):
    #                 if axis_todo[ia] == 1:
    #                     select += '%d' % i1
    #                 elif axis_todo[ia] == 2:
    #                     select += '%d' % i2
    #                 else:
    #                     select += ':'
    #                 if ia != n_dim-1:
    #                     select+=','
    #             select+=']'
    #             # print select
    #         Yf=eval(select)
    #         prob_lores = mathstools.normalise(Yf)
    #         import scipy.interpolate
    #         import pdb; pdb.set_trace()
    #         Fff = scipy.interpolate.interp2d(v1f,v2f,Yf)
    #         Yff = Fff(v1ff,v2ff)
    #         prob_hires = mathstools.normalise(Yff)

    #         pl.pcolormesh(X1f,X2f,Yf)
    #         pl.pcolormesh(X1ff,X2ff,Yff)
    #         pl.show()






    #     import pdb; pdb.set_trace()






def get_plot_bar_data(x,y):
           
    x = np.ravel(zip(x,x+1)) - (x[1]-x[0])/2.
    y = np.ravel(zip(y,y))

    return x,y



def plot_dist(X,bins='def',labels='def',contour=True,colormesh=True,scatter=False,contourf=False,use_fraction=None,color='b'):

    if use_fraction!=None:
        nx = X.shape[0]
        ns = int(nx*use_fraction)

        perm=np.random.permutation(nx)[:ns]
        X=X[perm,:]

    mdd = multi_dim_dist()
    mdd.color=color
    mdd.contour=contour
    mdd.contourf=contourf
    mdd.colormesh=colormesh
    mdd.scatter=scatter
    mdd.plot_dist(X,bins=bins,labels=labels)

def plot_dist_grid(X,y,labels='def'):

    mdd = multi_dim_dist()
    mdd.plot_dist_grid(X,y,labels)

def plot_dist_meshgrid(X,y,labels='def',contour=False,colormesh=True,scatter=False,contourf=False,use_fraction=None,color='b'):

    mdd = multi_dim_dist()
    mdd.color=color
    mdd.contour=contour
    mdd.contourf=contourf
    mdd.colormesh=colormesh
    mdd.scatter=scatter
    mdd.labels=labels
    mdd.plot_dist_meshgrid(X,y)

if __name__ == '__main__':

    bins_fwhm_centers = np.arange(0.7,1.7,0.1)
    bins_fwhm_edges = get_bins_edges(bins_fwhm_centers)
    bins_fwhm_centers2 = get_bins_centers(bins_fwhm_edges)

    print 'bins_fwhm_centers'
    print bins_fwhm_centers
    print 'bins_fwhm_edges'
    print bins_fwhm_edges
    print 'bins_fwhm_centers2'
    print bins_fwhm_centers2




            

