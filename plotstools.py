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
        self.color_step = 'b'
        self.y_offset_min = 0
        self.x_offset_max = 0
        self.x_offset_min = 0
        self.arr_plot_method = 'pcolormesh'

    def get_grids(self,x,y,bins_x,bins_y):

        if isinstance(bins_x,int):
            grid_x=np.linspace(x.min(),x.max(),bins_x)
            n_grid_x = bins_x
        elif isinstance(bins_x,list):
            grid_x=bins_x
            n_grid_x = len(bins_x)

        if isinstance(bins_y,int):
            grid_y=np.linspace(y.min(),y.max(),bins_y)
            n_grid_y = bins_y
        elif isinstance(bins_y,list):
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
        # pl.pcolormesh(xi, yi, zi , cmap=pl.cm.YlOrBr)

        if self.intensity:
            pl.pcolormesh(xi, yi, zi)

        if self.contours:
            # cp = pl.contour(xi, yi, zi,n_contours,cmap=pl.cm.Blues)
            contour_levels , contour_sigmas = mathstools.get_sigma_contours_levels(zi)
            cp = pl.contour(xi, yi, zi,levels=contour_levels,colors='r')
            # cp = pl.contourf(xi, yi, zi,levels=contour_levels, cmap=pl.cm.Blues)


        # cp = pl.contour(xi, yi, zi,levels=contour_levels,cmap=pl.cm.Blues)
        # pl.contourf(xi, yi, zi,levels=contour_levels,cmap=pl.cm.Blues)

        # n_contours = 5
        # # cp = pl.contour(xi, yi, zi,n_contours,cmap=pl.cm.jet)
        # # pl.contourf(xi, yi, zi,n_contours,cmap=pl.cm.jet)
        # pl.colorbar()
        # pl.clabel(cp, inline=1, fontsize=8, colors='k')
        # pl.colorbar()


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
            hist_levels, hist_bins , _ = pl.hist(X[:,ip],bins=bins[ip],histtype='step',normed=True,color=self.color_step)
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

def get_plot_bar_data(x,y):
           
    x = np.ravel(zip(x,x+1)) - (x[1]-x[0])/2.
    y = np.ravel(zip(y,y))

    return x,y



def plot_dist(X,bins='def',labels='def',contours=True,intensity=True,use_fraction=None):

    if use_fraction!=None:
        nx = X.shape[0]
        ns = int(nx*use_fraction)
        perm np.random.permutation(nx)[:ns]
        X=X[perm,:]

    mdd = multi_dim_dist()
    mdd.contours=contours
    mdd.intensity=intensity
    mdd.plot_dist(X,bins=bins,labels=labels)

def plot_dist_grid(X,y,labels='def'):

    mdd = multi_dim_dist()
    mdd.plot_dist_grid(X,y,labels)



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




            

