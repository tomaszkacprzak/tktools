import logging, sys
import pylab as pl
import numpy as np

default_log = logging.getLogger("plotstools") 
default_log.setLevel(logging.INFO)  
log_formatter = logging.Formatter("%(asctime)s %(name)s %(levelname)s   %(message)s ","%Y-%m-%d %H:%M:%S")
stream_handler = logging.StreamHandler(sys.stdout)
stream_handler.setFormatter(log_formatter)
default_log.addHandler(stream_handler)
log = default_log

def get_bins_centers(bins_edges):

    # ensure np array
    bins_edges=np.array(bins_edges)

    # get distance
    dx = bins_edges[1] - bins_edges[0]
    # print len(bins_edges)
    bins_centers = bins_edges[:-1] + dx/2.
    # print len(bins_centers)
    return bins_centers

def get_bins_edges(bins_centers):

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

def get_colorscale(n_colors):

    import colorsys
    HSV_tuples = [(x*1.0/n_colors, 0.75, 0.75) for x in range(n_colors)]
    RGB_tuples = map(lambda x: colorsys.hsv_to_rgb(*x), HSV_tuples)

    return RGB_tuples

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

def get_sigma_contours_levels(pdf,list_sigmas=[1,2,3]):

    import scipy
    import scipy.special

    # normalise
    pdf_norm = sum(pdf.flatten())
    pdf = pdf/pdf_norm

    max_pdf = max(pdf.flatten())
    min_pdf = 0.

    n_grid_prob = 2000
    grid_prob = np.linspace(min_pdf,max_pdf,n_grid_prob)

    list_levels = [] 
    conf_tol = 0.001
    diff = np.zeros(len(grid_prob))
    for sig in list_sigmas:

        confidence_level = scipy.special.erf( float(sig) / np.sqrt(2.) )

        log.debug('confindence %d sigmas %5.5f', sig, confidence_level)
        for il, lvl in enumerate(grid_prob):
            mass = sum(pdf[pdf > lvl]) 
            diff[il] = np.abs(confidence_level - mass) 
            # log.debug('diff %5.5f mass=%5.5f lvl=%5.5f at %5.2f' , diff[il], mass,lvl,float(il)/float(n_grid_prob))
        
        ib = diff.argmin()
        vb = grid_prob[ib]
        list_levels.append(vb)
        
        log.debug('confindence %5.5f level %5.5f/%5.5f at %5.2f', confidence_level, vb,max_pdf, float(ib)/float(n_grid_prob))

    return list_levels , list_sigmas

class multi_dim_dist():

    n_contours = 20
    color_step = 'b'


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
        k = kde.gaussian_kde(data.T,bw_method='silverman') 

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
        pl.pcolormesh(xi, yi, zi)
        # cp = pl.contour(xi, yi, zi,n_contours,cmap=pl.cm.Blues)
        contour_levels , contour_sigmas = get_sigma_contours_levels(zi)
        cp = pl.contour(xi, yi, zi,levels=contour_levels,colors='r')


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

        if bins=='def':
            bins = [100]*n_dims
            labels = [str(x) for x in range(n_dims)]
        
        iall=0
        for ip in range(n_dims):

            isub = ip*n_dims + ip + 1
            iall += 1
            log.info( 'panel %d ip %d ic %d isub %d' % (iall,ip,ip,isub) )
            ax=pl.subplot(n_dims,n_dims,isub)       
            pl.hist(X[:,ip],bins=bins[ip],histtype='step',normed=True,color=self.color_step)
            xticks=list(pl.xticks()[0]); del(xticks[0]); del(xticks[-1])
            yticks=list(pl.yticks()[0]); del(yticks[0]); del(yticks[-1])
            pl.xticks(xticks) ; pl.yticks(yticks)
            if iall==1:
                pl.ylabel('histogram')
            # panels in the middle
            # if (( isub % n_dims) != 1) and (isub <= n_dims*(n_dims-1) ):
            #     pl.yticks=[]
            #     pl.xticks=[]
            #     log.info('no xticks')
            #     log.info('no yticks')

            ax.yaxis.tick_right()

            adjust_limits(y_offset_min=0,x_offset_max=0,x_offset_min=0)

            if isub==n_dims**2:
                pl.xlabel(labels[ip])

            for ic in range(ip+1,n_dims):
                isub = ic*n_dims + ip +1
                iall += 1
                log.info( 'panel %d ip %d ic %d isub %d' % (iall,ip,ic,isub) )
                pl.subplot(n_dims,n_dims,isub)
                self.kde_plot(X[:,ip],X[:,ic],bins[ip],bins[ic])
                xticks=list(pl.xticks()[0]); del(xticks[0]); del(xticks[-1])
                yticks=list(pl.yticks()[0]); del(yticks[0]); del(yticks[-1])
                pl.xticks(xticks) ; pl.yticks(yticks)

                # if on the left edge
                if ( isub % n_dims) == 1:
                    pl.ylabel(labels[ic])
                    log.info('ylabel isub %d %s' % (isub,labels[ic]) )

                    # if not on the bottom
                    if (isub <= n_dims*(n_dims-1) ):
                        pl.xticks([])
                        log.info('no xticks')
                else:
                    pl.yticks([])

                # if on the bottom
                if (isub > n_dims*(n_dims-1) ):
                    pl.xlabel(labels[ip])
                    log.info('xlabel isub %d %s' % (isub,labels[ip]) )

                    # if not on the right side
                    if ( isub % n_dims) != 1:
                        pl.yticks([])
                        log.info('no yticks')
                else:
                    pl.xticks([])

def plot_dist(X,bins='def',labels='def'):


    mdd = multi_dim_dist()
    mdd.plot_dist(X,bins,labels)


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




            

