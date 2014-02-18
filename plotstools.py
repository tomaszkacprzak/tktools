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

def get_bins_centers(bins_hist):

    dx = bins_hist[1] - bins_hist[0]
    # print len(bins_hist)
    new_bins_hist = bins_hist[:-1] + dx/2.
    # print len(new_bins_hist)
    return new_bins_hist



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
        factor = 0.1
        k = kde.gaussian_kde(data.T) 
        k._factor = factor

        grid_x,grid_y,n_grid_x,n_grid_y = self.get_grids(x,y,bins_x,bins_y)    

        xi, yi = np.meshgrid( grid_x , grid_y )
        z = k(np.vstack([xi.flatten(), yi.flatten()]))
        zi = z.reshape(xi.shape)

        return xi,yi,zi


    def nice_plot(self,x,y,bins_x,bins_y):

        # pl.hist2d(x,y,bins=[bins_x,bins_y])
        # pl.imshow(like)
        
        # kde = KDE([x,y])
        # (x_axis,y_axis), like = kde.grid_evaluate(npix, [(min(x),max(x)),(min(y),max(y))] )
        # pl.imshow(like,extent=(min(x),max(x_axis),min(y_axis),max(y_axis)),interpolation='nearest')
        # pl.axis('tight')

        xi,yi,zi=self.kde_grid(x,y,bins_x,bins_y)
        grid_x,grid_y,n_grid_x,n_grid_y = self.get_grids(x,y,bins_x,bins_y)    

        # n_contours = min([n_grid_y,n_grid_x]) / 3
        # log.debug('n_contours = %d' % n_contours)
        n_contours = self.n_contours
        # pl.pcolormesh(xi, yi, zi)
        cp = pl.contour(xi, yi, zi,n_contours,cmap=pl.cm.Blues)
        pl.contourf(xi, yi, zi,n_contours,cmap=pl.cm.Blues)
        # pl.clabel(cp, inline=1, fontsize=8, colors='k')
        # pl.colorbar()



    def plot_dist(self,X,bins='def',labels='def'):


        n_dims = len(X)

        if bins=='def':
            bins = [50]*n_dims
            labels = [str(x) for x in range(n_dims)]
        
        iall=0
        for ip in range(n_dims):

            isub = ip*n_dims + ip + 1
            iall += 1
            log.info( 'panel %d ip %d ic %d isub %d' % (iall,ip,ip,isub) )
            ax=pl.subplot(n_dims,n_dims,isub)       
            pl.hist(X[ip],bins[ip],histtype='step',normed=True,color=self.color_step)
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
                self.nice_plot(X[ip],X[ic],bins[ip],bins[ic])
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




            

