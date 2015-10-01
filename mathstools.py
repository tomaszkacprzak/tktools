import logging, sys, plotstools, warnings
import pylab as pl
import numpy as np


default_log = logging.getLogger("mathstools") 
default_log.setLevel(logging.INFO)  
log_formatter = logging.Formatter("%(asctime)s %(name)s %(levelname)s   %(message)s ","%Y-%m-%d %H:%M:%S")
stream_handler = logging.StreamHandler(sys.stdout)
stream_handler.setFormatter(log_formatter)
default_log.addHandler(stream_handler)
log = default_log
log.propagate = False

def even_bins(x,n_split=10,bins=10,show_plots=False):

    hh,bb = np.histogram(x,bins=bins)
    hh_norm= hh / float(np.sum(hh))
    hh = hh_norm
    cc = np.cumsum(hh)
    cc = np.concatenate([np.zeros(1),cc])

    import scipy.interpolate

    ff = scipy.interpolate.interp1d(cc,bb)
    spp = np.linspace(0,1,n_split)
    bins_edges = ff(spp)

    if show_plots:
        pl.figure()
        pl.plot(bb,cc)
        # for ib,vb in enumerate(bins_edges):        pl.axvline(ib); pl.axhline(spp[ib])
        pl.xticks(bins_edges)
        pl.yticks(spp)
        pl.grid()
        # pl.xscale('log')

        pl.figure()
        pl.plot(cc,bb)
        pl.show()

    return bins_edges



def search_unsorted(sub_set,full_set):
    """
    @param full_set - an array of ints
    @param sub_set - an array of ints, all of ints in sub_set can be found in full_set
    @return indices of sub_set in full_set
    """
    l1 = sub_set
    l2 = full_set

    if len(l1)==0:
        raise Exception('sub_set empty')
    if len(l2)==0:
        raise Exception('full_set empty')

    a1 = np.argsort(sub_set)
    a2 = np.argsort(full_set)
    aa1 = np.argsort(a1)

    s1 = l1[a1]
    s2 = l2[a2]

    union12 = np.union1d(s2,s1)
    if len(union12) > len(s2):
        raise Exception('some of requested ids are not in the catalog' )

    ss = np.searchsorted(s2,s1)
    return a2[ss][aa1]




def get_bin_membership_matrix(x,bin_edges):

    digitized = np.array(np.digitize(x,bins=bin_edges)) -1
    M = np.zeros([len(x),len(bin_edges)-1])
    # for i in range(len(digitized)): 
    #     M[i,digitized[i]]=1
    #     if i % 1000 == 0 : print i, print len(digitized)
    M[range(len(x)),digitized]=1
    return M


def get_func_split(grid,func,n_split_by=100):

    n_grid = len(grid)
    dens = np.zeros_like(grid)

    for i in range(0,n_grid,n_split_by):

        i1 = i
        i2 = i1 + n_split_by
        if i2 > n_grid: i2 = n_grid
        
        dens[i1:i2] = func(grid[i1:i2])
        log.debug('part from %d to %d',i1,i2)

    return dens

def get_normalisation(log_post):

    interm_norm = max(log_post.flatten())
    log_post = log_post - interm_norm
    prob_post = np.exp(log_post)
    prob_norm = np.sum(prob_post)
    prob_post = prob_post / prob_norm
    log_post  = np.log(prob_post)
    log_norm = np.log(prob_norm) + interm_norm
       
    return prob_post , log_post , prob_norm , log_norm

def normalise(log_post):
    """
    @brief normalise the log probability
    @param log_post log probability, can be multidimensional
    @return returns the normalised probability (not log)
    """

    interm_norm = max(log_post.flatten())
    log_post_use = log_post - interm_norm
    prob_post = np.exp(log_post_use)
    prob_norm = np.sum(prob_post)
    prob_post = prob_post / prob_norm
      
    return prob_post

def get_marginals(X,y):
    """
    @brief get marginalised parameter distributions from grid
    @param X grid of parameters in format X.shape= n_combinations, n_params
    @param y probability corresponging to X, len(y) = X.shape[0]
    """

    n_combinations , n_dim = X.shape
    list_margs = []
    list_params = []

    for dim in range(n_dim):
        uniques, inverse = np.unique(X[:,dim],return_inverse=True)
        n_uniques = len(uniques)
        marg = np.zeros(n_uniques)
        log.debug('param %d found %d unique values' % (dim,n_uniques))
        
        # check if sorted
        for iu, vu in enumerate(uniques):
            select = inverse == iu
            marg[iu] = sum(y[select])

        list_margs.append(marg)
        list_params.append(uniques)

    return list_margs, list_params

def empty_lists(nx,ny):

    return [[None for _ in range(nx)] for _ in range(ny)]

def get_marginals_2d(X,y):
    """
    @brief get marginalised parameter distributions from grid
    @param X grid of parameters in format X.shape= n_combinations, n_params
    @param y probability corresponging to X, len(y) = X.shape[0]
    """

    n_combinations , n_dim = X.shape
    list_margs = empty_lists(n_dim,n_dim)
    list_params = empty_lists(n_dim,n_dim)

    for dim1 in range(n_dim):
        for dim2 in range(n_dim):
            
            uniques1, inverse1 = np.unique(X[:,dim1],return_inverse=True)
            uniques2, inverse2 = np.unique(X[:,dim2],return_inverse=True)
            n_uniques1 = len(uniques1)
            n_uniques2 = len(uniques2)
            marg = np.zeros([n_uniques1,n_uniques2])

            log.debug('marginals_2d: params %d %d found %d %d unique values' % (dim1,dim2,n_uniques1,n_uniques2))
       
            # check if sorted
            ia=0
            for ip1,vp1 in enumerate(uniques1):
                for ip2,vp2 in enumerate(uniques2):
                    select = (inverse1 == ip1) * (inverse2 == ip2)
                    marg[ip1,ip2] = sum(y[select])
                    ia += 1
                    
            list_margs[dim1][dim2] = marg
            list_params[dim1][dim2] = (uniques1,uniques2) 

    return list_margs, list_params



def estimate_confidence_interval(par_orig,pdf_orig,plot=False):
    import scipy
    import scipy.interpolate

    # upsample PDF
    n_upsample = 10000
    f = scipy.interpolate.interp1d(par_orig,pdf_orig)
    par = np.linspace(min(par_orig),max(par_orig),n_upsample) 
    pdf = f(par)

    sig = 1
    confidence_level = scipy.special.erf( float(sig) / np.sqrt(2.) )

    pdf_norm = sum(pdf.flatten())
    pdf = pdf/pdf_norm

    max_pdf = max(pdf.flatten())
    min_pdf = 0.

    max_par = par[pdf.argmax()]

    list_levels , _ = get_sigma_contours_levels(pdf,list_sigmas=[1])
    sig1_level = list_levels[0]

    diff = abs(pdf-sig1_level)
    ix = diff.argsort()
    par_lo = par[ix]
    par_hi = par[ix]
    sig_point_lo = par_lo[par_lo<=max_par][0]
    sig_point_hi = par_hi[par_hi>=max_par][0]
    err_hi = sig_point_hi - max_par
    err_lo = max_par - sig_point_lo 
    # if both are on the same side
    if (err_hi > 0) and (err_lo < 0):

        import pdb; pdb.set_trace()
        err_lo = err_hi
        # more options should be implemented here when needed
        warnings.warn('err_lo=err_hi')

    if (err_lo < 1e10):

        err_lo = err_hi
        warnings.warn('very small err_lo, something is wrong, setting to err_lo to err_hi')        

    if plot:
        pl.plot(par,pdf,'x-')
        pl.axvline(x=max_par,linewidth=1, color='c')
        pl.axvline(x=max_par - err_lo,linewidth=1, color='r')
        pl.axvline(x=max_par + err_hi,linewidth=1, color='r')

    log.debug('max %5.5f +%5.5f -%5.5f', max_par, err_hi , err_lo)

    return  max_par , err_hi , err_lo


def estimate_confidence_interval_reflect(par_orig,pdf_orig,plot=False):
    import scipy
    import scipy.interpolate


    # upsample PDF
    n_upsample = 10000
    f = scipy.interpolate.interp1d(par_orig,pdf_orig)
    par = np.linspace(min(par_orig),max(par_orig),n_upsample) 
    pdf = f(par)
    pdf /= np.sum(pdf)

    sig = 1
    confidence_level = scipy.special.erf( float(sig) / np.sqrt(2.) )

    pdf_norm = sum(pdf.flatten())
    pdf = pdf/pdf_norm

    max_pdf = max(pdf.flatten())
    min_pdf = 0.

    max_par = par[pdf.argmax()]

    # higher errorbar
    pdf_hi = pdf[pdf.argmax():]
    pdf_hi = np.concatenate([pdf_hi[::-1],pdf_hi])
    par_hi = par[pdf.argmax():] - par[pdf.argmax()]
    par_hi = np.concatenate([ -par_hi[::-1], par_hi ])
    pdf_hi /= np.sum(pdf_hi)
    
    # lower errorbar
    pdf_lo = pdf[:pdf.argmax()]
    pdf_lo = np.concatenate([pdf_lo,pdf_lo[::-1]])
    par_lo = par[:pdf.argmax()]
    par_lo = np.concatenate([ -par_lo[::-1], par_lo ])
    pdf_lo /= np.sum(pdf_lo)
    
    list_levels , _ = get_sigma_contours_levels(pdf_hi,list_sigmas=[1])
    sig1_level = list_levels[0]
    diff = abs(pdf_hi-sig1_level)
    ix = diff.argmin()
    err_hi = np.abs(par_hi[ix])
    
    list_levels , _ = get_sigma_contours_levels(pdf_lo,list_sigmas=[1])
    sig1_level = list_levels[0]
    diff = abs(pdf_lo-sig1_level)
    ix = diff.argmin()
    err_lo = np.abs(par_lo[ix])


    
    if plot:
        pl.figure(); 
        pl.plot(par_lo,pdf_lo); 
        pl.axvline(err_lo)
        pl.axvline(-err_lo)
        
        pl.figure();
        pl.plot(par_hi,pdf_hi);
        pl.axvline(err_hi)
        pl.axvline(-err_hi)

    log.debug('max %5.5f +%5.5f -%5.5f', max_par, err_hi , err_lo)

    return  max_par , err_hi , err_lo

def get_sigma_contours_levels(pdf,list_sigmas=[1,2,3]):

    import scipy
    import scipy.special

    # normalise
    pdf_norm = sum(pdf.flatten())
    pdf = pdf/pdf_norm
    # pdf=np.log(pdf)

    max_pdf = max(pdf.flatten())
    min_pdf = 0.

    n_grid_prob = 1e2
    grid_prob = np.linspace(min_pdf,max_pdf,n_grid_prob)
    grid_prob_hires = np.linspace(min_pdf,max_pdf,n_grid_prob*1e5)
    log.debug('confidence interval grid dx %1.4e' , grid_prob_hires[1]-grid_prob_hires[0])

    list_levels = [] 
    diff = np.zeros(len(grid_prob))
    # pl.figure()
    for sig in list_sigmas:

        confidence_level = scipy.special.erf( float(sig) / np.sqrt(2.) )

        log.debug('confindence %d sigmas %5.5f', sig, confidence_level)
        for il, lvl in enumerate(grid_prob):
            mass = sum(pdf[pdf > lvl]) 
            diff[il] = np.abs(mass) 
            # log.debug('diff %5.5f mass=%5.5f lvl=%5.5f at %5.2f' , diff[il], mass,lvl,float(il)/float(n_grid_prob))

        import scipy.interpolate
        f_interp=scipy.interpolate.interp1d(grid_prob,diff,'cubic')
        diff_hires=np.abs(f_interp(grid_prob_hires))
       
        ib = np.abs(diff_hires - confidence_level).argmin()
        vb = grid_prob_hires[ib]
        list_levels.append(vb)

        # pl.plot(grid_prob,diff,'rx')
        # pl.plot(grid_prob_hires,diff_hires,'b')
        # pl.axvline(grid_prob_hires[ib])
        
        log.debug('confindence %2.2f sigmas %5.5e level %5.5e/%5.5e', sig, confidence_level, vb,max_pdf)

    # pl.show()
    return list_levels , list_sigmas


def get_kl_divergence_from_samples(samples_p,samples_q,n_neighbor=10):

    # https://www.princeton.edu/~verdu/nearest.neigh.pdf


    from sklearn.neighbors import BallTree

    d = float(samples_p.shape[1])
    n = float(samples_p.shape[0])
    m = float(samples_q.shape[0])
    if m!=n:
        raise Exception('m!=n unsupported')

    ball_tree_p = BallTree(samples_p, leaf_size=5)        
    ball_tree_q = BallTree(samples_q, leaf_size=5)        
    log.info('querying ball 1')
    rho, rho_ind = ball_tree_p.query(samples_p, k=n_neighbor+1)        
    log.info('querying ball 2')
    nu, nu_ind   = ball_tree_q.query(samples_p, k=n_neighbor+1)        

    vec = nu[:,n_neighbor] / rho[:,n_neighbor]
    vec = vec[~np.isnan(vec)]
    vec = vec[~np.isnan(np.log(vec))]
    vec = vec[~np.isinf(vec)]
    vec = vec[~np.isinf(np.log(vec))]

    n = len(vec)
    m = len(vec)

    kl_pq = d / n * np.sum(np.log(vec))  + np.log(m/(n-1.))

    return kl_pq

def test_kl_divergence_from_samples():


    samples_p = np.random.randn(10000)[:,None]
    samples_q = np.random.randn(10000)[:,None]

    sig1=5
    grid_sig2 = np.linspace(1,10,10)
    list_kl_pq = []
    for sig2 in grid_sig2:

        kl_pq = get_kl_divergence_from_samples(samples_p*sig1,samples_q*sig2)
        list_kl_pq.append(kl_pq)
        print sig1,sig2,kl_pq

    pl.plot(grid_sig2,list_kl_pq)
    pl.show()

def get_1D_fwhm(profile,fwxm=0.5):

    # pl.figure()

    f0 = max(profile)
    x0 = np.argmax(profile)

    profile1 = profile[:x0]
    profile2 = profile[x0:]

    cut_val = f0 * fwxm

    diff = abs(profile1 - cut_val)
    f1 ,f2, x1, x2 = 0. , 0. , 0. , 0.
    x1 = np.argmin(diff) ;  f1 = profile1[x1]
    if( f1 < cut_val ):  x2 = x1+1
    else:       x2 = x1-1
    f2 = profile1[x2];
    a = (f1-f2)/(x1 - x2)
    b = f1 - a*x1;
    x31 = (cut_val - b)/a;
    # pl.plot(x1,f1,'dc',ms=10)
    # pl.plot(x2,f2,'dg',ms=10)
    # print x1,f1
    # print x2,f2

    diff = abs(profile2 - cut_val)
    f1 ,f2, x1, x2 = 0. , 0. , 0. , 0.
    x1 = np.argmin(diff) ;  f1 = profile2[x1]
    if( f1 < cut_val ):  x2 = x1-1
    else:       x2 = x1+1
    f2 = profile2[x2];
    a = (f1-f2)/(x1 - x2)
    b = f1 - a*x1;
    x32 = (cut_val - b)/a + x0
    # pl.plot(x0+x1,f1,'dc',ms=10)
    # pl.plot(x0+x2,f2,'dg',ms=10)
    # print x1,f1
    # print x2,f2
    fwhm = np.abs(x31-x32)

    # fwhm =  np.abs(2.* (x3-x0))

    # pl.plot(x0,f0,'ro')
    # pl.plot(profile,'r.-')
    # pl.plot(x31,cut_val,'bo')
    # pl.plot(x32,cut_val,'bo')
    # pl.axhline(f0*fwxm)


    return fwhm


def get_2D_fwhm(lores_img,n_angles=64):

    n_sub = 11
    img = np.kron(lores_img,np.ones([n_sub,n_sub]))

    n_pix = img.shape[0]
    s_box = float(n_pix)*np.sqrt(2.)/4. + 1e-7
    n_box = int(s_box)
    s_pix = 0.5


    x0,y0 = n_pix/2. ,n_pix/2.
    X,Y=np.meshgrid(np.arange(n_pix)-x0,np.arange(n_pix)-y0)

    x= np.concatenate( [ X.flatten()[:,None] ,  Y.flatten()[:,None] ],axis=1)
    i= img.flatten()[:,None]
    # print 'max image' , max(i)

    angles= np.linspace(0,2,n_angles)*np.pi
    # angles= np.array([0 , np.pi/4 , np.pi/2.])
    y1 = np.array( [ np.exp(1j*angles).real , np.exp(1j*angles).imag ] )
    y2 = np.array( [ np.exp(1j*(angles+np.pi/2.)).real , np.exp(1j*(angles+np.pi/2.)).imag ] )


    p = np.dot(x,y1)
    r = np.dot(x,y2)

    # s_upsampled_pixel = float(n_box)*2./n_sub
    grid_edges = np.linspace(-s_box/2.,s_box/2.,float(n_box)/float(n_sub)*4)
    grid_centers = plotstools.get_bins_centers(grid_edges)
    
    colors=plotstools.get_colorscale(len(angles))
    h=np.zeros([len(grid_centers),len(angles)])
    # pl.figure(100)
    for ia,ang in enumerate(angles):

        # select close pixel -- 
        select = (np.abs(r[:,ia]) < s_pix) * (np.abs(p[:,ia]) < s_box)

        #for angle measurement, select all pixels in box
        # select = (np.abs(r[:,ia]) < s_box) * (np.abs(p[:,ia]) < s_box)

        # pl.figure()
        # pl.scatter(x[:,0],x[:,1],c='y'); 
        # pl.scatter(x[select,0],x[select,1]);
        # pl.axis('equal')
        # pl.show()

        h1 , _ = np.histogram(p[select,ia][:,None],bins=grid_edges,weights=i[select])
        h2 , _ = np.histogram(p[select,ia][:,None],bins=grid_edges)
        h[:,ia] = h1/h2

        # # from scipy.interpolate import Rbf
        # # rbf = Rbf(p[:,ia], i.flatten())
        # # h[:,ia] = rbf(grid_centers)
        pl.plot(grid_centers,h[:,ia],'.',label=str(ang/np.pi),c=colors[ia])

    mean_prof = np.mean(h,axis=1)
    # max_prof = np.min(h,axis=1)
    # min_prof = np.max(h,axis=1)
    # mean_prof = (max_prof + min_prof)/2.
    dx=np.abs(grid_centers[2]-grid_centers[1])
    fwhm=get_1D_fwhm(mean_prof) * dx / float(n_sub)

    # pl.figure(100)
    # pl.plot(grid_centers,mean_prof,'ks-',label='mean',ms=5)
    # grid_centered=np.arange(img.shape[0]) - img.shape[0]/2
    # pl.plot(grid_centered, img[img.shape[0]/2,:],'gd')
    # pl.plot(grid_centered, img[:,img.shape[0]/2],'gd')
    # pl.plot(-fwhm/2.,0.5*max(mean_prof),'ro',ms=5)
    # pl.plot(+fwhm/2.,0.5*max(mean_prof),'ro',ms=5)
    # global true_fwhm
    # pl.plot(-true_fwhm/2.*n_sub,0.5*max(lores_img.flatten()),'rd',ms=5)
    # pl.plot(+true_fwhm/2.*n_sub,0.5*max(lores_img.flatten()),'rd',ms=5)
    # pl.legend()


    # print 'dx' , dx
    # print 'get_1D_fwhm' , fwhm / float(n_sub)
    # print 'get_1D_fwhm' , get_1D_fwhm(mean_prof)
    # print 'get_1D_fwhm x cross' , get_1D_fwhm(lores_img[lores_img.shape[0]/2,:])/float(n_sub)
    # print 'get_1D_fwhm y cross' , get_1D_fwhm(lores_img[:,lores_img.shape[0]/2])/float(n_sub)

    return fwhm

    # pl.figure()
    # pl.plot(p,i,'r.')
