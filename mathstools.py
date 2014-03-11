import logging, sys, plotstools
import pylab as pl
import numpy as np


default_log = logging.getLogger("mathstools") 
default_log.setLevel(logging.INFO)  
log_formatter = logging.Formatter("%(asctime)s %(name)s %(levelname)s   %(message)s ","%Y-%m-%d %H:%M:%S")
stream_handler = logging.StreamHandler(sys.stdout)
stream_handler.setFormatter(log_formatter)
default_log.addHandler(stream_handler)
log = default_log


def estimate_confidence_interval(par_orig,pdf_orig):
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
    ix1,ix2 = diff.argsort()[:2]
    par_x1 , par_x2 = par[ix1] , par[ix2]
    sig_point_lo = min([par_x1,par_x2])
    sig_point_hi = max([par_x1,par_x2])
    err_hi = sig_point_hi - max_par
    err_lo = max_par - sig_point_lo 
    # if both are on the same side
    if (err_hi > 0) and (err_lo < 0):
        err_lo = err_hi
        # more options should be implemented here when needed

    pl.figure()
    pl.plot(par,pdf,'x-')
    pl.axvline(x=max_par,linewidth=1, color='c')
    pl.axvline(x=max_par - err_lo,linewidth=1, color='r')
    pl.axvline(x=max_par + err_hi,linewidth=1, color='r')

    log.debug('max %5.5f +%5.5f -%5.5f', max_par, err_hi , err_lo)

    return  max_par , err_hi , err_lo

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

