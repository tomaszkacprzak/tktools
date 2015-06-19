import numpy as np
import warnings

def to_radians(x,unit='deg'):

    if unit=='deg':
        x_rad = x *np.pi/180.
    elif unit=='rad':
        x_rad = x 
    elif unit=='arcmin':
        x_rad = x *np.pi/180./60.
    elif unit=='arcsec':
        x_rad = x *np.pi/180./60./60.
    return x_rad

def from_radians(x_rad,unit='deg'):

    if unit=='deg':
        x = x_rad /np.pi*180.
    elif unit=='rad':
        x = x_rad 
    elif unit=='arcmin':
        x = x_rad /np.pi*180.*60.
    elif unit=='arcsec':
        x = x_rad /np.pi*180.*60.*60.
    return x


def get_gnomonic_projection(ra, de , ra_center, de_center, unit='rad'):
# http://mathworld.wolfram.com/GnomonicProjection.html
    
    ra_rad , de_rad = to_radians(ra,unit=unit), to_radians(de,unit=unit)
    ra_center_rad , de_center_rad = to_radians(ra_center,unit=unit), to_radians(de_center,unit=unit)

    cos_c = np.sin(de_center_rad)*np.sin(de_rad) + np.cos(de_center_rad) * np.cos(de_rad) * np.cos(ra_rad  - ra_center_rad)
    x_rad = np.cos(de_rad)*np.sin(ra_rad - ra_center_rad)
    y_rad = np.cos(de_center_rad) * np.sin(de_rad) - np.sin(de_center_rad)*np.cos(de_rad)*np.cos(ra_rad-ra_center_rad) 
    y_rad = y_rad/cos_c

    x , y = from_radians(x_rad,unit=unit), from_radians(y_rad,unit=unit)

    return x,y

def get_gnomonic_projection_shear(ra, de , ra_center, de_center, shear_g1 , shear_g2, unit='rad'):

    ra_rad, de_rad = to_radians(ra,unit=unit),  to_radians(de,unit=unit)
    ra_center_rad, de_center_rad = to_radians(ra_center,unit=unit), to_radians(de_center,unit=unit)

    # code from Jospeh
    # Here use a tangent plane to transform the sources' shear
    # from RA, DEC to a common local coordinate system around lens
    # phi_bar = ra_l[i] * pi/180.
    # th_bar = (90. - dec_l[i]) * pi/180.
    phi_bar = ra_center_rad
    th_bar = np.pi/2. - de_center_rad

    # phi = ra_s * (pi/180.)
    # theta = (90. - dec_s) * (pi/180.)
    phi = ra_rad
    theta = np.pi/2. - de_rad 

    gamma = np.abs(phi - phi_bar)

    cot_delta = ( np.sin(theta)  / np.tan(th_bar) - np.cos(theta)  * np.cos(gamma) ) /np.sin(gamma)
    cot_beta  = ( np.sin(th_bar) / np.tan(theta)  - np.cos(th_bar) * np.cos(gamma) ) /np.sin(gamma)
    delta = np.arctan(1./cot_delta)
    beta  = np.arctan(1./cot_beta )

    del_bar = np.pi - beta
    sgn = np.sign(phi - phi_bar)
    phase = np.exp(sgn *2.j *np.abs(delta - del_bar))

    # Now place shear in complex number, rotate by phase
    temp = (shear_g1 + 1.j*shear_g2) * phase

    gam1 = temp.real
    gam2 = temp.imag

    return gam1 , gam2

def get_inverse_gnomonic_projection(x,y,ra_center, de_center):  
    
    x_rad,y_rad = to_radians(x), to_radians(y)
    ra_center_rad, de_center_rad = to_radians(ra_center), to_radians(de_center)
    # http://mathworld.wolfram.com/GnomonicProjection.html
    # phi = dec
     # lambda = ra
    rho = np.sqrt( x_rad**2 + y_rad**2)
    c = np.arctan(rho)
    gamma = (y_rad*np.sin(c)*np.cos(de_center_rad))/(rho)
    if not np.isfinite(gamma): 
        warnings.warn( 'get_inverse_gnomonic_projection may be numerically unstable' )
        gamma = 0
    de_rad = np.arcsin( np.cos(c)*np.sin(de_center_rad) + gamma )
    ra_rad = ra_center_rad + np.arctan( (x_rad*np.sin(c))/(rho*np.cos(de_center_rad)*np.cos(c) - y_rad*np.sin(de_center_rad)*np.sin(c)) )
    ra,de = from_radians(ra_rad),from_radians(de_rad)
    return ra,de



#     The inverse transformation equations are

# phi =   sin^(-1)(coscsinphi_1+(ysinccosphi_1)/rho)  
# (4)
# lambda  =   lambda_0+tan^(-1)((xsinc)/(rhocosphi_1cosc-ysinphi_1sinc)), 
# (5)
# where

# rho =   sqrt(x^2+y^2)   
# (6)
# c   =   tan^(-1)rho



def get_angular_separation_vec(radec1_rad,radec2_rad):

    d_ra =  np.abs(radec1_rad[:,0]-radec2_rad[:,0])
    d_de =  np.abs(radec1_rad[:,1]-radec2_rad[:,1])
    theta = np.arccos( np.sin(de1_rad)*np.sin(de2_rad) + np.cos(de1_rad)*np.cos(de2_rad)*np.cos(d_ra))
    return theta


def get_angular_separation(ra1,de1,ra2,de2,unit='rad'):

    ra1_rad = to_radians(ra1,unit=unit)
    de1_rad = to_radians(de1,unit=unit)
    ra2_rad = to_radians(ra2,unit=unit)
    de2_rad = to_radians(de2,unit=unit)
    d_ra =  np.abs(ra1_rad-ra2_rad)
    d_de =  np.abs(de1_rad-de2_rad)
    theta_rad = np.arccos( np.sin(de1_rad)*np.sin(de2_rad) + np.cos(de1_rad)*np.cos(de2_rad)*np.cos(d_ra))
    theta = from_radians(theta_rad,unit=unit)
    return theta


def get_projection_matrix(center_ra,center_dec):

    delta = 1e-5

    ds_ra = get_angular_separation(0.,0.,delta,0.)
    ds_dec = get_angular_separation(0.,0.,0.,delta)

    P = [[ds_ra/delta, 0],[0,ds_dec/delta]]

    return P


def get_midpoint( halo1_ra , halo1_de , halo2_ra , halo2_de , unit='rad' ):

    
    halo1_ra_rad , halo1_de_rad = to_radians(halo1_ra,unit=unit) , to_radians(halo1_de,unit=unit)
    halo2_ra_rad , halo2_de_rad = to_radians(halo2_ra,unit=unit) , to_radians(halo2_de,unit=unit)

    # lon <-> RA , lat <-> DEC
    # http://www.movable-type.co.uk/scripts/latlong.html
    Bx = np.cos(halo2_de_rad) * np.cos(halo2_ra_rad - halo1_ra_rad)
    By = np.cos(halo2_de_rad) * np.sin(halo2_ra_rad - halo1_ra_rad)
    mid_de_rad = np.arctan2( np.sin(halo1_de_rad) + np.sin(halo2_de_rad) , np.sqrt( (np.cos(halo1_de_rad) + Bx)**2 + By**2 ) )
    mid_ra_rad = halo1_ra_rad + np.arctan2(By , np.cos(halo1_de_rad) + Bx)

    mid_de , mid_ra = from_radians(mid_de_rad,unit=unit) , from_radians(mid_ra_rad,unit=unit)

    return mid_ra, mid_de

def deg2rad(ra_deg,de_deg):

    ra_rad = ra_deg*np.pi/180.
    de_rad = de_deg*np.pi/180.    

    return ra_rad,de_rad

def rad2deg(ra_rad,de_rad):

    ra_deg = ra_rad * 180. / np.pi
    de_deg = de_rad * 180. / np.pi

    return ra_deg , de_deg


def arcsec2deg(ra_arcsec,de_arcsec):

    return ra_arcsec/3600. , de_arcsec/3600.

def deg2arcsec(ra_deg,de_deg):

    return ra_deg*3600, de_deg*3600

def deg2arcmin(ra_deg,de_deg):

    return ra_deg*60, de_deg*60


def rad2arcsec(ra_rad,de_rad):

    ra_arcsec = ra_rad/np.pi*180*3600. 
    de_arcsec = de_rad/np.pi*180*3600.

    return ra_arcsec , de_arcsec

def arcmin2rad(ra_arcmin,de_arcmin):

    ra_rad = ra_arcmin*np.pi/180/60. 
    de_rad = de_arcmin*np.pi/180/60. 

    return ra_rad , de_rad

def rad2arcmin(ra_rad,de_rad):

    ra_arcmin = ra_rad/np.pi*180*60. 
    de_arcmin = de_rad/np.pi*180*60.

    return ra_arcmin , de_arcmin

def radec2pix(nside,ra,dec,nest=True):
    import healpy as hp;
    x1,x2=np.pi/2-np.deg2rad(dec),np.deg2rad(ra)
    return hp.ang2pix(nside,x1,x2,nest=nest)

def pix2radec(nside,pix,nest=True):
    import healpy as hp;
    x1,x2 = hp.pix2ang(nside,pix,nest=nest)
    ra,dec=np.rad2deg(x2),np.rad2deg(np.pi/2-x1)
    return ra,dec

