import numpy as np

def distortion_to_shear(E1,E2):

    xi = E1 + E2*1j
    q = np.sqrt( (1 - np.abs(xi))/(1 + np.abs(xi)) )
    e = (1-q)/(1+q) * np.exp(1j*np.angle(xi))
    e1 = e.real
    e2 = e.imag

    return e1,e2

def add_shear(e1,e2,g1,g2):

    e = e1+1j*e2
    g = g1+1j*g2
    es = (e+g)/(1+g.conjugate()*e)
    es1 = es.real
    es2 = es.imag

    return es1,es2




