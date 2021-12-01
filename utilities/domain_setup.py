import numpy as np
from numpy import pi

def setup_domain_1D(st,en,h):
    vec = np.arange(st,en + h,h)
    return vec

def setup_domain(Lx = 2.*pi, Ly = 2.*pi, Lz = 2.*pi, nx = 32, ny = 32, nz = 32, \
        zPeriodic = False):
    nxf, nyf, nzf = np.float64((nx,ny,nz))
    dx, dy, dz = (Lx/nxf, Ly/nyf, Lz/nzf)

    x = setup_domain_1D(0.,Lx-dx,dx)
    y = setup_domain_1D(0.,Ly-dy,dy)
    if zPeriodic:
        z = setup_domain_1D(0.,Lz-dz,dz)
    else:
        z = setup_domain_1D(0.5*dz, Lz - 0.5*dz, dz)
        #z = np.arange(0.5*dz, Lz+0.5*dz, dz)

    xmesh, ymesh, zmesh = np.meshgrid(x, y, z, indexing='ij')

    return dx, dy, dz, x, y, z, xmesh, ymesh, zmesh
