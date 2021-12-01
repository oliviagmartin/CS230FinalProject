import h5py
import numpy as np
import time

def get_domain_size(fname, dsname):
    buf   = h5py.File(fname,'r')
    nx,ny,nz = np.array(buf[dsname][:,:,:]).shape
    buf.close()
    return nx, ny, nz

def read_fortran_data(fname, dsname, order = 'F'):
    buf   = h5py.File(fname,'r')
    nz,ny,nx = np.array(buf[dsname][:,:,:]).shape
    f = np.empty((nx,ny,nz), dtype = np.dtype(np.float64, align = True),\
            order = order)
    f[:] = np.transpose(buf[dsname][:])
    buf.close()
    return f

if __name__ == '__main__':
    u = read_fortran_data('/Users/ryanhass/Documents/MATLAB/CS_230/DNS_test_data.h5','uVel')
    print(u[:10,0,0])
