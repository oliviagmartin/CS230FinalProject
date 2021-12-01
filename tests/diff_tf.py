# ME 451B Homework 5 (Autumn 2020)
# This is the startup code for the linear stability analysis of the Rayleigh-Benard-Poiseuille flow.
# In this file the class of differential operator is defined and implemented. The default parameters
# for the class construction is based on the problem configuration. The first derivative in x and y
# are based on the spactral method, and the first derivative in z uses the 6th order compact finite
# difference scheme.
#
# HOW TO USE:
# from diff import DiffOps
#
# Programmer: Hang Song (songhang@stanford.edu)
# Translation to Tensorflow: Kyle Pietrzyk, Ryan Hass, and Olivia Martin

import numpy as np
from numpy import pi
from scipy.fft import fft, ifft, fftshift
from scipy.sparse import lil_matrix
from scipy.sparse.linalg import splu
import sys
import code
import tensorflow as tf
import time

import matplotlib.pyplot as plt


class DiffOps:
    def __init__(self, nx = 256, ny = 128, nz = 64, Lx = 4.0*pi, Ly = 2.0*pi, Lz = 1.0):
        self.NX, self.NY, self.NZ = nx, ny, nz
        self.buf_cmplx = np.empty((nx,ny,nz), dtype=np.dtype(np.complex128, align=True), order='F')
        self.buf_z = np.empty((nz, nx*ny), dtype=np.float64, order='F')
        self.rhs = np.empty((nz, nx*ny), dtype=np.float64, order='F')
        #Lx, Ly = 4.0*pi, 2.0*pi
        kx = fftshift(np.arange(-nx//2, nx//2)) * (2.0*pi/Lx)
        ky = fftshift(np.arange(-ny//2, ny//2)) * (2.0*pi/Ly)
        self.KX, self.KY = np.meshgrid(kx, ky, indexing='ij')


        A = lil_matrix((nz, nz), dtype=np.float64)
        B = lil_matrix((nz, nz), dtype=np.float64)
        idz = np.float64(nz/Lz)

        A[0, 0], A[-1, -1] = 1.00, 1.00
        A[1, 1], A[-2, -2] = 1.00, 1.00
        A[2, 2], A[-3, -3] = 1.00, 1.00
        A[0, 1], A[-1, -2] = 3.00, 3.00
        A[1, 2], A[-2, -1] = 0.25, 0.25
        A[1, 0], A[-2, -3] = 0.25, 0.25
        A[2, 3], A[-3, -2] = 163./508., 163./508.
        A[2, 1], A[-3, -4] = 163./508., 163./508.

        B[ 0, [ 0, 1, 2, 3]] = [-17./6.*idz, 3./2.*idz, 3./2.*idz, -1./6*idz]
        B[-1, [-1,-2,-3,-4]] = [ 17./6.*idz,-3./2.*idz,-3./2.*idz,  1./6*idz]
        B[ 1, [ 0, 2]]       = [-3./4.*idz, 3./4.*idz]
        B[-2, [-3,-1]]       = [-3./4.*idz, 3./4.*idz]
        B[ 2, [ 0, 1, 3, 4]] = [-3./127.*idz, -393./508.*idz, 393./508.*idz, 3./127.*idz]
        B[-3, [-5,-4,-2,-1]] = [-3./127.*idz, -393./508.*idz, 393./508.*idz, 3./127.*idz]

        for i in range(3, nz-3):
            A[i, [i-1, i, i+1]] = [1./3., 1.0, 1./3.]
            B[i, [i-2, i-1, i+1, i+2]] = [-1./36.*idz, -7./9.*idz, 7./9.*idz, 1./36.*idz]

        self.LU_tf = tf.linalg.lu(A.toarray())
        self.LU = splu(A.tocsc())
        self.B = B.tocsc()


    def i_imag_tensor(self, shape):
        return tf.dtypes.complex(real = tf.zeros(shape), imag = tf.ones(shape))

    def create_zeros_tensor(self, shape, xyz):
        tensor = np.ones(shape)
        if xyz == 'x':
            tensor[:, :, self.NX//2, :, :] = 0
        elif xyz == 'y':
            tensor[:, :, :, self.NY//2, :] = 0
        return tf.cast(tensor, tf.complex64)

    def create_zeros_tensor_pointed(self, shape, xyz):
        tensor = np.ones(shape)
        if xyz == 'x':
            tensor[:, self.NX//2, :, :] = 0
        elif xyz == 'y':
            tensor[:, :, self.NY//2, :] = 0
        return tf.cast(tensor, tf.complex64)

    def ddx(self, f):
        '''
        Input: f -- 3D array in Fortran ordering
        Return dfdx
        '''
        nfm, nfq, nfx, nfy, nfz = tuple(f.shape.as_list())

        # FFT
        f_reshaped = tf.reshape(tf.transpose(f, perm = [0,1,3,4,2]), ( nfm * nfq * nfy * nfz, nfx) )
        f_reshaped = tf.signal.fft(tf.cast(f_reshaped, tf.complex64))
        f_reshaped = tf.transpose(tf.reshape(f_reshaped, (nfm, nfq, nfy, nfz, nfx) ), perm = [0,1,4,2,3])

        # Multiply by j, KX, and set NX/2 = 0
        f_reshaped *= self.i_imag_tensor(shape = f_reshaped.shape)
        f_reshaped = tf.reshape(f_reshaped, (nfm * nfq, nfx, nfy, nfz) )
        f_reshaped *= tf.cast(self.KX[:,:,None], tf.complex64)
        f_reshaped = tf.reshape(f_reshaped, (nfm, nfq, nfx, nfy, nfz) )
        f_reshaped *= self.create_zeros_tensor(shape = f_reshaped.shape, xyz = 'x')

        # Inverse FFT
        f_reshaped = tf.reshape(tf.transpose(f_reshaped, perm = [0,1,3,4,2]), ( nfm * nfq * nfy * nfz, nfx) )
        f_reshaped = tf.math.real(tf.signal.ifft(f_reshaped))
        f_reshaped = tf.transpose(tf.reshape(f_reshaped, (nfm, nfq, nfy, nfz, nfx) ), perm = [0,1,4,2,3])

        return f_reshaped

    def ddy(self, f):
        '''
        Input: f -- 3D array in Fortran ordering
        Return dfdy
        '''
        nfm, nfq, nfx, nfy, nfz = tuple(f.shape.as_list())

        # FFT
        f_reshaped = tf.reshape(tf.transpose(f, perm = [0,1,2,4,3]), ( nfm * nfq * nfx * nfz, nfy) )
        f_reshaped = tf.signal.fft(tf.cast(f_reshaped, tf.complex64))
        f_reshaped = tf.transpose(tf.reshape(f_reshaped, (nfm, nfq, nfx, nfz, nfy) ), perm = [0,1,2,4,3])

        # Multiply by j, KX, and set NX/2 = 0
        f_reshaped *= self.i_imag_tensor(shape = f_reshaped.shape)
        f_reshaped = tf.reshape(f_reshaped, (nfm * nfq, nfx, nfy, nfz) )
        f_reshaped *= tf.cast(self.KY[:,:,None], tf.complex64)
        f_reshaped = tf.reshape(f_reshaped, (nfm, nfq, nfx, nfy, nfz) )
        f_reshaped *= self.create_zeros_tensor(shape = f_reshaped.shape, xyz = 'y')

        # Inverse FFT
        f_reshaped = tf.reshape(tf.transpose(f_reshaped, perm = [0,1,2,4,3]), ( nfm * nfq * nfx * nfz, nfy) )
        f_reshaped = tf.math.real(tf.signal.ifft(f_reshaped))
        f_reshaped = tf.transpose(tf.reshape(f_reshaped, (nfm, nfq, nfx, nfz, nfy) ), perm = [0,1,2,4,3])

        return f_reshaped

    def ddz(self, f):
        '''
        Input: f -- 3D array in Fortran ordering
        Return dfdz
        '''
        nfm, nfq, nfx, nfy, nfz = tuple(f.shape.as_list())


        f_reshaped = tf.transpose(tf.reshape(f, ( nfm * nfq, nfx, nfy, nfz ) ), perm = [0,2,1,3]) # Account for the Fortran formatting
        f_reshaped = tf.transpose(tf.reshape(f_reshaped, ( nfm * nfq * nfx * nfy, nfz ) )) # This is self.buf_z

        f_reshaped = tf.tensordot(tf.cast(self.B.toarray(), tf.float32), f_reshaped, axes = 1) # this is self.rhs

        f_reshaped = tf.linalg.lu_solve(lower_upper = tf.cast(self.LU_tf[0], tf.float32), perm = self.LU_tf[1], rhs = f_reshaped)
        f_reshaped = tf.reshape(tf.transpose(f_reshaped), ( nfm * nfq, nfy, nfx, nfz ) )
        f_reshaped = tf.reshape(tf.transpose(f_reshaped, perm = [0,2,1,3]), ( nfm, nfq, nfx, nfy, nfz ) )

        return f_reshaped

        '''
        #### Previous Version
        self.buf_z = np.transpose(f.numpy().reshape((self.NX*self.NY, self.NZ), order='F'))
        self.rhs = self.B.dot(self.buf_z)
        fz = np.transpose(self.LU.solve(self.rhs)).reshape((self.NX, self.NY, self.NZ), order='F')
        return fz
        '''

    def ddx_pointed(self, f):
        '''
        Input: f -- 3D array in Fortran ordering
        Return dfdx
        '''
        nfm, nfx, nfy, nfz = tuple(f.shape.as_list())

        # FFT
        f_reshaped = tf.reshape(tf.transpose(f, perm = [0,2,3,1]), ( nfm * nfy * nfz, nfx) )
        f_reshaped = tf.signal.fft(tf.cast(f_reshaped, tf.complex64))
        f_reshaped = tf.transpose(tf.reshape(f_reshaped, (nfm, nfy, nfz, nfx) ), perm = [0,3,1,2])

        # Multiply by j, KX, and set NX/2 = 0
        f_reshaped *= self.i_imag_tensor(shape = f_reshaped.shape)
        f_reshaped *= tf.cast(self.KX[:,:,None], tf.complex64)
        f_reshaped *= self.create_zeros_tensor_pointed(shape = f_reshaped.shape, xyz = 'x')

        # Inverse FFT
        f_reshaped = tf.reshape(tf.transpose(f_reshaped, perm = [0,2,3,1]), ( nfm * nfy * nfz, nfx) )
        f_reshaped = tf.math.real(tf.signal.ifft(f_reshaped))
        f_reshaped = tf.transpose(tf.reshape(f_reshaped, (nfm, nfy, nfz, nfx) ), perm = [0,3,1,2])

        return f_reshaped

    def ddy_pointed(self, f):
        '''
        Input: f -- 3D array in Fortran ordering
        Return dfdy
        '''
        nfm, nfx, nfy, nfz = tuple(f.shape.as_list())
        
        # FFT
        f_reshaped = tf.reshape(tf.transpose(f, perm = [0,1,3,2]), ( nfm * nfx * nfz, nfy) )
        f_reshaped = tf.signal.fft(tf.cast(f_reshaped, tf.complex64))
        f_reshaped = tf.transpose(tf.reshape(f_reshaped, (nfm, nfx, nfz, nfy) ), perm = [0,1,3,2])

        # Multiply by j, KX, and set NX/2 = 0
        f_reshaped *= self.i_imag_tensor(shape = f_reshaped.shape)
        f_reshaped *= tf.cast(self.KY[:,:,None], tf.complex64)
        f_reshaped *= self.create_zeros_tensor_pointed(shape = f_reshaped.shape, xyz = 'y')

        # Inverse FFT
        f_reshaped = tf.reshape(tf.transpose(f_reshaped, perm = [0,1,3,2]), ( nfm * nfx * nfz, nfy) )
        f_reshaped = tf.math.real(tf.signal.ifft(f_reshaped))
        f_reshaped = tf.transpose(tf.reshape(f_reshaped, (nfm, nfx, nfz, nfy) ), perm = [0,1,3,2])

        return f_reshaped

    def ddz_pointed(self, f):
        '''
        Input: f -- 3D array in Fortran ordering
        Return dfdz
        '''
        nfm, nfx, nfy, nfz = tuple(f.shape.as_list())


        f_reshaped = tf.transpose(f, perm = [0,2,1,3]) # Account for the Fortran formatting
        f_reshaped = tf.transpose(tf.reshape(f_reshaped, ( nfm * nfx * nfy, nfz ) )) # This is self.buf_z

        f_reshaped = tf.tensordot(tf.cast(self.B.toarray(), tf.float32), f_reshaped, axes = 1) # this is self.rhs

        f_reshaped = tf.linalg.lu_solve(lower_upper = tf.cast(self.LU_tf[0], tf.float32), perm = self.LU_tf[1], rhs = f_reshaped)
        f_reshaped = tf.reshape(tf.transpose(f_reshaped), ( nfm, nfy, nfx, nfz ) )
        f_reshaped = tf.transpose(f_reshaped, perm = [0,2,1,3])

        return f_reshaped


    def tic(self):
        self.start_time = time.time()

    def toc(self):
        print('Elapsed: %s' % (time.time() - self.start_time))


def test(NX = 256, NY = 128, NZ = 64):
    #NX, NY, NZ = 256, 128, 64
    dz = 1.0 / NZ
    x = np.linspace(0, 4*pi, num=NX, endpoint=False)
    y = np.linspace(0, 2*pi, num=NY, endpoint=False)
    z = np.linspace(-0.5*(1.0-dz), 0.5*(1.0-dz), num=NZ)
    X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
    f   = np.empty((NX, NY, NZ), dtype=np.float64, order='F')
    fxe = np.empty((NX, NY, NZ), dtype=np.float64, order='F')
    fye = np.empty((NX, NY, NZ), dtype=np.float64, order='F')
    fze = np.empty((NX, NY, NZ), dtype=np.float64, order='F')
    fxn = np.empty((NX, NY, NZ), dtype=np.float64, order='F')
    fyn = np.empty((NX, NY, NZ), dtype=np.float64, order='F')
    fzn = np.empty((NX, NY, NZ), dtype=np.float64, order='F')


    f  [:] = np.cos(3.0 * X) * np.cos(2.0 * Y) * np.cos(4*np.pi*Z)
    f      = tf.cast(f[None, None, :, :, :], tf.float32)
    fxe[:] =-np.sin(3.0 * X) * np.cos(2.0 * Y) * np.cos(4*np.pi*Z) * 3.0
    fxe    = tf.cast(fxe[None, None, :, :, :], tf.float32)
    fye[:] =-np.cos(3.0 * X) * np.sin(2.0 * Y) * np.cos(4*np.pi*Z) * 2.0
    fye    = tf.cast(fye[None, None, :, :, :], tf.float32)
    fze[:] =-np.cos(3.0 * X) * np.cos(2.0 * Y) * np.sin(4*np.pi*Z) * 4.0 * np.pi
    fze    = tf.cast(fze[None, None, :, :, :], tf.float32)

    diff = DiffOps(NX, NY, NZ)

    # Test derivative functions for intaking/taking the derivative of all quantities (p,u,v,w)
    fxn = diff.ddx(f)
    print("Linf error dfdx: {:.5E}".format(np.max(np.abs(fxe-fxn))))
    fyn = diff.ddy(f)
    print("Linf error dfdy: {:.5E}".format(np.max(np.abs(fye-fyn))))
    fzn = diff.ddz(f)
    print("Linf error dfdz: {:.5E}".format(np.max(np.abs(fze-fzn))))
    print("L2 error dfdz: {:.5E}".format(np.sqrt(np.sum((fze-fzn)**2)/NZ)))

    # Test the pointed functions (take derivatives of specified quantities)
    # Note: the input shape is different than above. Here, it is (N_samples, NX, NY, NZ)
    fxn_pointed = diff.ddx_pointed(f[:,0,:,:,:])
    print("Linf error dfdx_pointed: {:.5E}".format(np.max(np.abs(fxe[:,0,:,:,:]-fxn_pointed))))
    fyn_pointed = diff.ddy_pointed(f[:,0,:,:,:])
    print("Linf error dfdy_pointed: {:.5E}".format(np.max(np.abs(fye[:,0,:,:,:]-fyn_pointed))))
    fzn_pointed = diff.ddz_pointed(f[:,0,:,:,:])
    print("Linf error dfdz_pointed: {:.5E}".format(np.max(np.abs(fze[:,0,:,:,:]-fzn_pointed))))
    print("L2 error dfdz: {:.5E}".format(np.sqrt(np.sum((fze[:,0,:,:,:]-fzn_pointed)**2)/NZ)))



'''
if __name__ == '__main__':
    if len(sys.argv) < 4:
        print("Can specify grid resolution via:")
        print("   python3 diff.py <NX> <NY> <NZ>")
        print("---------------------------------")
        print("Using default domain size instead: 256 X 128 X 64")
        test()
    else:
        NX, NY, NZ = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3])
        print("Domain size: {} X {} X {}".format(NX, NY, NZ))
        test(NX = NX, NY = NY, NZ = NZ)
'''
