import code
from utilities.domain_setup import setup_domain_1D
import numpy as np
from numpy import pi
from utilities.read_fortran_data import read_fortran_data, get_domain_size
import sys
from time import time
from utilities.io_mod import load_dataset_V2
from utilities.convDiff import ddx, ddy, ddz
from utilities.diff_tf import DiffOps
import tensorflow as tf
from utilities.problem_parameters import nx, ny, nzC, nzF, Lx, Ly, Lz
import tensorflow as tf
import matplotlib.pyplot as plt
from matplotlib import cm

dx = 6.*3.14159265/192;
dy = 3.*3.14159265/192;
dz = 1/64;

#loss_CONV = self.mean_square(ddx(self.fields['u1'], 0, dx) + \
#        ddy(self.fields['u2'], 0, dy) + \
#        ddz(self.fields['u3'], 0, dz))

#loss_spect = self.mean_square(self.dop.ddx_pointed(self.fields['u1']) + \
#        self.dop.ddy_pointed(self.fields['u2']) + self.dop.ddz_pointed(self.fields['u3']))



def testConv(NX = 256, NY = 128, NZ = 64):

    dx = 4.*3.14159265/256;
    dy = 2.*3.14159265/128;
    dz = 1/64;

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

    #diff = DiffOps(NX, NY, NZ)

    # Test derivative functions for intaking/taking the derivative of all quantities (p,u,v,w)
    #fxn = diff.ddx(f)
    #print("Linf error dfdx: {:.5E}".format(np.max(np.abs(fxe-fxn))))
    #fyn = diff.ddy(f)
    #print("Linf error dfdy: {:.5E}".format(np.max(np.abs(fye-fyn))))
    #fzn = diff.ddz(f)
    #print("Linf error dfdz: {:.5E}".format(np.max(np.abs(fze-fzn))))
    #print("L2 error dfdz: {:.5E}".format(np.sqrt(np.sum((fze-fzn)**2)/NZ)))

    # Test the pointed functions (take derivatives of specified quantities)
    # Note: the input shape is different than above. Here, it is (N_samples, NX, NY, NZ)
    fxn_pointed = ddx(f[:,0,:,:,:], 0, dx)
    print("Linf error dfdx_pointed: {:.5E}".format(np.max(np.abs(fxe[:,0,:,:,:]-fxn_pointed))))
    fyn_pointed = ddy(f[:,0,:,:,:], 0, dy)
    print("Linf error dfdy_pointed: {:.5E}".format(np.max(np.abs(fye[:,0,:,:,:]-fyn_pointed))))
    fzn_pointed = ddz(f[:,0,:,:,:], 0, dz)
    print("Linf error dfdz_pointed: {:.5E}".format(np.max(np.abs(fze[:,0,:,:,:]-fzn_pointed))))
    print("L2 error dfdz: {:.5E}".format(np.sqrt(np.sum((fze[:,0,:,:,:]-fzn_pointed)**2)/NZ)))


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


testConv()
test()
