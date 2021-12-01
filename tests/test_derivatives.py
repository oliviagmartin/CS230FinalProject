import sys
sys.path.insert(0,'/Users/ryanhass/Documents/MATLAB/CS_230/Final_project/utilities')
#from derivative_operators import ddx, ddy
from diff_tf import DiffOps
import numpy as np
from domain_setup import setup_domain
from convDiff import ddx, ddy, ddz
import tensorflow as tf

def test_spectral(A,B,C,nx,ny,nz):
    # Define spatial domain
    coefs = [A, B, C]
    Lx, Ly, Lz = tuple([2.*np.pi*i for i in coefs])
    dx, dy, dz, xmesh, ymesh, zmesh = setup_domain(Lx,Ly,Lz,nx,ny,nz)

    xcos = np.cos(xmesh)
    xsin = np.sin(xmesh)
    ycos = np.cos(ymesh)
    ysin = np.sin(ymesh)
    zcos = np.cos(zmesh)
    zsin = np.sin(zmesh)

    xycos = np.multiply(xcos,ycos)
    xzcos = np.multiply(xcos,zcos)
    yzcos = np.multiply(ycos,zcos)

    f = np.multiply(xcos,yzcos)
    f = tf.convert_to_tensor(f, dtype=tf.float32)
    #f = tf.expand_dims(tf.expand_dims(f, axis=0), axis = 0)
    f = tf.expand_dims(f, axis=0)
    #print(f.get_shape().as_list())

    dops = DiffOps(nx = nx, ny = ny, nz = nz, Lx = Lx, Ly = Ly, Lz = Lz)

    #dfdx = dops.ddx(f)
    dfdx = tf.squeeze(ddx(f, 0, dx))
    dfdx_true = np.multiply(-1.*xsin,yzcos)

    #fdy = dops.ddy(f)
    dfdy = tf.squeeze(ddy(f, 0, dy))
    dfdy_true = np.multiply(-1.*ysin,xzcos)

    comparex = dfdx - dfdx_true
    comparey = dfdy - dfdy_true
    assert np.amax(comparex) < 1.e-12, "ddx test 1 FAILED. max difference = {}".\
            format(np.amax(comparex))
    print("ddx test 1 PASSED!")
    assert np.amax(comparey) < 1.e-12, "ddy test 1 FAILED. max difference = {}".\
            format(np.amax(comparey))
    print("ddy test 1 PASSED!")

    xzsin = np.multiply(xsin,zsin)
    xsinycos = np.multiply(xsin,ycos)
    ycoszsin = np.multiply(ycos,zsin)

    g = np.multiply(xsin,ycoszsin)

    dgdx = dops.ddx(g)
    dgdx_true = np.multiply(xcos,ycoszsin)

    dgdy = dops.ddy(g)
    dgdy_true = np.multiply(-1.*ysin,xzsin)


    comparex = dgdx - dgdx_true
    comparey = dgdy - dgdy_true
    assert np.amax(comparex) < 1.e-12, "ddx test 2 FAILED. max difference = {}".\
            format(np.amax(comparex))
    print("ddx test 2 PASSED!")
    assert np.amax(comparey) < 1.e-12, "ddy test 2 FAILED. max difference = {}".\
            format(np.amax(comparey))
    print("ddy test 2 PASSED!")
    return

#if len(sys.argv) < 7:
#    print("Usage:")
#    print("  python3 test_derivatives.py <Lx_on2pi> <Ly_on2pi> <Lz_on2pi> <nx> <ny> <nz>")
#    sys.exit()

Lx_on2pi, Ly_on2pi, Lz_on2pi, nx, ny, nz = (sys.argv[1], sys.argv[2], sys.argv[3],\
        sys.argv[4], sys.argv[5], sys.argv[6])
Lx_on2pi, Ly_on2pi, Lz_on2pi = np.float64((Lx_on2pi, Ly_on2pi, Lz_on2pi))
nx, ny, nz = (int(nx),int(ny),int(nz))
test_spectral(Lx_on2pi,Ly_on2pi,Lz_on2pi,nx,ny,nz)
