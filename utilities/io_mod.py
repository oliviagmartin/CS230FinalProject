from utilities.domain_setup import setup_domain_1D
import numpy as np
from numpy import pi
from utilities.read_fortran_data import read_fortran_data
from scipy import interpolate
import code
import utilities.problem_parameters as params
import os


def get_x_vec(datadir,X,buff,tid_vec,dsname):
    # Load in input LR data stored in AWS S3 storage
    for n, tid in enumerate(tid_vec):
        tidstr = str(tid).zfill(7)
        if n > 0:
            X[n-1,:,:,:,:] = buff

        fname1 = 'Run01_' + tidstr + '.h5'
        command = "sudo aws s3 cp s3://lesdata/data/" + fname1 + " " + datadir
        os.system(command)

        for i, ds in enumerate(dsname):
            buff[i,:,:,:] = read_fortran_data(datadir + 'Run01_' + \
                    tidstr + '.h5', ds)

        os.system('sudo rm -f ' + datadir + fname1)
    X[-1,:,:,:,:] = buff
    return X

def get_y_vec(datadir, Y, buff, zF, zC, tid_vec, prof_ids, navg = 1):
    # Output:
    #   Y --> labels vector of dimension [n_examples, nprofs, nzC]

    # Loads HR data from AWS S3 storage
    nzC = np.size(zC)
    nzF = np.size(zF)
    for n, tid in enumerate(tid_vec):


        fname1 = 'Run01_budget0_t' + str(tid).zfill(6) + '_n' + \
                str(navg).zfill(6) + '_nzF' + str(nzF) + '.stt'

        command = "sudo aws s3 cp s3://lesdata/data/" + fname1 + " " + datadir
        os.system(command)

        fname2 = datadir + 'Run01_budget0_t' + str(tid).zfill(6) + '_n' + \
                str(navg).zfill(6) + '_nzF' + str(nzF) + '.stt'

        # Read in average profiles from disk. These correspond to the "Fine" grid
        avgF = np.genfromtxt(fname2,dtype = np.float32).T


        os.system('sudo rm -f ' + datadir + fname1)
        # Interpolate the average profiles to the zC locations (i.e. "Course" grid)
        avgC = np.zeros((avgF.shape[0],nzC))
        for i in range(avgF.shape[0]):
            tck = interpolate.splrep(zF.T, avgF[i,:], s=0)
            avgC[i,:] = interpolate.splev(np.squeeze(zC.T), tck, der=0)

        # Copy the profiles we need to the buffer array
        for i, prof in enumerate(prof_ids):
            buff[i,:] = avgC[prof,:]

        # Copy the flattened buffer to the Y-vector
        Y[n,:,:] = buff
    return Y

def load_dataset_V2(data_directory, nx, ny, nz, nzF, x_tid_vec_train, x_tid_vec_test,\
        y_tid_vec_train, y_tid_vec_test, inc_prss = True, navg = 1):
    # Inputs:
    #   data_directory  --> directory path where raw data resides
    #   nx, ny, nz      --> number of grid points in computational domain
    #   nzC             --> number of grid points in z for the "Fine grid"
    #   x_tid_vec_test  --> vector of time ID's from the simulation for the training features
    #   x_tid_vec_train --> "                                             " test features
    #   y_tid_vec_test  --> "                                             " training labels
    #   y_tid_vec_train --> "                                             " test labels
    #   inc_prss        --> are we including pressure in the input layer?
    #   navg            --> how many snapshots the profiles are averaged over

    tsteps_train = x_tid_vec_train.size
    tsteps_test  = x_tid_vec_test.size
    assert tsteps_train == y_tid_vec_train.size
    assert tsteps_test  == y_tid_vec_test.size

    ncube = nx*ny*nz
    dsname = ['uVel','vVel','wVel'] # Dataset names in the hdf5 files
    if inc_prss:
        nfields = 4
        nprofs  = 14 # meanU, <uu>, <uv>, <uw>, <vv>, <vw>, <ww>, <tau11>, <tau12>
                     # <tau13>, <tau22>, <tau23>, <tau33>, meanP
        prof_id = (0,3,4,5,6,7,8,17,18,13,19,14,20,16)
        dsname.append('prss')
    else:
        nfields = 3
        nprofs  = 7 # meanU, <uu>, <uv>, <uw>, <vv>, <vw>, <ww>
        prof_id = (0,3,4,5,6,7,8)
    assert len(prof_id) == nprofs

    # Define the low and high resolution computational domains (we actually only need the z-vectors)
    nzC = nz
    Lz = params.Lz
    zC = setup_domain_1D(0.5*Lz/nzC, Lz - 0.5*Lz/nzC, Lz/nzC)
    zF = setup_domain_1D(0.5*Lz/nzF, Lz - 0.5*Lz/nzF, Lz/nzF)

    buff_x = np.empty((nfields,nx,ny,nz), dtype = np.float32)
    buff_y = np.empty((nprofs,nz),        dtype = np.float32)

    # Initialize training and test features and labels
    train_set_x = np.empty((tsteps_train, nfields, nx, ny, nz), dtype = np.float32)
    test_set_x  = np.empty((tsteps_test,  nfields, nx, ny, nz), dtype = np.float32)
    train_set_y = np.empty((tsteps_train, nprofs, nz), dtype = np.float32)
    test_set_y  = np.empty((tsteps_test,  nprofs, nz), dtype = np.float32)

    train_set_x = get_x_vec(data_directory, train_set_x, buff_x, x_tid_vec_train, dsname)
    test_set_x  = get_x_vec(data_directory, test_set_x,  buff_x, x_tid_vec_test,  dsname)

    train_set_y = get_y_vec(data_directory, train_set_y, buff_y, zF, zC, y_tid_vec_train, prof_id, navg = navg)
    test_set_y  = get_y_vec(data_directory, test_set_y,  buff_y, zF, zC, y_tid_vec_test,  prof_id, navg = navg)

    #train_set_y = train_set_y.reshape((1, train_set_y.shape[0]))
    #test_set_y = test_set_y.reshape((1, test_set_y.shape[0]))

    return train_set_x, train_set_y, test_set_x, test_set_y

def normalize_data(train_set_x, train_set_y, test_set_x, test_set_y, inc_prss = True):
    # Normalize data by mean flow velocity
    
    ncube = params.nx*params.ny*params.nzC
    vel_scale = 1./params.Uinf
    vel_scale_sq = vel_scale**2.

    # Confirm input array dimensions
    if inc_prss:
        assert train_set_x.shape[1] == 4
        assert test_set_x.shape[1] == 4

        assert train_set_y.shape[1] == 14
        assert test_set_y.shape[1] == 14
    else:
        assert train_set_x.shape[1] == 3
        assert test_set_x.shape[1] == 3

        assert train_set_y.shape[1] == 7
        assert test_set_y.shape[1] == 7

    assert train_set_y.shape[2] == params.nzC
    assert test_set_y.shape[2] == params.nzC

    train_set_x[:,:3] = train_set_x[:,:3] * vel_scale
    train_set_y[:,0,:] = train_set_y[:,0,:] * vel_scale
    train_set_y[:,1:7,:] = train_set_y[:,1:7,:] * vel_scale_sq

    test_set_x[:,:3] = test_set_x[:,:3] * vel_scale
    test_set_y[:,0,:] = test_set_y[:,0,:] * vel_scale
    test_set_y[:,1:7,:] = test_set_y[:,1:7,:] * vel_scale_sq

    if inc_prss:
        train_set_x[:,3:] = train_set_x[:,3:] * vel_scale_sq
        train_set_y[:,7:,:] = train_set_y[:,7:,:] * vel_scale_sq

    return train_set_x, train_set_y, test_set_x, test_set_y


################## TESTS ###############################
def test_load_dataset_V2(data_directory, nx, ny, nz, nzF, x_tid_vec_train, \
            x_tid_vec_test, y_tid_vec_train, y_tid_vec_test, \
            inc_prss = True, navg = 1):
    X_train, Y_train, X_test, Y_test = \
            load_dataset_V2(data_directory, nx, ny, nz, nzF, x_tid_vec_train, \
            x_tid_vec_test, y_tid_vec_train, y_tid_vec_test, \
            inc_prss = inc_prss, navg = navg)
    print("Shape of X_train: {}".format(X_train.shape))
    print("Shape of X_test: {}".format(X_test.shape))
    print("Shape of Y_train: {}".format(Y_train.shape))
    print("Shape of Y_test: {}".format(Y_test.shape))
    print(X_train[:10])
    print(X_test[:10])
    print(Y_train[0,0,:10])
    print(Y_test[0,0,:10])

    print('\n --------------------------------------------------------- \n')
    X_train, Y_train, X_test, Y_test = normalize_data(X_train, Y_train, X_test, \
            Y_test, inc_prss = inc_prss)
    print(X_train[:10])
    print(X_test[:10])
    print(X_test[0,-10:])
    print(Y_train[0,0,:10])
    print(Y_test[0,0,:10])
    return None


'''
if __name__ == '__main__':
    data_directory = '/Users/ryanhass/Documents/MATLAB/CS_230/data/'
    nx = 192
    ny = 192
    nz = 64
    nzF = 256
    Lx = 6.*pi
    Ly = 3.*pi
    Lz = 1.

    x_tid_vec_test = np.array([179300])
    x_tid_vec_train = np.array([179300])
    y_tid_vec_test = np.array([25400])
    y_tid_vec_train = np.array([25400])
    test_load_dataset_V2(data_directory, nx, ny, nz, nzF, x_tid_vec_train, \
            x_tid_vec_test, y_tid_vec_train, y_tid_vec_test, \
            inc_prss = True, navg = 840)
'''
