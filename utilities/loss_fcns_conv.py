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

# Array shapes:
# X, Yhat: [m, nvars, nx, ny, nzC]
# Y:       [m, nprofs, nzC]

class Loss:
    def __init__(self, n_examples, inc_mom = False):
        # Inputs:
        #   n_examples --> Numer of training examples in current mini-batch
        #   inc_mom    --> logical whether or not to include "momentum" terms
        #                  in model (e.g. tauij, L_mom, pressure)
        self.inc_mom = inc_mom
        if self.inc_mom:
            self.nprofs = 14
            self.nvars = 4
        else:
            self.nprofs = 7
            self.nvars = 3


        self.m = n_examples
        self.nx, self.ny, self.nz = nx, ny, nzC

        # Initialize derivative operator
        self.dop = DiffOps(nx, ny, nzC, Lx, Ly, Lz)

        # Allocate memory for the flow variables
        self.fields = {}
        self.fields["u1"] = tf.zeros((n_examples,nx,ny,nzC), dtype = tf.float32)
        self.fields["u2"] = tf.zeros((n_examples,nx,ny,nzC), dtype = tf.float32)
        self.fields["u3"] = tf.zeros((n_examples,nx,ny,nzC), dtype = tf.float32)
        if self.inc_mom:
            self.fields["p"] = tf.zeros((n_examples,nx,ny,nzC), dtype = tf.float32)

        # Allocate memory for the current state
        current_state = tf.zeros((n_examples,self.nprofs,nzC), dtype = tf.float32)

    def mean_square(self,f):
        return tf.math.reduce_mean(tf.square(f), name = 'mean_square')

    def confirm_dimensions(self,nx1,nx2,nx3):
        assert self.nx == nx1
        assert self.ny == nx2
        assert self.nz == nx3
        return None

    def xy_avg(self,f):
        _, nx1, nx2, nx3 = tuple(f.shape.as_list())
        self.confirm_dimensions(nx1, nx2, nx3)
        return tf.math.reduce_mean(f, axis = (1,2), keepdims = False, name = 'xy_avg')

    def L_mass(self):
        # Compute mass conservation loss
        dx = Lx/nx;
        dy = Ly/ny;
        dz = Lz/nzC;

        return self.mean_square(ddx(self.fields['u1'], 0, dx) + \
                ddy(self.fields['u2'], 0, dy) + \
                self.dop.ddz_pointed(self.fields['u3']))

    def L_mom(self):
        # Compute the residual of the pressure Poisson equations

        # tauij[:,:,:,0] --> tau_11
        # tauij[:,:,:,1] --> tau_12
        # tauij[:,:,:,2] --> tau_13
        # tauij[:,:,:,3] --> tau_22
        # tauij[:,:,:,4] --> tau_23
        # tauij[:,:,:,5] --> tau_33

#        Inertial_term = self.dop.ddx(self.dop.ddx( self.u*self.u ) ) +\
#                 2.*self.dop.ddx(self.dop.ddy( self.u*self.v ) ) +\
#                 2.*self.dop.ddx(self.dop.ddz( self.u*self.w ) ) +\
#                    self.dop.ddy(self.dop.ddy( self.v*self.v ) ) +\
#                 2.*self.dop.ddy(self.dop.ddz( self.v*self.w ) ) +\
#                    self.dop.ddz(self.dop.ddz( self.w*self.w ) )
#        Pressure_term = self.dop.ddx(self.dop.ddx(self.p)) + \
#           self.dop.ddy(self.dop.ddy(self.p)) + self.dop.ddz(self.dop.ddz(self.p))
#        Stress_term   = self.dop.ddx(self.dop.ddx(self.tauij[:,:,:,0])) + \
#                     2.*self.dop.ddx(self.dop.ddy(self.tauij[:,:,:,1])) + \
#                     2.*self.dop.ddx(self.dop.ddz(self.tauij[:,:,:,2])) + \
#                        self.dop.ddy(self.dop.ddy(self.tauij[:,:,:,3])) + \
#                     2.*self.dop.ddy(self.dop.ddz(self.tauij[:,:,:,4])) + \
#                        self.dop.ddz(self.dop.ddz(self.tauij[:,:,:,5]))

#        return self.mean_square(Inertial_term + Pressure_term + Stress_term)
        return None

    def set_fields(self,Yhat):
        self.fields['u1'] = Yhat[:,0,:,:,:]
        self.fields['u2'] = Yhat[:,1,:,:,:]
        self.fields['u3'] = Yhat[:,2,:,:,:]
        if self.inc_mom:
            self.fields['p'] = Yhat[:,3,:,:,:]

        return None

    def compute_averages(self):
        # Stack in the following order: mean(U), <u1u1> ,<u1u2>, <u1u3>, <u2u2>, <u2u3>, <u3u3>
        self.current_state = tf.transpose( tf.stack([\
                self.xy_avg(self.fields['u1']),\
                self.xy_avg(tf.math.multiply(self.fields['u1'],self.fields['u1'])) - \
                        tf.math.multiply(self.xy_avg(self.fields['u1']),self.xy_avg(self.fields['u1'])), \
                self.xy_avg(tf.math.multiply(self.fields['u1'],self.fields['u2'])) - \
                        tf.math.multiply(self.xy_avg(self.fields['u1']),self.xy_avg(self.fields['u2'])), \
                self.xy_avg(tf.math.multiply(self.fields['u1'],self.fields['u3'])) - \
                        tf.math.multiply(self.xy_avg(self.fields['u1']),self.xy_avg(self.fields['u3'])), \
                self.xy_avg(tf.math.multiply(self.fields['u2'],self.fields['u2'])) - \
                        tf.math.multiply(self.xy_avg(self.fields['u2']),self.xy_avg(self.fields['u2'])), \
                self.xy_avg(tf.math.multiply(self.fields['u2'],self.fields['u3'])) - \
                        tf.math.multiply(self.xy_avg(self.fields['u2']),self.xy_avg(self.fields['u3'])), \
                self.xy_avg(tf.math.multiply(self.fields['u3'],self.fields['u3'])) - \
                        tf.math.multiply(self.xy_avg(self.fields['u3']),self.xy_avg(self.fields['u3']))], \
                ), perm = [1,0,2] )

        if self.inc_mom:
            # TODO: Needs to be implemented
            None
        return None

    def MSE(self, Y, axis):
        return tf.math.reduce_mean(tf.square(Y-self.current_state), axis=axis)

    def Lcontent(self, Y):
        mse = self.MSE(Y, axis = (0,2))
        Lcont = tf.math.reduce_sum(mse) #self.L_uiuj() + self.L_U()
        return Lcont

    def compute_loss(self, Yhat, Y, lambda_p = 0.5, lambda_tau = 0.5):
        # Inputs:
        #   Yhat       --> NN output layer
        #                  Type: TensorFlow tensor
        #                  Dimension: [m,nvars,nx,ny,nzC]
        #   Y          --> "labels" (Type: TensorFlow tensor)
        #                  Type: TensorFlow tensor
        #                  Dimension: [m,nprfs,nzC]
        #   lambda_p   --> hyperparameter of model that determines the ...
        #                  ... relative importance of the physics loss term.
        #                  Type: np.float32
        #   lambda_tau --> hyperparameter of model that determines the ...
        #                  ... relative importance of the residual stress ...
        #                  ... in the content loss.
        #                  Type: np.float32

        nx,ny,nz = self.nx, self.ny, self.nz
        m = Yhat.shape[0]

        # Verify dimensions of input arrays

        assert self.nvars == Yhat.shape[1]
        assert nx == Yhat.shape[2]
        assert ny == Yhat.shape[3]
        assert nzC == Yhat.shape[4]

        assert m == Y.shape[0]
        assert self.nprofs == Y.shape[1]
        assert nzC == Y.shape[2]

        #self.extract_avg_profiles_from_labels_array(Y)
        self.set_fields(Yhat)
        self.compute_averages()

        # Compute loss functions
        Lphys = self.L_mass()
        Lcont = self.Lcontent(Y)

        if self.inc_mom:
            # TODO: compute momentume relavant loss terms
            #Lphys += self.L_mom()
            #Lcontent = (1. - lambda_tau)*Lcontent
            #Lcontent += lambda_tau*self.L_tauij()
            None
        total_loss = lambda_p*Lphys + (1. - lambda_p)*Lcont
        return total_loss

    def predict_average_profiles(self, Yhat, lambda_p = 0.5, lambda_tau = 0.5):
        # Inputs:
        #   Yhat       --> NN output layer
        #                  Type: TensorFlow tensor
        #                  Dimension: [m,nvars,nx,ny,nz]
        #   lambda_p   --> hyperparameter of model that determines the ...
        #                  ... relative importance of the physics loss term.
        #                  Type: np.float32
        #   lambda_tau --> hyperparameter of model that determines the ...
        #                  ... relative importance of the residual stress ...
        #                  ... in the content loss.
        #                  Type: np.float32

        nx, ny, nz = self.nx, self.ny, self.nz
        m = Yhat.shape[0]

        # Verify dimensions of input arrays

        assert self.nvars == Yhat.shape[1]
        assert nx == Yhat.shape[2]
        assert ny == Yhat.shape[3]
        assert nz == Yhat.shape[4]

        # Compute average profiles.
        self.set_fields(Yhat)
        self.compute_averages()

        return self.current_state

    def tic(self):
        self.start_time = time.time()

    def toc(self):
        print('Elapsed: %s' % (time.time() - self.start_time))



def load_data_for_loss_tests(datadir,tidx,tidy,navg,inc_prss = False):
    tidx_vec = np.array([tidx])
    tidy_vec = np.array([tidy])

    Yhat, Y, _, _ = load_dataset_V2(datadir, nx, ny, nzC, nzF, tidx_vec, \
            tidx_vec, tidy_vec, tidy_vec, inc_prss = inc_prss, navg = navg)

    Yhat = Yhat.reshape((1,3,nx,ny,nzC), order = 'F')
    Y = tf.cast(Y, tf.float32)
    Yhat = tf.cast(Yhat, tf.float32)
    return Y, Yhat




'''
if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage:")
        print("  python3 loss_fcns.py <datadir>")
        sys.exit()

#### For milestone ######
    datadir = sys.argv[1] + '/'

    # fname parameters
    navg = 840
    tidx = 179300
    tidy = 25400

    # Load channel data for loss tests
    Y, Yhat = load_data_for_loss_tests(datadir,tidx,tidy,navg,\
            inc_prss = False)

    # Create placeholders for tf operations
    # Load data into Loss class
    L_test = Loss(1,inc_mom = False)
    L_test.set_fields(Yhat)
    L_test.compute_averages()
    Lphys = L_test.L_mass()
    Lcont = L_test.Lcontent(Y)
    #code.interact(local=locals())

    # TODO: Plot some ground-truth profiles to confirm proper initialization of the class

###### For final project ######
    # Test L_P

    # Test L_tauij

    # Test L_mom
    #Lmom = L_mom(u,v,w,p,tauij,A,B,dop,nx,ny,nz)
    #assert Lmom < 1.e-4, 'Lmom = {}'.format(Lmom)
    #print("L_mom test PASSED!")

'''
