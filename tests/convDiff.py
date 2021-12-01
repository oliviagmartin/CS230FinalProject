import sys
import numpy as np
import code
import tensorflow.compat.v1 as tf
from tensorflow.python.framework import ops
import h5py
import os
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
#import tensorflow.contrib.slim as slim

#Comments:
#inpt is a 5-dimensional vector with dim 0 = training examples, dim 1 = y-axis,
#dim2 = x-axis, dim3 = z-axis, dim4 = number of channels (one)

def periodic_padding(inpt, pad):
    L = inpt[:,:pad[0][0],:,:,:]
    if pad[0][1] > 0:
        R = inpt[:,-pad[0][1]:,:,:,:]
    else:
        R = inpt[:,:0,:,:,:]
    inpt_pad = tf.concat([R, inpt, L], axis=1)

    L = inpt_pad[:,:,:pad[1][0],:,:]
    if pad[1][1] > 0:
        R = inpt_pad[:,:,-pad[1][1]:,:,:]
    else:
        R = inpt_pad[:,:,:0,:,:]
    inpt_pad = tf.concat([R, inpt_pad, L], axis=2)

    L = inpt_pad[:,:,:,:pad[2][0],:]
    if pad[2][1] > 0:
        R = inpt_pad[:,:,:,-pad[2][1]:,:]
    else:
        R = inpt_pad[:,:,:,:0,:]
    inpt_pad = tf.concat([R, inpt_pad, L], axis=3)

    return inpt_pad

def ddx(inpt, channel, dx, scope='ddx', name=None):
    #inpt_shape = inpt.get_shape().as_list()
    #print(inpt_shape)
    #inpt = tf.transpose(inpt, perm = [0,2,1,3])

    #inpt_shape = inpt.get_shape().as_list()
    #print(inpt_shape)
    var = tf.expand_dims( inpt, axis=4)

    with tf.compat.v1.variable_scope(scope):
        ddx1D = tf.constant([-1./60., 3./20., -3./4., 0., 3./4., -3./20., 1./60.], dtype=tf.float32)
        ddx3D = tf.reshape(ddx1D, shape=(-1,1,1,1,1))

    strides = [1,1,1,1,1]
    var_pad = periodic_padding(var, ((3,3),(0,0),(0,0)) )

    #plt.matshow(var_pad[0,:,:,0,0])
    #plt.colorbar()
    #plt.show()

    output = tf.nn.conv3d(var_pad, ddx3D, strides, padding = 'VALID',
                          data_format = 'NDHWC', name=name)
    output = tf.scalar_mul(1./dx, output)

    #output = tf.transpose(output, perm = [0,2,1,3,4])

    return tf.squeeze(output, axis = 4)

def ddy(inpt, channel, dy, scope='ddy', name=None):

    #inpt = tf.transpose(inpt, perm = [0,2,1,3])

    #inpt_shape = inpt.get_shape().as_list()
    var = tf.expand_dims( inpt, axis=4)

    with tf.compat.v1.variable_scope(scope):
        ddy1D = tf.constant([-1./60., 3./20., -3./4., 0., 3./4., -3./20., 1./60.], dtype=tf.float32)
        ddy3D = tf.reshape(ddy1D, shape=(1,-1,1,1,1))

    strides = [1,1,1,1,1]
    var_pad = periodic_padding( var, ((0,0),(3,3),(0,0)) )
    output = tf.nn.conv3d(var_pad, ddy3D, strides, padding = 'VALID',
                          data_format = 'NDHWC', name=name)
    output = tf.scalar_mul(1./dy, output)

    #output = tf.transpose(output, perm = [0,2,1,3,4])

    return tf.squeeze(output, axis = 4)


def ddz(inpt, channel, dz, scope='ddz', name=None):

    #inpt = tf.transpose(inpt, perm = [0,1,3,2])

    inpt_shape = inpt.get_shape().as_list()
    var = tf.expand_dims( inpt, axis=4)

    with tf.compat.v1.variable_scope(scope):
        ddz1D = tf.constant([-1./60., 3./20., -3./4., 0., 3./4., -3./20., 1./60.], dtype=tf.float32)
        ddz3D = tf.reshape(ddz1D, shape=(1,1,-1,1,1))

    strides = [1,1,1,1,1]
    var_pad = periodic_padding( var, ((0,0),(0,0),(3,3)) )

    output = tf.nn.conv3d(var_pad, ddz3D, strides, padding = 'VALID',
                          data_format = 'NDHWC', name=name)
    output = tf.scalar_mul(1./dz, output)

    #output = tf.transpose(output, perm = [0,1,2,3, 4])

    return tf.squeeze(output, axis = 4)

def conv3d_withPeriodicPadding(inpt, filtr, strides, name=None):
    ### Does not work for large strides ###
    inpt_shape = inpt.get_shape().as_list()
    filtr_shape = filtr.get_shape().as_list()
    pad = []

    for i_dim in range(3):
        # Compute pad assuming output_size = input_size / stride and odd filter sizes
        padL = int( 0.5*(filtr_shape[i_dim] - 1) )
        padR = padL
        pad_idim = (padL,padR)
        pad.append(pad_idim)

    inpt_pad = periodic_padding(inpt, pad)
    output = tf.nn.conv3d(inpt_pad, filtr, strides, use_bias=False, scope='conv', name=None)

    return output


def conv3d(inpt, f, output_channels, s, use_bias=False, scope='conv', name=None):
    inpt_shape = inpt.get_shape().as_list()
    with tf.variable_scope(scope):
        filtr = tf.get_variable(initializer=tf.contrib.layers.xavier_initializer(),
                                shape=[f,f,f,inpt_shape[-1]],name='filtr')

    strides = [1,1,1,1,1]
    output = conv3d_withPeriodicPadding(inpt,filtr,strides,name)

    if use_bias:
        with tf.variable_scope(scope):
            bias = tf.get_variable(intializer=tf.zeros_initializer(
                [1,1,1,1,output_channels],dtype=tf.float32),name='bias')
            output = output + bias;

    return output


#inpt = tf.random.uniform(
#    shape = [1,100,100,100,1], minval=0, maxval=3, dtype=tf.dtypes.float32, seed=None, name=None
#)

#Test
#dx = 25/100
#xval1 = tf.transpose(tf.math.sin(tf.linspace(0, 25, 100, name=None, axis=0)))
#xmat1 = tf.transpose(tf.broadcast_to(tf.transpose(tf.broadcast_to(xval1, [100,100], name = None)), [100,100,100]))
#inpt1 = tf.expand_dims(xmat1, axis=0 )

#dx = 50/100
#xval2 = tf.transpose(tf.math.sin(tf.linspace(0, 50, 100, name=None, axis=0)))
#xmat2 = tf.transpose(tf.broadcast_to(tf.transpose(tf.broadcast_to(xval2, [100,100], name = None)), [100,100,100]))
#inpt2 = tf.expand_dims( xmat2, axis=0 )

#inpt3 = tf.concat((inpt1, inpt2), axis = 0)
#inpt4 = tf.expand_dims( inpt3, axis=4 )

#deriv = ddx(inpt4, 0, dx).numpy()

#fig2 = plt.matshow(deriv[0,:,:,0,0])
#plt.colorbar()
#plt.show()

#fig2 = plt.matshow(deriv[1,:,:,0,0])
#plt.colorbar()
#plt.show()

#l = tf.constant([1, 2, 3])

#b = tf.transpose(tf.broadcast_to(tf.transpose(tf.broadcast_to(l, [3,3])), [3,3,3]))
#print(b[:,:,0])

#print(tf.shape(deriv))
#dotprod = tf.tensordot(inpt4[0,:,:,0,0], deriv[0,:,:,0,0], axes = 1)
#print(tf.shape(dotprod))
#plt.matshow(dotprod)
#plt.colorbar()
#plt.show()

#dx = 25/100
#xval = tf.transpose(tf.math.sin(tf.linspace(0, 25, 100, name=None, axis=0)))
#xmat = tf.broadcast_to(xval, [100,100,100], name = None)
#inpt2 = tf.expand_dims( xmat, axis=0 )
#inpt4 = tf.expand_dims( inpt2, axis=4 )
#fig3 = plt.matshow(xmat[0,:,:])
#plt.colorbar()
#plt.show()

#deriv = ddz(inpt4, 0, dx).numpy()

#fig4 = plt.matshow(deriv[0,0,:,:,0])
#plt.colorbar()
#plt.show()
