This repo contains a series of scripts used to initialize and train a NN
to correction low resolution flow simulation data.

Authors: Ryan Hass, Kyle Pietrzyk, Olivia Martin
CS230 Final Project - Fall 2021

hyperparameter_search.py - Initializes the hyperparameters, runs the NN, and
saves the output data in txt and .h files.

Turbulence_Learning_Project_NOMAIN - Initializes and trains the NN

utilities/convDiff.py - computes spatial derivatives of the flow field using a
convolutional filter

utilities/diff_tf.py - computes spectral derivates in x and y and uses a finite
difference to compute derivatives in z

utilities/domain_setup.py - sets up the spatial domain

utilities/io_mod.py - contains functions to import the train/test data from an
AWS S3 storage location and normalize the input data

utilities/loss_fcns_conv.py - computes the average flow profiles, the physics loss,
the content loss, and the total cost

utilities/problem_parameters.py - contains dimensions of flow geometry and mean velocity

utilities/read_fortran_data.py - reads in Fortran data

tests - This folder contains scripts used to test the derivative functions.
        None of this code is used to train the NN.
