import numpy as np
import code
import tensorflow as tf
import time
import h5py
import os
import math
import matplotlib.pyplot as plt
from utilities.tf_utils import random_mini_batches
from utilities.io_mod import load_dataset_V2, normalize_data
from utilities.loss_fcns_conv import Loss
from utilities.domain_setup import setup_domain_1D
import utilities.problem_parameters as params
from Turbulence_Learning_Project_NOMAIN import NN_model
from matplotlib import cm
import random

# Define function to rename output files so that we have them for each hyperparameter search
def renameFiles(direct, iteration):

    fname1 = 'Network_Configuration0.npy '
    fname1NEW = 'Network_Configuration0' + str(iteration) + '.npy'
    fname2 = 'Parameters0.h5 '
    fname2NEW = 'Parameters0' + str(iteration) + '.h5'

    command2 = 'mv ' + direct + fname2 + direct + fname2NEW
    command1 = 'mv ' + direct + fname1 + direct + fname1NEW
    os.system(command2)
    os.system(command1)

# Randomly select ylabels for the training/test sets
def formYlabel(Ylength, min = 17000, max = 23800):
    # array of all labels
    allY = np.arange(min, max, 100)
    y_tid_vec_train = []
    # randomly assign labels
    for i in range(Ylength):
        index = random.randint(0, np.shape(allY)[0] - 1)
        y_tid_vec_train.append(allY[index])

    return np.array(y_tid_vec_train)

# Save the predicted output profiles
def save_OutputProfiles(direct, y_test, y_predict, Y_predict_avg_profiles):
    # Save the network parameters (i.e., weights and biases).
    file_name = direct + 'DevPredict'
    with h5py.File(file_name + '.h5', "w") as f:
        f_params = f.create_group("DevProfiles")
        for i in range(np.shape(Y_predict_avg_profiles)[0]):
            f_params.create_dataset("testNumber" + str(i), data = Y_predict_avg_profiles[i,:,:])

        f_params = f.create_group("DevSpatFields")
        for i in range(np.shape(y_predict)[0]):
            f_params.create_dataset("testNumber" + str(i), data = y_predict[i,:,:,:,:])

    fileName = direct + 'DevPredict.txt'
    f = open(fileName, "w")
    for i in range(len(y_test)):
        f.write('Test Number: ' + str(y_test[i]) + '\n')
    f.close()

# Define function to run the model
def runModel(learning_rate = 0.001, lambda_p = 0.5, lambda_tau = 0.5, \
    N_epochs = 4, minibatch_size = 2, renameFile = True, iteration = 1, direct = 'hpSearch/'):

    # File path from code to data directory.
    data_directory = "../Troubleshooting_Data/"

    # Data and simulation parameters.
    nx = params.nx
    ny = params.ny
    nz = params.nzC
    nzF = params.nzF
    Lx = params.Lx
    Ly = params.Ly
    Lz = params.Lz
    navg = 1

    # Set up the domain
    zC = setup_domain_1D(0.5*Lz/nz , Lz - 0.5*Lz/nz , Lz/nz)
    zF = setup_domain_1D(0.5*Lz/nzF, Lz - 0.5*Lz/nzF, Lz/nzF)

    # Define the training and test sets by inputing the time id numbers.
    x_tid_vec_train = np.arange(146000, 172700, 100)
    y_tid_vec_train = formYlabel(np.shape(x_tid_vec_train)[0], min = 17000, max = 23800)
    x_tid_vec_test = np.arange(176100, 179400, 100)
    y_tid_vec_test = formYlabel(np.shape(x_tid_vec_test)[0], min = 23900, max = 24700)


######################################################################


    # Get the training and test data from disk.
    X_train, Y_train, X_test, Y_test = \
        load_dataset_V2(data_directory, nx, ny, nz, nzF, x_tid_vec_train, \
        x_tid_vec_test, y_tid_vec_train, y_tid_vec_test, \
        inc_prss = False, navg = navg)

    # Normalize the data
    X_train, Y_train, X_test, Y_test = normalize_data(train_set_x = \
        X_train, train_set_y = Y_train, test_set_x = X_test, \
        test_set_y = Y_test, inc_prss = False)

    # Get the dimensions of the training and test sets.
    N_samples_train = X_train.shape[0]
    N_X_quantities  = X_train.shape[1]
    N_Y_quantities  = Y_train.shape[1]
    Nx              = X_train.shape[2]
    Ny              = X_train.shape[3]
    Nz              = X_train.shape[4]

    N_X_train_unroll = [N_X_quantities, Nx, Ny, Nz]
    N_Y_train_unroll = [N_Y_quantities, Nz]

    N_samples_test  = X_test.shape[0]

    N_X_test_unroll = [N_X_quantities, Nx, Ny, Nz]
    N_Y_test_unroll = [N_Y_quantities, Nz]

    # Roll up the training and testing data (i.e. make it shape (N_samples x N_inputs))
    X_train = np.reshape(X_train, (N_samples_train, N_X_train_unroll[0]*N_X_train_unroll[1]*N_X_train_unroll[2]*N_X_train_unroll[3]) )
    Y_train = np.reshape(Y_train, (N_samples_train, N_Y_train_unroll[0]*N_Y_train_unroll[1]) )
    X_test = np.reshape(X_test, (N_samples_test, N_X_test_unroll[0]*N_X_test_unroll[1]*N_X_test_unroll[2]*N_X_test_unroll[3]) )
    Y_test = np.reshape(Y_test, (N_samples_test, N_Y_test_unroll[0]*N_Y_test_unroll[1]) )

    # Provide the transforms to be completed on the forward pass.
    FP_transform_sequence = [ \
                            {"Ttype" : "INPUT", "Neurons" : X_train.shape[1]}, \
                            {"Ttype" : "LINEAR", "Neurons" : 25}, \
                            {"Ttype" : "RELU"},\
			    {"Ttype" : "LINEAR", "Neurons" : 25}, \
			    {"Ttype" : "RELU"}, \
			    {"Ttype" : "LINEAR", "Neurons" : 25}, \
			    {"Ttype" : "RELU"}, \
                            {"Ttype" : "LINEAR", "Neurons" : X_train.shape[1]}, \
                            {"Ttype" : "RELU"} \
                            ]

    # Define the model
    model = NN_model(FP_transform_sequence = FP_transform_sequence, learning_rate = learning_rate)

    # Line to load preivously trained network
    #model.load_network()

    # Train the network
    cost, costsPhys, costsCont, cost_test, diff_profile, y_predict, mini_cost, mini_Ploss, mini_Closs, Y_predict_avg_profiles = \
        model.train(X_train, Y_train, X_test, Y_test, N_X_train_unroll, \
        N_Y_train_unroll, N_X_test_unroll, N_Y_test_unroll, N_epochs = N_epochs, \
        minibatch_size = minibatch_size, save_NN = True, print_cost = True, \
        display_step_epochs = 1, change_learning_rate = False, \
        lambda_p = lambda_p, lambda_tau = lambda_tau)

    '''
    #### Test the loading of a network and its the parameters
    prediction1 = model.predict(X_predict = np.array([X_test[0]]))

    model2 = NN_model(FP_transform_sequence = [])
    model2.load_network()
    prediction2 = model2.predict(X_predict = np.array([X_test[0]]))
    print(prediction1-prediction2)
    '''

    if renameFile == True:
        renameFiles('../Troubleshooting_Data/', iteration)

    # Save the output velocity profiles
    save_OutputProfiles(direct, x_tid_vec_test, y_predict, Y_predict_avg_profiles)


    return cost, costsPhys, costsCont, cost_test, diff_profile, y_predict, mini_cost, mini_Ploss, mini_Closs

def plotField(field):
    # Plot the output velocity fields
    dx = 6*np.pi/192;
    dy = 3*np.pi/192;
    x = np.arange(0, 6*np.pi, dx)
    y = np.arange(0, 3*np.pi, dy)
    X, Y = np.meshgrid(x, y)

    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    surf = ax.plot_surface(X, Y, field, cmap=cm.coolwarm)
    fig.colorbar(surf)
    ax.view_init(azim=0, elev=90)

    plt.show()

    return

def plotCost(cost,learning_rate, lambda_p,lambda_tau, direct):
    # Plot the training cost vs. iteration number for all hyperparameter sets
    numIterations = np.shape(cost)[0]
    numEpochs = np.shape(cost)[1]
    for i in range(np.shape(cost)[0]):
        name = r'$\alpha = ' + str(round(learning_rate[i],4)) + ' ' + '; \lambda_{p} = ' + str(round(lambda_p[i],3)) + ' ' + '; \lambda_{tau} = ' + str(round(lambda_tau[i],3)) + '$'
        plt.plot(range(1,numEpochs), cost[i,1:], label=name)

    plt.xlabel("Epoch")
    plt.ylabel("Cost")
    plt.legend(loc='upper right')
    plt.show()

    figName = direct + 'HPSearch_Costs.png'

    plt.savefig(figName, dpi=300)
    #plt.close()

    return

# Save the hyperparameter values in a txt file
def save_HyperParams(direct, learning_rate, lambda_p, lambda_tau):
    fileName = direct + 'hyperParameters.txt'
    f = open(fileName, "w")
    for i in range(len(learning_rate)):
        f.write('iteration: ' + str(i+1) + '\n' + 'learning rate: ' + \
            str(learning_rate[i]) + '\n' + 'lambda_p: ' + str(lambda_p[i]) + '\n' + \
            'lambda_tau: ' + str(lambda_tau[i]) + '\n\n\n')
    f.close()

# Save the training costs in a .h file
def save_TrainingCosts(direct, costs_train, costs_phys, costs_cont):
    # Save the network parameters (i.e., weights and biases).
    file_name = direct + 'trainingCosts'
    with h5py.File(file_name + '.h5', "w") as f:
        f_params = f.create_group("Costs")
        for i in range(np.shape(costs_train)[0]):
            f_params.create_dataset("iteration" + str(i), data = costs_train[i,:])
        f_params = f.create_group("PhysCosts")
        for i in range(np.shape(costs_phys)[0]):
            f_params.create_dataset("iteration" + str(i), data = costs_phys[i,:])
        f_params = f.create_group("ContCosts")
        for i in range(np.shape(costs_cont)[0]):
            f_params.create_dataset("iteration" + str(i), data = costs_cont[i,:])

if __name__ == "__main__":

    # Number of random hyperparameters to search
    numVals = 1

    learning_rate = [0.00001]

    # Choose random values of lambda_p (physics loss) and lambda_tau
    lambda_p = [0.2];
    lambda_tau = [0.5];

    # vector of training example numbers
    x_tid_vec_train = np.arange(146000, 172700, 100)
    N_datafiles = np.shape(x_tid_vec_train)[0]

    # minibatch size
    minibatch_size = 2
    N_epochs = 1000
    numIterations = N_epochs * math.ceil(N_datafiles/minibatch_size);

    # Initialize arrays to store training costs
    costs_train = np.zeros((numVals,numIterations))
    costs_phys = np.zeros((numVals,numIterations))
    costs_cont = np.zeros((numVals,numIterations))
    costs_test = np.zeros((numVals,))

    # Directory to store data
    direct = '../Troubleshooting_Data/hpSearch/'
    os.system('mkdir ' + direct)

    # Loop through hyperparameters and train the network
    for i in range(numVals):
        cost_train, cost_phys, cost_cont, cost_test, diff_profiles, y_predict, mini_cost, mini_Ploss, mini_Closs = \
            runModel(learning_rate = \
            learning_rate[i], lambda_p = lambda_p[i], lambda_tau = \
            lambda_tau[i], N_epochs = N_epochs, iteration = i, direct = direct, minibatch_size = minibatch_size)
        # Store losses
        costs_train[i,:] = mini_cost
        costs_phys[i,:] = mini_Ploss
        costs_cont[i,:] = mini_Closs
        costs_test[i] = cost_test

    # Save a txt file containing the hyperparameters
    # Save .h5 file with training costs
    save_HyperParams(direct, learning_rate, lambda_p, lambda_tau)
    save_TrainingCosts(direct, costs_train, costs_phys, costs_cont)

    # Generate plot of cost vs. epoch
    #plotCost(mini_cost, learning_rate,lambda_p,lambda_tau, direct)

    # Plot random velocity profile
    #plotField(y_predict[1,1,:,:,32],X,Y)

    code.interact(local=locals())
