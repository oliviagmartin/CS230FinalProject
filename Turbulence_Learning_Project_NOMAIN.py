#import sys
#from tensorflow.python.framework import ops
#import tensorflow.experimental.numpy as tnp
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


'''

Example transform sequence input:

    FP_transform_sequence = [ \
                            {"Ttype" : "INPUT", "Neurons" : 1000}, \
                            {"Ttype" : "LINEAR", "Neurons" : 25}, \
                            {"Ttype" : "RELU"}, \
                            {"Ttype" : "LINEAR", "Neurons" : 25}, \
                            {"Ttype" : "RELU"}, \
                            {"Ttype" : "LINEAR", "Neurons" : 1000}, \
                            {"Ttype" : "SOFTMAX"} \
                            ]

NOTE: - FP_transform_sequence defines the sequence of transforms to be applied to the input (during forward propagation) to get to the output (i.e., the prediction;
        what you will plug into the loss function for training).
      - The example input for FP_transform_sequence above corresponds to LINEAR -> RELU -> LINEAR -> RELU -> LINEAR -> SOFTMAX.

'''


class NN_model(tf.keras.Model):
    # Set the optimizer, initializers (for weights and biases), and build the forward pass according to FP_transform_sequence.
    def __init__(self, FP_transform_sequence = [], learning_rate = 0.0001):
        super(NN_model, self).__init__()

        # Gather variables into the class.
        self.FP_transform_sequence = FP_transform_sequence
        self.learning_rate = learning_rate

        # Set the optimizer (we will just use Adam for now).
        self.optimizer = self.get_optimizer(opt_type = "Adam", learning_rate = self.learning_rate)

        # Set the weight initializer (we will just use Xavier for now).
        self.initializer_weights = self.get_weight_initializer(init_type = "Xavier")

        # Set the bias initializer (we will just use zeros for now).
        self.initializer_biases = self.get_bias_initializer(init_type = "Zeros")

        # Create the neural network transform based on FP_transform_sequence (forward pass).
        self.create_NN_transform_sequence()

    def call(self, X, is_training = False):
        # Call the forward pass.
        # NOTE: is_training can be used if there are transforms with different
        # behaviors during training versus utilization (e.g. dropout).
        return self.NN_transform_sequence(X)

    def get_weight_initializer(self, init_type = "Xavier"):
        # Select weight initialization based on init_type.
        if init_type == "Xavier":
            initialization = tf.keras.initializers.GlorotUniform(seed = None)
        else:
            print("ERROR: Invalid 'init_type' for weight initializer selection.")

        return initialization

    def get_bias_initializer(self, init_type = "Zeros"):
        # Select bias initialization based on init_type.
        if init_type == "Zeros":
            initialization = tf.keras.initializers.Zeros()
        else:
            print("ERROR: Invalid 'init_type' for bias initializer selection.")

        return initialization

    def get_optimizer(self, opt_type = "Adam", learning_rate = 0.0001):
        # Select the optimizer based on opt_type.
        if opt_type == "Adam":
            optimizer = tf.keras.optimizers.Adam(learning_rate = learning_rate, beta_1 = 0.9, beta_2 = 0.999, epsilon = 1e-07)
        else:
            print("ERROR: Invalid 'opt_type' for optimizer selection.")

        return optimizer


    def create_NN_transform_sequence(self):
        # Create the NN transform sequence based on the provided forward pass sequence.
        # Convolutional tranforms can be easily added in here as well.
        self.NN_transform_sequence = tf.keras.Sequential()
        for i in range(len(self.FP_transform_sequence)):
            if self.FP_transform_sequence[i]["Ttype"] == "INPUT":
                self.NN_transform_sequence.add( tf.keras.layers.InputLayer(input_shape = (self.FP_transform_sequence[i]["Neurons"], )) )
            elif self.FP_transform_sequence[i]["Ttype"] == "LINEAR":
                self.NN_transform_sequence.add( tf.keras.layers.Dense(units = self.FP_transform_sequence[i]["Neurons"], activation = None, use_bias = True, kernel_initializer = self.initializer_weights, bias_initializer = self.initializer_biases) )
            elif self.FP_transform_sequence[i]["Ttype"] == "RELU":
                self.NN_transform_sequence.add( tf.keras.layers.Activation(tf.nn.relu) )
            elif self.FP_transform_sequence[i]["Ttype"] == "SIGMOID":
                self.NN_transform_sequence.add( tf.keras.layers.Activation(tf.math.sigmoid) )
            elif self.FP_transform_sequence[i]["Ttype"] == "TANH":
                self.NN_transform_sequence.add( tf.keras.layers.Activation(tf.math.tanh) )
            elif self.FP_transform_sequence[i]["Ttype"] == "SOFTMAX":
                self.NN_transform_sequence.add( tf.keras.layers.Activation(tf.nn.softmax) )
            else:
                print("ERROR: Provided transform is not encoded in the NN_model class.")


    def train(self, X_train, Y_train, X_test, Y_test, N_X_train_unroll, N_Y_train_unroll, N_X_test_unroll, N_Y_test_unroll, N_epochs = 4, minibatch_size = 32, save_NN = True, print_cost = True, display_step_epochs = 1, change_learning_rate = False, lambda_p = 0.5, lambda_tau = 0.5):
        # Train the network.

        # Get the total number of samples.
        total_samples = X_train.shape[0]

        # Initialize the costs for plotting.
        costs = []
        costsPhys = []
        costsCont = []

        mini_cost = []
        mini_Ploss = []
        mini_Closs = []

        # Do the training loop.
        for epoch in range(N_epochs):

            # Define a cost related to the current epoch.
            epoch_cost = 0.
            epoch_Ploss = 0.
            epoch_Closs = 0.

            # Calculate the number of minibatches in the training set from the provided
            # minibatch size.
            num_minibatches = int(total_samples / minibatch_size)

            # Randomize the training data and create minibatches.
            minibatches = self.random_mini_batches(X = X_train, Y = Y_train, mini_batch_size = minibatch_size)

            # Run the prescribed optimization on each minibatch.
            for minibatch in minibatches:

                # Divide the current minibatch into inputs and outputs.
                (minibatch_X, minibatch_Y) = minibatch

                # Run the optimizer on the current minibatch.
                minibatch_loss, phys_loss, cont_loss = self.run_optimization(X_train = minibatch_X, Y_train = minibatch_Y, N_X_train_unroll = N_X_train_unroll, N_Y_train_unroll = N_Y_train_unroll, lambda_p = lambda_p, lambda_tau = lambda_tau)

                # Add the loss to the total cost of the epoch.
                epoch_cost += minibatch_loss / minibatch_size
                epoch_Ploss += phys_loss / minibatch_size
                epoch_Closs += cont_loss / minibatch_size

                mini_cost.append(minibatch_loss / minibatch_size)
                mini_Ploss.append(phys_loss / minibatch_size)
                mini_Closs.append(cont_loss / minibatch_size)


            # Print the cost every 'display_step_epochs' epochs. Also, save the costs every
            # 5 epochs for plotting purposes.
            if print_cost and epoch % display_step_epochs == 0:
                print ("Cost after epoch %i: %f" % (epoch, epoch_cost))

            # save the physics and content costs
            costs.append(epoch_cost)
            costsPhys.append(epoch_Ploss)
            costsCont.append(epoch_Closs)

            # Save the weights and network configuration every 5 epochs. (this can be coded more formally, but for now it is fine)
            if save_NN and epoch % 50 == 0 and epoch != 0:
                self.save_network()
                print("Network Configuration and Parameters Saved!")

            # Added ability to change the learning rate mid training. (this can be coded more formally, but for now it is fine)
            if change_learning_rate and epoch % 5 == 0 and epoch != 0:
                import utilities.neural_network_parameters as NN_params
                self.learning_rate = NN_params.learning_rate
                self.optimizer = self.get_optimizer(opt_type = "Adam", learning_rate = self.learning_rate)


        # Plot cost vs. epochs (figure is saved in the same folder as the code).
        plt.plot(np.squeeze(costs))
        plt.ylabel('Cost')
        plt.xlabel('Iterations (per 5 iterations)')
        #plt.title("Learning rate =" + str(self.learning_rate))
        plt.savefig('Cost_vs_Iteration.png', dpi=300)
        plt.close()

        # Save the network. Includes saving both the configuration (i.e., layers/forward pass transform)
        # and the parameters (i.e., weights and biases). The configuration is saved in a '.npy' file and
        # the parameters are saved in an '.h5' file found in the same folder as the code.
        if save_NN:
            self.save_network()
            print("Network Configuration and Parameters Saved!")

        # Calculate the predicted profiles for the test set
        cost_test, difference_profiles, y_predict, y_predict_avg_profiles = self.evaluate_test_set(X_test = X_test, Y_test = Y_test, N_X_test_unroll = N_X_test_unroll, N_Y_test_unroll = N_Y_test_unroll, lambda_p=lambda_p,lambda_tau=lambda_tau)

        # Return costs and profiles
        return costs, costsPhys, costsCont, cost_test, difference_profiles, y_predict, mini_cost, mini_Ploss, mini_Closs, y_predict_avg_profiles


    def run_optimization(self, X_train, Y_train, N_X_train_unroll, N_Y_train_unroll, lambda_p = 0.5, lambda_tau = 0.5):
        # Run the optimization algorithm.

        # Get the number of samples in the minibatch.
        m = X_train.shape[0]

        with tf.GradientTape() as tape:
            # This is where the forward pass + the cost calculation go. tf.GradientTape()
            # records the calculations made from input to calculating the cost, and calculates
            # the gradients. This is then used to update weights and biases

            # Run the forward pass to get the predictions.
            #self.tic()
            Y_predict = self.call(X = X_train, is_training = True)
            #print('Forward Pass Time')
            #self.toc()

            # Unroll Y_predict and Y_train for the loss function.
            Y_predict = tf.reshape(Y_predict, (m, N_X_train_unroll[0], N_X_train_unroll[1], N_X_train_unroll[2], N_X_train_unroll[3]) )
            Y_train = tf.cast(tf.reshape(Y_train, (m, N_Y_train_unroll[0], N_Y_train_unroll[1]) ), tf.float32)

            # Initiate Loss class.
            self.loss = Loss(n_examples = m, inc_mom = False)
            #self.tic()
            # Run the loss function(s).
            loss = self.loss.compute_loss(Yhat = Y_predict, Y = Y_train, lambda_p = lambda_p, lambda_tau = lambda_tau)
            # Compute content loss
            cont_loss = self.loss.Lcontent(Y = Y_train)
            # Compute physics loss
            phys_loss = self.loss.L_mass()


        # Get the recorded gradients staring at the loss function for the network with
        # respect to the weights and biases.
        gradients = tape.gradient(loss, self.trainable_variables)

        # Apply the gradients to the weights and biases using the previously defined optimizer.
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))

        # Return the loss for printing purposes.
        return loss, phys_loss, cont_loss


    def random_mini_batches(self, X, Y, mini_batch_size, seed = 0):
        # Create a list of random minibatches from (X, Y).
        m = X.shape[0]
        mini_batches = []
        np.random.seed(seed)

        # Shuffle (X, Y).
        permutation = list(np.random.permutation(m))
        shuffled_X = X[permutation, :]
        shuffled_Y = Y[permutation, :].reshape((m, Y.shape[1]))

        # Partition (shuffled_X, shuffled_Y). Minus the end case.
        num_complete_minibatches = math.floor(m/mini_batch_size) # number of mini batches of size mini_batch_size in your partitionning
        for k in range(0, num_complete_minibatches):
            mini_batch_X = shuffled_X[k * mini_batch_size : k * mini_batch_size + mini_batch_size, :]
            mini_batch_Y = shuffled_Y[k * mini_batch_size : k * mini_batch_size + mini_batch_size, :]
            mini_batch = (mini_batch_X, mini_batch_Y)
            mini_batches.append(mini_batch)

        # Handling the end case (last mini-batch < mini_batch_size)
        if m % mini_batch_size != 0:
            mini_batch_X = shuffled_X[num_complete_minibatches * mini_batch_size : m, :]
            mini_batch_Y = shuffled_Y[num_complete_minibatches * mini_batch_size : m, :]
            mini_batch = (mini_batch_X, mini_batch_Y)
            mini_batches.append(mini_batch)

        # Return mini_batches = (mini_batch_X, mini_batch_Y).
        return mini_batches


    def get_weights_and_biases(self):
        return self.get_weights()

    # Save network configuration and parameters
    def save_network(self, network_configuration_file_name = "Network_Configuration0", parameters_file_name = "Parameters0"):
        # Save the neural network (i.e., the forward pass configuration and the weights and biases).
        self.save_network_configuration(file_name = network_configuration_file_name)
        self.save_parameters(file_name = parameters_file_name)

    # Save network configuration
    def save_network_configuration(self, file_name = "Network_Configuration0"):
        # Save the network transform sequence (i.e., FP_transform_sequence).
        np.save('../Troubleshooting_Data/' + file_name + '.npy', self.FP_transform_sequence)

    # Save network parameters
    def save_parameters(self, file_name = "Parameters0"):
        # Save the network parameters (i.e., weights and biases).
        parameters = self.get_weights_and_biases()
        with h5py.File('../Troubleshooting_Data/' + file_name + '.h5', "w") as f:
            f_params = f.create_group("Parameters")
            for i in range(len(parameters)):
                f_params.create_dataset("p" + str(i), data = parameters[i])

    # Load the previously saved network
    def load_network(self, network_configuration_file_name = "Network_Configuration0", parameters_file_name = "Parameters0"):
        # Load a previously saved neural network (i.e., the forward pass configuration and the weights and biases).
        self.load_network_configuration(file_name = network_configuration_file_name)
        parameters = self.load_parameters(file_name = parameters_file_name)

        # If the loaded FP_transform_sequence does not have an input layer, the parameters cannot be loaded into
        # the network. So, we must create on if the input layer is not already there.
        if not self.FP_transform_sequence[0]["Ttype"] == "INPUT":
            # Insert the missing input layer.
            self.FP_transform_sequence.insert(0, {"Ttype" : "INPUT", "Neurons" : parameters[0].shape[0]})

        # Rebuild the NN based on the provided forward pass sequence.
        self.create_NN_transform_sequence()
        print("Loaded Configuration: '" + network_configuration_file_name + "'")

        # Set the weights and biases to the previously saved parameters.
        self.set_weights(parameters)
        print("Loaded Parameters: '" + parameters_file_name + "'")

        print("Network Loading Successful!")


    def load_network_configuration(self, file_name = "Network_Configuration0"):
        # Load the network transform sequence (i.e., FP_transform_sequence).
        self.FP_transform_sequence = list(np.load(file_name + '.npy', allow_pickle = True))


    def load_parameters(self, file_name = "Parameters0"):
        # Load the network parameters (i.e., weights and biases).
        parameters = []
        with h5py.File(file_name + '.h5', "r") as f:
            f_params_keys = list(f["Parameters"].keys())
            for i in range(len(f_params_keys)):
                parameters.append( f["Parameters"][f_params_keys[i]][:] )

        return parameters

    def evaluate_test_set(self, X_test, Y_test, N_X_test_unroll, N_Y_test_unroll,lambda_p,lambda_tau):
        # Evaluate X_test and create the averaged profiles for the Y_predict.

        # Obtain the averaged profiles from X_test.
        Y_predict, Y_predict_avg_profiles = self.predict_avg_profiles(X_predict = X_test, N_X_test_unroll = N_X_test_unroll)

        # Unroll Y_test.
        Y_test = tf.cast(tf.reshape(Y_test, (Y_test.shape[0], N_Y_test_unroll[0], N_Y_test_unroll[1]) ), tf.float32)

        # Calculate the difference profiles (if you plot these, they should land within the margins of error for the averaged profiles in Z)
        profile_differences = Y_predict_avg_profiles - Y_test
        cost = self.loss.compute_loss(Yhat = Y_predict, Y = Y_test, lambda_p = lambda_p, lambda_tau = lambda_tau)

        return cost, profile_differences, Y_predict, Y_predict_avg_profiles


    def predict(self, X_predict):
        # Predict the velocity/pressure fields with the NN.
        return self.call(X = X_predict, is_training = False)


    def predict_avg_profiles(self, X_predict, N_X_test_unroll):
        # Predict the averaged profiles with the NN.

        # Get the number of samples in the minibatch.
        m = X_predict.shape[0]

        # Predict the velocity/pressure fields.
        Y_predict = self.predict(X_predict = X_predict)

        # Unroll Y_predict.
        Y_predict = tf.reshape(Y_predict, (m, N_X_test_unroll[0], N_X_test_unroll[1], N_X_test_unroll[2], N_X_test_unroll[3]) )

        # Initiate Loss class.
        self.loss = Loss(n_examples = m, inc_mom = False)
        #self.loss = Loss(nx = N_X_test_unroll[1], ny = N_X_test_unroll[2], nz = N_X_test_unroll[3], Lx = 4.0*np.pi, Ly = 2.0*np.pi, Lz = 1.0, n_examples = m)

        # Compute the average profiles.
        Y_predict_avg_profiles = self.loss.predict_average_profiles(Yhat = Y_predict)

        return Y_predict, Y_predict_avg_profiles


    def tic(self):
        self.start_time = time.time()

    def toc(self):
        print('Elapsed: %s' % (time.time() - self.start_time))








#if __name__ == "__main__":
#
#    # File path from code to data directory.
#    data_directory = "../Troubleshooting_Data/"
#
    # Data and simulation parameters.
#    nx = params.nx
#    ny = params.ny
#    nz = params.nzC
#    nzF = params.nzF
#    Lx = params.Lx
#    Ly = params.Ly
#    Lz = params.Lz
#    navg = 840

#    zC = setup_domain_1D(0.5*Lz/nz , Lz - 0.5*Lz/nz , Lz/nz)
#    zF = setup_domain_1D(0.5*Lz/nzF, Lz - 0.5*Lz/nzF, Lz/nzF)

    # Define the training and test sets by inputing the time id numbers.
#    x_tid_vec_train = np.array([179300, 179300, 179300])
#    y_tid_vec_train = np.array([25400, 25400, 25400])
#    x_tid_vec_test = np.array([179300, 179300])
#    y_tid_vec_test = np.array([25400, 25400])


######################################################################


    # Get the training and test data from disk.
#    X_train, Y_train, X_test, Y_test = \
#        load_dataset_V2(data_directory, nx, ny, nz, nzF, x_tid_vec_train, \
#        x_tid_vec_test, y_tid_vec_train, y_tid_vec_test, \
#        inc_prss = False, navg = navg)

    # Normalize the data
#    X_train, Y_train, X_test, Y_test = normalize_data(train_set_x = \
#        X_train, train_set_y = Y_train, test_set_x = X_test, \
#        test_set_y = Y_test, inc_prss = False)

    # Get the dimensions of the training and test sets.
#    N_samples_train = X_train.shape[0]
#    N_X_quantities  = X_train.shape[1]
#    N_Y_quantities  = Y_train.shape[1]
#    Nx              = X_train.shape[2]
#    Ny              = X_train.shape[3]
#    Nz              = X_train.shape[4]

#    N_X_train_unroll = [N_X_quantities, Nx, Ny, Nz]
#    N_Y_train_unroll = [N_Y_quantities, Nz]

#    N_samples_test  = X_test.shape[0]

#    N_X_test_unroll = [N_X_quantities, Nx, Ny, Nz]
#    N_Y_test_unroll = [N_Y_quantities, Nz]

    # Roll up the training and testing data (i.e. make it shape (N_samples x N_inputs))
#    X_train = np.reshape(X_train, (N_samples_train, N_X_train_unroll[0]*N_X_train_unroll[1]*N_X_train_unroll[2]*N_X_train_unroll[3]) )
#    Y_train = np.reshape(Y_train, (N_samples_train, N_Y_train_unroll[0]*N_Y_train_unroll[1]) )
#    X_test = np.reshape(X_test, (N_samples_test, N_X_test_unroll[0]*N_X_test_unroll[1]*N_X_test_unroll[2]*N_X_test_unroll[3]) )
#    Y_test = np.reshape(Y_test, (N_samples_test, N_Y_test_unroll[0]*N_Y_test_unroll[1]) )



    # Provide the transforms to be completed on the forward pass.
#    FP_transform_sequence = [ \
#                            {"Ttype" : "INPUT", "Neurons" : X_train.shape[1]}, \
#                            {"Ttype" : "LINEAR", "Neurons" : 25}, \
#                            {"Ttype" : "RELU"}, \
#                            {"Ttype" : "LINEAR", "Neurons" : X_train.shape[1]}, \
#                            {"Ttype" : "RELU"} \
#                            ]

    # Define the model
#    model = NN_model(FP_transform_sequence = FP_transform_sequence)

    # Train the model   # shape (inputs x N_samples)
#    model.train(X_train = X_train, Y_train = Y_train, X_test = X_test, \
#        Y_test = Y_test, N_X_train_unroll = N_X_train_unroll, \
#        N_Y_train_unroll = N_Y_train_unroll, N_X_test_unroll = \
#        N_X_test_unroll, N_Y_test_unroll = N_Y_test_unroll)


    #### Test the loading of a network and its the parameters
#    prediction1 = model.predict(X_predict = np.array([X_test[0]]))

#    model2 = NN_model(FP_transform_sequence = [])
#    model2.load_network()
#    prediction2 = model2.predict(X_predict = np.array([X_test[0]]))
#    print(prediction1-prediction2)
#    '''

#    code.interact(local=locals())"""
