import numpy as np

def relu(x, derivative = False):
    res = x
    if derivative == True:
        return (res > 0)*1
    else:
        return res * (res > 0)

def sigmoid(x, derivative = False):
    res = 1/(1 + np.exp(-x))
    if derivative == True:
        return res * (1 - res)
    else:
        return res


def train_a_network(train_inputs, n_x, m, l_1, n_z, alpha, epochs = 100):
    '''
    Parameters:
    ___________
    train_inputs = Training data
    n_x = Number of features
    m = Number of data points
    l_1 = Number of neurons in first layer
    n_z = Number of latent variables 
    alpha = learning rate
    epochs = Number of full passes of the data
    
    Returns:
    _________
    Dictionary that has the parameters of the network, along with two lists of loss behaviour that can be plotted.
    '''

    W_1 = np.random.randn(l_1, n_x) * np.sqrt(2/n_x) # Kaiming initialization
    b_1 = np.zeros(shape=(l_1, 1))

    W_mu = np.random.randn(n_z, l_1) * np.sqrt(2/l_1) # Kaiming initialization
    b_mu = np.zeros(shape = (n_z,1))

    W_sigma = np.random.randn(n_z, l_1) * np.sqrt(2/l_1) # Kaiming initialization
    b_sigma = np.zeros(shape = (n_z, 1))

    eps = np.random.randn(n_z, m)

    W_2 = np.random.randn(n_x, n_z) * np.sqrt(2/n_z) # Kaiming initialization
    b_2 = np.zeros(shape = (n_x, 1)) # Shape of bias should be (n_x, 1) or (n_x, m)? I'm thinking same bias means 1?

    # train_truth = (np.random.sample(size=(n_x,1)) >= 0.5)*1

    losses_mse = []
    losses_kl = []

    for i in range(epochs):
        # Forward pass
        z_1 = W_1.dot(train_inputs) + b_1
        a_1 = relu(z_1)

        z_mu = W_mu.dot(a_1) + b_mu
        # There is no activation for mu layer
        z_sigma = W_sigma.dot(a_1) + b_sigma

        sampled_vector = z_mu + np.multiply(np.exp(z_sigma * 0.5), eps) # Elem-wise multiplication for eps and var.
        # Also treat z_sigma as a log-var instead, bypass the problem with negatives.

        z_2 = W_2.dot(sampled_vector) + b_2
        # Should there be a final activation? 
        # I don't think so - if we're going for MSE loss we're going to compare every single feature vector with the 
        # reconstructed one. So the range of the activation function has to be the whole real line, but most common ones
        # are not that way, like ReLU, tanh, sigmoid.

        # Loss - go with simple L2 loss
        loss_mse = np.mean(np.square(train_inputs - z_2))
        loss_kl = np.mean(np.sum(np.exp(z_sigma) + np.square(z_mu) - z_sigma - 0.5*np.ones(shape = z_sigma.shape),axis=1)) 
        loss = loss_mse + loss_kl

        losses_mse.append(loss_mse)
        losses_kl.append(loss_kl)
        # Warning - KL loss, taking mean is obscuring how some are close to N(0,1) and some are not.

        if i % 10 == 0:
            print('The loss from pass {} is: '.format(i) + str(loss))

        # Backward pass
        # Need one for W_1, b_1, W_mu, b_mu, W_sigma, b_sigma, W_2, b_2
        # This is for MSE loss
        grad_z_2 = -2 * (train_inputs - z_2) # (n_x, m)
        grad_W_2 = grad_z_2.dot(sampled_vector.T) # (n_z, m)
        grad_b_2 = (1/m) * np.sum(grad_z_2, keepdims = True)
        grad_s = W_2.T.dot(grad_z_2) # (n_z, m), gradient of sampled vector
        grad_z_mu = grad_s # (n_z, m)
        grad_z_sigma = np.multiply(grad_s, 0.5 * np.exp(z_sigma * 0.5) * eps)
        grad_W_mu = grad_z_mu.dot(a_1.T)
        grad_b_mu = (1/m) * np.sum(grad_z_mu, keepdims=True)
        grad_W_sigma = grad_z_sigma.dot(a_1.T)
        grad_b_sigma = (1/m) * np.sum(grad_z_sigma, keepdims=True)
        grad_a_1 = W_mu.T.dot(grad_z_mu)
        grad_z_1 = np.multiply(grad_a_1, relu(z_1, derivative=True))
        grad_b_1 = (1/m) * np.sum(grad_z_1, keepdims = True)
        grad_W_1 = grad_z_1.dot(train_inputs.T)

        # This is for KL loss
        grad_z_sigma += np.exp(z_sigma) - 1
        grad_z_mu += 2*z_mu # Seriously, wth. There is likely something from KL that can give intuition on this.
        grad_W_sigma += grad_z_sigma.dot(a_1.T)
        grad_b_sigma += (1/m) * np.sum(grad_z_sigma, keepdims = True)
        grad_W_mu += grad_z_mu.dot(a_1.T)
        grad_b_mu += (1/m) * np.sum(grad_z_mu, keepdims = True)

        # Do your gradient updates.
        b_2 -= alpha* grad_b_2
        W_2 -= alpha* grad_W_2
        b_mu -= alpha* grad_b_mu
        W_mu -= alpha* grad_W_mu
        b_sigma -= alpha* grad_b_sigma
        W_sigma -= alpha* grad_W_sigma
        b_1 -= alpha* grad_b_1
        W_1 -= alpha* grad_W_1
        
        result_dict = {
            'b_2' : b_2,
            'W_2' : W_2,
            'b_mu' : b_mu,
            'W_mu' : W_mu,
            'b_sigma': b_sigma,
            'W_sigma': W_sigma,
            'b_1': b_1,
            'W_1': W_1,
            'losses_mse':losses_mse
            'losses_kl': losses_kl
        }
        return result_dict
    
def rc_loss_of_dps(inputs, params_dict):
    ''' 
    Send a bunch of inputs through a network specified by params_dict to obtain the reconstruction loss. Only allows for the most
    basic architecture
    
    X -> FC -> RELU -> SAMPLE -> FC -> X_hat
    
    Paramters:
    __________
    inputs: The data points intended to be sent through the network.
    params_dict: Dictionary of parameters returned by train_a_network
    
    Returns:
    ________
    Reconstruction loss for those inputs.
    '''
    W_1 = params_dict['W_1']
    b_1 = params_dict['b_1']
    W_sigma = params_dict['W_sigma']
    b_sigma = params_dict['b_sigma']
    W_mu = params_dict['W_mu']
    b_mu = params_dict['b_mu']
    W_2 = params_dict['W_2']
    b_2 = params_dict['b_2']
    
    z_1_test = W_1.dot(inputs) + b_1
    a_1_test = relu(z_1_test)
    z_mu_test = W_mu.dot(a_1_test) + b_mu
    z_sigma_test = W_sigma.dot(a_1_test) + b_sigma
    test_sampled_eps = np.random.normal(size = z_sigma_test.shape)
    
    # Reparameterization trick
    sampled_vector_test = z_mu_test + np.multiply(test_sampled_eps, np.exp(.5 * z_sigma_test))
    
    # Reconstruct the input
    z_2_test = W_2.dot(sampled_vector_test) + b_2
    
    # Find the reconstruction loss of the decoded test vectors
    rcloss = np.mean(0.5*np.square(z_2_test - test_inputs))
    return rcloss

