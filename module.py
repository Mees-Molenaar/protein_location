# Import
import numpy as np
import matplotlib.pyplot as plt

# Open the training data and save some important variables. If you want to train on your own text, just change the .txt file in the data variable.
#data = open('shakespeare.txt', 'r').read()
data = open('nescio.txt', 'r').read()
chars = list(set(data))
data_size = len(data)
vocab_size = len(chars)

# Simple character embedding
char_to_idx = {char:i for i, char in enumerate(chars)}
idx_to_char = {i:char for i, char in enumerate(chars)}

class Module:
    r"""
    Parent class for recurrent neural networks (RNNs).
    
    This class is the base of RNNs (conventional RNN and Long short-term memory RNN).
    It contains the initialization of the dictionaries containing the weights and gradients,
    these weights and gradients are also initialized by this class.
    Furthermore, it contains the reset method of the hidden layers. Most importantly, this 
    class contains the training method. 
    
    Parameters
    ----------
    seq_len (int): Number of layers connected to each other.
    hidden_sz (int): The number of features in the hidden state h.
    vocab_sz (int): The nomber of possible inputs and outputs.
    weights (list): List containing the names of the needed weights.
    hidden (list): List containing the names of the needed hidden layers.
    
    
    """
    def __init__(self, seq_len, hidden_sz, vocab_sz, weights, hidden):
        self.seq_len = seq_len
        self.hidden_sz = hidden_sz
        self.vocab_sz = vocab_sz
        
        self.params = dict() # Dictionary of parameters including weights and gradients
        self.hidden = dict() # Dictionary of hidden
        self.sm_ps = dict()  # Dictionary of softmax pro values
        
        
        self.make_hidden_dict(hidden)
        self.make_params_dict(weights)
        
        # Initialize weights, hidden and cell states.
        self.init_weights()
        self.reset_hidden()
        
        # Start with zero loss
        self.loss = 0
        
        # Start with no adagrad memory setup
        self.adagrad_mem = False
        
    def make_params_dict(self, weights):
        """
        Set up the dictionary to contain the model weights, biases and gradients.
        Based on the weights the model needs.
        """
        
        for weight in weights:     
            if weight == 'Why':
                size = (self.vocab_sz, self.hidden_sz)
            elif weight.startswith('Wx'):
                size = (self.hidden_sz, self.vocab_sz)
            elif 'B' in weight and weight != 'By':
                size = (self.hidden_sz, 1)
            elif 'B' in weight and weight == 'By':
                size = (self.vocab_sz, 1)
            else:
                size = (self.hidden_sz, self.hidden_sz)
            
            self.params[weight] = {'size': size}
        
    def init_weights(self):
        """
        Initializes weights and biases based on the inputs hidden sz and
        vocab_sz.
        """
        
        for param in self.params:    
            # Initialize a weight matrix
            x,y = self.params[param]['size']
            if y != 1:
                self.params[param]['weight'] = np.random.randn(x, y) * 0.01
            # Initialize a bias
            else:
                self.params[param]['bias'] = np.zeros((x, y))
                
    def init_grads(self):
        """
        Initialize gradients for biases and weights
        """
        for param in self.params:
            # Initialize gradients for weights
            if 'weight' in self.params[param].keys():
                self.params[param]['grad'] = np.zeros_like(self.params[param]['weight'])
            # Initialize gradients for biases
            else:
                self.params[param]['grad'] = np.zeros_like(self.params[param]['bias'])
                
    def init_adagrad_mem(self):
        """
        Initialize memory matrices needed for Adagrad.
        """
        
        self.adagrad_mem = True
        
        for param in self.params:
            # Initialize gradients for weights
            if 'weight' in self.params[param].keys():
                self.params[param]['ada_mem'] = np.zeros_like(self.params[param]['weight'])
            # Initialize gradients for biases
            else:
                self.params[param]['ada_mem'] = np.zeros_like(self.params[param]['bias'])
                
    def update_grads(self, optimizer, lr):
        """
        Update gradients based on the optimizer you choose.
        
        Inputs
        ------
        optimizer (string): Optimizer you want to use to update the gradients.
        lr (float): Learning rate used to optimize the gradients.
        """

        if optimizer == 'Adagrad':
            if not self.adagrad_mem:
                self.init_adagrad_mem()
                
            for param in self.params:
                mem = self.params[param]['ada_mem']
                grad = self.params[param]['grad']
                self.params[param]['ada_mem'] += grad * grad
                
                # Update weight
                if 'weight' in self.params[param].keys():
                    self.params[param]['weight'] += -1 * lr * grad / np.sqrt(mem + 1e-8)
                # Update bias
                else:
                    self.params[param]['bias'] += -1 * lr * grad / np.sqrt(mem + 1e-8)
    
    def make_hidden_dict(self, hidden):
        """
        Create a dictionary of hidden layers
        """
        
        for layer in hidden:
            self.hidden[layer] = {}
            
    
    def reset_hidden(self):
        """
        Reset hidden layers and possible cell state
        """
        for layer in self.hidden.keys():
            self.hidden[layer][-1] = np.zeros((self.hidden_sz, 1))
        

    def plot_losses(self):
        """
        Plot the cross entropy loss against the number of training sequences
        """

        if hasattr(self, 'losses'):
            plt.plot(self.losses)
            plt.xlabel('Number of training sequences')
            plt.ylabel('Cross Entropy Loss')
            plt.show()
        else:
            print('Error: No losses recorded, train the model!')

    def train(self, data, optimizer, lr, epochs, progress=True):
        """
        Train the model by chopping the data in sequences followed by performing
        the forward pass, backward pass and update the gradients.
        
        Inputs
        ------
        data (string): Data used to train the network.
        optimizer (string): Optimizer you want to use to update the gradients
        lr (float): Learning rate used to update the gradients.
        epochs (int): The number of epochs to train the network.
        progress (Boolean): If True, shows the progress of training the network.
        
        Outputs
        -------
        smooth_loss (float): The loss of the current trained network.
        
        """
        self.losses = []
        smooth_loss = -np.log(1.0 / self.vocab_sz) * self.seq_len # Loss at iteration 0

        # Loop over the amount of epochs
        for epoch in range(epochs):
            n = 0

            # Reset hidden state
            self.reset_hidden()

            data_len = len(data)

            # Loop over amount of sequences in the data
            sequences_amount = int(data_len // self.seq_len)
            for j in range(sequences_amount):

                start_pos = self.seq_len * j

                # Embed the inputs and targets
                xs = [char_to_idx[ch] for ch in data[start_pos:start_pos + self.seq_len]]
                targets = [char_to_idx[ch] for ch in data[start_pos + 1:start_pos + self.seq_len + 1]]

                # Forward pass
                self.forward(xs, targets)

                # Backward pass
                self.backward(xs, targets)

                # Update weight matrices
                self.update_grads(optimizer, lr)

                smooth_loss = smooth_loss * 0.999 + self.loss * 0.001

                if progress and n % 1000 == 0:
                    print(f'Epoch {epoch + 1}: {n} / {sequences_amount}: {smooth_loss}')

                n += 1
                self.losses.append(smooth_loss)
                
    def predict(self, start, n):
        """
        Predict a sequence of text based on a starting string.
        
        Inputs
        ------
        start (string): Sequence of characters that contain the start of the predicted text.
        n (int): Length of the prediction.
        
        Outputs
        -------
        txt (string): A string that is predicted by the RNN.
        """
        pass