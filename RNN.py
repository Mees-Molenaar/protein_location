import numpy as np
import matplotlib.pyplot as plt
from module import Module

class RNN(Module):
    r""" Simple recurrent neural network (RNN) class for an input sequence.
    
        This RNN initializes weight and gradients. And contains the forward
        and backward pass. The network is optimized using Adagrad.
        The train method is used to train the network.
        
        Parameters
        ----------
        seq_len : Number of layers connected to each others. 
        hidden_sz : The number of features in the hidden state h.
        vocab_sz : The number of possible inputs and outputs.
        
        
        Inputs (train)
        --------------
        data : Data used to train the network.
        optimizer : The optimizer that is used to train the network.
        lr : The learning rate used to train the network.
        epochs : The number of epochs to train the network.
        progress : If True, shows the progress of training the network.
        
        Inputs (predict)
        ----------------
        start : Start of a sentence that the network uses as initial sequence.
        n : Length of the prediction.   
        
        Output (train)
        --------------
        smooth_loss : The loss of the current trained network.
        
        Output (predict)
        ----------------
        txt : A string that is predicted by the RNN. 
    """
    
    def __init__(self, seq_len, hidden_sz, vocab_sz):
        self.weight_params = ['Wxh', 'Whh', 'Why', 'Bh', 'By']
        self.hidden_params = ['hs']
        
        super().__init__(seq_len, hidden_sz, vocab_sz)
        
    def forward(self, xs, targets):
        """
        Forward pass of the RNN
        """
        
        y_preds = {}

        self.loss = 0

        for i in range(len(xs)):
            x = xs[i]
            x_vec = np.zeros((self.vocab_sz, 1)) # vectorize the input
            x_vec[x] = 1

            # Calculate the new hidden, which is based on the input and the previous hidden layer
            self.hidden['hs'][i] = np.tanh(np.dot(self.params['Wxh']['weight'], x_vec)
                                + np.dot(self.params['Whh']['weight'], self.hidden['hs'][i - 1]) 
                                + self.params['Bh']['bias'])
            # Predict y
            y_preds[i] = np.dot(self.params['Why']['weight'], self.hidden['hs'][i]) + \
            self.params['By']['bias'] 

            self.sm_ps[i] = np.exp(y_preds[i]) / np.sum(np.exp(y_preds[i])) # Softmax probabilty
            self.loss += -np.log(self.sm_ps[i][targets[i], 0]) #Negative loss likelyhood

        self.hidden['hs'][-1] = self.hidden['hs'][len(xs) - 1]
        
    def backward(self, xs, targets):
        """
        Backward pass of the RNN
        """
        self.init_grads()
    
        # Initialize empty next hidden layer for the first backprop
        dhnext = np.zeros_like(self.hidden['hs'][0])

        for i in reversed(range(len(xs))):
            # X to vector
            x = xs[i]    
            x_vec = np.zeros((self.vocab_sz, 1))
            x_vec[x] = 1

            dy = np.copy(self.sm_ps[i])
            dy[targets[i]] -= 1 # backprop into y. see http://cs231n.github.io/neural-networks-case-study/#grad if confused here

            self.params['By']['grad'] += dy   
            self.params['Why']['grad'] += np.dot(dy, self.hidden['hs'][i].T)
            dh = np.dot(self.params['Why']['weight'].T, dy) + dhnext
            dhraw = (1 - self.hidden['hs'][i] * self.hidden['hs'][i]) * dh  
            self.params['Wxh']['grad'] += np.dot(dhraw, x_vec.T)
            self.params['Whh']['grad'] += np.dot(dhraw, self.hidden['hs'][i-1].T)
            self.params['Bh']['grad'] += dhraw
            dhnext = np.dot(self.params['Whh']['weight'].T, dhraw)

        # Clip to prevent exploding gradients
        for dparam in self.params.keys():
            np.clip(self.params[dparam]['grad'], -5, 5, out=self.params[dparam]['grad'])
    
    def predict(self, start, n):
        """
        Predict a sequence of text based on a starting string.
        """
        seed_idx = char_to_idx[start[-1]]
        x = np.zeros((self.vocab_sz, 1))
        x[seed_idx] = 1
        
        txt = [ch for ch in start]
        
        idxes = []
        
        h = self.hidden['hs'][-1]
        
        for _ in range(n):
            
            # Calculate the hidden
            h = np.tanh(np.dot(self.params['Wxh']['weight'], x) \
                        + np.dot(self.params['Whh']['weight'], h) \
                        + self.params['Bh']['bias'])
            # Calculate y
            y = np.dot(self.params['Why']['weight'], h) \
                + self.params['By']['bias']

            sm_p = np.exp(y) / np.sum(np.exp(y)) # Softmax probabilty
            # Determine character based on weighted probability (is using the softmax probability)
            idx = np.random.choice(range(self.vocab_sz), p=sm_p.ravel())
            idxes.append(idx)
            
            # Save X for next iteration
            x = np.zeros((self.vocab_sz, 1))
            x[idx] = 1
            
        prediction = [idx_to_char[idx] for idx in idxes]
        
        txt += prediction
        
        return txt