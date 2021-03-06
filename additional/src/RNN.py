import numpy as np
import matplotlib.pyplot as plt
from module import Module

class RNN(Module):
    r""" Simple recurrent neural network (RNN) class for an input sequence.
    
        This RNN initializes weight and gradients. And contains the forward
        and backward pass. The network is optimized using Adagrad.
        The train method is used to train the network.
        
        Parameter
        ---------
        seq_len (int): Number of layers connected to each other.
        hidden_sz (int): The number of features in the hidden state h.
        vocab_sz (int): The number of possible input and outputs.

        Inputs (forward/ backward)
        --------------------------
        xs (string): List of consecutive characters. The forward and backward pass are calculated for each character.
        targets (string): List of the targets of each character (that is x + 1), since the next
        character is predicted.

        Inputs (predict)
        ----------------
        start (string): Start of a sentence that you want to use as start of your predicton.
        n (int): Length of the prediction you want to perform

        Output (predict)
        ----------------
        txt (string): A string that is predicted by the network.
        """
    
    def __init__(self, seq_len, hidden_sz, vocab_sz):
        self.weight_params = ['Wxh', 'Whh', 'Why', 'Bh', 'By']
        self.hidden_params = ['hs']
        
        super().__init__(seq_len, hidden_sz, vocab_sz, self.weight_params,
        self.hidden_params, embedding_dim)
        
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

            embedded = np.dot(self.params['emb']['weight'].T, x_vec)

            # Calculate the new hidden, which is based on the input and the previous hidden layer
            self.hidden['hs'][i] = np.tanh(np.dot(self.params['Wxh']['weight'], embedded)
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

            # Embed the vector
            embedded = np.dot(self.params['emb']['weight'].T, x_vec)

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

            # Back to embedding
            dq = np.dot(self.params['Wxh']['weight'].t, dhraw)
            self.params['emb']['grad'] += (np.dot(dq, x_vec.T)).T

        # Clip to prevent exploding gradients
        for dparam in self.params.keys():
            np.clip(self.params[dparam]['grad'], -5, 5, out=self.params[dparam]['grad'])
    
    def predict(self, start, n):
        """
        Predict a sequence of text based on a starting string.
        """
        # Vectorize the input
        seed_idx = char_to_idx[start[-1]]
        x = np.zeros((self.vocab_sz, 1))
        x[seed_idx] = 1
        
        txt = [ch for ch in start]
        
        idxes = []
        
        h = self.hidden['hs'][-1]
        
        for _ in range(n):

            # Embed the vector
            embedded = np.dot(self.params['emb']['weight'].T, x) = 1
            
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