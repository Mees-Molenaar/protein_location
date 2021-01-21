import numpy as np
import matplotlib.pyplot as plt
from module import Module

# Sigmoid helper function
def sigmoid(x):
    return 1 / (1 + np.exp(-1 * x))

# Derivative sigmoid helper function
def dsigmoid(x):
    return sigmoid(x) * (1 - sigmoid(x))

class LSTM(Module):
    
    def __init__(self, seq_len, hidden_sz, vocab_sz):
        self.weight_params = ['Whf', 'Wxf', 'Whi', 'Wxi', 'Whc', 'Wxc'
                             , 'Who', 'Wxo', 'Why', 'Bf', 'Bi', 'Bc'
                             , 'Bo', 'By']    
        self.hidden_params = ['hs', 'ct', 'ho', 'hc', 'hi', 'hf']
        
        super().__init__(seq_len, hidden_sz, vocab_sz)

    def forward(self, xs, targets):
        """
        Forward pass LSTM
        """
        
        y_preds = {}
        
        self.loss = 0
        
        for i in range(len(xs)):
            # Vectorize the input
            x = xs[i]
            x_vec = np.zeros((self.vocab_sz, 1)) 
            x_vec[x] = 1
            
            
            # Calculate new hidden and cell state
            self.hidden['hf'][i] = np.dot(self.params['Whf']['weight'], self.hidden['hs'][i - 1]) \
                                   + np.dot(self.params['Wxf']['weight'], x_vec) \
                                   + self.params['Bf']['bias']
            
            self.hidden['hi'][i] = np.dot(self.params['Whi']['weight'], self.hidden['hs'][i - 1]) \
                                   + np.dot(self.params['Wxi']['weight'], x_vec) \
                                   + self.params['Bi']['bias']
            
            self.hidden['hc'][i] = np.dot(self.params['Whc']['weight'], self.hidden['hs'][i - 1]) \
                                   + np.dot(self.params['Wxc']['weight'], x_vec) \
                                   + self.params['Bc']['bias']
            
            self.hidden['ho'][i] = np.dot(self.params['Who']['weight'], self.hidden['hs'][i - 1]) \
                                   + np.dot(self.params['Wxo']['weight'], x_vec) \
                                   + self.params['Bo']['bias']
            
            
            f_t = sigmoid(self.hidden['hf'][i])
            i_t = sigmoid(self.hidden['hi'][i])
            cwave_t = np.tanh(self.hidden['hc'][i])
            o_t = sigmoid(self.hidden['ho'][i])
            
            self.hidden['ct'][i] = f_t * self.hidden['ct'][i - 1] + i_t * cwave_t
            self.hidden['hs'][i] = o_t * np.tanh(self.hidden['ct'][i])
            
            # Predict y
            y_preds[i] = np.dot(self.params['Why']['weight'], self.hidden['hs'][i]) \
                         + self.params['By']['bias']
            
            self.sm_ps[i] = np.exp(y_preds[i]) / np.sum(np.exp(y_preds[i])) # Softmax probability
            self.loss += -np.log(self.sm_ps[i][targets[i], 0]) # Negative loss likelyhood
            
        self.hidden['ct'][-1] = self.hidden['ct'][len(xs) - 1]
        self.hidden['hs'][-1] = self.hidden['hs'][len(xs) - 1]
        
    def backward(self, xs, targets):
        """
        Backward pass for LSTM
        """
        
        # Initialize gradients
        self.init_grads()
        
        # Start with an empty next layer for the cell state and hidden state
        dhnext = np.zeros_like(self.hidden['hs'][0])
        dcnext = np.zeros_like(self.hidden['ct'][0])
        
        # Loop over inputs and calculate gradients
        for i in reversed(range(len(xs))):
            # One hot encoding
            x = xs[i]
            x_vec = np.zeros((self.vocab_sz, 1))
            x_vec[x] = 1
            
            dy = np.copy(self.sm_ps[i])
            dy[targets[i]] -= 1 # backprop into y. see http://cs231n.github.io/neural-networks-case-study/#grad if confused here
            
            self.params['By']['grad'] += dy
            self.params['Why']['grad'] += np.dot(dy, self.hidden['hs'][i].T)
            # h branches to ouput, and next layer. Therefore, we need the gradient of next layer is added
            dh = np.dot(self.params['Why']['weight'].T, dy) + dhnext 
            
            # Calculations for o
            do = dh * np.tanh(self.hidden['ct'][i]) # Weet dit niet zeker
            do = dsigmoid(self.hidden['ho'][i]) * do
            
            self.params['Wxo']['grad'] += np.dot(do, x_vec.T)
            self.params['Who']['grad'] += np.dot(do, self.hidden['hs'][i-1].T)
            self.params['Bo']['grad'] += do
            
            # Calculations for dc
            dc= dh * sigmoid(self.hidden['ho'][i]) 
            dc = (1-np.square(np.tanh(self.hidden['ct'][i]))) * dc #Weet dit ook niet zeker
            dc = dc + dcnext
            
            # Calculation dcwave
            dcwave_t = sigmoid(self.hidden['hi'][i]) * dc
            # C branches to next layer, therefore we need the gradient of that layer added.
            dcwave_t = dcwave_t * (1-np.square(np.tanh(self.hidden['hc'][i])))
            
            self.params['Wxc']['grad'] += np.dot(dcwave_t, x_vec.T)
            self.params['Whc']['grad'] += np.dot(dcwave_t, self.hidden['hs'][i-1].T)
            self.params['Bc']['grad'] += dcwave_t
            
            # Calculating di
            di = sigmoid(self.hidden['hc'][i]) * dc
            di = di * dsigmoid(self.hidden['hi'][i])
            
            self.params['Wxi']['grad'] += np.dot(di, x_vec.T)
            self.params['Whi']['grad'] += np.dot(di, self.hidden['hs'][i-1].T)
            self.params['Bi']['grad'] += di
            
            #Calculating df
            df = self.hidden['ct'][i-1] * dc
            df = dsigmoid(self.hidden['hf'][i]) * df
            
            self.params['Wxf']['grad'] += np.dot(df, x_vec.T)
            self.params['Whf']['grad'] += np.dot(df, self.hidden['hs'][i-1].T)
            self.params['Bf']['grad'] += df
            
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
        
        hs = self.hidden['hs'][-1]
        ct = self.hidden['ct'][-1]
        
        for _ in range(n):
            # Calculate new hidden and cell state
            hf = np.dot(self.params['Whf']['weight'], hs) \
                                   + np.dot(self.params['Wxf']['weight'], x) \
                                   + self.params['Bf']['bias']

            hi = np.dot(self.params['Whi']['weight'], hs) \
                                   + np.dot(self.params['Wxi']['weight'], x) \
                                   + self.params['Bi']['bias']

            hc = np.dot(self.params['Whc']['weight'], hs) \
                                   + np.dot(self.params['Wxc']['weight'], x) \
                                   + self.params['Bc']['bias']

            ho = np.dot(self.params['Who']['weight'], hs) \
                                   + np.dot(self.params['Wxo']['weight'], x) \
                                   + self.params['Bo']['bias']


            f_t = sigmoid(hf)
            i_t = sigmoid(hi)
            cwave_t = np.tanh(hc)
            o_t = sigmoid(ho)

            ct = f_t * ct + i_t * cwave_t
            hs = o_t * np.tanh(ct)

            # Predict y
            y = np.dot(self.params['Why']['weight'], hs) \
                         + self.params['By']['bias']
            
            sm_p = np.exp(y) / np.sum(np.exp(y)) # Softmax probability
            # Determine character based on weighted probability
            idx = np.random.choice(range(self.vocab_sz), p=sm_p.ravel())
            idxes.append(idx)
            
            # Save X for next iteration
            x = np.zeros((self.vocab_sz,1 ))
            x[idx] = 1
            
        prediction = [idx_to_char[idx] for idx in idxes]
        
        txt += prediction
        
        return txt