#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np
import matplotlib.pyplot as plt


# In[3]:


# Random seed
np.random.seed(420)


# In[4]:


data = open('shakespeare.txt', 'r').read()
#data = open('nescio.txt', 'r').read()
chars = list(set(data))
data_size = len(data)
vocab_size = len(chars)


# In[5]:


print(f'Data set is length {data_size}')
print(f'Vocab set is length {vocab_size}')


# In[6]:


# Simple character embedding
char_to_idx = {char:i for i, char in enumerate(chars)}
idx_to_char = {i:char for i, char in enumerate(chars)}


# In[7]:


# hyperparameters
hidden_size = 100
seq_length = 25
learning_rate = 1e-1
epochs = 100


# In[8]:


# weight parameters
Wxh = np.random.randn(hidden_size, vocab_size) * 0.01 # I think its times 0.01 to avoid exploding gradients
Whh = np.random.randn(hidden_size, hidden_size) * 0.01
Why = np.random.randn(vocab_size, hidden_size) * 0.01

# bias
bh = np.zeros((hidden_size, 1))
by = np.zeros((vocab_size, 1))


# What should happen:
# 
# x = data[0]
# y = data[1]
# 
# h = np.tanh(np.dot(Wxh, x) + np.dot(Whh, h))
# 
# y_pred = np.dot(Why, h)
# 
# loss = y - y_pred (Simplified, we will use Cross-entropy loss for it)

# In[9]:


def forward(xs, hidden=None):
    """Calculate the forward pass"""
    
    y_preds = {}
    hs = {}
        
    hs[-1] = np.copy(hidden)
    
    for i in range(len(xs)):
        x = xs[i]
        x_vec = np.zeros((vocab_size, 1)) # vectorize the input
        x_vec[x] = 1

        # Calculate the new hidden, which is based on the input and the previous hidden layer
        hs[i] = np.tanh(np.dot(Wxh, x_vec) + np.dot(Whh, hidden) + bh)
        # Predict y
        y_preds[i] = np.dot(Why, hs[i]) + by
    
    prev_hidden = hs[-1]

    return y_preds, prev_hidden, hs


# In[10]:


def loss_function(y_preds, target): 
    """Calculate the cross-entropy loss, 
    which is based on the softmax functon and the negative log likelyhood."""
    
    softmax_probs = {}
    loss = 0
    
    for i in range(len(y_preds)):
        softmax_probs[i] = np.exp(y_preds[i]) / np.sum(np.exp(y_preds[i])) # Softmax probabilty

        loss += -np.log(softmax_probs[i][target[i], 0]) #Negative loss likelyhood
    
    return softmax_probs, loss


# In[11]:


def initialize_gradients():
    """Initialize the gradients to 0."""
    
    dWxh, dWhh, dWhy = np.zeros_like(Wxh), np.zeros_like(Whh), np.zeros_like(Why)
    dby, dbh = np.zeros_like(by), np.zeros_like(bh)

    return dWxh, dWhh, dWhy, dby, dbh


# In[12]:


def backward(softmax_probs, hs, xs, dhnext=None):
    """Perform the backward pass"""
    
    # Initialize gradients
    dWxh, dWhh, dWhy, dby, dbh = initialize_gradients()
    
    # Set up dhnext if it is first time backpop
    if not dhnext:
        dhnext = np.zeros_like(hs[0])
    
    for i in reversed(range(len(xs))):
        # X to vector
        x = xs[i]
        x_vec = np.zeros((vocab_size, 1))
        x_vec[x] = 1

        dy = np.copy(softmax_probs[i])
        dy[x] -= 1 # backprop into y. see http://cs231n.github.io/neural-networks-case-study/#grad if confused here

        dby += dy   
        dWhy += np.dot(dy, hs[i].T)
        ds = (np.dot(dWhy.T, dy)  + dhnext) * (1 - hs[i] * hs[i])
        dWxh += np.dot(ds, x_vec.T)
        dWhh += np.dot(ds, hs[i-1].T)
        dbh += ds
        dhnext = np.dot(Whh.T, ds)

        # Clip to prevent exploding gradients

        for dparam in [dWhy, dWxh, dWhh, dbh, dby]:
            np.clip(dparam, -5, 5, out=dparam)
    
    return dWxh, dWhh, dWhy, dby, dbh, dhnext


# In[13]:


def update_gradients(Wxh, Whh, Why, by, bh, dWxh, dWhh, dWhy, dby, dbh, learning_rate):
    """Update the gradients using stochastic gradient descent."""
    Wxh -= learning_rate * dWxh
    Whh -= learning_rate * dWhh
    Why -= learning_rate * dWhy
    bh -= learning_rate * dbh
    by -= learning_rate * dby
    
    return Wxh, Whh, Why, bh, by


# In[24]:


def RNN(data, seq_length, epochs):
    """Perform RNN over the data"""
    data_len = len(data)
    
    # Initialize weights
    Wxh = np.random.randn(hidden_size, vocab_size) * 0.01
    Whh = np.random.randn(hidden_size, hidden_size) * 0.01
    Why = np.random.randn(vocab_size, hidden_size) * 0.01

    # bias
    bh = np.zeros((hidden_size, 1))
    by = np.zeros((vocab_size, 1))

    # Initialize gradients
    dWxh, dWhh, dWhy, dby, dbh = initialize_gradients()
    
    # Store losses
    losses = []
    smooth_loss = -np.log(1.0/vocab_size)*seq_length # loss at iteration 0
    
    # Loop over the epochs
    for i in range(epochs):
        n = 0
        # Loop over the amount of sequences
        sequences_amount = int(data_len // seq_length)
        for j in range(sequences_amount):
            
            start_pos = seq_length * j
            # Reset and go from the start of data
            if n == 0 or start_pos + seq_length + 1 >= data_len:
                prev_hidden = np.zeros((hidden_size, 1))
                
            # Embed the inputs and targets
            xs = [char_to_idx[ch] for ch in data[start_pos:start_pos+seq_length]]
            targets = [char_to_idx[ch] for ch in data[start_pos+1:start_pos+seq_length+1]]

            # Forward pass
            y_preds, prev_hidden, hs = forward(xs, prev_hidden)

            #Loss
            softmax_probs, loss = loss_function(y_preds, targets)

            #Backward
            dWxh, dWhh, dWhy, dby, dbh, dhnext = backward(softmax_probs, hs, xs)

            # Update gradients
            Wxh, Whh, Why, bh, by = update_gradients(Wxh, Whh, Why, by, bh, dWxh, dWhh, dWhy, dby, dbh, learning_rate)
            
            smooth_loss = smooth_loss * 0.999 + loss * 0.001
            
            if n % 10000 == 0:
                losses.append(smooth_loss)
                print(f'{i + 1}: {n} / {sequences_amount}: {smooth_loss}')
                
            n += 1
            
        print(f'Finished epoch {i + 1}.')
            
    return losses


# In[25]:


epochs = 10
losses = RNN(data, seq_length, epochs)


# In[ ]:


plt.plot(losses)

