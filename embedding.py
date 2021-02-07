"""
Functions to embed the sequence.
First the sequences are tokenized and based on the tokenization the embedding
is created.
"""

import numpy as np

# Set-up numpy generator for random numbers
random_number_generator = np.random.default_rng()

# First tokenize the protein sequence (or any sequence) in kmers.
def tokenize(protein_seqs, kmer_sz):
    kmers = set()
    # Loop over protein sequences
    for protein_seq in protein_seqs:
        # Loop over the whole sequence
        for i in range(len(protein_seq) - (kmer_sz - 1)):
            # Add kmers to the set, thus only unique kmers will remain
            kmers.add(protein_seq[i: i + kmer_sz])
            
    # Map kmers for one hot-encoding
    kmer_to_id = dict()
    id_to_kmer = dict()
    
    for ind, kmer in enumerate(kmers):
        kmer_to_id[kmer] = ind
        id_to_kmer[ind] = kmer
        
    vocab_sz = len(kmers)
    
    assert vocab_sz == len(kmer_to_id.keys())
    
    # Tokenize the protein sequence to integers
    tokenized = []
    for protein_seq in protein_seqs:
        sequence = []
        for i in  range(len(protein_seq) - (kmer_sz -1)):
            # Convert kmer to integer
            kmer = protein_seq[i: i + kmer_sz]
            sequence.append(kmer_to_id[kmer])
            
        tokenized.append(sequence)
            
    
    return tokenized, vocab_sz, kmer_to_id, id_to_kmer

# Embedding dictionary to embed the tokenized sequence
def embed(embedding_dim, vocab_sz, rng):
    embedding = {}
    for i in range(vocab_sz):
        # Use random number generator to fill the embedding with embedding_dimension random numbers 
        embedding[i] = rng.random(size=(embedding_dim, 1))
        
    return embedding

if __name__ == '__main__':
    # Globals
    KMER_SIZE = 3 # Choose a Kmer_size (this is a hyperparameter which can be optimized)
    EMBEDDING_DIM = 10 # Also a hyperparameter
    
    # Store myoglobin protein sequence in a list of protein sequences
    protein_seqs = ['MGLSDGEWQLVLNVWGKVEADIPGHGQEVLIRLFKGHPETLEKFDKFKHLKSEDEMKASEDLKKHGATVLTALGGILKKKGHHEAEIKPLAQSHATKHKIPVKYLEFISECIIQVLQSKHPGDFGADAQGAMNKALELFRKDMASNYKELGFQG']

    # Tokenize the protein sequence
    tokenized_seqs, vocab_sz, kmer_to_id, id_to_kmer = tokenize(protein_seqs, KMER_SIZE)
    
    embedding = embed(EMBEDDING_DIM, vocab_sz, random_number_generator)
    
    assert vocab_sz == len(embedding)
    
    # Embed the tokenized protein sequence
    for protein_seq in tokenized_seqs:
        for token in protein_seq:
            print(embedding[token])
            break