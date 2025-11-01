import numpy as np
from tokenizers import Tokenizer
import sys
from scipy.spatial.distance import jensenshannon
from collections import Counter
import editdistance 

# Corpus-level Jaccard (CJ)
def corpus_jaccard(tokens_1, tokens_2):
    """
    Calculate the corpus-level Jaccard index between two sets of tokenized corpora.
    
    Args:
        tokens_1 (list of list): Tokenized sentences from corpus 1.
        tokens_2 (list of list): Tokenized sentences from corpus 2.
    
    Returns:
        float: Corpus-level Jaccard index.
    """
    set_1 = set(token for sentence in tokens_1 for token in sentence)
    set_2 = set(token for sentence in tokens_2 for token in sentence)
    
    intersection = len(set_1.intersection(set_2))
    union = len(set_1.union(set_2))
    
    return intersection / union if union > 0 else 0.0

# Mean Pairwise Jaccard (MPJ)
def mean_pairwise_jaccard(tokens_1, tokens_2):
    """
    Calculate the mean pairwise Jaccard index between two sets of tokenized corpora. 
    This calculates the Jaccard index for each pair of sentences from the two corpora and returns the mean.

    Args:
        tokens_1 (list of list): Tokenized sentences from corpus 1.
        tokens_2 (list of list): Tokenized sentences from corpus 2.

    Returns:
        float: Mean pairwise Jaccard index.
    """
    jaccard_indices = []
    
    for sent_1, sent_2 in zip(tokens_1, tokens_2):
        set_1 = set(sent_1)
        set_2 = set(sent_2)
        
        intersection = len(set_1.intersection(set_2))
        union = len(set_1.union(set_2))
        
        jaccard_index = intersection / union if union > 0 else 0.0
        jaccard_indices.append(jaccard_index)

    return np.mean(jaccard_indices) if jaccard_indices else 0.0


def main():
    tokenizer_path = sys.argv[1]
    corpus_1_path = sys.argv[2]
    corpus_2_path = sys.argv[3]

    # Load the tokenizer
    try:
        tokenizer = Tokenizer.from_file(tokenizer_path)
    except Exception as e:
        print(f"Error loading tokenizer: {e}")
        sys.exit(1)

    # Load the corpora
    try:
        with open(corpus_1_path, 'r', encoding='utf-8') as f:
            corpus_1 = f.readlines()
        with open(corpus_2_path, 'r', encoding='utf-8') as f:
            corpus_2 = f.readlines()
    except Exception as e:
        print(f"Error loading corpora: {e}")
        sys.exit(1)
        
    assert len(corpus_1) == len(corpus_2), "Corpora must have the same number of lines."

    print(f"Loaded {len(corpus_1)} sentences from each corpus. Tokenizing...")

    # Tokenize the corpora
    tokens_1 = [tokenizer.encode(line.strip()).tokens for line in corpus_1]
    tokens_2 = [tokenizer.encode(line.strip()).tokens for line in corpus_2]

    print(f"Loaded and tokenized {len(tokens_1)} sentences from each corpus. Calculating overlap...")

    # Calculate corpus-level Jaccard index
    cj_index = corpus_jaccard(tokens_1, tokens_2)
    print(f"Corpus-level Jaccard Index: {cj_index:.4f}")

    # Calculate mean pairwise Jaccard index
    mpj_index = mean_pairwise_jaccard(tokens_1, tokens_2)
    print(f"Mean Pairwise Jaccard Index: {mpj_index:.4f}")

    print()

if __name__ == "__main__":
    main()