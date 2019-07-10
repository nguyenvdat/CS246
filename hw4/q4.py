import numpy as np
import math
from collections import defaultdict
import matplotlib.pyplot as plt
import matplotlib as mpl


def hash_fun(a, b, p, n_buckets, x):
    y = x % p
    hash_val = (a * y + b) % p
    return hash_val % n_buckets + 1


def get_hash_counts(delta, epsilon, stream_file):
    hash_params = []
    p = 123457
    n_buckets = math.ceil(math.e / epsilon)
    with open('q4/data/hash_params.txt') as f:
        for line in f:
            a, b = line.split()
            hash_params.append((int(a), int(b)))
    # the count for each hash function
    hash_counts = [defaultdict(int) for x in hash_params]
    hash_functions = [lambda x: hash_fun(
        a, b, p, n_buckets, x) for (a, b) in hash_params]
    all_words = set()
    with open(stream_file) as f:
        for line in f:
            word = int(line)
            all_words.add(word)
            for i in range(len(hash_counts)):
                hash_counts[i][hash_functions[i](word)] += 1
    counts = defaultdict(int)  # count for each word
    for word in list(all_words):
        counts[word] = min([h[f(word)]
                            for h, f in zip(hash_counts, hash_functions)])
    return counts


def draw_plot(hash_counts, true_counts):
    hash_counts_list = np.array([hash_counts[word]
                                 for word in sorted(hash_counts)])
    true_counts_list = np.array([true_counts[word]
                                 for word in sorted(true_counts)])
    error = (hash_counts_list - true_counts_list) / true_counts_list
    word_freq = true_counts_list/np.sum(true_counts_list)
    mpl.rcParams['agg.path.chunksize'] = 10000

    plt.plot(word_freq, error)
    plt.title('Relative error vs Word frequency')
    plt.xlabel('Word Frequency')
    plt.ylabel('Error')
    plt.xscale('log')
    plt.yscale('log')
    plt.show()


def get_true_counts(true_counts_file):
    true_counts = defaultdict(int)
    with open(true_counts_file) as f:
        for line in f:
            word, count = line.split()
            true_counts[int(word)] = int(count)
    return true_counts


if __name__ == "__main__":
    true_counts_file = 'q4/data/counts_tiny.txt'
    stream_file = 'q4/data/words_stream_tiny.txt'
    delta = math.exp(-5)
    epsilon = math.e * 10**-4
    true_counts = get_true_counts(true_counts_file)
    hash_counts = get_hash_counts(delta, epsilon, stream_file)
    draw_plot(hash_counts, true_counts)
