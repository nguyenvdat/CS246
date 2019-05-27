# Authors: Jessica Su, Wanzi Zhou, Pratyaksh Sharma, Dylan Liu, Ansh Shukla

import numpy as np
import random
import time
import pdb
import unittest
from PIL import Image
import matplotlib.pyplot as plt
import os

# Finds the L1 distance between two vectors
# u and v are 1-dimensional np.array objects
# TODO: Implement this
def l1(u, v):
    return np.sum(np.abs(u - v))

# Loads the data into a np array, where each row corresponds to
# an image patch -- this step is sort of slow.
# Each row in the data is an image, and there are 400 columns.
def load_data(filename):
    return np.genfromtxt(filename, delimiter=',')

# Creates a hash function from a list of dimensions and thresholds.
def create_function(dimensions, thresholds):
    def f(v):
        boolarray = [v[dimensions[i]] >= thresholds[i] for i in range(len(dimensions))]
        return "".join(map(str, map(int, boolarray)))
    return f

# Creates the LSH functions (functions that compute L K-bit hash keys).
# Each function selects k dimensions (i.e. column indices of the image matrix)
# at random, and then chooses a random threshold for each dimension, between 0 and
# 255.  For any image, if its value on a given dimension is greater than or equal to
# the randomly chosen threshold, we set that bit to 1.  Each hash function returns
# a length-k bit string of the form "0101010001101001...", and the L hash functions 
# will produce L such bit strings for each image.
def create_functions(k, L, num_dimensions=400, min_threshold=0, max_threshold=255):
    functions = []
    for i in range(L):
        dimensions = np.random.randint(low = 0, 
                                   high = num_dimensions,
                                   size = k)
        thresholds = np.random.randint(low = min_threshold, 
                                   high = max_threshold + 1, 
                                   size = k)

        functions.append(create_function(dimensions, thresholds))
    return functions

# Hashes an individual vector (i.e. image).  This produces an array with L
# entries, where each entry is a string of k bits.
def hash_vector(functions, v):
    return np.array([f(v) for f in functions])

# Hashes the data in A, where each row is a datapoint, using the L
# functions in "functions."
def hash_data(functions, A):
    return np.array(list(map(lambda v: hash_vector(functions, v), A)))

# Retrieve all of the points that hash to one of the same buckets 
# as the query point.  Do not do any random sampling (unlike what the first
# part of this problem prescribes).
# Don't retrieve a point if it is the same point as the query point.
def get_candidates(hashed_A, hashed_point, query_index):
    return filter(lambda i: i != query_index and \
        any(hashed_point == hashed_A[i]), range(len(hashed_A)))

# Sets up the LSH.  You should try to call this function as few times as 
# possible, since it is expensive.
# A: The dataset.
# Return the LSH functions and hashed data structure.
def lsh_setup(A, k = 24, L = 10):
    functions = create_functions(k = k, L = L)
    hashed_A = hash_data(functions, A)
    return (functions, hashed_A)

# Run the entire LSH algorithm
def lsh_search(A, hashed_A, functions, query_index, num_neighbors = 10):
    hashed_point = hash_vector(functions, A[query_index, :])
    candidate_row_nums = get_candidates(hashed_A, hashed_point, query_index)
    
    distances = map(lambda r: (r, l1(A[r], A[query_index])), candidate_row_nums)
    best_neighbors = sorted(distances, key=lambda t: t[1])[:num_neighbors]

    return [t[0] for t in best_neighbors]

# Plots images at the specified rows and saves them each to files.
def plot(A, row_nums, base_filename):
    plt.rcParams.update({'font.size': 7})
    fig,ax = plt.subplots(3,5, figsize=(5, 4))
    for i, row_num in enumerate(row_nums):
        patch = np.reshape(A[row_num, :], [20, 20])
        im = Image.fromarray(patch)
        if im.mode != 'RGB':
            im = im.convert('RGB')
        # im.save(base_filename + "-" + str(row_num) + ".png")
        if i == 0:
            ax[0][0].imshow(im)
            ax[0][0].set_title("Original row: " + str(row_num))
        else:
            ax[(i - 1)%2 + 1][(i - 1)//2].imshow(im)
            ax[(i - 1)%2 + 1][(i - 1)//2].set_title('Row: ' + str(row_num))
    for i in range(3):
        for j in range(5):
            ax[i][j].axis('off')
    fig.savefig('../tex/' + base_filename + ".png")

# Finds the nearest neighbors to a given vector, using linear search.
def linear_search(A, query_index, num_neighbors):
    #TODO
    distances = np.sum(np.abs(A - A[query_index]), axis=1)
    distances[query_index] = 1e6
    return list(np.argpartition(-distances, -num_neighbors)[-num_neighbors:])

# TODO: Write a function that computes the error measure
def compute_error(A, query_indexes, linear_neighbors, lsh_neighbors):
    def error_one_query(query_index, neighbors):
        distance = sum(map(lambda i: l1(A[query_index], A[i]), neighbors))
        return distance
    distance_ratios = map(lambda i: error_one_query(query_indexes[i], lsh_neighbors[i])/error_one_query(query_indexes[i], linear_neighbors[i]), range(len(lsh_neighbors)))
    return sum(distance_ratios)/len(lsh_neighbors)


# TODO: Solve Problem 4
def problem4(A):
    # A = load_data('q4/data/patches.csv')
    problem4_1(A)

def problem4_1(A):
    num_neighbors = 3
    query_indexes = [100 * j for j in range(1, 11)]
    start = time.time()
    linear_neighbors = list(map(lambda query_index: linear_search(A, query_index, num_neighbors), query_indexes))
    end = time.time()
    print('Time linear search: ' + str(end - start))

    L = 10
    k = 24
    functions, hashed_A = lsh_setup(A, k, L)
    start = time.time()
    lsh_neighbors = list(map(lambda query_index: lsh_search(A, hashed_A, functions, query_index, num_neighbors), query_indexes))
    end = time.time()
    print('Time lsh search: ' + str(end - start))

def problem4_2(A):
    num_neighbors = 3
    query_indexes = [100 * j for j in range(1, 11)]
    start = time.time()
    linear_neighbors = list(map(lambda query_index: linear_search(A, query_index, num_neighbors), query_indexes))
    end = time.time()
    print('Time linear search: ' + str(end - start))
    Ls = np.arange(10, 21, 2)
    k = 24
    errors = []
    for L in Ls:
        functions, hashed_A = lsh_setup(A, k, L)
        lsh_neighbors = list(map(lambda query_index: lsh_search(A, hashed_A, functions, query_index, num_neighbors), query_indexes))
        error = compute_error(A, query_indexes, linear_neighbors, lsh_neighbors)
        errors.append(error)
    plt.plot(Ls, errors, 'ro')
    plt.axis([0, 6, 5, 25])
    plt.ylabel('Error')
    plt.xlabel('L')
    plt.show()

    L = 10
    ks = [16, 18, 20, 22, 24]
    errors = []
    for k in ks:
        print(k)
        functions, hashed_A = lsh_setup(A, k, L)
        lsh_neighbors = list(map(lambda query_index: lsh_search(A, hashed_A, functions, query_index, num_neighbors), query_indexes))
        error = compute_error(A, query_indexes, linear_neighbors, lsh_neighbors)
        errors.append(error)
    plt.plot(ks, errors, 'ro')
    plt.axis([15, 25, 1, 1.05])
    plt.ylabel('Error')
    plt.xlabel('k')
    plt.show()

#### TESTS #####

class TestLSH(unittest.TestCase):
    def test_l1(self):
        u = np.array([1, 2, 3, 4])
        v = np.array([2, 3, 2, 3])
        self.assertEqual(l1(u, v), 4)

    def test_hash_data(self):
        f1 = lambda v: sum(v)
        f2 = lambda v: sum([x * x for x in v])
        A = np.array([[1, 2, 3], [4, 5, 6]])
        self.assertEqual(f1(A[0,:]), 6)
        self.assertEqual(f2(A[0,:]), 14)

        functions = [f1, f2]
        self.assertTrue(np.array_equal(hash_vector(functions, A[0, :]), np.array([6, 14])))
        self.assertTrue(np.array_equal(hash_data(functions, A), np.array([[6, 14], [15, 77]])))

    def test_error(self):
        A = np.array([[1, 2, 3], [4, 5, 6], [0, 0, 0], [1, 2, 0]])
        query_indexes = [0, 1]
        linear_neighbors = [[2, 3], [0, 3]]
        lsh_neighbors = [[1, 2], [0, 2]]
        self.assertEqual(error(A, query_indexes, linear_neighbors, lsh_neighbors), 59/42)


    ### TODO: Write your tests here (they won't be graded, 
    ### but you may find them helpful)


if __name__ == '__main__':
#    unittest.main() ### TODO: Uncomment this to run tests
    # test = TestLSH()
    # test.test_l1()
    # test.test_error()
    # plt.plot([1,2,3,4], [1,4,9,16], 'ro')
    # plt.axis([0, 6, 0, 20])
    # plt.ylabel('Error')
    # plt.xlabel('L')
    # plt.show()
    A = load_data('q4/data/patches.csv')
    plot(A, 200, 'test')
    # problem4()
