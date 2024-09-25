import sys
import pandas as pd
import numpy as np
import mykmeanssp

def is_float(string):
    if string.replace(".", "").replace("-", "").isnumeric():
        return True
    else:
        return False


def read_vectors_from_txt(filename, delimiter=","):
    vectors = []
    with open(filename, 'r') as f:
        for line in f:
            # Split the line based on the delimiter and convert each element to float
            vector = [float(x) for x in line.strip().split(delimiter)]
            vectors.append(vector)
    return vectors


def input_requirements(k, eps, fn1, fn2, iteration='300'):
    if fn1[-3:] == 'txt':
        vectors_1 = pd.DataFrame(read_vectors_from_txt(fn1))
    else:
        vectors_1 = pd.read_csv(fn1, header=None)
    if fn2[-3:] == 'txt':
        vectors_2 = pd.DataFrame(read_vectors_from_txt(fn2))
    else:
        vectors_2 = pd.read_csv(fn2, header=None)
    vectors_1.set_index(0, inplace=True)
    vectors_2.set_index(0, inplace=True)
    vectors = vectors_1.join(vectors_2, how="inner", lsuffix="_1", rsuffix="_2").sort_index()

    # Check for k's validity
    valid = False
    if k.isdigit():
        k = int(k)
        if 1 < k < len(vectors):
            valid = True
    if not valid:
        print("Invalid number of clusters!")
        return vectors, k, iteration, eps, valid

    # Check for iteration's validity
    valid = False
    if iteration.isdigit():
        iteration = int(iteration)
        if 1 < iteration < 1000:
            valid = True
    if not valid:
        print("Invalid maximum iteration!")
        return vectors, k, iteration, eps, valid

    valid = False
    if is_float(eps):
        eps = float(eps)
        if eps >= 0:
            valid = True
    if not valid:
        print("Invalid epsilon!")
        return vectors, k, iteration, eps, valid

    #If all are valid, return all
    return vectors, k, iteration, eps, valid


def k_meanspp(vectors, k, iter, eps):
    np.random.seed(1234)
    n = vectors.shape[0]
    centers = np.zeros((k, vectors.shape[1]))
    chosen_index = np.random.choice(n, 1)
    centers[0] = vectors[chosen_index]

    relevant_vectors = np.delete(np.copy(vectors), chosen_index, axis=0)
    #distance_vec = np.zeros((len(relevant_vectors), k))
    for i in range(k-1):
        distance_vec = np.zeros((len(relevant_vectors), k))
        for j in range(i+1):
            distance_vec[:,j] = np.linalg.norm(relevant_vectors - centers[j], axis = 1)
        min_dist_vec = distance_vec[:, :i+1].min(1)
        weighted_pd = min_dist_vec / np.sum(min_dist_vec)
        choice_of_weighted_ind = np.random.choice(len(weighted_pd), 1, p=weighted_pd)
        centers[i+1] = relevant_vectors[choice_of_weighted_ind]
        relevant_vectors = np.delete(relevant_vectors, choice_of_weighted_ind, axis = 0)

    centroids_indices = np.zeros(k, dtype=int)
    for i in range(k):
        centroids_indices[i] = np.argmin(np.linalg.norm(vectors - centers[i], axis = 1))

    final_centroids = mykmeanssp.fit(centers.tolist(), vectors.tolist(), k, iter, eps, vectors.shape[0], vectors.shape[1])
    #final_centroids = centers
    final_centroids = np.array(final_centroids).reshape(k, vectors.shape[1])

    return centroids_indices, final_centroids


vectors = None
k = None
iterations = None
valid = False
eps = None
if len(sys.argv) == 5:
    vectors, k, iterations, eps, valid = input_requirements(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], '300')
if len(sys.argv) == 6:
    vectors, k, iterations, eps, valid = input_requirements(sys.argv[1], sys.argv[3], sys.argv[4], sys.argv[5], sys.argv[2])

if valid:
    indices, centroids = k_meanspp(np.array(vectors), k, iterations, eps)
    for i in range(len(indices)-1):
        print(str(indices[i]) + ",", end="")
    print(str(indices[-1]))

    for i in range(k):
        cluster_formatted_4_digits = [f"{x:.4f}" for x in centroids[i]]
        for j in range(len(cluster_formatted_4_digits)-1):
            print(str(cluster_formatted_4_digits[j])+",", end="")
        print(str(cluster_formatted_4_digits[-1]))