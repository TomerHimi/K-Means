"""
Created on Tue Nov 10 19:00:26 2020
@author: Tomer Himi

"""
import sys
import numpy as np
from scipy.io import wavfile

def train(sample, centroids, old_centroids, new_centroids): 
    """fuction for K-Means implementaion on a wav file
    parm: sample: the name of wav file
    parm: centroids: ndarray of the given centroids
    parm: old_centroids: array for storing old centroids
    parm: new_centroids: array for storing new_centrids
    type: sample: str
    type: centroids: ndarray
    type: old_centroids: ndarray
    type: new_centroids: ndarray
    return: flag for checking convergence
    rtype: converge: bool
    """
    clusters = np.zeros(sample.shape)
    distances = np.zeros((sample.shape[0], centroids.shape[0]))
    for i in range(centroids.shape[0]):  #calc the distance to every centroid
        distances[:, i] = np.linalg.norm(sample - new_centroids[i], axis = 1)
    clusters = np.argmin(distances, axis = 1)
    old_centroids = np.copy(new_centroids)
    for i in range(centroids.shape[0]):  #update of centroids
        temp_cluster = sample[clusters == i]
        if temp_cluster.shape[0] > 0:
            new_centroids[i] = np.round(np.mean(temp_cluster, axis = 0))
        else:
            new_centroids[i] = old_centroids[i]
    return np.linalg.norm(new_centroids - old_centroids) == 0  #check convergence of K-Means

def main():
    sample, centroids = sys.argv[1], sys.argv[2]
    fs, y = wavfile.read(sample)  #reading_sample_wav
    centroids = np.loadtxt(centroids) #reading_centroids
    output_file = open(f"output.txt","w")
    iterations = 0
    converge = False 
    old_centroids = np.zeros(centroids.shape)  #store old centers
    new_centroids = np.copy(centroids)  #store new centers
    while iterations < 30 and not converge:  #K-Means implementation
        converge = train(y, centroids, old_centroids, new_centroids) 
        iterations += 1
        output_file.write(f"[iter {iterations-1}]:{','.join([str(i) for i in new_centroids])}\n")  #writes the centroids to output.txt
        print(f"[iter {iterations-1}]:{','.join([str(i) for i in new_centroids])}")
    
if __name__ == "__main__":
    main()