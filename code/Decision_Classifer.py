import pickle
import keras
import data
import numpy as np
import sklearn
import cv2
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from numpy import linalg as LA
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import euclidean_distances
import time
import os
import Saab_function as sf

fake_tag = '_fake_128'
real_tag = '_real_128'
fake_real_size = 8000
split_ratio = '0_4'
train_size = 9600
test_size = 6400

def main():
    path = sf.get_path(fake_tag,real_tag,fake_real_size,split_ratio)
    print(path)
    train_patches, test_patches, train_labels, test_labels = sf.get_patch(path,train_size,test_size)
    sf.majority_voting(path,test_labels)

if __name__ == '__main__':
    main()
