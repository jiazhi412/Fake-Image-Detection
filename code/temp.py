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
from tensorflow.python.platform import flags
import saab
import pickle
import keras
import sklearn
from sklearn.cluster import KMeans
from numpy import linalg as LA
import run_file as rf
import numpy as np
import time
import apply_Saab
import os

fake_tag = '_fake_128'
real_tag = '_real_128'
fake_real_size = 8000
train_size = 15200
test_size = 800

flags.DEFINE_string("output_path", None, "The output dir to save params")
flags.DEFINE_string("use_classes", "0,1", "Supported format: 0,1,5-9")
flags.DEFINE_string("kernel_sizes", "5,5", "Kernels size for each stage. Format: '3,3'")
flags.DEFINE_string("num_kernels", "32,64", "Num of kernels for each stage. Format: '4,10'")
flags.DEFINE_float("energy_percent", None, "Energy to be preserved in each stage")
flags.DEFINE_integer("use_num_images", -1, "Num of images used for training")
FLAGS = flags.FLAGS

def main():
    path = sf.get_path(fake_tag,real_tag,fake_real_size)
    train_patches, test_patches, train_labels, test_labels = sf.get_patch(path,train_size,test_size)

    sf.getkernels(path,train_patches,train_labels,FLAGS)
    sf.majority_voting(path,test_labels)

if __name__ == '__main__':
    main()
