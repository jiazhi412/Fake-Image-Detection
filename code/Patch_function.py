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
from sklearn.model_selection import train_test_split

def extract_patch(fake_tag,real_tag,fake_real_size,path):
    # 1.proprocess
    # real image
    foldername = "../Dataset/celeba-dataset/" + str(fake_real_size) + real_tag
    facial_cor = rf.collect_coordinates(foldername)
    print("real facial cor shape:", facial_cor.shape)
    patches_all_img = rf.extract_patch(foldername, facial_cor)
    np.save(path + '/ten_real_patch_' + str(fake_real_size), patches_all_img)

    # fake image
    if fake_tag == '_fake_1024':
        foldername = "../Dataset/ProGAN_generated_images/" + str(fake_real_size) + fake_tag
    else:
        foldername = "../Dataset/ProGAN_128-20190523T053406Z-001/" + str(fake_real_size) + fake_tag
    facial_cor = rf.collect_coordinates(foldername)
    print("fake facial cor shape:", facial_cor.shape)
    patches_all_img = rf.extract_patch(foldername, facial_cor)
    np.save(path + '/ten_fake_patch_' + str(fake_real_size), patches_all_img)

def comb_patch(path,fake_real_size): #interlude real and fake
    fake_patches = np.load(path + '/ten_fake_patch_' + str(fake_real_size) + '.npy')
    real_patches = np.load(path + '/ten_real_patch_' + str(fake_real_size) + '.npy')
    # create labels for patches, 1:fake, 0:real
    fake_labels = np.ones((fake_patches.shape[0], 11))  # 500
    real_labels = np.zeros((real_patches.shape[0], 11))  # 500

    patch_comb = []
    label_comb = []
    for i in range(fake_real_size):
        patch_comb.append(fake_patches[i, :])
        patch_comb.append(real_patches[i, :])
        label_comb.append(fake_labels[i, :])
        label_comb.append(real_labels[i, :])

    patch_comb = np.array(patch_comb)
    label_comb = np.array(label_comb)
    print(patch_comb.shape, label_comb.shape)

    np.save(path+'/patch_' + str(fake_real_size*2), patch_comb)
    np.save(path+'/label_' + str(fake_real_size*2), label_comb)

def split_patch_built_in(path,fake_real_size,train_size,test_size):
    patches = np.load(path + '/patch_' + str(fake_real_size*2) + '.npy')
    labels = np.load(path + '/label_' + str(fake_real_size*2) + '.npy')

    train_patches, test_patches, train_labels, test_labels = train_test_split(patches, labels, test_size=test_size)
    print(train_patches.shape, '\n', test_patches.shape, '\n', train_labels.shape, '\n', test_labels.shape)

    np.save(path+'/patch_' + str(train_size), train_patches)
    np.save(path+'/patch_' + str(test_size), test_patches)
    np.save(path+'/label_' + str(train_size), train_labels)
    np.save(path+'/label_' + str(test_size), test_labels)

def split_patch(path,fake_real_size,train_size,test_size):
    patches = np.load(path + '/patch_' + str(fake_real_size*2) + '.npy')
    labels = np.load(path + '/label_' + str(fake_real_size*2) + '.npy')

    train_patches = []
    test_patches = []
    train_labels = []
    test_labels = []
    count = 0
    while count < train_size:
        train_patches.append(patches[count, :])
        train_labels.append(labels[count, :])
        count = count + 1

    while count < fake_real_size*2:
        test_patches.append(patches[count, :])
        test_labels.append(labels[count, :])
        count = count + 1

    train_patches = np.array(train_patches)
    train_labels = np.array(train_labels)
    test_patches = np.array(test_patches)
    test_labels = np.array(test_labels)

    print(train_patches.shape, '\n', test_patches.shape, '\n', train_labels.shape, '\n', test_labels.shape)

    np.save(path+'/patch_' + str(train_size), train_patches)
    np.save(path+'/patch_' + str(test_size), test_patches)
    np.save(path+'/label_' + str(train_size), train_labels)
    np.save(path+'/label_' + str(test_size), test_labels)

def get_patch(path,train_size,test_size):
    train_patches = np.load(path + '/patch_' + str(train_size) + '.npy')
    test_patches = np.load(path + '/patch_' + str(test_size) + '.npy')
    train_labels = np.load(path + '/label_' + str(train_size) + '.npy')
    test_labels = np.load(path + '/label_' + str(test_size) + '.npy')
    return train_patches,test_patches,train_labels,test_labels

