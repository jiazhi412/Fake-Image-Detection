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


#real flag
size = '500'
fake_tag = '_fake_1024'
real_tag = '_real_128'

#save flag
tag = 'patch/' + fake_tag + real_tag
isExist = os.path.exists(tag)
if isExist != True:
    os.mkdir(tag)

def main():
    # # 1.proprocess
    # # real image
    # foldername = "../Dataset/celeba-dataset/celeba-dataset/" + size + real_tag
    # facial_cor = rf.collect_coordinates(foldername)
    # print("fake facial cor shape:", facial_cor.shape)
    # patches_all_img = rf.extract_patch(foldername, facial_cor)
    # np.save(tag + '/ten_real_patch_'+size, patches_all_img)
    #
    # # fake image
    # if fake_tag == '_fake_1024':
    #     foldername = "../Dataset/ProGAN_generated_images/ProGAN_generated_images/" + size + fake_tag
    # else:
    #     foldername = "../Dataset/ProGAN_128-20190523T053406Z-001/" + size + fake_tag
    # facial_cor = rf.collect_coordinates(foldername)
    # print("fake facial cor shape:", facial_cor.shape)
    # patches_all_img = rf.extract_patch(foldername, facial_cor)
    # np.save(tag + '/ten_fake_patch_' + size, patches_all_img)
    #
    # 2. preprocess saab
    fake = np.load(tag + '/ten_fake_patch_' + size+'.npy')
    real = np.load(tag + '/ten_real_patch_'+ size + '.npy')
    patch,label = comb(fake, real)
    print(patch.shape, '\n', label.shape, '\n')

    np.save(tag + '/patch_' + str(int(size)*2), patch)
    np.save(tag + '/label_' + str(int(size)*2), label)




    split_size = 500
    patch_part,label_part = split(patch,label,split_size)

    np.save(tag + '/patch_' + str(split_size), patch_part)
    np.save(tag + '/label_' + str(split_size), label_part)

    split_size = 600
    patch_part, label_part = split(patch, label, split_size)

    np.save(tag + '/patch_' + str(split_size), patch_part)
    np.save(tag + '/label_' + str(split_size), label_part)

    split_size = 700
    patch_part, label_part = split(patch, label, split_size)

    np.save(tag + '/patch_' + str(split_size), patch_part)
    np.save(tag + '/label_' + str(split_size), label_part)

    split_size = 800
    patch_part, label_part = split(patch, label, split_size)

    np.save(tag + '/patch_' + str(split_size), patch_part)
    np.save(tag + '/label_' + str(split_size), label_part)

    split_size = 900
    patch_part, label_part = split(patch, label, split_size)

    np.save(tag + '/patch_' + str(split_size), patch_part)
    np.save(tag + '/label_' + str(split_size), label_part)








def comb(fake_patches,real_patches):
    # create labels for patches, 1:fake, 0:real
    fake_labels = np.ones((fake_patches.shape[0], 11))  # 500
    real_labels = np.zeros((real_patches.shape[0], 11))  # 500

    # concatenate fake and real
    patch = np.concatenate((fake_patches, real_patches), axis=0)  # 1000,11,32,32,3
    # train_patch = np.reshape(train_patch_1, (-1, 32, 32, 3))

    label = np.concatenate((fake_labels, real_labels), axis=0)
    # train_label = np.reshape(train_label_1, (-1,1))

    print(patch.shape, label.shape)

    # # shuffle data and label
    # idx = np.random.permutation(len(train_patch))
    # train_shuffled, label_shuffled = train_patch[idx], train_label[idx]

    # X_train = change_dim(X_train_1)
    # X_test = change_dim(X_test_1)

    return patch,label

def split(patch,label,num):
    patch_part = []
    label_part = []
    trials = int(num/2)
    for i in range(trials):
        patch_part.append(patch[i,:])
        label_part.append(label[i, :])


    for i in range(trials):
        patch_part.append(patch[i+int(patch.shape[0]/2), :])
        label_part.append(label[i+int(label.shape[0]/2), :])
    patch_part = np.array(patch_part)
    label_part = np.array(label_part)
    return patch_part,label_part


if __name__ == '__main__':
	main()