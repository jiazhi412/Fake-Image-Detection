import os
import glob
import numpy as np
import time
from PIL import Image
# import cv2
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split


def read_patches():
	fake_patches = np.load("ten_fake_patch_6000.npy")
	real_patches = np.load("ten_real_patch_6000.npy")

	return fake_patches, real_patches

def change_dim(arr):
	# adjust patches size to 68, #_img, 32,32,3
	landmarks_patch = []

	for i in range(11):
		landmarks_patch.append(arr[:,i,:,:,:])

	landmarks_patch = np.array(landmarks_patch)

	return landmarks_patch

def fit_to_saab(fake_patches, real_patches, test_ratio):

	# create labels for patches, 1:fake, 0:real
	fake_labels = np.ones((fake_patches.shape[0],11)) # 500
	real_labels = np.zeros((real_patches.shape[0],11)) # 500

	# concatenate fake and real
	train_patch_1 = np.concatenate((fake_patches, real_patches), axis=0) # 1000,11,32,32,3
	# train_patch = np.reshape(train_patch_1, (-1, 32, 32, 3))

	train_label_1 = np.concatenate((fake_labels, real_labels), axis=0) 
	# train_label = np.reshape(train_label_1, (-1,1))

	print(train_patch_1.shape, train_label_1.shape)

	# # shuffle data and label
	# idx = np.random.permutation(len(train_patch))
	# train_shuffled, label_shuffled = train_patch[idx], train_label[idx]

	# split training and testing
	X_train_1, X_test_1, y_train, y_test = train_test_split(train_patch_1, train_label_1, test_size=test_ratio, shuffle = True)
	# X_train = change_dim(X_train_1)
	# X_test = change_dim(X_test_1)


	return X_train_1, X_test_1, y_train, y_test




if __name__ == "__main__":
	fake = np.load("ten_fake_patch_6000.npy")
	real = np.load("ten_real_patch_6000.npy")
	X_train, X_test, y_train, y_test = fit_to_saab(fake, real)
	print(X_train.shape, '\n', X_test.shape, '\n', y_train.shape, '\n', y_test.shape)

	# np.save('result/whole_train_patch_noshuffle', X_train)
	# np.save('result/whole_test_patch_noshuffle', X_test)
	# np.save('result/whole_train_label_noshuffle', y_train)
	# np.save('result/whole_test_label_noshuffle', y_test)

    # test
	np.save('result/train_patch_6000', X_train)
	np.save('result/test_patch_6000', X_test)
	np.save('result/train_label_6000', y_train)
	np.save('result/test_label_6000', y_test)




