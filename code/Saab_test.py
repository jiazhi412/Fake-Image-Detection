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

def main():
	start = time.time()

	# read data
	# train_images_1, train_labels_1, test_images_1, test_labels_1, class_list = data.import_data("0-9")
	train_images = np.load('result/train_patch_6000.npy')
	test_images = np.load('result/test_patch_6000.npy')
	train_labels = np.load('result/train_label_6000.npy')
	test_labels = np.load('result/test_label_6000.npy')
	class_list = [0, 1]

	acc_stat = {}

	for s in range(11):
		fr = open('llsr_weights' + str(s) + '.pkl', 'rb')
		weights = pickle.load(fr, encoding='latin1')
		fr.close()
		fr = open('llsr_bias' + str(s) + '.pkl', 'rb')
		biases = pickle.load(fr, encoding='latin1')
		fr.close()

		train_images_Sep = train_images[:, s, :, :, :]
		test_images_Sep = test_images[:, s, :, :, :]
		train_labels_Sep = train_labels[:, s]
		test_labels_Sep = test_labels[:, s]

		print('Training image size:', train_images.shape)
		print('Testing_image size:', test_images.shape)

		# load feature
		fr = open('feat' + str(s) + '.pkl', 'rb')
		feat = pickle.load(fr, encoding='latin1')
		fr.close()

		feature_train = feat['training_feature']
		feature = feat['testing_feature']

		# feature = np.concatenate((feature_train,feature_test),axis=0)
		labels_comb = np.concatenate((train_labels, test_labels), axis=0)

		# test_labels=feat['testing_labels']
		feature = np.absolute(feature)
		feature = feature.reshape(feature.shape[0], -1)
		print("S4 shape:", feature.shape)
		print('--------Finish Feature Extraction subnet--------')

		# feature normalization
		std_var = (np.std(feature, axis=1)).reshape(-1, 1)
		feature = feature / std_var
		# relu
		for i in range(feature.shape[0]):
			for j in range(feature.shape[1]):
				if feature[i, j] < 0:
					feature[i, j] = 0

		num_clusters = [120, 84, 2]
		use_classes = 2
		for k in range(len(num_clusters)):
			weight = weights['%d LLSR weight' % k]
			bias = biases['%d LLSR bias' % k]
			feature = np.matmul(feature, weight) + bias
			print(k, ' layer LSR weight shape:', weight.shape)
			print(k, ' layer LSR output shape:', feature.shape)
			if k != len(num_clusters) - 1:
				# Relu
				for i in range(feature.shape[0]):
					for j in range(feature.shape[1]):
						if feature[i, j] < 0:
							feature[i, j] = 0
			else:
				pred_labels = np.argmax(feature, axis=1)
				acc_test = sklearn.metrics.accuracy_score(test_labels_Sep, pred_labels)
				print('testing acc for landmark' + str(s) + ' is {}'.format(acc_test))
				acc_stat['%d testing acc for landmark' % s] = acc_test

				fw = open('pred_labels' + str(s) + '.pkl', 'wb')
				pickle.dump(pred_labels, fw)
				fw.close()

	end =  time.time()
	print(end - start)


    # save data
	fw = open('acc_stat.pkl', 'wb')
	pickle.dump(acc_stat, fw)
	fw.close()

if __name__ == '__main__':
	main()

