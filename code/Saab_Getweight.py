import pickle
import keras
import data
from keras.datasets import cifar10
import numpy as np
import sklearn
# import cv2
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from numpy import linalg as LA
# import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import euclidean_distances

tag = '200'

# read data
train_images = np.load(tag + '/train_patch_' + tag + '.npy')
test_images = np.load(tag + '/test_patch_' + tag + '.npy')
train_labels = np.load(tag + '/train_label_' + tag + '.npy')
test_labels = np.load(tag + '/test_label_' + tag + '.npy')
class_list = [0, 1]

def main():
	# read data
	# train_images_1, train_labels_1, test_images_1, test_labels_1, class_list = data.import_data("0-9")
	# train_images = np.load('result/train_patch_6000.npy')
	# test_images = np.load('result/test_patch_6000.npy')
	# train_labels = np.load('result/train_label_6000.npy')
	# test_labels = np.load('result/test_label_6000.npy')
	# class_list = [0,1]

	for s in range(11):
		train_images_Sep = train_images[:, s, :, :, :]
		test_images_Sep = test_images[:, s, :, :, :]
		train_labels_Sep = train_labels[:, s]
		test_labels_Sep = test_labels[:, s]

		print('Training image size:', train_images.shape)
		print('Testing_image size:', test_images.shape)

		# load feature
		# fr = open('feat' + str(s) + '.pkl', 'rb')
		fr = open(tag + '/pred_labels' + str(s) + '.pkl', 'rb')
		feat = pickle.load(fr, encoding='Latin')
		fr.close()
		feature = feat['training_feature']
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
		use_classes = class_list
		weights = {}
		bias = {}
		for k in range(len(num_clusters)):
			if k != len(num_clusters) - 1:
				num_clus = int(num_clusters[k] / len(use_classes))
				labels = np.zeros((feature.shape[0], num_clusters[k]))
				for n in range(len(use_classes)):
					idx = (train_labels_Sep == use_classes[n])
					index = np.where(idx == True)[0]
					feature_special = np.zeros((index.shape[0], feature.shape[1]))
					for i in range(index.shape[0]):
						feature_special[i] = feature[index[i]]
					kmeans = KMeans(n_clusters=num_clus).fit(feature_special)
					pred_labels = kmeans.labels_
					for i in range(feature_special.shape[0]):
						labels[index[i], pred_labels[i] + n * num_clus] = 1

				# least square regression
				A = np.ones((feature.shape[0], 1))
				feature = np.concatenate((A, feature), axis=1)
				weight = np.matmul(LA.pinv(feature), labels)
				feature = np.matmul(feature, weight)
				weights['%d LLSR weight' % k] = weight[1:weight.shape[0]]
				bias['%d LLSR bias' % k] = weight[0].reshape(1, -1)
				print(k, ' layer LSR weight shape:', weight.shape)
				print(k, ' layer LSR output shape:', feature.shape)

				pred_labels = np.argmax(feature, axis=1)
				num_clas = np.zeros((num_clusters[k], len(use_classes)))
				for i in range(num_clusters[k]):
					for t in range(len(use_classes)):
						for j in range(feature.shape[0]):
							if pred_labels[j] == i and train_labels_Sep[j] == t:
								num_clas[i, t] += 1
				acc_train = np.sum(np.amax(num_clas, axis=1)) / feature.shape[0]
				print(k, ' layer LSR training acc is {}'.format(acc_train))

				# Relu
				for i in range(feature.shape[0]):
					for j in range(feature.shape[1]):
						if feature[i, j] < 0:
							feature[i, j] = 0
			else:
				# least square regression
				labels = keras.utils.to_categorical(train_labels_Sep, 2)

				# labels = train_labels
				A = np.ones((feature.shape[0], 1))
				feature = np.concatenate((A, feature), axis=1)
				weight = np.matmul(LA.pinv(feature), labels)
				feature = np.matmul(feature, weight)
				weights['%d LLSR weight' % k] = weight[1:weight.shape[0]]
				bias['%d LLSR bias' % k] = weight[0].reshape(1, -1)
				print(k, ' layer LSR weight shape:', weight.shape)
				print(k, ' layer LSR output shape:', feature.shape)

				pred_labels = np.argmax(feature, axis=1)
				acc_train = sklearn.metrics.accuracy_score(train_labels_Sep, pred_labels)
				print('training acc is {}'.format(acc_train))
		# save data
		fw = open('llsr_weights' + str(s) + '.pkl', 'wb')
		pickle.dump(weights, fw)
		fw.close()
		fw = open('llsr_bias' + str(s) + '.pkl', 'wb')
		pickle.dump(bias, fw)
		fw.close()

if __name__ == '__main__':
	main()

