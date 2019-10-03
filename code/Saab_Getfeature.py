import pickle
import numpy as np
import data
import saab
import keras
import sklearn
import matplotlib.pyplot as plt

def main():

	# read data
	# train_images, train_labels, test_images, test_labels, class_list = data.import_data("0-9")

	# read data
    # train_images, train_labels, test_images, test_labels, class_list = data.import_data(FLAGS.use_classes)
	train_images = np.load('result/train_patch_6000.npy')
	test_images = np.load('result/test_patch_6000.npy')
	train_labels = np.load('result/train_label_6000.npy')
	test_labels = np.load('result/test_label_6000.npy')
	class_list = [0,1]

	for i in range(11):
		# load data
		fr = open('pca_params' + str(i) + '.pkl', 'rb')
		pca_params = pickle.load(fr, encoding='latin1')
		fr.close()

		train_images_Sep = train_images[:, i, :, :, :]
		test_images_Sep = test_images[:, i, :, :, :]

		print('Training image size:', train_images.shape)
		print('Testing_image size:', test_images.shape)

		feat = {}
		# Training
		print('--------Training--------')
		feature = saab.initialize(train_images_Sep, pca_params)
		print("S4 shape:", feature.shape)
		print('--------Finish Feature Extraction subnet--------')
		feat['training_feature'] = feature

		print('--------Testing--------')
		feature = saab.initialize(test_images_Sep, pca_params)
		print("S4 shape:", feature.shape)
		print('--------Finish Feature Extraction subnet--------')
		feat['testing_feature'] = feature

		# save data
		fw = open('feat' + str(i) + '.pkl', 'wb')
		pickle.dump(feat, fw)
		fw.close()


if __name__ == '__main__':
	main()
