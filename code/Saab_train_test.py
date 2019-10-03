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

flags.DEFINE_string("output_path", None, "The output dir to save params")
flags.DEFINE_string("use_classes", "0,1", "Supported format: 0,1,5-9")
flags.DEFINE_string("kernel_sizes", "5,5", "Kernels size for each stage. Format: '3,3'")
flags.DEFINE_string("num_kernels", "32,64", "Num of kernels for each stage. Format: '4,10'")
flags.DEFINE_float("energy_percent", None, "Energy to be preserved in each stage")
flags.DEFINE_integer("use_num_images", -1, "Num of images used for training")
FLAGS = flags.FLAGS

fake_tag = '_fake_128'
real_tag = '_real_128'

train_size = 1000
test_size = 200

tag = '200_test/' + fake_tag + real_tag +'/' + str(train_size)
isExist = os.path.exists(tag)
if isExist != True:
    os.mkdir(tag)

def main():
    # 1.getkernel
    # read data
    train_images = np.load(tag + '/patch_' + str(train_size) + '.npy')
    test_images = np.load(tag + '/patch_' + str(test_size) + '.npy')
    train_labels = np.load(tag + '/label_' + str(train_size) + '.npy')
    test_labels = np.load(tag + '/label_' + str(test_size) + '.npy')
    class_list = [0, 1]

    print('Training image size:', train_images.shape)
    print('Testing_image size:', test_images.shape)

    kernel_sizes = saab.parse_list_string(FLAGS.kernel_sizes)
    if FLAGS.num_kernels:
        num_kernels = saab.parse_list_string(FLAGS.num_kernels)
    else:
        num_kernels = None
    energy_percent = FLAGS.energy_percent
    use_num_images = FLAGS.use_num_images
    print('Parameters:')
    print('use_classes:', class_list)
    print('Kernel_sizes:', kernel_sizes)
    print('Number_kernels:', num_kernels)
    print('Energy_percent:', energy_percent)
    print('Number_use_images:', use_num_images)

    for i in range(11):
        train_images_Sep = train_images[:, i, :, :, :]
        train_labels_Sep = train_labels[:, i]

        pca_params = saab.multi_Saab_transform(train_images_Sep, train_labels_Sep,
                                               kernel_sizes=kernel_sizes,
                                               num_kernels=num_kernels,
                                               energy_percent=energy_percent,
                                               use_num_images=use_num_images,
                                               use_classes=class_list)
        # save data
        fw = open(tag + '/pca_params' + str(i) + '.pkl', 'wb')
        pickle.dump(pca_params, fw)
        fw.close()

    # 2.getfeature
    for i in range(11):
        # load data
        fr = open(tag + '/pca_params' + str(i) + '.pkl', 'rb')
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
        fw = open(tag + '/feat' + str(i) + '.pkl', 'wb')
        pickle.dump(feat, fw)
        fw.close()

    # 3.getkernel
    # read data

    for s in range(11):
        train_images_Sep = train_images[:, s, :, :, :]
        test_images_Sep = test_images[:, s, :, :, :]
        train_labels_Sep = train_labels[:, s]
        test_labels_Sep = test_labels[:, s]

        print('Training image size:', train_images.shape)
        print('Testing_image size:', test_images.shape)

        # load feature
        fr = open(tag + '/feat' + str(s) + '.pkl', 'rb')
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
        fw = open(tag + '/llsr_weights' + str(s) + '.pkl', 'wb')
        pickle.dump(weights, fw)
        fw.close()
        fw = open(tag + '/llsr_bias' + str(s) + '.pkl', 'wb')
        pickle.dump(bias, fw)
        fw.close()

    # 4.saab test
    # read data
    class_list = [0, 1]
    start = time.time()

    acc_stat = {}
    for s in range(11):
        fr = open(tag + '/llsr_weights' + str(s) + '.pkl', 'rb')
        weights = pickle.load(fr, encoding='latin1')
        fr.close()
        fr = open(tag + '/llsr_bias' + str(s) + '.pkl', 'rb')
        biases = pickle.load(fr, encoding='latin1')
        fr.close()

        train_images_Sep = train_images[:, s, :, :, :]
        test_images_Sep = test_images[:, s, :, :, :]
        train_labels_Sep = train_labels[:, s]
        test_labels_Sep = test_labels[:, s]

        print('Training image size:', train_images.shape)
        print('Testing_image size:', test_images.shape)

        # load feature
        fr = open(tag + '/feat' + str(s) + '.pkl', 'rb')
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

                fw = open(tag + '/pred_labels' + str(s) + '.pkl', 'wb')
                pickle.dump(pred_labels, fw)
                fw.close()

    end = time.time()
    print(end - start)

    # save data
    fw = open(tag + '/acc_stat.pkl', 'wb')
    pickle.dump(acc_stat, fw)
    fw.close()

    # 5. final decision
    start = time.time()

    train_labels_Sep = train_labels[:, 0]
    test_labels_Sep = test_labels[:, 0]

    pred_labels_all = []

    for i in range(11):
        fr = open(tag + '/pred_labels' + str(i) + '.pkl', 'rb')
        pred_labels = pickle.load(fr, encoding='latin1')
        fr.close()
        pred_labels_all.append(pred_labels)

    pred_labels_all = np.array(pred_labels_all)
    decision = np.zeros((pred_labels_all.shape[1]))  # 500

    for i in range(pred_labels_all.shape[1]):
        real = 0
        fake = 0
        for j in range(pred_labels_all.shape[0]):
            if pred_labels_all[j, i] == 0:
                real = real + 1
            else:
                fake = fake + 1
        if real > fake:
            decision[i] = 0
        else:
            decision[i] = 1

    acc_test = sklearn.metrics.accuracy_score(test_labels_Sep, decision)
    print('Final testing acc is {}'.format(acc_test))

    end = time.time()
    print(end - start)


if __name__ == '__main__':
	main()