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

def get_path(fake_tag,real_tag,fake_real_size,test_size): #8000_fake_128_real_128
    path = str(fake_real_size) + fake_tag + real_tag
    isExist = os.path.exists(path)
    if isExist != True:
        os.mkdir(path)
    path = path + '/' + str(test_size)
    isExist = os.path.exists(path)
    if isExist != True:
        os.mkdir(path)
    return path

def get_patch(path,train_size,test_size):
    train_patches = np.load(path + '/patch_' + str(train_size) + '.npy')
    test_patches = np.load(path + '/patch_' + str(test_size) + '.npy')
    train_labels = np.load(path + '/label_' + str(train_size) + '.npy')
    test_labels = np.load(path + '/label_' + str(test_size) + '.npy')
    return train_patches,test_patches,train_labels,test_labels

def getkernels(path,train_patches,train_labels,FLAGS):
    start = time.time()
    print('-------Getkernels-------')
    print('Training image size:', train_patches.shape)

    kernel_sizes = saab.parse_list_string(FLAGS.kernel_sizes)
    class_list = saab.parse_list_string(FLAGS.use_classes)
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
        train_images_Sep = train_patches[:, i, :, :, :]
        train_labels_Sep = train_labels[:, i]

        pca_params = saab.multi_Saab_transform(train_images_Sep, train_labels_Sep,
                                               kernel_sizes=kernel_sizes,
                                               num_kernels=num_kernels,
                                               energy_percent=energy_percent,
                                               use_num_images=use_num_images,
                                               use_classes=class_list)
        # save data
        fw = open(path + '/pca_params' + str(i) + '.pkl', 'wb')
        pickle.dump(pca_params, fw)
        fw.close()

    end = time.time()
    print("Time for getkernels: " + str(end - start))

def getfeatures(path,train_patches,test_patches):
    start = time.time()
    print('-------Getfeatures-------')
    print('Training image size:', train_patches.shape)
    print('Testing_image size:', test_patches.shape)
    # 2.getfeature
    for i in range(11):
        # load data
        fr = open(path + '/pca_params' + str(i) + '.pkl', 'rb')
        pca_params = pickle.load(fr, encoding='latin1')
        fr.close()

        train_images_Sep = train_patches[:, i, :, :, :]
        test_images_Sep = test_patches[:, i, :, :, :]

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
        fw = open(path + '/feat' + str(i) + '.pkl', 'wb')
        pickle.dump(feat, fw)
        fw.close()

    end = time.time()
    print("Time for getfeatures: " + str(end - start))

def getweights(path,train_labels,FLAGS):
    start = time.time()
    print('-------Getweights-------')
    for s in range(11):
        train_labels_Sep = train_labels[:, s]

        # load feature
        fr = open(path + '/feat' + str(s) + '.pkl', 'rb')
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

        num_clusters = saab.parse_list_string(FLAGS.num_clusters)
        use_classes = saab.parse_list_string(FLAGS.use_classes)

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
        fw = open(path + '/llsr_weights' + str(s) + '.pkl', 'wb')
        pickle.dump(weights, fw)
        fw.close()
        fw = open(path + '/llsr_bias' + str(s) + '.pkl', 'wb')
        pickle.dump(bias, fw)
        fw.close()

    end = time.time()
    print("Time for getweights: " + str(end - start))

def test(path,test_size,FLAGS):
    start = time.time()
    print('-------Test-------')
    test_labels = np.load(path + '/label_' + str(test_size) + '.npy')

    acc_stat = {}
    for s in range(11):
        fr = open(path + '/llsr_weights' + str(s) + '.pkl', 'rb')
        weights = pickle.load(fr, encoding='latin1')
        fr.close()
        fr = open(path + '/llsr_bias' + str(s) + '.pkl', 'rb')
        biases = pickle.load(fr, encoding='latin1')
        fr.close()

        test_labels_Sep = test_labels[:, s]


        # load feature
        fr = open(path + '/feat' + str(s) + '.pkl', 'rb')
        feat = pickle.load(fr, encoding='latin1')
        fr.close()

        feature_train = feat['training_feature']
        feature = feat['testing_feature']

        # feature = np.concatenate((feature_train,feature_test),axis=0)
        # labels_comb = np.concatenate((train_labels, test_labels), axis=0)

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

        num_clusters = saab.parse_list_string(FLAGS.num_clusters)
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

                fw = open(path + '/pred_labels' + str(s) + '.pkl', 'wb')
                pickle.dump(pred_labels, fw)
                fw.close()


    # save data
    fw = open(path + '/acc_stat.pkl', 'wb')
    pickle.dump(acc_stat, fw)
    fw.close()

    end = time.time()
    print("Time for getweights: " + str(end - start))
    
def majority_voting(path,test_labels):
    start = time.time()
    print('-------Majority voting-------')
    test_labels_Sep = test_labels[:, 0]

    pred_labels_all = []
    for i in range(11):
        fr = open(path + '/pred_labels' + str(i) + '.pkl', 'rb')
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
    print("Time for majority voting: " + str(end-start))