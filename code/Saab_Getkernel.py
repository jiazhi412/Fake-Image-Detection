from tensorflow.python.platform import flags
import pickle
import data
import saab
import numpy as np

flags.DEFINE_string("output_path", None, "The output dir to save params")
flags.DEFINE_string("use_classes", "0,1", "Supported format: 0,1,5-9")
flags.DEFINE_string("kernel_sizes", "5,5", "Kernels size for each stage. Format: '3,3'")
flags.DEFINE_string("num_kernels", "32,64", "Num of kernels for each stage. Format: '4,10'")
flags.DEFINE_float("energy_percent", None, "Energy to be preserved in each stage")
flags.DEFINE_integer("use_num_images", -1, "Num of images used for training")
FLAGS = flags.FLAGS

def main():
    # read data
    # train_images, train_labels, test_images, test_labels, class_list = data.import_data(FLAGS.use_classes)
    train_images = np.load('result/train_patch_6000.npy')
    test_images = np.load('result/test_patch_6000.npy')
    train_labels = np.load('result/train_label_6000.npy')
    test_labels = np.load('result/test_label_6000.npy')
    class_list = [0,1]


    print('Training image size:', train_images.shape)
    print('Testing_image size:', test_images.shape)

    kernel_sizes=saab.parse_list_string(FLAGS.kernel_sizes)
    if FLAGS.num_kernels:
        num_kernels=saab.parse_list_string(FLAGS.num_kernels)
    else:
        num_kernels=None
    energy_percent=FLAGS.energy_percent
    use_num_images=FLAGS.use_num_images
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
        fw = open('pca_params'+ str(i) + '.pkl', 'wb')
        pickle.dump(pca_params, fw)
        fw.close()



if __name__ == '__main__':
    main()

















