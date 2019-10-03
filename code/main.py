from tensorflow.python.platform import flags
import Patch_function as pf
import Saab_function as sf

flags.DEFINE_string("output_path", None, "The output dir to save params")
flags.DEFINE_string("use_classes", "0,1", "Supported format: 0,1,5-9")
flags.DEFINE_string("kernel_sizes", "5,5", "Kernels size for each stage. Format: '3,3'")
flags.DEFINE_string("num_kernels", "32,64", "Num of kernels for each stage. Format: '4,10'")
flags.DEFINE_string("num_clusters", "120,84,2", "Num of clusters for each fully connected layer. Format: '120,84,2'")
flags.DEFINE_float("energy_percent", None, "Energy to be preserved in each stage")
flags.DEFINE_integer("use_num_images", -1, "Num of images used for training")
FLAGS = flags.FLAGS


fake_tag = '_fake_128'
real_tag = '_real_128'
fake_real_size = 6000

train_size = 11000
test_size = fake_real_size*2 - train_size

def main():
    path = sf.get_path(fake_tag, real_tag, fake_real_size, test_size)
    # # 1.extract fake and real patches from images
    # pf.extract_patch(fake_tag,real_tag,fake_real_size,path)

    # 2.combine fake and real patches to be train and test
    pf.comb_patch(path, fake_real_size)
    pf.split_patch(path, fake_real_size, train_size, test_size)

    train_patches, test_patches, train_labels, test_labels = pf.get_patch(path, train_size, test_size)
    # 3.getkernel
    sf.getkernels(path, train_patches, train_labels, FLAGS)

    # 4.getfeature
    sf.getfeatures(path, train_patches, test_patches)

    # 5.getweight
    sf.getweights(path, train_labels, FLAGS)

    # 6.saab test
    sf.test(path,test_size,FLAGS)

    # 7. final decision
    sf.majority_voting(path,test_labels)

if __name__ == '__main__':
	main()