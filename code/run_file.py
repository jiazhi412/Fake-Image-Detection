import os
import glob
import detect_face_features
from collections import OrderedDict
import numpy as np
import time
from PIL import Image
# import cv2
from matplotlib import pyplot as plt

def image_path(foldername): # directory for all images name in folder
    img_paths = []
    img_dir = foldername # Enter Directory of all images
    data_path = os.path.join(img_dir,'*g')
    files = glob.glob(data_path)
    for f in files:
        img_path = str(f)
        img_paths.append(img_path)
    # print(len(img_paths))

    return img_paths

def read_images(foldername):
    imgs = []
    img_dir = foldername # Enter Directory of all images
    data_path = os.path.join(img_dir,'*g')
    files = glob.glob(data_path)
    for f in files:
        img = cv2.imread(f)
        img_s = cv2.resize(img, (128, 128))
        img_s = cv2.cvtColor(img_s, cv2.COLOR_BGR2RGB) 

        img_np = np.array(img_s)
        imgs.append(img_np)

    imgs = np.array(imgs)
    return imgs


def collect_coordinates(foldername):
    
    arg_1 = "shape_predictor_68_face_landmarks.dat"
    
    arg_2_list = image_path(foldername)
    
    facial_cor = []
    for i in range(len(arg_2_list)):
        print("image:", arg_2_list[i])

        facial_features_cordinates = detect_face_features.main(arg_1, arg_2_list[i])
        

        facial_cor.append(facial_features_cordinates)

    facial_cor = np.array(facial_cor)

    ten_facial = np.zeros((len(arg_2_list), 11, 2), dtype=int)

    for i in range(len(arg_2_list)): # why these index??
        ten_facial[i,:,:] = np.array([facial_cor[i,19,:], facial_cor[i,24,:], facial_cor[i,27,:], facial_cor[i,33,:], 
            facial_cor[i,36,:], facial_cor[i,39,:], facial_cor[i,42,:], facial_cor[i,45,:], 
            facial_cor[i,48,:], facial_cor[i,54,:], facial_cor[i,66,:]])

        for k in range(11):
            if (ten_facial[i][k][0]+16>128 or ten_facial[i][k][1]+16>128):
                print("Exceed upper boundary", arg_2_list[i])
            if (ten_facial[i][k][0]<16 or ten_facial[i][k][1]<16):
                print("Exceed lower boundary", arg_2_list[i])
    


    return ten_facial

def extract_patch(foldername, facial_cor):

    imgs = read_images(foldername)
    print(imgs.shape)
    

    # patches_all_img = np.zeros((imgs.shape[0],68,32,32,3))

    patches_all_img = []
    

    for i in range(facial_cor.shape[0]): # number of images
        # cv2.imshow("facial features", imgs[i])
        # cv2.waitKey(0)

        patch_per_img = []

        for j in range(facial_cor.shape[1]): # number of facial landmarks 68
            x,y = facial_cor[i,j] # location for specific landmark

            # print("image:", i, "facial landmark:", j,x,y)

            patch_per_img.append(imgs[i,y-16:y+16,x-16:x+16,:])
            # patches_all_img[i,j,:,:,:] = imgs[i,y-16:y+16,x-16:x+16,:]

        # print(len(patch_per_img), patch_per_img[0].shape)

        patch_per_img = np.array(patch_per_img)

        patches_all_img.append(patch_per_img)

    patches_all_img = np.array(patches_all_img)

    print("image113232", patches_all_img.shape)

    return patches_all_img





if __name__ == "__main__":

    start = time.time()

    foldername = "G:/MCL/Summer Intern/Code/Dataset/celeba-dataset/celeba-dataset/6000"
    # foldername = "G:/MCL/Summer Intern/Code/Dataset/ProGAN_128-20190523T053406Z-001/6000"
    # foldername = "../celeba-dataset/test_imgs"
    #foldername = "../ProGAN_generated_images/one"

    facial_cor = collect_coordinates(foldername)
    # np.save('ten_real_cor', facial_cor)
    # facial_cor = np.load("ten_fake_cor.npy")

    print("fake facial cor shape:", facial_cor.shape)

    patches_all_img = extract_patch(foldername, facial_cor)
    np.save('ten_real_patch_6000', patches_all_img)

    end = time.time()
    print(end-start)

    # for i in range(11):
    #     plt.imshow(patches_all_img[0,i,:,:,:])
    #     plt.axis('off')
    #     plt.savefig("ten_patch/test/{}.png".format(i))


    # for i in range(68):
    #     if (i >=0 and i<17):
    #         plt.imshow(patches_all_img[0,i,:,:,:])
    #         plt.axis('off')
    #         plt.savefig("../real_patch_examples/jaw_{}.png".format(i))
    #     if (i>=17 and i<22):
    #         plt.imshow(patches_all_img[0,i,:,:,:])
    #         plt.axis('off')
    #         plt.savefig("../real_patch_examples/left_eyebrow_{}.png".format(i))
    #     if (i>=22 and i<27):
    #         plt.imshow(patches_all_img[0,i,:,:,:])
    #         plt.axis('off')
    #         plt.savefig("../real_patch_examples/right_eyebrow_{}.png".format(i))
    #     if (i>=27 and i<36):
    #         plt.imshow(patches_all_img[0,i,:,:,:])
    #         plt.axis('off')
    #         plt.savefig("../real_patch_examples/nose_{}.png".format(i))
    #     if (i>=36 and i<42):
    #         plt.imshow(patches_all_img[0,i,:,:,:])
    #         plt.axis('off')
    #         plt.savefig("../real_patch_examples/lefteye_{}.png".format(i))
    #     if (i>=42 and i<48):
    #         plt.imshow(patches_all_img[0,i,:,:,:])
    #         plt.axis('off')
    #         plt.savefig("../real_patch_examples/righteye_{}.png".format(i))
    #     if (i>=48 and i<68):
    #         plt.imshow(patches_all_img[0,i,:,:,:])
    #         plt.axis('off')
    #         plt.savefig("../real_patch_examples/mouse_{}.png".format(i))

        # plt.imshow(patches_all_img[0,i,:,:,:])
        # plt.axis('off')
        # plt.savefig("fake_patch_examples/fp_{}.png".format(i))





    
                                            



