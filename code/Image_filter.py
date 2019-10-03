import run_file as rf
import numpy as np
import detect_face_features
import shutil
import os

if __name__ == "__main__":
    # foldername = "G:/MCL/Summer Intern/Code/Dataset/ProGAN_128-20190523T053406Z-001/ProGAN_128"
    # foldername = "G:/MCL/Summer Intern/Code/Dataset/celeba-dataset/celeba-dataset/img_align_celeba_resize"
    foldername = "../dataset/ProGAN_generated_images/ProGAN_generated_images/dataset"
    arg_1 = "shape_predictor_68_face_landmarks.dat"

    arg_2_list = rf.image_path(foldername)

    facial_cor = []
    for i in range(len(arg_2_list)):
        ten_facial = np.zeros((11, 2), dtype=int)
        try:
            print("image:", arg_2_list[i])
            facial_features_cordinates = detect_face_features.main(arg_1, arg_2_list[i])
            ten_facial[:, :] = np.array(
                [facial_features_cordinates[19, :], facial_features_cordinates[ 24, :], facial_features_cordinates[27, :], facial_features_cordinates[33, :],
                 facial_features_cordinates[36, :], facial_features_cordinates[ 39, :], facial_features_cordinates[42, :], facial_features_cordinates[45, :],
                 facial_features_cordinates[48, :], facial_features_cordinates[ 54, :], facial_features_cordinates[66, :]])

            flag = False

            for k in range(11):
                if (ten_facial[k][0] + 16 > 128 or ten_facial[k][1] + 16 > 128):
                    print("Exceed upper boundary", arg_2_list[i])
                    flag = True
                    break
                if (ten_facial[k][0] < 16 or ten_facial[k][1] < 16):
                    print("Exceed lower boundary", arg_2_list[i])
                    flag = True
                    continue
            if flag:
                continue
        except:
            continue
        else:
            facial_cor.append(facial_features_cordinates)
            shutil.copy(arg_2_list[i], os.path.join(foldername, 'filtered'))


