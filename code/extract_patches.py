import numpy
import cv2
import os
import glob
import matplotlib.pyplot as plt
from PIL import Image
from skimage.transform import resize

def create_dataset():
    img_dir = "celebahq_generated_images" # Enter Directory of all images 
    data_path = os.path.join(img_dir,'*g')
    files = glob.glob(data_path)
    data = []
    for f1 in files:
        img = Image.open(f1)        
        pic = img.resize((256, 256), Image.ANTIALIAS)
        pic_np = np.array(pic)
#         img = cv2.imread(f1)
#         img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) 
        data.append(pic_np)
        
    data = np.array(data)
    return data[:100]