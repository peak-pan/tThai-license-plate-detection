import numpy as np
import pandas as pd
from PIL import Image
import tensorflow as tf
import tensorflow_hub as hub
import requests
import numpy as np
import matplotlib.pyplot as plt

def gen_img(imgid_list,crop_test):
    esrgn_path = "https://tfhub.dev/captain-pool/esrgan-tf2/1"
    model = hub.load(esrgn_path)
    def preprocessing(img):
        imageSize = (tf.convert_to_tensor(image_plot.shape[:-1]) // 4) * 4
        cropped_image = tf.image.crop_to_bounding_box(
            img, 0, 0, imageSize[0], imageSize[1])
        preprocessed_image = tf.cast(cropped_image, tf.float32)
        return tf.expand_dims(preprocessed_image, 0)
    
    # Employ the model
    def srmodel(img):
        #Preprocess low resolution image
        preprocessed_image = preprocessing(img) 
        new_image = model(preprocessed_image)  
        return tf.squeeze(new_image) / 255.0
        
    for img_id in tqdm(imgid_list[500:]):
        x1,y1,x2,y2,path = get_hr_image(img_id=img_id,df=crop_test)
        img = cv2.imread(path)
        #print(x1,y1,x2,y2)
        license_plate = img[y1:y2, x1:x2]
        image_plot = cv2.cvtColor(license_plate, cv2.COLOR_BGR2RGB)
        # Plot the image
        fig = plt.figure()
        fig.figsize=(5, 5)
        ax = fig.add_subplot(111)
        ax.imshow(image_plot)
        ax.axis('off')
        #plt.show()
        hr_image = srmodel(image_plot)
        fig2 = plt.figure()
        fig2.figsize=(5, 5)
        ax2 = fig2.add_subplot(111)
        ax2.imshow(hr_image)
        ax2.axis('off')
        save_path = os.path.join("GAN",img_id + '.jpg')
        fig2.savefig(save_path,bbox_inches='tight', pad_inches=0)
        #plt.show()

#run image gen function
gen_img(imgid_list=imgid_list,crop_test=crop_test)