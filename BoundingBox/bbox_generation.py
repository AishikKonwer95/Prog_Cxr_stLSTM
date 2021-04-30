#pip install SimpleITK
#pip install medpy

import os
import cv2
import numpy as np
from medpy.io import load
import matplotlib.pyplot as plt
import skimage.io as io

def show_img(ori_img):
    plt.imshow(ori_img[:, :],cmap='gray')  # channel_last
    plt.show()

cf=1
	
data_path = '/home/aishik/Downloads/preds'
for filename in os.listdir(data_path):
    if filename.endswith(".nii.gz") :
        print(cf)
        cf+=1
        #print(os.path.join(data_path, filename))
        example_filename = os.path.join(data_path, filename)
        image, image_header = load(example_filename)
        print(image.shape)
        #from medpy.filter import otsu
        #threshold = otsu(image_data)
        #output_data = image_data > threshold
        original = image.copy()
        gray = image.copy()
        #gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        #print(gray.shape)
        #edged = cv2.Canny(gray, 30, 200) 
        edged = gray.copy()
        #show_img(gray)  latest commented
        #cv2_imshow(gray)
        #cv2.waitKey(0) 
        #thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
        hb,wb,_ = gray.shape
        inback=np.zeros((hb,wb))
        # Find contours, obtain bounding box, extract and save ROI
        ROI_number = 0
        cnts = cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE) 
        cnts = cnts[0] if len(cnts) == 2 else cnts[1]
        for c in cnts:
            x,y,w,h = cv2.boundingRect(c)
            #print(x,y,w,h)
            #cv2.rectangle(image, (x, y), (x + w, y + h), (36,255,12), 1)
            #cv2.rectangle(image, (x, y), (x + round(w/3), y + h), (36,255,12), 1)
            #cv2.rectangle(image, (x, y), (x + round(2*w/3), y + h), (36,255,12), 1)
            cv2.rectangle(inback, (x, y), (x + w, y + h), (36,255,12), 5)
            cv2.rectangle(inback, (x, y), (x + round(w/3), y + h), (36,255,12), 5)
            cv2.rectangle(inback, (x, y), (x + round(2*w/3), y + h), (36,255,12), 5)
            ROI = original[y:y+h, x:x+w]
            hh,ww=ROI.shape
            #print (hh,ww)
            upper = ROI[:,0:round(ww/3)]
            mid = ROI[:,round(ww/3):round(2*ww/3)]
            lower = ROI[:,round(2*ww/3):ww]
            #cv2.imwrite('ROI_{}_upper.png'.format(ROI_number), upper)
            #cv2.imwrite('ROI_{}_mid.png'.format(ROI_number), mid)
            #cv2.imwrite('ROI_{}_lower.png'.format(ROI_number), lower)
            ROI_number += 1

        #from google.colab.patches import cv2_imshow
        #cv2_imshow(gray)
        #show_img(inback)     
        #cv2.imwrite('gray.nii',gray)
        #nib.save(gray,'xyz.nii.gz')
        #show_img(image)
        #cv2.imshow('image', image)
        #cv2.waitKey()
        #final=cv2.addWeighted(original, 1, gray, 1, 0)
        #show_img(final)
        #setting alpha=1, beta=1, gamma=0 gives direct overlay of two images
        from medpy.io import save
        save(inback, '/home/aishik/Downloads/bbox/BB_{}'.format(filename), image_header)
    else:
        continue
