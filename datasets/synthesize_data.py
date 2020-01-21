import os
import cv2
import sys
import random
import math
import imutils
import numpy as np
from tqdm import tqdm
from skimage.measure import compare_ssim 

_STRIDE = 8
spc = 1000
image_width = 320
image_height = 540
save_width = 280
save_height = 500

data_dir = '/home/cheer/Project/Do_Dont/Rico_Data/test_data/materials'
background_dir = '/home/cheer/program/slim/flowers/flower_photos'

def compare_frame(frameA, frameB):
  grayA = cv2.cvtColor(frameA, cv2.COLOR_BGR2GRAY)
  grayB = cv2.cvtColor(frameB, cv2.COLOR_BGR2GRAY)

  score, diff = compare_ssim(grayA, grayB, full=True)
  diff = (diff * 255).astype("uint8")

  thresh = cv2.threshold(diff, 180, 255, cv2.THRESH_BINARY_INV)[1]
  opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, np.ones((10, 10)))
  opening = cv2.dilate(opening, np.ones((20, 20)))
  cnts = cv2.findContours(opening.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
  cnts = cnts[0] if imutils.is_cv2() else cnts[1]
  return cnts

def convert_box(cnts):
  box = []
  for c in cnts:
    x, y, w, h = cv2.boundingRect(c)
    #if w > 20 and h > 20:
    box.append([x, y, x+w, y+h])
  box = np.array(box)
  return box

def resample(keep_list):
  if len(keep_list) in range(int(_STRIDE / 2) + 1, _STRIDE):
    sample_list = keep_list + random.sample(keep_list, _STRIDE - len(keep_list))
    sample_list.sort()
  else:  
    keep_list = keep_list * math.ceil(_STRIDE/len(keep_list))
    sample_list = random.sample(keep_list, _STRIDE)
    sample_list.sort()
  return sample_list

def main():
  label_list = os.listdir(data_dir)
  background_list = os.listdir(background_dir)
  print (label_list)
  for label in label_list:
    print ('synthesizing category', label)
    material_list = os.listdir(os.path.join(data_dir, label))
    sample_len = spc / len(material_list)
    sample_count = 0
    background_folder = random.choice(background_list)
    background_images = os.listdir(os.path.join(background_dir, background_folder))
    
    for material in material_list:
      keep_list = []
      material_image = []
      cap = cv2.VideoCapture(os.path.join(data_dir, label, material))
      i = 0
      while(cap.isOpened()):
        ret, frame = cap.read()
        if ret:
          sys.stdout.write('\rreading image {}'.format(i))
          sys.stdout.flush()
          frame = cv2.resize(frame, (image_width, image_height))
          material_image.append(frame)
          i += 1
        else:
          cap.release()
          break

      keep_list.append(0)
      for i in tqdm(range(1, len(material_image))):
        cnts = compare_frame(material_image[i-1], material_image[i])
        if len(cnts):
          keep_list.append(i)
      sample_list = resample(keep_list)
      if label == '1':
        sample_list = [13, 14, 15, 56, 57, 58, 62, 68]
      print (sample_list)
      box_list = []
      for i in range(1, _STRIDE):
        if label == '1':
          box_list.append([[5, 90, 315, 395]])
        else:
          cnts = compare_frame(material_image[sample_list[i-1]], material_image[sample_list[i]])
          box_list.append(convert_box(cnts))
      
      for _ in tqdm(range(int(sample_len))):
        background = cv2.imread(os.path.join(background_dir, background_folder, random.choice(background_images)))
        background = cv2.resize(background, (image_width, image_height))
        save_dir = os.path.join('/home/cheer/Project/Do_Dont/Rico_Data/synthetic_data', label, str(sample_count))
        if not os.path.exists(save_dir):
          os.makedirs(save_dir)    
        cv2.imwrite(os.path.join(save_dir, '000.jpg'), cv2.resize(background, (save_width, save_height)))
        for i in range(1, _STRIDE):
          for box in box_list[i-1]:
            crop = material_image[sample_list[i]][box[1]:box[3], box[0]:box[2]]
            background[box[1]:box[3], box[0]:box[2]] = crop
          cv2.imwrite(os.path.join(save_dir, '{:03}'.format(i) + '.jpg'), cv2.resize(background, (save_width, save_height)))  
        with open(os.path.join('/home/cheer/Project/Do_Dont/Rico_Data/synthetic_data', 'label.txt'), 'a') as label_file: 
          label_file.write(save_dir + ' '  + label + '\n')    
        sample_count += 1

if __name__ == '__main__':
  main()
