import shutil
import imutils
import os
import cv2
import sys
import math
import random
import numpy as np
from tqdm import tqdm
from PIL import Image, ImageSequence
from skimage.measure import compare_ssim

_STRIDE = 8

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

def find_max(box_list):
  if len(box_list) == 0:
    return []
  boxes = []
  for box in box_list:
    box = np.append(box, (box[2] - box[0]) * (box[3] - box[1]))
    boxes.append(box)
  boxes = np.array(boxes)
  idx = np.argsort(boxes[:,4])
  box_max = boxes[idx[-1]][:4]
  box_max = np.array(box_max, dtype = np.int32)
  return box_max

def resample(keep_list):
  if len(keep_list) in range(int(_STRIDE / 2) + 1, _STRIDE):
    sample_list = keep_list + random.sample(keep_list, _STRIDE - len(keep_list))
    sample_list.sort()
  else:  
    keep_list = keep_list * math.ceil(_STRIDE/len(keep_list))
    sample_list = random.sample(keep_list, _STRIDE)
    sample_list.sort()
  return sample_list

def rico():
  total = 0
  success = 0
  remove = 0
  for root, dirs, files in os.walk('/home/cheer/Project/Do_Dont/Rico_Data/animations'):
    for name in files:
      file_name = os.path.join(root, name)
      save_dir = '/home/cheer/Project/Do_Dont/Rico_Data/'
      try:
        gif = Image.open(file_name)
        frame_iter = ImageSequence.Iterator(gif)
        duration = gif.info['duration']
        frame_num = 0
        for frame in frame_iter:
          frame_num += 1
        with open(os.path.join(save_dir, 'duration.txt'), 'a') as duration_file: 
          duration_file.write(os.path.splitext(file_name)[0] + ' ' + str(frame_num) + ' ' + str(duration) + ' ' + str(frame_num * duration / 1000.0) + '\n')    
        success += 1
      except:
        pass
      total += 1

      sys.stdout.write('\rtotal num:{} / success num:{}'.format(total, success))
      sys.stdout.flush()

def video():
  for root, dirs, files in os.walk('/home/cheer/Project/Do_Dont/Rico_Data/test_data/videos'):
    for name in files:
      file_name = os.path.join(root, name)
      save_dir = os.path.splitext(file_name)[0].replace('/videos', '/images')
      cap = cv2.VideoCapture(file_name)
      images = []
      keep_list = []
      i = 0
      while(cap.isOpened()):
        ret, frame = cap.read()
        if ret:
          sys.stdout.write('\rreading image {}'.format(i))
          sys.stdout.flush()
          images.append(frame)
          i += 1
        else:
          cap.release()
          break
      
      keep_list.append(0)
      for i in tqdm(range(1, len(images))):
        cnts = compare_frame(images[i-1], images[i])
        if len(cnts):
          keep_list.append(i)
      sample_list = resample(keep_list)

      if not os.path.exists(save_dir):
        os.makedirs(save_dir) 
      for i in range(_STRIDE):
        cv2.imwrite(os.path.join(save_dir, '{:03}'.format(i) + '.jpg'), images[sample_list[i]])                

      box_old = np.zeros(4, dtype = np.int32)
      box_list = ''
      for i in range(1, _STRIDE):
        cnts = compare_frame(images[sample_list[0]], images[sample_list[i]])
        diff_box = convert_box(cnts)
        if len(diff_box):
          max_box = find_max(diff_box)
          box_old = max_box
          max_box = max_box.tolist()
          print (max_box)
          box_list = box_list + str(max_box)[1:-1].replace(' ', '') + ' '
        else:
          max_box = box_old
          max_box = max_box.tolist()
          print (max_box)
          box_list = box_list + str(max_box)[1:-1].replace(' ', '') + ' '
       
      with open(os.path.join('/home/cheer/Project/Do_Dont/Rico_Data/test_data', 'label.txt'), 'a') as label_file: 
        label_file.write(os.path.splitext(file_name)[0].replace('videos', 'images') + ' ' + box_list + '\n')
      

if __name__ == '__main__':
  #rico()
  video()
