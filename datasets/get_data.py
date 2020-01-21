from skimage.measure import compare_ssim
import imutils
import shutil
import os
import re
import cv2
import json
import sys
import random
import numpy as np
from PIL import Image, ImageSequence

global item
item = []

def get_ads():
  if re.search('ads', text, re.IGNORECASE) or re.search('adview', text, re.IGNORECASE):
    return text
  else:
    return ''
        

def tranverse(dict_or_list, path = []):
  global item
  is_item = 1
  if isinstance(dict_or_list, dict):
    iterator = dict_or_list.iteritems()
  else:
    iterator = enumerate(dict_or_list)
  key_list = []
  value_list = []
  for k, v in iterator:
    if k == 'visibility' and v == 'gone':
      is_item = 0
    if k == 'visible-to-user' and not v:
      is_item = 0
    key_list.append(k)
    value_list.append(v)
    if isinstance(v, list) and re.search('children', str(k), re.IGNORECASE):
      del key_list[-1]
      del value_list[-1]
      tranverse(v, path + [k])
    elif isinstance(v, dict):
      is_item = 0
      tranverse(v, path + [k])
  if is_item:
    item.append(zip(key_list, value_list))


def IoU(boxes, diff_box_list):
  max_iou = []
  for box in boxes:
    iou_list = []
    for diff_box in diff_box_list:
      if box == [0, 0, 0, 0] or box == [] or diff_box == []:
        iou = 0
      else:
        boxA = box
        boxB = diff_box
        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[2], boxB[2])
        yB = min(boxA[3], boxB[3])
        interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
        boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
        boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)  
        iou = interArea / float(boxAArea + boxBArea - interArea)
        iou_list.append(iou)
    if len(iou_list):
      max_iou.append(max(iou_list))
    else:
      max_iou.append(0)
  return boxes[np.argsort(max_iou)[-1]]

def compare_frame(frameA, frameB):
  grayA = cv2.cvtColor(frameA, cv2.COLOR_BGR2GRAY)
  grayB = cv2.cvtColor(frameB, cv2.COLOR_BGR2GRAY)

  score, diff = compare_ssim(grayA, grayB, full=True)
  diff = (diff * 255).astype("uint8")
  #sys.stdout.write('\rSSIM: {}'.format(score))
  #sys.stdout.flush()

  thresh = cv2.threshold(diff, 180, 255, cv2.THRESH_BINARY_INV)[1]
  opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, np.ones((10, 10)))
  opening = cv2.dilate(opening, np.ones((20, 20)))
  cnts = cv2.findContours(opening.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
  cnts = cnts[0] if imutils.is_cv2() else cnts[1]

  return cnts

def resize_box(image, value):
  if len(image) > len(image[0]):
    value[0] = value[0] * len(image[0]) / 1440
    value[1] = value[1] * len(image) / 2560
    value[2] = value[2] * len(image[0]) / 1440
    value[3] = value[3] * len(image) / 2560
  else:
    value[0] = value[0] * len(image[0]) / 2560
    value[1] = value[1] * len(image) / 1440
    value[2] = value[2] * len(image[0]) / 2560
    value[3] = value[3] * len(image) / 1440
  return value

def convert_box(cnts):
  box = []
  for c in cnts:
    x, y, w, h = cv2.boundingRect(c)
    if w > 20 and h > 20:
      box.append([x, y, x+w, y+h])
  box = np.array(box)
  return box

def rico():
  global item
  total = 0
  success = 0
  remove = 0
  for root, dirs, files in os.walk('/media/cheer/Elements/Rico_Data/animations'):
    for name in files:
      file_name = os.path.join(root, name)
      save_dir = os.path.splitext(file_name)[0].replace('/animations', '/split_image').replace('/gifs', '')
      boxes = []
      item = []
      images = []
      diff_list = []
      try:
        with open(file_name.replace('animations', 'filtered_traces').replace('gifs', 'view_hierarchies').replace('.gif', '.json'), 'r') as json_file:
          data = json.load(json_file)
        tranverse(data['activity']['root'])

        gif = Image.open(file_name)
        frame_iter = ImageSequence.Iterator(gif)
        for frame in frame_iter:
          images.append(cv2.cvtColor(np.array(frame.copy().convert('RGB')), cv2.COLOR_RGB2BGR))
        image_first = images[0].copy()       
        for it in item:
          for key, value in it:
            if key == 'bounds':
              value = resize_box(image_first, value)
              boxes.append(value)

        if not os.path.exists(save_dir):
          os.makedirs(save_dir) 
        cv2.imwrite(os.path.join(save_dir, '000.jpg'), images[0])

        for i in range(len(images) - 1):
          i += 1
          cnts = compare_frame(images[i-1], images[i])
          box_diff = convert_box(cnts)
          if len(box_diff):
            cnts_c = compare_frame(image_first, images[i])
            box_diff_c = convert_box(cnts_c)
            if len(box_diff_c):
              for box in box_diff_c:
                diff_list.append(box)         
            cv2.imwrite(os.path.join(save_dir, '{:03}'.format(i) + '.jpg'), images[i])

        iou_box = IoU(boxes, diff_list)
        for it in item:
          for key, value in it:
            if value == iou_box:
              iou_it = it
        for key, value in iou_it:
          if key == 'class':
            class_name = value
          if key == 'bounds':
            iou_bound = value
        if len(os.listdir(save_dir)) < 4 or len(os.listdir(save_dir)) > 20:
          shutil.rmtree(save_dir)
          remove += 1
          success -= 1
        else:
          with open(os.path.join(root.replace('/animations', '/split_image').replace('/gifs', ''), 'class_name.txt'), 'a') as label_file:
            label_file.write(os.path.splitext(os.path.basename(file_name))[0] + ' ' + class_name + ' ' + str(iou_bound)[1:-1].replace(' ', '') + '\n')
            success += 1
      except:
        pass
      total += 1

      sys.stdout.write('\rtotal num:{} / success num:{} / remove:{}'.format(total, success, remove))
      sys.stdout.flush()


if __name__ == '__main__':
  rico()
