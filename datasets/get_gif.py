import os
import cv2
import numpy as np
from glob import glob
from PIL import Image, ImageSequence

data_dir = '/media/cheer/Elements/Dribbble_Data'

def get_list():
  gif_list = glob(os.path.join(data_dir, 'All_images', '*'))
  #with open(os.path.join(data_dir, 'gif_list.txt'), 'w') as gif_list_file:
  #  for gif_file in gif_list:
  #    gif_list_file.write(gif_file + '\n')
  print (len(gif_list))

def check():
  file_i = 0
  frame_i = 0
  images = []
  with open(os.path.join(data_dir, 'gif_list.txt'), 'r') as gif_list_file:
    gif_list = gif_list_file.readlines()
  cv2.namedWindow('UI', cv2.WINDOW_NORMAL)
  gif = Image.open(gif_list[file_i].strip())
  frame_iter = ImageSequence.Iterator(gif)
  for frame in frame_iter:
    images.append(cv2.cvtColor(np.array(frame.copy().convert('RGB')), cv2.COLOR_RGB2BGR))
  while True:
    k = cv2.waitKey(10)
    image = images[frame_i]
    frame_i += 1
    if frame_i == len(images):
      frame_i = 0
    if k == ord('q'):
      cv2.destroyAllWindows()
      break
    elif k == ord('n'):
      file_i += 1
      images = []
      frame_i = 0
      gif = Image.open(gif_list[file_i].strip())
      frame_iter = ImageSequence.Iterator(gif)
      for frame in frame_iter:
        images.append(cv2.cvtColor(np.array(frame.copy().convert('RGB')), cv2.COLOR_RGB2BGR))
      
    elif k == ord('b'):
      file_i -= 1
      images = []
      frame_i = 0
      gif = Image.open(gif_list[file_i].strip())
      frame_iter = ImageSequence.Iterator(gif)
      for frame in frame_iter:
        images.append(cv2.cvtColor(np.array(frame.copy().convert('RGB')), cv2.COLOR_RGB2BGR))
    cv2.imshow('UI', image)  
    
if __name__ == '__main__':
  get_list()
  #check()
