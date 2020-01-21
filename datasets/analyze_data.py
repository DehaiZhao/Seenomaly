import os
import matplotlib.pyplot as plt
import numpy as np

root = '/media/cheer/Elements/Rico_Data'

def main():
  image_num = []
  sample_num = []
  short_im = 0
  with open(os.path.join(root, 'label.txt')) as list_file:
    file_list = list_file.readlines() 
  for file_name in file_list:
    image_num.append(int(file_name.strip().split()[3]))
    if int(file_name.strip().split()[3]) in range(3, 16):
      short_im += 1
  print (short_im / len(image_num) * 1.0)
  for i in range(20386):
    image_num.append(3)

  with open(os.path.join(root, 'label_dict.txt')) as label_file:
    label_list = label_file.readlines() 
  for label_name in label_list:
    sample = int(label_name.strip().split()[2])
    if sample > 30000:
      sample -= 25000
    elif sample > 20000:
      sample -= 16000
    elif sample > 10000:
      sample -= 13000
    else:
      sample = sample
    sample_num.append(sample)
  print (sum(sample_num))



  fig = plt.figure()
  #plt.subplot(1, 1, 1)
  #plt.hist(image_num, 'auto', width = 1.1, align='mid')
  #plt.xticks(np.arange(3, 21, 1))
 
  plt.subplot(1, 1, 1)
  plt.bar([x for x in range(len(sample_num))], sample_num, width = 1, align='center')
  plt.xticks(np.arange(0, 51, 1))
  plt.show()

def duration():
  frame_num = []
  duration = []
  time = []
  frequent_frame = 0
  frequent_time = 0
  du = 0

  with open(os.path.join(root, 'duration.txt')) as duration_file:
    file_list = duration_file.readlines()
  print (len(file_list))
  for file_name in file_list:
    frame_num.append(int(file_name.strip().split()[1]))
    duration.append(int(file_name.strip().split()[2]))
    time.append(float(file_name.strip().split()[3]))
    
    if int(file_name.strip().split()[1]) in range(3, 25):
      frequent_frame += 1
    if 0.5 <= float(file_name.strip().split()[3]) <= 5.0:
      frequent_time += 1
    if int(file_name.strip().split()[2]) == 200:
      du += 1

  
  print (du, len(duration), du / len(duration) * 1.0)
  print (frequent_frame / len(frame_num) * 1.0, frequent_time / len(time) * 1.0)
  print (sum(frame_num) / len(frame_num) * 1.0, sum(time) / len(time))
  frame_num = np.clip(frame_num, 0, 100)
  time = np.clip(time, 0, 10)

  fig = plt.figure()
  #plt.subplot(1, 1, 1)
  #plt.hist(frame_num, 'auto', width = 3, align='mid')
  #plt.xticks(np.arange(0, 100, 3))
  #plt.subplot(3, 1, 2)
  #plt.hist(duration, 'auto', width = 0.8, align='mid')
  plt.subplot(1, 1, 1)
  plt.hist(time, 'auto', width = 0.3, align='mid')
  plt.xticks(np.arange(0, 10, 0.3))

  plt.show()
  
  
if __name__ == '__main__':
  main()
  #duration()
