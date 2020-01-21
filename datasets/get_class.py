import os
import glob
from tqdm import tqdm

root = '/media/cheer/Elements/Rico_Data'
split_image = os.path.join(root, 'split_image')

def rico():
  class_dict = dict()
  clip_dict = dict()
  class_id = 0
  class_list = []
  app_list = os.listdir(split_image)
  for app in tqdm(app_list):
    trace_list = glob.glob(os.path.join(split_image, app, 'trace_*'))
    for trace in trace_list:
      try:
        with open(os.path.join(trace, 'class_name.txt'), 'r') as class_file:
          lines = class_file.readlines()
        for line in lines:
          class_name = line.split()[1]
          if class_dict.get(class_name) is None:
            class_dict[class_name] = 0
          else:
            class_dict[class_name] = class_dict.get(class_name) + 1          
      except:
        pass
  sorted_list = sorted(class_dict.items(), key = lambda item:item[1], reverse = True)
  print (len(sorted_list))
  exit(1)
  for item in sorted_list[:50]:
    class_list.append(item[0] + ' ' + str(class_id) + ' ' + str(item[1]) + '\n')
    clip_dict[item[0]] = class_id 
    class_id += 1
  with open(os.path.join(root, 'label_dict.txt'), 'w') as label_dict:
    label_dict.writelines(class_list)

  for app in tqdm(app_list):
    trace_list = glob.glob(os.path.join(split_image, app, 'trace_*'))
    for trace in trace_list:
      try:
        with open(os.path.join(trace, 'class_name.txt'), 'r') as class_file:
          lines = class_file.readlines()
        for line in lines:
          image_name = line.split()[0]
          class_name = line.split()[1] 
          box = line.split()[2]
          image_num = len(os.listdir(os.path.join(trace, image_name)))
          with open(os.path.join(root, 'label.txt'), 'a') as label_file: 
            label_file.write(os.path.join(trace, image_name) + ' ' + str(clip_dict[class_name]) + ' ' + box + ' ' + str(image_num) + '\n')
      except:
        pass    
  
  
if __name__ == '__main__':
  rico()
