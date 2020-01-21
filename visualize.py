import matplotlib.pyplot as plt
import os
import numpy as np
from scipy.misc import imsave
from scipy import misc
from sklearn.decomposition import PCA
from scipy.spatial import distance
from tqdm import tqdm
import pickle
import json
import random
import math
from matplotlib.pyplot import imshow
from PIL import Image
from sklearn.manifold import TSNE
import rasterfairy

batch_size = 1
image_size = 224
num_images_to_plot = 400
net_name = 'vae'
_STRIDE = 8
num_classes = 50 
dataset_dir = '/home/cheer/Project/AutoTest/Rico_Data'
save_dir = '/home/cheer/Project/Do_Dont/demo'

def _sample_image(file_name):
  file_list = os.listdir(file_name)
  image_list = []
  if len(file_list) in range(int(_STRIDE / 2) + 1, _STRIDE):
    sample_list = file_list + random.sample(file_list, _STRIDE - len(file_list))
    sample_list.sort()
  else:  
    file_list = file_list * int(math.ceil(_STRIDE * 1.0 / len(file_list)))
    file_list.sort()
    sample_list = random.sample(file_list, _STRIDE)
    sample_list.sort()
  for sample in sample_list:
    image_list.append(os.path.join(file_name, sample))
  return image_list

def main():
  images, pca_features = pickle.load(open(os.path.join(save_dir, 'features.p'), 'r'))
  for i, f in list(zip(images, pca_features))[0:5]:
    print (i, f[0], f[1], f[2])

  if len(images) > num_images_to_plot:
    sort_order = sorted(random.sample(range(len(images)), num_images_to_plot))
    images = [images[i] for i in sort_order]
    pca_features = [pca_features[i] for i in sort_order]

  X = np.array(pca_features)
  tsne = TSNE(n_components=2, learning_rate=150, perplexity=30, angle=0.2, verbose=2).fit_transform(X)

  tx, ty = tsne[:,0], tsne[:,1]
  tx = (tx-np.min(tx)) / (np.max(tx) - np.min(tx))
  ty = (ty-np.min(ty)) / (np.max(ty) - np.min(ty)) 
  #random.shuffle(tx)
  #random.shuffle(ty) 

  T = np.arctan2(ty, tx)
  print (type(T))
  print (len(T))

  #plt.scatter(tx,ty,s=55, c=T, alpha=.3)
  #plt.savefig(os.path.join(save_dir, 'image', 'scatter.png'))

  width = 4000
  height = 3000
  max_dim = 120

  full_list = []

  for gif_file in images:
    sample_list = _sample_image(gif_file)
    full_list.append(sample_list)
  full_list = np.array(full_list)
  for i in range(_STRIDE):
    image_list = full_list[:, i]
    full_image = Image.new('RGBA', (width, height))
    for img, x, y in tqdm(zip(image_list, tx, ty)):
      tile = Image.open(img)
      rs = max(1, tile.width/max_dim, tile.height/max_dim)
      tile = tile.resize((int(tile.width/rs), int(tile.height/rs)), Image.ANTIALIAS)
      full_image.paste(tile, (int((width-max_dim)*x), 3000-int((height-max_dim)*y)), mask=tile.convert('RGBA'))

    #plt.figure(figsize = (16,12))
    #imshow(full_image)
    full_image.save(os.path.join(save_dir, 'image', str(i) + '.png'))
    #plt.show()

  tsne_path = os.path.join(save_dir, 'grid.json')

  data = [{"path":os.path.abspath(img), "point":[float(x), float(y)]} for img, x, y in zip(images, tx, ty)]
  with open(tsne_path, 'w') as outfile:
    json.dump(data, outfile)

  print "saved t-SNE result to %s" % tsne_path
  exit(1)

  arrangements = rasterfairy.getRectArrangements(num_images_to_plot)
  nx = arrangements[0][1] 
  ny = arrangements[0][0]
  grid_assignment = rasterfairy.transformPointCloud2D(tsne, target=(nx, ny))
  print 'finished', arrangements


  tile_width = 72
  tile_height = 56

  full_width = tile_width * nx
  full_height = tile_height * ny
  aspect_ratio = float(tile_width) / tile_height

  grid_image = Image.new('RGB', (full_width, full_height))

  grid_list = grid_assignment[0]

  for i in range(_STRIDE):
    image_list = full_list[:, i]
    for img, grid_pos in tqdm(zip(image_list, grid_list)):
      idx_x, idx_y = grid_pos
      x, y = tile_width * idx_x, tile_height * idx_y
      tile = Image.open(img)
      tile_ar = float(tile.width) / tile.height  # center-crop the tile to match aspect_ratio
      if (tile_ar > aspect_ratio):
          margin = 0.5 * (tile.width - aspect_ratio * tile.height)
          tile = tile.crop((margin, 0, margin + aspect_ratio * tile.height, tile.height))
      else:
          margin = 0.5 * (tile.height - float(tile.width) / aspect_ratio)
          tile = tile.crop((0, margin, tile.width, margin + float(tile.width) / aspect_ratio))
      tile = tile.resize((tile_width, tile_height), Image.ANTIALIAS)
      grid_image.paste(tile, (int(x), int(y)))

    plt.figure(figsize = (16,12))
    grid_image.save(os.path.join(save_dir, 'image', str(i) + '_grid.png'))

if __name__ == '__main__':
  main()

