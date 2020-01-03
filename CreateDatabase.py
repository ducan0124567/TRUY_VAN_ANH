import os.path
import time
from platform import platform

import cv2
import numpy as np
from glob2 import glob
from sklearn.cluster import KMeans

import ImageUtils as IU

# using multithreads
cv2.setUseOptimized(True)
cv2.setNumThreads(8)

# Calculate time
t0 = time.time()
print('Exploring folders...')
# get path
folders_path = os.path.realpath('') + '/database/'

# get all folders in path
folders = glob(folders_path + '*/')

print('Exploring files...')

img_files = []

# now only get image files in each folder (category)
for folder in folders:
    files_path = folder
    print('Folder: ' + files_path)

    img_files.extend(glob(files_path + '*.JPG'))
    img_files.extend(glob(files_path + '*.JPEG'))
    img_files.extend(glob(files_path + '*.BMP'))
    img_files.extend(glob(files_path + '*.PNG'))


if len(img_files) > 0:
    print('Search complete,', len(img_files), 'image(s) found.')
else:
    exit()

print('Creating pixel features for color indexing...')
# a vector containing the pixels of all images to calculate the color indexing with
pixels_vector = IU.ImgPathToPixelVector(img_files[0])
# read all images and create a vector of them to index
if len(img_files) > 1:
    for i in range(1, len(img_files)):
        img_file = img_files[i]
        reshaped_image = IU.ImgPathToPixelVector(img_file)
        pixels_vector = np.vstack((pixels_vector, reshaped_image))

        percent = (i + 1) / len(img_files) * 100.0
        percent_text = 'Preparing indexed color data ' + str(int(percent)) + '%'
        print(percent_text, end='\r', flush=True)

print('Calculating cluster centers...')
kmeans = KMeans(n_clusters=IU.n_indexed_colors, n_init=1, tol=0.001, max_iter=100, random_state=0, n_jobs=1,
                algorithm='full')
kmeans.fit(pixels_vector)

centers = kmeans.cluster_centers_
print('Cluster centers\' calculation complete !!!')

print('Saving indexed color classes...')
centers = np.uint8(centers)
np.save(IU.histsogram_centers_file_name, centers)

pixels_vector = None

# create the feature vector for every image
for i, img_file in enumerate(img_files):
    # read image fi
    img = cv2.imread(img_file, 1)
    # convert images to RGB
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    features_vector = IU.CreateImageFeaturesVector(img, centers)

    # Save features vector to file
    features_vector_file_name = img_file + '.npy'
    np.save(features_vector_file_name, features_vector)

    percent = (i + 1) / len(img_files) * 100
    percent_text = 'Creating images feature vector ' + str(int(percent)) + '%'
    print(percent_text, end='\r', flush=True)

t1 = time.time()
print('Complete, time elapsed:', t1 - t0)
