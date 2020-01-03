import math
import os.path
import time
import tkinter as tk
from tkinter import filedialog

import cv2
import numpy as np
from glob2 import glob
from matplotlib import pyplot as plt
from scipy.spatial import distance

import ImageUtils as IU

# using multithreads
cv2.setUseOptimized(True)
cv2.setNumThreads(8)

print('Exploring folders...')
# get the path
folders_path = os.path.realpath('') + '/database/'
# get all folders in path
folders = glob(folders_path + '*/')

print('Exploring files...')

feature_files = []

for folder in folders:
    files_path = folder
    print('Folder: ' + files_path)
    # features are saved as a numpy ndarray
    feature_files.extend(glob(files_path + '*.npy'))

if len(feature_files) > 0:
    print('Search complete,', len(feature_files), 'image feature files found.')
else:
    print('No feature files found, exiting script.')
    exit()

feature_vectors = []

# loop over all files and load the features
for i, feature_file in enumerate(feature_files):
    img_path = feature_file.replace('.npy', '')
    feature_vectors.append([img_path, np.load(feature_file)])

    percent = (i + 1) / len(feature_files) * 100
    percent_text = 'Loading features ' + str(int(percent)) + '%'
    print(percent_text, end='\r', flush=True)

print('Loading indexed color lookup table...')

centers = np.load(IU.histsogram_centers_file_name)

root_window = tk.Tk()
root_window.withdraw()

widow_title = 'Choose image'
image_file_types = ('Image files', '*.jpg *.jpeg *.bmp *.png *.tiff *.tif *.JPG *.JPEG *.BMP *.PNG *.TIFF *.TIF')
file_types = (image_file_types, ("all files", "*.*"))
img_to_test_path = filedialog.askopenfilename(initialdir=folders_path, title=widow_title, filetypes=file_types)

print('Selected image:', img_to_test_path)

print('Reading image and creating features..')
t0 = time.time()
search_img = cv2.imread(img_to_test_path, 1)
search_img = cv2.cvtColor(search_img, cv2.COLOR_BGR2RGB)
img_features_vector = IU.CreateImageFeaturesVector(search_img, centers)

# Sort the list of feature vectors based on euclidean distance
feature_vectors.sort(key=lambda feature_vector: distance.euclidean(feature_vector[1], img_features_vector))
print('Search complete, time elapsed:', time.time() - t0)

print('K = ')
k = int(input())
n_matching_images_to_show = k
n_matching_images_to_show_real = n_matching_images_to_show + 1
check = math.sqrt(n_matching_images_to_show_real) % 1
if check == 0:
    n_cols = math.sqrt(n_matching_images_to_show_real)
    n_rows = math.sqrt(n_matching_images_to_show_real)
else:
    n_cols = math.sqrt(n_matching_images_to_show_real) / 1 + 1
    n_rows = math.sqrt(n_matching_images_to_show_real) / 1 + 1

results_figure = plt.figure()

results_figure.add_subplot(n_rows, n_cols, 1)
plt.axis('off')
plt.title('Search image')
plt.imshow(search_img)

if n_matching_images_to_show > n_rows * n_cols:
    n_matching_images_to_show = n_rows * n_cols

if n_matching_images_to_show > len(feature_vectors):
    n_matching_images_to_show = len(feature_vectors)

for i in range(0, n_matching_images_to_show):
    results_figure.add_subplot(n_rows, n_cols, i + 2)
    plt.axis('off')
    plt.title(str(i + 1))
    # read image file
    match = cv2.imread(feature_vectors[i][0], 1)
    # convert all images to RGB
    match = cv2.cvtColor(match, cv2.COLOR_BGR2RGB)
    plt.imshow(match)

plt.show()
