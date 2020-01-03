import cv2
import numpy as np
from skimage.feature import greycomatrix, greycoprops
from sklearn.cluster import KMeans

histsogram_centers_file_name = 'HistogramCenters.npy'
n_indexed_colors = 256
n_color_histogram_categories = 64
dct2_size = 100
GLCM_resize_size = 200
GLCM_step = 20

# using multithreads
cv2.setUseOptimized(True)
cv2.setNumThreads(8)


def ImgPathToPixelVector(img_path):
    img = cv2.imread(img_path)
    img = cv2.resize(img, (200, 200))
    reshaped_image = img.reshape((-1, 3))
    reshaped_image = np.float32(reshaped_image)
    return reshaped_image


def RGBToIndex(img, color_classes):
    # reconstruct the kmeans from center information
    kmeans = KMeans(n_clusters=n_indexed_colors, random_state=0)
    kmeans.cluster_centers_ = color_classes
    pixel_vector = img.reshape(-1, 3)
    labels = kmeans.predict(pixel_vector)
    return_img = labels
    rows, cols, channels = img.shape
    return return_img.reshape(rows, cols)


def CreateColorHistogram(img):
    histogram = cv2.calcHist([img], [0], None, [n_color_histogram_categories], [0, 256])
    histogram = cv2.normalize(histogram, None)

    ch1_histogram = cv2.calcHist([img], [1], None, [n_color_histogram_categories], [0, 256])
    ch1_histogram = cv2.normalize(ch1_histogram, None)
    histogram = np.vstack((histogram, ch1_histogram))

    ch2_histogram = cv2.calcHist([img], [2], None, [n_color_histogram_categories], [0, 256])
    ch2_histogram = cv2.normalize(ch2_histogram, None)
    histogram = np.vstack((histogram, ch2_histogram))

    return histogram


def CreateIndexedColorHistogram(img, color_classes):
    indexed_img = RGBToIndex(img, color_classes)
    indexed_img = indexed_img.astype(np.uint8)
    histogram = cv2.calcHist([indexed_img], [0], None, [n_indexed_colors], [0, n_indexed_colors])
    histogram = cv2.normalize(histogram, None)
    return histogram


def CreateDCT2(img):
    # gray image for the dct
    grey_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # normalise the image
    NormImg = np.float32(grey_img) / 255.0
    Dct2 = cv2.dct(NormImg)
    Dct2Out = np.zeros([dct2_size, dct2_size])
    Dct2Out = Dct2[:dct2_size, :dct2_size]
    return Dct2Out.reshape(-1, 1)


# greycomatrix
def CreateGLCM(img):
    grey_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    resized_img = cv2.resize(grey_img, (GLCM_resize_size, GLCM_resize_size))

    energy_features = []
    contrast_features = []

    sz = resized_img.shape
    for i in range(0, sz[0], GLCM_step):
        for j in range(0, sz[1], GLCM_step):
            patch = resized_img[i:i + GLCM_step, j:j + GLCM_step]

            glcm = greycomatrix(patch, [1], [0], 256, symmetric=True, normed=True)
            energy_features.append(greycoprops(glcm, 'energy')[0, 0])
            contrast_features.append(greycoprops(glcm, 'contrast')[0, 0])

    out_glsm_features = np.array(energy_features)
    out_glsm_features = np.vstack((out_glsm_features, contrast_features))

    return out_glsm_features.reshape(-1, 1)


def CreateImageFeaturesVector(img, colors_classes):
    features_vector = CreateColorHistogram(img)

    indexed_histogram_features = CreateIndexedColorHistogram(img, colors_classes)
    features_vector = np.vstack((features_vector, indexed_histogram_features))

    dct2_features = CreateDCT2(img)
    features_vector = np.vstack((features_vector, dct2_features))

    GLSM_features = CreateGLCM(img)
    features_vector = np.vstack((features_vector, GLSM_features))

    return features_vector
