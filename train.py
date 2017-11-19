import argparse
import numpy as np
import cv2
import glob
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
from skimage.feature import hog

IMAGE_SIZE = (64, 64)

# Hog parameters.
HOG_CELL_SIZE = (8, 8)
HOG_CELLS_PER_BLOCK = (2, 2)
HOG_ANGLE_BINS = 12

# Color histogram parameters.
COLOR_BINS = 8


def load_dataset(positive_dirs, negative_dirs):
    positives = []
    negatives = []

    for d in positive_dirs:
        for file in glob.glob(d + "/*.png"):
            positives.append(cv2.imread(file))

    for d in negative_dirs:
        for file in glob.glob(d + "/*.png"):
            positives.append(cv2.imread(file))

    print(np.array(positives).shape)
    print(np.array(negatives).shape)

    X = np.vstack((positives, negatives))
    y = np.vstack((np.ones(len(positives)), np.zeros(len(negatives))))
    return X, y

#
# Get HOG and LUV histogram features from images.
# This function should not change the order of X.
#
def extract_features(X, colorspace, visualize=False):
    features = []
    for img in X:
        img = cv2.cvtColor(img, colorspace)

        hist_features = []
        hog_features = []
        hog_images = []
        for i in range(3):
            features, hog_image = hog(img, orientations=HOG_ANGLE_BINS, pixels_per_cell=HOG_CELL_SIZE, cells_per_block=HOG_CELLS_PER_BLOCK, block_norm='L2-Hys', visualize=False, transform_sqrt=False, feature_vector=True)
            hog_features.append(features)
            hog_images.append(hog_image)
            hist_features.append(np.histogram(img, COLOR_BINS))

        print(np.float64(hist_features).ravel().shape)
        print(np.float64(hog_features).ravel().shape)

        features.append(np.concatenate((np.float64(hist_features).ravel(), np.float64(hog_features).ravel())))

    return np.array(features)

def get_colorspace(str):
    if str == 'RGB':
        return cv2.COLOR_BGR2RGB
    elif str == 'LUV':
        return cv2.COLOR_BGR2LUV
    elif str == 'HLS':
        return cv2.COLOR_BGR2HLS
    elif str == 'YUV':
        return cv2.COLOR_BGR2YUV
    elif str == 'YCrCb':
        return cv2.COLOR_BGR2YCrCb
    else:
        raise Exception('Invalid colorspace provided: ' + str)

def main():
    parser = argparse.ArgumentParser(description='Train classifier')
    parser.add_argument('-p', '--posisitve-dirs', nargs='+', default=[], help='Directory(s) containing positive (vehicle) training images.')
    parser.add_argument('-n', '--negative-dirs', nargs='+', default=[], help='Directory(s) containing negative (non-vehicle) training images.')
    parser.add_argument('-c', '--colorspace', default='LUV', help='Colorspace to use for features. Can be RGB, LUV, HLS, YUV, or YCrCb.')
    args = parser.parse_args()

    if (not args.posisitve_dirs or not args.negative_dirs):
        raise Exception('You must provide both positive and negative training data.')

    X, y = load_dataset(args.posisitve_dirs, args.negative_dirs)
    X = extract_features(X, get_colorspace(args.colorspace))

    print(X.shape)

if __name__ == '__main__':
    main()