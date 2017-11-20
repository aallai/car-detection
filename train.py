import argparse
import numpy as np
import cv2
import glob
import pickle
import matplotlib.pyplot as plt
from sklearn import svm, model_selection
from sklearn.preprocessing import StandardScaler
from skimage.feature import hog
from sklearn.model_selection import train_test_split

TRAINING_IMAGE_SIZE = (64, 64)

# Hog parameters.
HOG_CELL_SIZE = (8, 8)
HOG_CELLS_PER_BLOCK = (2, 2)
HOG_ANGLE_BINS = 12

# Color histogram parameters.
COLOR_BINS = 32

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

def load_dataset(positive_dirs, negative_dirs):
    positives = []
    negatives = []

    for d in positive_dirs:
        for file in glob.glob(d + "/*.png"):
            positives.append(cv2.imread(file))

    for d in negative_dirs:
        for file in glob.glob(d + "/*.png"):
            negatives.append(cv2.imread(file))

    images = np.vstack((positives, negatives))
    labels = np.hstack((np.ones(len(positives)), np.zeros(len(negatives))))
    return images, labels

def visualize_hog(original, hog_image, colorspace):
    fig = plt.figure()
    plt.title('HOG features for colorspace {}.'.format(colorspace))
    plt.subplot(1, 4, 1)
    plt.imshow(original, cmap='gray')

    for i in range(3):
        plt.subplot(1, 4, 2 + i)
        plt.imshow(hog_image[i], cmap='gray')

    plt.savefig('./output_images/hog.png')

#
# Get HOG and histogram features from images.
# This function should not change the order of X.
#
def extract_features(X, colorspace, flatten=False, visualize=False):

    data = []
    images = []
    for img in X:
        img = cv2.cvtColor(img, get_colorspace(colorspace))

        hist_features = []
        hog_features = []
        hog_images = []

        for i in range(3):
            if visualize:
                features, hog_image = hog(img[...,i], orientations=HOG_ANGLE_BINS, pixels_per_cell=HOG_CELL_SIZE, cells_per_block=HOG_CELLS_PER_BLOCK, block_norm='L2-Hys', visualise=True, transform_sqrt=False, feature_vector=flatten)
                hog_images.append(hog_image)
            else:
                features = hog(img[...,i], orientations=HOG_ANGLE_BINS, pixels_per_cell=HOG_CELL_SIZE, cells_per_block=HOG_CELLS_PER_BLOCK, block_norm='L2-Hys', visualise=False, transform_sqrt=False, feature_vector=flatten)

            hog_features.append(features)
            hist_features.append(np.histogram(img, COLOR_BINS)[0])

        if flatten:
            data.append(np.concatenate((np.array(hist_features).ravel(), np.array(hog_features).ravel())))
        else:
            data.append((hist_features, hog_features))

        if visualize:
            images.append(hog_images)

    if (visualize):
        visualize_hog(X[0], images[0], colorspace)

    return np.array(data)

def train(X, y):
    scaler = StandardScaler().fit(X)
    normed_X = scaler.transform(X)

    X_train, X_test, y_train, y_test = train_test_split(normed_X, y, test_size=0.2)

    classifier = svm.SVC(kernel='rbf')
    classifier.fit(X_train, y_train)

    print("Classifier accuracy:")
    print(classifier.score(X_test, y_test))

    return classifier, scaler

def main():
    parser = argparse.ArgumentParser(description='Train classifier')
    parser.add_argument('-p', '--posisitve-dirs', nargs='+', default=[], help='Directory(s) containing positive (vehicle) training images.')
    parser.add_argument('-n', '--negative-dirs', nargs='+', default=[], help='Directory(s) containing negative (non-vehicle) training images.')
    parser.add_argument('-c', '--colorspace', default='LUV', help='Colorspace to use for features. Can be RGB, LUV, HLS, YUV, or YCrCb.')
    parser.add_argument('-v', '--visualize', action='store_true', help='Visualize hog features.')
    args = parser.parse_args()

    if (not args.posisitve_dirs or not args.negative_dirs):
        raise Exception('You must provide both positive and negative training data.')

    images, y = load_dataset(args.posisitve_dirs, args.negative_dirs)
    X = extract_features(images, args.colorspace, True, args.visualize)

    print("Dataset dimensions:")
    print(X.shape)

    classifier, scaler = train(X, y)

    with open('model.pkl', 'wb') as file:
        pickle.dump((classifier, scaler, args.colorspace), file)

if __name__ == '__main__':
    main()