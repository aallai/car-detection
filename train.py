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

# Apply random rotation, translation and shear.
def jitter(image, angle=15, shift=4, shear=0.1):
    angle = np.random.uniform(-angle, angle)
    rotation = cv2.getRotationMatrix2D((image.shape[0]/2, image.shape[1]/2), angle, 1)

    dx, dy = np.random.uniform(-shift, shift, (2,))
    shear_x, shear_y = np.random.uniform(-shear, shear, (2,))

    trans = np.array([[1.0, shear_x, dx],[shear_y, 1.0, dy]])

    image = cv2.warpAffine(image, trans, image.shape[:2])
    image = cv2.warpAffine(image, rotation, image.shape[:2])
    return image

def load_dataset(positive_dirs, negative_dirs):
    positives = []
    negatives = []

    for d in positive_dirs:
        for file in glob.glob(d + "/*.png"):
            positives.append(cv2.imread(file))

    for d in negative_dirs:
        for file in glob.glob(d + "/*.png"):
            img = cv2.imread(file)
            negatives.append(img)
            negatives.append(jitter(img))

    print("Number of positive training examples:")
    print(len(positives))
    print("Number of negative training examples:")
    print(len(negatives))

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

        #hist_features = []
        hog_features = []
        hog_images = []

        for i in range(3):
            if visualize:
                features, hog_image = hog(img[...,i], orientations=HOG_ANGLE_BINS, pixels_per_cell=HOG_CELL_SIZE, cells_per_block=HOG_CELLS_PER_BLOCK, block_norm='L2-Hys', visualise=True, transform_sqrt=False, feature_vector=flatten)
                hog_images.append(hog_image)
            else:
                features = hog(img[...,i], orientations=HOG_ANGLE_BINS, pixels_per_cell=HOG_CELL_SIZE, cells_per_block=HOG_CELLS_PER_BLOCK, block_norm='L2-Hys', visualise=False, transform_sqrt=False, feature_vector=flatten)

            hog_features.append(features)

        if flatten:
            data.append(np.array(hog_features).ravel())
        else:
            data.append(hog_features)

        if visualize:
            images.append(hog_images)

    if (visualize):
        visualize_hog(X[0], images[0], colorspace)

    return np.array(data)

def train(X, y):
    scaler = StandardScaler().fit(X)
    normed_X = scaler.transform(X)

    X_train, X_test, y_train, y_test = train_test_split(normed_X, y, test_size=0.2)

    classifier = svm.SVC(kernel='linear', probability=True)
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