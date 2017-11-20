import pickle
import train
import glob
import argparse
import cv2
from train import extract_features

def slide_window(img, height, width, x_overlap = 0.5, y_overlap = 0.5):

    windows = []

    for i in range((img.shape[0] - height) // (int(height * (1-y_overlap))) + 1):
        for j in range((img.shape[1] - width) // (int(width * (1-x_overlap))) + 1):
            v1_x = int(j * width * (1-x_overlap))
            v1_y = int(i * height * (1-y_overlap))
            v2_x = int(v1_x + width)
            v2_y = int(v1_y + height)

            windows.append([[v1_x, v1_y], [v2_x, v2_y]])

    return windows

class Pipeline:

    def __init__(self, model):
        with open(model, 'rb') as file:
            self.classifier, self.scaler, self.colorspace = pickle.load(file)

    def predict(self, image):
        resized = cv2.resize(image, train.TRAINING_IMAGE_SIZE)
        features = extract_features([resized], self.colorspace, flatten=True)
        features = self.scaler.transform(features)
        return self.classifier.predict(features)

    def process_image(self, image, visualize=False):
        # Only search bottom half of image
        half_image = image.shape[0]//2
        img = image[half_image:,...]

        # TODO: investigate computing features only once per image.
        # Search at small resolutions in top of half image.
        windows =[]
        windows += slide_window(img, 64, 128, 0.5, 0.5)
        windows += slide_window(img, 128, 128, 0.75, 0.75)
        windows += slide_window(img, 256, 256, 0.75, 0.75)

        hits = []
        for window in windows:
            sub_image = img[window[0][1]:window[1][1], window[0][0]:window[1][0], ...]
            if self.predict(sub_image) == 1:
                hits.append(window)

        if visualize:
            for window in hits:
                window[0][1] += half_image
                window[1][1] += half_image
                image = cv2.rectangle(image, tuple(window[0]), tuple(window[1]), (0, 0, 255), 6)

        return hits, image

def main():
    parser = argparse.ArgumentParser(description='Run detection pipeline.')
    parser.add_argument('-t', '--test', action='store_true', help='Test single-image processing.')
    parser.add_argument('-m', '--model', default='model.pkl', help='Trained SVM model to use for prediction.')
    args = parser.parse_args()

    if args.test:
        p = Pipeline(args.model)

        i = 0
        for file in glob.glob('./test_images/*.jpg'):
            img = cv2.imread(file)
            _, img = p.process_image(img, True)
            cv2.imwrite('./output_images/detections{}.jpg'.format(i), img)
            i += 1


if __name__ == '__main__':
    main()

