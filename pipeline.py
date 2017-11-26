import pickle
import train
import glob
import argparse
import cv2
import time
import numpy as np
from train import extract_features
from moviepy.editor import VideoFileClip
from scipy.ndimage.measurements import label

HIT_BUFFER_LENGTH = 25
MIN_HITS = 20
CONFIDENCE_THRESHOLD = 0.80

def slide_window(img, width, height, x_overlap = 0.5, y_overlap = 0.5):

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

        return self.classifier.predict_proba(features)[0][1]

    def process_image(self, image, visualize=False, ):
        # Only search bottom half of image
        img = image[image.shape[0]//2:,image.shape[1]//2:,...]

        # TODO: investigate computing features only once per image.
        windows =[]
        windows += slide_window(img[:img.shape[0]//2, ...], 90, 90, 0.75, 0.75)
        windows += slide_window(img[:img.shape[0]//2, ...], 128, 128, 0.75, 0.75)

        hits = []
        for window in windows:
            sub_image = img[window[0][1]:window[1][1], window[0][0]:window[1][0], ...]
            if self.predict(sub_image) >= CONFIDENCE_THRESHOLD:
                hits.append(window)

        for window in hits:
            window[0][0] += image.shape[1]//2
            window[1][0] += image.shape[1]//2
            window[0][1] += image.shape[0]//2
            window[1][1] += image.shape[0]//2
            if visualize:
                image = cv2.rectangle(image, tuple(window[0]), tuple(window[1]), (0, 0, 255), 6)

        return hits, image

    # Implement heat map meathod from the lectures.
    def draw_bounding_boxes(self, image):
        heatmap = np.zeros_like(image[:,:,0])

        for frame in self.hit_buffer:
            for hit in frame:
                heatmap[hit[0][1]:hit[1][1], hit[0][0]:hit[1][0]] += 1

        heatmap[heatmap < MIN_HITS] = 0

        labeled, num_labels = label(heatmap)
        windows = []

        for i in range(1, num_labels+1):
            nonzero = (labeled == i).nonzero()

            y = np.array(nonzero[0])
            x = np.array(nonzero[1])

            top_left = (np.min(x), np.min(y))
            bottom_right = (np.max(x), np.max(y))

            image = cv2.rectangle(image, top_left, bottom_right, (0, 0, 255), 6)

        return image

    def process_video_image(self, image):

        color_converted = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        hits, _ = self.process_image(color_converted)

        self.hit_buffer.append(hits)
        if len(self.hit_buffer) > HIT_BUFFER_LENGTH:
            self.hit_buffer.pop(0)

        return self.draw_bounding_boxes(image)

    def process_video(self, video_file, output_file):
        self.hit_buffer = []
        video = VideoFileClip(video_file)
        output = video.fl_image(self.process_video_image)
        output.write_videofile(output_file, audio=False)

def main():
    parser = argparse.ArgumentParser(description='Run detection pipeline.')
    parser.add_argument('-t', '--test', action='store_true', help='Test single-image processing.')
    parser.add_argument('-m', '--model', default='model.pkl', help='Trained SVM model to use for prediction.')
    parser.add_argument('-v', '--video', type=str, default=None, help='Run video pipeline on specified video clip.')
    parser.add_argument('-o', '--output-file', type=str, default='output_video.mp4', help='Ouput video to specified file.')
    args = parser.parse_args()

    if args.test:
        p = Pipeline(args.model)

        i = 0
        for file in glob.glob('./test_images/*.jpg'):
            img = cv2.imread(file)
            _, img = p.process_image(img, True)
            cv2.imwrite('./output_images/detections{}.jpg'.format(i), img)
            i += 1

    elif args.video:
        p = Pipeline(args.model)
        p.process_video(args.video, args.output_file)

    else:
        parser.print_help()

if __name__ == '__main__':
    main()

