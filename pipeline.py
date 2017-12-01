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

HIT_BUFFER_LENGTH = 12
MIN_HITS = 20
CONFIDENCE_THRESHOLD = 0.6

IMAGE_SHAPE = (720, 1280, 3)

class Pipeline:

    def __init__(self, model):
        with open(model, 'rb') as file:
            self.classifier, self.scaler, self.colorspace = pickle.load(file)
        self.frame_counter = 0

    def slide_window(self, image, y_start, x_start, y_end, x_end, scale, x_overlap = 0.5, y_overlap = 0.5):

        image = cv2.resize(image, (int(image.shape[1] / scale), int(image.shape[0] / scale)))

        hog_image, scaled_img = extract_features([image], self.colorspace)[0]

        height_blocks = train.TRAINING_IMAGE_SIZE[0]//train.HOG_CELL_SIZE[0] - train.HOG_CELLS_PER_BLOCK[0] + 1
        width_blocks = train.TRAINING_IMAGE_SIZE[1]//train.HOG_CELL_SIZE[1] - train.HOG_CELLS_PER_BLOCK[1] + 1
        y_block_start = int(y_start / scale)//train.HOG_CELL_SIZE[0]
        x_block_start = int(x_start / scale)//train.HOG_CELL_SIZE[1]
        y_block_end = int(y_end / scale)//train.HOG_CELL_SIZE[0] - 1
        x_block_end = int(x_end / scale)//train.HOG_CELL_SIZE[1] - 1
        y_block_step = height_blocks - int(height_blocks * y_overlap)
        x_block_step = width_blocks - int(width_blocks * x_overlap)

        hits = []
        for i in range((y_block_end - y_block_start - height_blocks) // y_block_step + 1):
            for j in range((x_block_end - x_block_start - width_blocks) // x_block_step + 1):
                v1_x = j * x_block_step + x_block_start
                v1_y = i * y_block_step + y_block_start
                v2_x = v1_x + width_blocks
                v2_y = v1_y + height_blocks

                v1_x_spatial = int(v1_x*train.HOG_CELL_SIZE[1]*train.SPATIAL_FEATURE_SCALE)
                v1_y_spatial = int(v1_y*train.HOG_CELL_SIZE[1]*train.SPATIAL_FEATURE_SCALE)

                spatial_features = scaled_img[v1_y_spatial:v1_y_spatial+int(train.TRAINING_IMAGE_SIZE[1]*train.SPATIAL_FEATURE_SCALE),
                                              v1_x_spatial:v1_x_spatial+int(train.TRAINING_IMAGE_SIZE[0]*train.SPATIAL_FEATURE_SCALE), ...]

                features = np.concatenate([hog_image[:, v1_y:v2_y, v1_x:v2_x, ...].ravel(), spatial_features.ravel()])

                if self.predict([features]) >= CONFIDENCE_THRESHOLD:
                    v1_x *= int(train.HOG_CELL_SIZE[1] * scale)
                    v1_y *= int(train.HOG_CELL_SIZE[0] * scale)
                    v2_x *= int(train.HOG_CELL_SIZE[1] * scale)
                    v2_y *= int(train.HOG_CELL_SIZE[0] * scale)

                    hits.append([[v1_x, v1_y], [v2_x, v2_y]])

        return hits

    def predict(self, features):
        features = self.scaler.transform(features)
        return self.classifier.predict_proba(features)[0][1]

    def process_image(self, image, visualize=False):

        hits =[]
        hits += self.slide_window(image, IMAGE_SHAPE[0]//2, IMAGE_SHAPE[1]//2, IMAGE_SHAPE[0]//2 + 128, IMAGE_SHAPE[1]//2, 1, 0.90, 0.90)
        hits += self.slide_window(image, IMAGE_SHAPE[0]//2, IMAGE_SHAPE[1]//2, 3*(IMAGE_SHAPE[0]//4), IMAGE_SHAPE[1], 1.5, 0.90, 0.90)
        hits += self.slide_window(image, IMAGE_SHAPE[0]//2, IMAGE_SHAPE[1]//2, 3*(IMAGE_SHAPE[0]//4), IMAGE_SHAPE[1], 2, 0.75, 0.75)

        if visualize:
            for window in hits:
                image = cv2.rectangle(image, tuple(window[0]), tuple(window[1]), (0, 0, 255), 6)

        return hits, image

    # Implement heat map method from the lectures.
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

        self.frame_counter += 1

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
            hits, img = p.process_image(img, True)
            cv2.imwrite('./output_images/detections{}.jpg'.format(i), img)
            i += 1

    elif args.video:
        p = Pipeline(args.model)
        p.process_video(args.video, args.output_file)

    else:
        parser.print_help()

if __name__ == '__main__':
    main()

