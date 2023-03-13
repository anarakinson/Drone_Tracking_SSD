import os
import tensorflow as tf
import cv2
import numpy as np
from object_detection.utils import label_map_util, config_util, visualization_utils as viz_utils
from object_detection.builders import model_builder
import time

from utils.paths import paths, files
from utils.labelmap import read_label_map

LABELS = read_label_map("Tensorflow/workspace/annotations/label_map_2.pbtxt")
TRESH = 0.75

class DetectionModel():
    def __init__(self):
        # load pipeline config and build model
        self.model_config = config_util.get_configs_from_pipeline_file(files['PIPELINE_CONFIG'])
        self.detection_model = model_builder.build(model_config=self.model_config['model'], is_training=False)

        # restore checkpoint
        ckpt = tf.train.Checkpoint(model=self.detection_model)
        checkpoint_num = tf.train.latest_checkpoint(paths['CHECKPOINT_PATH'])
        print("checkpoint_num", checkpoint_num)
        ckpt.restore(os.path.join(checkpoint_num)).expect_partial()


    @tf.function
    def detect_fn(self, image):
        image, shapes = self.detection_model.preprocess(image)
        # print(shapes)
        # print(image)
        # print(tf.get_static_value(shapes))
        # print(tf.get_static_value(image))
        prediction_dict = self.detection_model.predict(image, shapes)
        detections = self.detection_model.postprocess(prediction_dict, shapes)
        return detections


def draw_bboxes(
    frame,
    x_min,
    x_max,
    y_min,
    y_max,
    score,
    label,
):

    frame = cv2.rectangle(
        frame,
        (x_min, y_max),
        (x_max, y_min),
        (0, 255, 0),
        2,
    )

    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(
        frame,
        f'{label} - {str(score)[:4]}',
        (x_min, y_min - 10),
        font,
        0.75,
        (0, 255, 0),
        2,
        cv2.LINE_AA
    )

def draw_result(
    frame,
    tresh,
    detection_scores,
    detection_boxes,
    detection_labels,
    width,
    height,
):

    for score, coords, label_id in zip(detection_scores, detection_boxes, detection_labels):
        y_min, x_min, y_max, x_max = coords
        if score < tresh:
            continue

        x_min = (x_min * width).astype("int")
        x_max = (x_max * width).astype("int")
        y_min = (y_min * height).astype("int")
        y_max = (y_max * height).astype("int")

        label = LABELS[label_id + 1]
        draw_bboxes(frame, x_min, x_max, y_min, y_max, score, label)


def main():
    category_index = label_map_util.create_category_index_from_labelmap(files['LABELMAP'])

    start_time = time.time()
    detection_model = DetectionModel()
    end_time = time.time()
    print(f"Model loaded. Time passed: {round(end_time - start_time, 2)} sec")

    # open camera
    cap = cv2.VideoCapture(0)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # get detections
    while cap.isOpened():
        start_time = time.time()

        ret, img = cap.read()
        input_tensor = tf.convert_to_tensor(np.expand_dims(img, 0), dtype=tf.float32)

        # detection
        detections = detection_model.detect_fn(input_tensor)

        num_detections = int(detections.pop('num_detections'))
        detections = {
            key : value[0, :num_detections].numpy()
            for key, value in detections.items()
        }
        detections['num_detections'] = num_detections

        # detection_classes should be int
        detections['detection_classes'] = detections['detection_classes'].astype('int')

        # draw bboxes
        img_detections = img.copy()
        draw_result(
            img_detections,
            TRESH,
            detections['detection_scores'],
            detections['detection_boxes'],
            detections['detection_classes'],
            width,
            height,
        )

        # print framerate
        end_time = time.time()
        total_time = end_time - start_time
        fps = 1 / total_time

        cv2.putText(img_detections, f'FPS: {round(fps, 2)}', (20,70), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,0), 2)
        img_detections = cv2.resize(img_detections, (640, 640))
        cv2.imshow("detections", img_detections)

        # to cancel press "q"
        if (cv2.waitKey(10) & 0xff == ord('q')):
            break

    # Release and destroy all windows before termination
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
