import os
import sys
import glob
import time

import cv2
import numpy as np

import tensorflow as tf
from object_detection.utils import label_map_util, config_util, visualization_utils as viz_utils
from object_detection.builders import model_builder

from deepsort.deepsort_tracker import DeepSort
from utils.paths import paths, files
from utils.labelmap import read_label_map
from detect import DetectionModel


TRESH = 0.75
LABELS = read_label_map("Tensorflow/workspace/annotations/label_map_2.pbtxt")


def get_tracker_detections(
    tresh,
    detection_scores,
    detection_boxes,
    detection_labels,
    width,
    height,
):
    #                    detections format:
    #  [xmin ymin width height]  confidence     class/name
    # [([33, 69, 604, 397], 0.5679649710655212, 'Drone')]
    detections = []

    detection_scores = np.array(detection_scores)
    detection_boxes = np.array(detection_boxes)
    detection_labels = np.array(detection_labels)

    detection_boxes = detection_boxes[detection_scores > tresh]
    detection_labels = detection_labels[detection_scores > tresh]
    detection_scores = detection_scores[detection_scores > tresh]

    for score, coords, label_id in zip(detection_scores, detection_boxes, detection_labels):
        # if score < tresh:
        #     continue
        y_min, x_min, y_max, x_max = coords

        x_min = (x_min * width).astype("int")
        x_max = (x_max * width).astype("int")
        y_min = (y_min * height).astype("int")
        y_max = (y_max * height).astype("int")
        w = x_max - x_min
        h = y_max - y_min

        label = LABELS[label_id + 1]

        detections.append(([x_min, y_min, w, h], score, label))

    return detections


def main():

    cap = cv2.VideoCapture(0)
    # cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
    # cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 320)

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))


    # create models
    detection_model = DetectionModel()

    os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

    object_tracker = DeepSort(
        max_age=7,
        n_init=2,
        nms_max_overlap=1.0,
        max_cosine_distance=0.8,
        nn_budget=None,
        override_track_class=None,
        embedder="custom",
        half=True,
        bgr=True,
        embedder_gpu=True,
        embedder_model_name=None,
        embedder_wts=None,
        polygon=False,
        today=None,
    )


    while cap.isOpened():

        ret, img = cap.read()

        # img = cv2.rotate(img, cv2.ROTATE_180)

        start = time.perf_counter()

        # detection
        input_tensor = tf.convert_to_tensor(np.expand_dims(img, 0), dtype=tf.float32)
        detections = detection_model.detect_fn(input_tensor)

        num_detections = int(detections.pop('num_detections'))
        detections = {
            key : value[0, :num_detections].numpy()
            for key, value in detections.items()
        }
        detections['num_detections'] = num_detections

        # detection_classes should be int
        detections['detection_classes'] = detections['detection_classes'].astype('int')


        # tracking
        tracker_detections = get_tracker_detections(
            TRESH,
            detections['detection_scores'],
            detections['detection_boxes'],
            detections['detection_classes'],
            width,
            height,
        )


        if len(tracker_detections) > 0:
            tracks = object_tracker.update_tracks(tracker_detections, frame=img) # bbs expected to be a list of detections, each in tuples of ( [left,top,w,h], confidence, detection_class )

            for track in tracks:
                if not track.is_confirmed():
                    continue
                track_id = track.track_id
                ltrb = track.to_ltrb()
                confidence = track.get_det_conf()

                bbox = ltrb

                cv2.rectangle(img, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (0, 255, 0), 2)
                cv2.putText(
                    img,
                    f"ID: {str(track_id)} - {str(confidence)[:4]}",
                    (int(bbox[0]), int(bbox[1] - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.75,
                    (0, 255, 0),
                    2,
                )


        # print framerate
        end = time.perf_counter()
        total_time = end - start
        fps = 1 / total_time

        cv2.putText(img, f'FPS: {round(fps, 2)}', (20,70), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,0), 2)
        img = cv2.resize(img, (640, 640))
        cv2.imshow('img', img)

        # to cancel press "q"
        if (cv2.waitKey(10) & 0xff == ord('q')):
            break

    # Release and destroy all windows before termination
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
