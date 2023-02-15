from deepsort.embedder.embedder_pytorch import (
    MobileNetv2_Embedder as Embedder,
)
import cv2


#############
# embedder must return list of np.arrays with 1280 features of visual NN
#############



half=True
bgr=True
embedder_gpu=True
embedder_model_name=None
embedder_wts=None
polygon=False
today=None


class Tracker():
    def __init__(
        self,
        half=True,
        bgr=True,
        embedder_gpu=True,
        embedder_model_name=None,
        embedder_wts=None,
        polygon=False,
        today=None,
    ):

        self.embedder = Embedder(
            half=half,
            max_batch_size=16,
            bgr=bgr,
            gpu=embedder_gpu,
            model_wts_path=embedder_wts,
        )

    def generate_embeds(self, frame, raw_dets):
        crops = self.crop_bb(frame, raw_dets)
        return self.embedder.predict(crops)

    def generate_embeds_poly(self, frame, polygons, bounding_rects):
        crops = self.crop_poly_pad_black(frame, polygons, bounding_rects)
        return self.embedder.predict(crops)

    @staticmethod
    def crop_bb(frame, raw_dets):
        crops = []
        im_height, im_width = frame.shape[:2]
        for detection in raw_dets:
            l, t, w, h = [int(x) for x in detection[0]]
            r = l + w
            b = t + h
            crop_l = max(0, l)
            crop_r = min(im_width, r)
            crop_t = max(0, t)
            crop_b = min(im_height, b)
            crops.append(frame[crop_t:crop_b, crop_l:crop_r])
        return crops

    @staticmethod
    def crop_poly_pad_black(frame, polygons, bounding_rects):
        masked_polys = []
        im_height, im_width = frame.shape[:2]
        for polygon, bounding_rect in zip(polygons, bounding_rects):
            mask = np.zeros(frame.shape, dtype=np.uint8)
            polygon_mask = np.array([polygon]).astype(int)
            cv2.fillPoly(mask, polygon_mask, color=(255, 255, 255))

            # apply the mask
            masked_image = cv2.bitwise_and(frame, mask)

            # crop masked image
            x, y, w, h = bounding_rect
            crop_l = max(0, x)
            crop_r = min(im_width, x + w)
            crop_t = max(0, y)
            crop_b = min(im_height, y + h)
            cropped = masked_image[crop_t:crop_b, crop_l:crop_r].copy()
            masked_polys.append(np.array(cropped))
        return masked_polys

tracker = Tracker()
print(tracker.embedder)


img = cv2.imread("drone2.jpg")
raw_dets = [
    ([33, 69, 604, 397], 0.5679649710655212, 'Drone'),
    ([39, 39, 404, 297], 0.9710655213421223, 'Drone'),
]
pred = tracker.generate_embeds(img, raw_dets)
print(pred)
print(len(pred))
print(type(pred))
for p in pred:
    print(len(p))
    print(type(p))
print()
# model
