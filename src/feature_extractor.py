import numpy as np
from cv2 import COLOR_BGR2GRAY, COLOR_BGR2HSV, COLOR_BGR2LAB, cvtColor
from skimage.feature import hog, local_binary_pattern
from skimage.segmentation import slic

COLORSPACES = (COLOR_BGR2HSV, COLOR_BGR2LAB)


class FeatureExtractor:
    def __init__(self, image=None, colorspace=COLOR_BGR2HSV, debug=False):
        if image is None:
            raise ValueError("Image cannot be 'None'. Load an image using cv2.imread()")

        self.bgr_image = image
        self.color_image = cvtColor(
            image.copy(), colorspace if colorspace in COLORSPACES else COLOR_BGR2HSV
        )
        self.gray_image = cvtColor(image.copy(), COLOR_BGR2GRAY)
        self.debug = debug

    def extract(self):
        if self.debug:
            self._debug()

        self._color_features()
        self._hog_features()
        self._lbp_features()
        self._superpixel_features()

        all_features = []
        all_features.extend(self.color_features)
        all_features.extend(self.hog_features)
        all_features.extend(self.lbp_features)
        all_features.extend(self.superpixel_features)

        self.all_features = np.array(all_features, dtype=np.float32)

        return self.all_features

    def _color_features(self, bins=32):
        features = []

        image = self.color_image.astype("float32") / 255.0

        for i in range(image.shape[2]):
            histogram = np.histogram(image[:, :, i], bins=bins, range=(0, 1))[0]
            histogram = histogram / (np.sum(histogram) + 1e-6)
            features.extend(histogram)

        self.color_features = features

    def _hog_features(
        self,
        orientations=9,
        pixels_per_cell=(8, 8),
        cells_per_block=(2, 2),
        block_norm="L2-Hys",
        feature_vector=True,
    ):
        image = self.gray_image.astype("float32") / 255.0

        hog_features = hog(
            image,
            orientations=orientations,
            pixels_per_cell=pixels_per_cell,
            cells_per_block=cells_per_block,
            block_norm=block_norm,
            feature_vector=feature_vector,
        )

        self.hog_features = hog_features.tolist()

    def _lbp_features(self, radius=1, points=8):
        image = self.gray_image
        lbp = local_binary_pattern(image, points, radius, method="uniform")

        bins = int(points + 2)
        histogram, _ = np.histogram(lbp, bins=bins, range=(0, bins))

        histogram = histogram.astype("float32")
        histogram /= histogram.sum() + 1e-6

        self.lbp_features = histogram.tolist()

    def _superpixel_features(self, n_segments=50, max_segments=50):
        image = cvtColor(self.bgr_image, COLOR_BGR2LAB).astype("float32") / 255.0
        segments = slic(image, n_segments=n_segments, compactness=10, start_label=0)

        features = []

        segment_ids = list(np.unique(segments))[:max_segments]
        for segment_id in segment_ids:
            mask = segments == segment_id
            for color in range(image.shape[2]):
                features.append(np.mean(image[:, :, color][mask]))

        expected_length = max_segments * image.shape[2]
        while len(features) < expected_length:
            features.append(0.0)

        self.superpixel_features = features

    def _debug(self):
        print("Extracting color, hog, lbp and superpixel features.")
