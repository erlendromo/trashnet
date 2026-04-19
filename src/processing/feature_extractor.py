import numpy as np
from cv2 import COLOR_BGR2GRAY, COLOR_BGR2HSV, COLOR_BGR2LAB, cvtColor
from skimage.feature import hog, local_binary_pattern
from skimage.segmentation import slic


class FeatureExtractor:
    def __init__(
        self,
        image=None,
        hog=True,
        lbp=True,
        superpixels=True,
        debug=False,
    ):
        if image is None:
            raise ValueError("image cannot be 'None'")

        self.image = image
        self.hog = hog
        self.lbp = lbp
        self.superpixels = superpixels
        self.debug = debug

    def extract(self):
        if self.debug:
            self._debug()

        features = []

        bgr_image = self.image.copy()
        gray_image = cvtColor(bgr_image, COLOR_BGR2GRAY)
        lab_image = cvtColor(bgr_image, COLOR_BGR2LAB).astype("float32") / 255.0

        features.extend(self._color_features(image=lab_image))

        if self.hog:
            features.extend(self._hog_features(image=gray_image))

        if self.lbp:
            features.extend(self._lbp_features(image=gray_image))

        if self.superpixels:
            features.extend(self._superpixel_features(image=lab_image))

        self.features = features

        return self.features

    def _color_features(self, image, bins=32):
        features = []

        for i in range(image.shape[2]):
            histogram = np.histogram(image[:, :, i], bins=bins, range=(0, 1))[0]
            histogram = histogram / (np.sum(histogram) + 1e-6)
            features.extend(histogram)

        return features

    def _hog_features(
        self,
        image,
        orientations=9,
        pixels_per_cell=(8, 8),
        cells_per_block=(2, 2),
        block_norm="L2-Hys",
        feature_vector=True,
    ):
        image = image.astype("float32") / 255.0

        features = hog(
            image,
            orientations=orientations,
            pixels_per_cell=pixels_per_cell,
            cells_per_block=cells_per_block,
            block_norm=block_norm,
            feature_vector=feature_vector,
        )

        return features.tolist()

    def _lbp_features(self, image, radius=1, points=8):
        lbp = local_binary_pattern(image, points, radius, method="uniform")

        bins = int(points + 2)
        histogram, _ = np.histogram(lbp, bins=bins, range=(0, bins))

        histogram = histogram.astype("float32")
        histogram /= histogram.sum() + 1e-6

        return histogram.tolist()

    def _superpixel_features(self, image, n_segments=50, max_segments=50):
        features = []

        segments = slic(image, n_segments=n_segments, compactness=10, start_label=0)
        segment_ids = list(np.unique(segments))[:max_segments]

        for segment_id in segment_ids:
            mask = segments == segment_id

            for color in range(image.shape[2]):
                vals = image[:, :, color][mask]
                features.append(np.mean(vals) if vals.size > 0 else 0.0)

        expected_length = max_segments * image.shape[2]

        while len(features) < expected_length:
            features.append(0.0)

        return features

    def _debug(self):
        hog = ", hog features" if self.hog else ""
        lbp = ", lbp features" if self.lbp else ""
        superpixels = ", superpixel features" if self.superpixels else ""

        print(f"Extracting LAB color features{hog}{lbp}{superpixels}.")
