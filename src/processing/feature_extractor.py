import cv2
import numpy as np
from skimage.feature import local_binary_pattern, graycomatrix, graycoprops, hog


class FeatureExtractor:
    def __init__(self, image, lbp=True, glcm=True, hsv=True, gabor=True, sift=False, hu=False, hog=False, superpixel=False):
        if image is None:
            raise ValueError("image cannot be 'None'")

        self.image = image
        self.lbp = lbp
        self.glcm = glcm
        self.hsv = hsv
        self.gabor = gabor
        self.sift = sift
        self.hu = hu
        self.hog = hog
        self.superpixel = superpixel


    def extract(self):
        """
        Combine features into a single feature vector.

        Returns:
            feature vector
        """

        features = []

        image = self.image.copy()

        if self.lbp:
            features.append(self._extract_lbp_features(image))

        if self.glcm:
            features.append(self._extract_glcm_features(image))

        if self.hsv:
            features.append(self._extract_hsv_histogram(image))

        if self.gabor:
            features.append(self._extract_gabor_features(image))

        if self.sift:
            features.append(self._extract_sift_features(image))

        if self.hu:
            features.append(self._extract_hu_features(image))

        if self.hog:
            features.append(self._extract_hog_features(image))

        if self.superpixel:
            features.append(self._extract_superpixel_features(image))

        if len(features) == 0:
            raise ValueError("No features selected for extraction.")

        combined_features = np.concatenate(features)

        self.features = combined_features
        return combined_features


    def _extract_lbp_features(self, image):
        """
        Extract improved LBP features using:
        - Multi-scale LBP
        - Rotation-invariant uniform patterns

        This is much stronger than basic LBP for material classification.

        Parameters
        ----------
        image
            Input BGR image (from cv2.imread)

        Returns
        -------
        feature_vector
            Concatenated normalized LBP histogram features
        """

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Multi-scale settings
        radii = [1, 2, 3]
        n_points_list = [8, 16, 24]

        feature_vector = []

        for radius, n_points in zip(radii, n_points_list):
            lbp = local_binary_pattern(
                gray,
                P=n_points,
                R=radius,
                method="uniform"   # rotation-invariant uniform LBP
            )

            n_bins = int(lbp.max() + 1)

            hist, _ = np.histogram(
                lbp.ravel(),
                bins=n_bins,
                range=(0, n_bins)
            )

            hist = hist.astype("float")
            hist /= (hist.sum() + 1e-7)

            feature_vector.extend(hist)

        return np.array(feature_vector)


    def _extract_hog_features(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        features = hog(
            image,
            orientations=9,
            pixels_per_cell=(8, 8),
            cells_per_block=(2, 2),
            block_norm="L2-Hys",
            feature_vector=True,
        )

        return features.tolist()


    def _extract_superpixel_features(self, image, max_segments=50):
        features = []

        segments = slic(image, n_segments=50, compactness=10, start_label=0)
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


    def _extract_sift_features(self, image, max_features=200):
        """
        Extract SIFT descriptors and convert them into
        a fixed-length feature vector.

        Since SIFT returns variable numbers of keypoints,
        we create a stable representation by taking:

        - mean descriptor
        - std descriptor

        Final size:
        128 (mean) + 128 (std) = 256 features

        Parameters
        ----------
        image : np.ndarray
            Input BGR image

        max_features : int
            Maximum number of SIFT keypoints

        Returns
        -------
        feature_vector : np.ndarray
            Fixed-length SIFT feature vector
        """

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Create SIFT detector
        sift = cv2.SIFT_create(nfeatures=max_features)

        keypoints, descriptors = sift.detectAndCompute(gray, None)

        # Handle images with no keypoints
        if descriptors is None or len(descriptors) == 0:
            return np.zeros(256)

        # Use mean + std of descriptors for fixed length
        mean_desc = np.mean(descriptors, axis=0)
        std_desc = np.std(descriptors, axis=0)

        feature_vector = np.concatenate([mean_desc, std_desc])

        return feature_vector


    def _extract_glcm_features(self, image):
        """
        Extract Gray-Level Co-occurrence Matrix (GLCM) features.

        Parameters:
            image : np.ndarray
                Input BGR image

        Returns:
            features : np.ndarray
                GLCM feature vector:
                contrast, dissimilarity, homogeneity,
                energy, correlation, ASM
        """

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Reduce gray levels for faster GLCM computation
        gray = (gray / 8).astype(np.uint8)

        glcm = graycomatrix(
            gray,
            distances=[1],
            angles=[0, np.pi/4, np.pi/2, 3*np.pi/4],
            levels=32,
            symmetric=True,
            normed=True
        )

        properties = [
            "contrast",
            "dissimilarity",
            "homogeneity",
            "energy",
            "correlation",
            "ASM"
        ]

        features = []

        for prop in properties:
            values = graycoprops(glcm, prop)
            features.extend(values.flatten())

        return np.array(features)


    def _extract_hog_features(self, image):
        """
        Extract Histogram of Oriented Gradients (HOG) features.

        Parameters:
            image : np.ndarray
                Input BGR image

        Returns:
            hog_features : np.ndarray
                HOG feature vector
        """

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        hog_features = hog(
            gray,
            orientations=9,
            pixels_per_cell=(8, 8),
            cells_per_block=(2, 2),
            block_norm="L2-Hys",
            transform_sqrt=True,
            feature_vector=True
        )

        return hog_features


    def _extract_hsv_histogram(self, image, bins=(8, 8, 8)):
        """
        Extract HSV color histogram features.

        Parameters:
            image : np.ndarray
                Input BGR image
            bins : tuple
                Number of bins for H, S, V channels

        Returns:
            hist : np.ndarray
                Normalized flattened HSV histogram
        """

        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        hist = cv2.calcHist(
            [hsv],
            channels=[0, 1, 2],
            mask=None,
            histSize=bins,
            ranges=[0, 180, 0, 256, 0, 256]
        )

        hist = cv2.normalize(hist, hist).flatten()

        return hist


    def _extract_gabor_features(self, image):
        """
        Extract Gabor texture features using multiple
        orientations and frequencies.

        This is excellent for:
        - paper texture
        - cardboard roughness
        - plastic smoothness
        - material surface patterns

        Parameters
        ----------
        image : np.ndarray
            Input BGR image (from cv2.imread)

        Returns
        -------
        feature_vector : np.ndarray
            Gabor feature vector using:
            mean + std response from multiple filters
        """

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        feature_vector = []

        # Recommended Gabor settings
        ksize = 21              # kernel size
        sigma = 5.0             # Gaussian envelope
        lambdas = [8, 16]       # wavelengths (frequencies)
        gammas = [0.5, 1.0]     # aspect ratios
        psi = 0                 # phase offset

        # Multiple orientations
        thetas = [
            0,
            np.pi / 4,
            np.pi / 2,
            3 * np.pi / 4
        ]

        for theta in thetas:
            for lambd in lambdas:
                for gamma in gammas:

                    kernel = cv2.getGaborKernel(
                        (ksize, ksize),
                        sigma,
                        theta,
                        lambd,
                        gamma,
                        psi,
                        ktype=cv2.CV_32F
                    )

                    filtered = cv2.filter2D(
                        gray,
                        cv2.CV_8UC3,
                        kernel
                    )

                    # Statistical summary of filter response
                    mean = np.mean(filtered)
                    std = np.std(filtered)

                    feature_vector.extend([mean, std])

        return np.array(feature_vector)


    def _extract_hu_features(self, image):
        """
        Extract Hu Moments for shape description.

        Useful for:
        - bottles
        - cans
        - boxes
        - containers
        - glass / metal / cardboard shape structure

        Hu Moments are:
        - scale invariant
        - rotation invariant
        - translation invariant

        Parameters
        ----------
        image : np.ndarray
            Input BGR image (from cv2.imread)

        Returns
        -------
        hu_features : np.ndarray
            7-dimensional Hu Moments feature vector
        """

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Optional smoothing to reduce noise
        gray = cv2.GaussianBlur(gray, (5, 5), 0)

        # Binary threshold for contour/shape extraction
        _, thresh = cv2.threshold(
            gray,
            0,
            255,
            cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )

        # Compute image moments
        moments = cv2.moments(thresh)

        # Compute Hu Moments
        hu_features = cv2.HuMoments(moments).flatten()

        # Log transform for better numerical stability
        # (standard practice)
        for i in range(len(hu_features)):
            if hu_features[i] != 0:
                hu_features[i] = -np.sign(hu_features[i]) * np.log10(
                    abs(hu_features[i])
                )

        return hu_features
