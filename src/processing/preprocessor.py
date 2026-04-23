import cv2
import numpy as np
import matplotlib.pyplot as plt
from collections import deque
from scipy.ndimage import binary_opening, binary_closing


class Preprocessor:
    def __init__(self, image_path, size=256, noise=False, segment=False, visualize=False):
        self.image = cv2.imread(image_path)
        if self.image is None:
            raise ValueError("invalid image_path, unable to load image")

        self.size = max(128, min(size, 512))
        self.noise = noise
        self.segment = segment
        self.visualize = visualize


    def process(self):
        preprocessed_image = self.image.copy()
        mask = None

        preprocessed_image = self._resize(preprocessed_image)

        if self.noise:
            preprocessed_image = self._apply_noise(preprocessed_image)

        if self.segment:
            preprocessed_image = self._add_white_border(preprocessed_image)
            mask = self._region_grow_adaptive_thresholding(preprocessed_image)

        self.preprocessed_image = preprocessed_image

        if self.visualize and mask is not None:
            self._plot_segmentation_result(preprocessed_image, mask)

        return preprocessed_image


    def _resize(self, image):
        return cv2.resize(image, (self.size, self.size))


    def _apply_noise(self, image, kernel=(5, 5)):
        return cv2.GaussianBlur(image, kernel, 0)


    def _add_white_border(self, image, border_width=1, white_value=255):
        """
        Adds a white border to an image without changing its size.
        The border is applied by overwriting edge pixels.

        Parameters:
            image: Input image (H, W) or (H, W, C)
            border_width: Thickness of border in pixels
            white_value: Value used for the border (255 for uint8 images)

        Returns:
            Image with white border
        """
        border_image = image.copy()

        # Top and bottom borders
        border_image[:border_width, ...] = white_value
        border_image[-border_width:, ...] = white_value

        # Left and right borders
        border_image[:, :border_width, ...] = white_value
        border_image[:, -border_width:, ...] = white_value

        return border_image


    def _region_grow_adaptive_thresholding(
        self,
        image,
        threshold=40,
        max_white_distance=80,
        iterations=1
    ):
        """
        Adaptive mean-updating region growing with a hard upper limit
        from the original white value.

        Logic:
        1. Seed starts at top-left corner (assumed white border)
        2. Pixel must satisfy BOTH:
            - close enough to current region mean (adaptive threshold)
            - not too far from original white reference (absolute limit)

        This prevents leakage into darker foreground objects while still
        allowing gradual background variation.

        Parameters:
            image : np.ndarray
                Grayscale (H, W) or color (H, W, C)

            threshold : float
                Relative adaptive threshold to current region mean

            max_white_distance : float
                Absolute maximum allowed distance from white reference

            iterations : int
                Morphological smoothing passes

        Returns:
            np.ndarray
                Binary mask (H, W), dtype uint8
        """

        h, w = image.shape[:2]
        mask = np.zeros((h, w), dtype=bool)

        seed_pixel = image[0, 0].astype(np.float32)
        white_reference = np.full(
            image.shape[2],
            255.0,
            dtype=np.float32
        )

        region_sum = seed_pixel.copy()
        region_count = 1
        region_mean = seed_pixel.copy()

        queue = deque([(0, 0)])
        mask[0, 0] = True

        neighbors = [(-1, 0), (1, 0), (0, -1), (0, 1)]

        while queue:
            x, y = queue.popleft()

            for dx, dy in neighbors:
                nx, ny = x + dx, y + dy

                # skip out of bounds
                if nx < 0 or nx >= h or ny < 0 or ny >= w:
                    continue

                # already visited
                if mask[nx, ny]:
                    continue

                pixel = image[nx, ny].astype(np.float32)

                # grayscale image
                if image.ndim == 2:
                    adaptive_difference = abs(pixel - region_mean)
                    white_difference = abs(pixel - white_reference)

                # color image
                else:
                    adaptive_difference = np.linalg.norm(pixel - region_mean)
                    white_difference = np.linalg.norm(pixel - white_reference)

                # adaptive thresholding
                if (
                    adaptive_difference <= threshold
                    and white_difference <= max_white_distance
                ):
                    mask[nx, ny] = True
                    queue.append((nx, ny))

                    # update running mean
                    region_sum += pixel
                    region_count += 1
                    region_mean = region_sum / region_count

        # smoothen the mask
        for _ in range(iterations):
            mask = binary_opening(mask)   # removes small noise
            mask = binary_closing(mask)   # fills small holes

        # discard mask if too small
        area = np.sum(mask)
        total = mask.size
        min_ratio = 0.05 if image.mean() > 200 else 0.15

        if area / total < min_ratio:
            return np.zeros_like(mask, dtype=np.uint8)

        return mask.astype(np.uint8)


    def _region_grow(self, image, threshold=80, iterations=1):
        """
        Simple region growing segmentation starting from top-left corner.

        Parameters:
            image: Grayscale (H, W) or color (H, W, C)
            threshold: Maximum allowed difference from seed value

        Returns:
            Binary mask (H, W) where region == 1
        """
        h, w = image.shape[:2]
        mask = np.zeros((h, w), dtype=np.uint8)

        seed = image[0, 0].astype(np.float32)

        queue = deque()
        queue.append((0, 0))
        mask[0, 0] = 1

        # 4-neighborhood
        neighbors = [(-1, 0), (1, 0), (0, -1), (0, 1)]

        while queue:
            x, y = queue.popleft()

            for dx, dy in neighbors:
                nx, ny = x + dx, y + dy

                # skip out-of-bounds
                if nx < 0 or nx >= h or ny < 0 or ny >= w:
                    continue

                # skip border
                if mask[nx, ny] == 1:
                    continue

                pixel = image[nx, ny].astype(np.float32)

                # handle grayscale vs color
                if pixel.shape == ():  # grayscale scalar
                    difference = abs(pixel - seed)
                else:
                    difference = np.linalg.norm(pixel - seed)

                if difference <= threshold:
                    mask[nx, ny] = 1
                    queue.append((nx, ny))

        mask = mask.astype(bool)

        # smoothen the mask
        for _ in range(iterations):
            mask = binary_opening(mask)   # removes small noise
            mask = binary_closing(mask)   # fills small holes

        # discard mask if less than threshold
        area = np.sum(mask)
        total = mask.size
        min_ratio = 0.05 if image.mean() > 200 else 0.15

        if area / total < min_ratio:
            return np.zeros_like(mask, dtype=np.uint8)

        return mask.astype(np.uint8)


    def _plot_segmentation_result(self, image, mask, alpha=0.4):
        """
        Plots original image, mask, and overlay.

        Parameters:
            image: Input image (H, W) or (H, W, C)
            mask: Binary mask (H, W)
            alpha: Transparency for overlay
        """

        # Ensure mask is binary
        mask = (mask > 0).astype(np.uint8)

        # Handle grayscale -> RGB for consistent overlay display
        if image.ndim == 2:
            image_disp = np.stack([image] * 3, axis=-1)
        else:
            image_disp = image.copy()

        # Create overlay: highlight mask area
        overlay = image_disp.copy()
        overlay[mask == 1] = (
            overlay[mask == 1] * (1 - alpha) + np.array([255, 0, 0]) * alpha
        )

        fig, axes = plt.subplots(1, 3, figsize=(12, 4))

        # Original
        axes[0].imshow(image, cmap='gray' if image.ndim == 2 else None)
        axes[0].set_title("Original")
        axes[0].axis("off")

        # Mask
        axes[1].imshow(mask, cmap='gray')
        axes[1].set_title("Mask")
        axes[1].axis("off")

        # Overlay
        axes[2].imshow(overlay.astype(np.uint8))
        axes[2].set_title("Overlay")
        axes[2].axis("off")

        plt.tight_layout()
        plt.show()
