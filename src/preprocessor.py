from cv2 import (
    COLOR_BGR2GRAY,
    COLOR_BGR2HSV,
    COLOR_BGR2LAB,
    GaussianBlur,
    cvtColor,
    resize,
)

COLORSPACES = (COLOR_BGR2HSV, COLOR_BGR2LAB, COLOR_BGR2GRAY)


class Preprocessor:
    def __init__(self, image=None, size=256, color=COLOR_BGR2HSV, noise=False):
        if image is None:
            raise ValueError("Image cannot be 'None'. Load an image using cv2.imread()")

        self.image = image
        self.size = max(128, min(size, 512))
        self.color = color if color in COLORSPACES else COLOR_BGR2HSV
        self.noise = noise

    def preprocess(self):
        self.processed_image = self.image.copy()

        self._resize()
        self._convert_color()

        if self.noise:
            self._apply_noise()

        self._normalize()

        return self.processed_image

    def _resize(self):
        self.processed_image = resize(self.processed_image, (self.size, self.size))

    def _convert_color(self):
        self.processed_image = cvtColor(self.processed_image, self.color)

    def _apply_noise(self, kernel=(5, 5)):
        self.processed_image = GaussianBlur(self.processed_image, kernel, 0)

    def _normalize(self):
        self.processed_image = self.processed_image.astype("float32") / 255.0

        if len(self.processed_image.shape) == 2:
            self.processed_image = self.processed_image[..., None]
