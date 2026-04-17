from cv2 import GaussianBlur, resize


class Preprocessor:
    def __init__(self, image=None, size=256, noise=False, debug=False):
        if image is None:
            raise ValueError("Image cannot be 'None'. Load an image using cv2.imread()")

        self.image = image
        self.size = max(128, min(size, 512))
        self.noise = noise
        self.debug = debug

    def process(self):
        if self.debug:
            self._debug()

        self.processed_image = self.image.copy()

        self._resize()

        if self.noise:
            self._apply_noise()

        return self.processed_image

    def _resize(self):
        self.processed_image = resize(self.processed_image, (self.size, self.size))

    def _apply_noise(self, kernel=(5, 5)):
        self.processed_image = GaussianBlur(self.processed_image, kernel, 0)

    def _debug(self):
        size = f"{self.size}x{self.size}"
        noise = "with noise" if self.noise else "without noise"

        print(f"Processing image (resize to {size}, {noise}).")
