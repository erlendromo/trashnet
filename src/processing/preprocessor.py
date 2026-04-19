from cv2 import GaussianBlur, imread, resize


class Preprocessor:
    def __init__(self, image_path=None, size=256, noise=False, debug=False):
        self.image = imread(image_path)
        if self.image is None:
            raise ValueError("invalid image_path, unable to load image")

        self.size = max(128, min(size, 512))
        self.noise = noise
        self.debug = debug

    def process(self):
        if self.debug:
            self._debug()

        preprocessed_image = self.image.copy()

        preprocessed_image = self._resize(preprocessed_image)

        if self.noise:
            preprocessed_image = self._apply_noise(preprocessed_image)

        self.preprocessed_image = preprocessed_image

        return self.preprocessed_image

    def _resize(self, image):
        return resize(image, (self.size, self.size))

    def _apply_noise(self, image, kernel=(5, 5)):
        return GaussianBlur(image, kernel, 0)

    def _debug(self):
        size = f"{self.size}x{self.size}"
        noise = "with noise" if self.noise else "without noise"

        print(f"Preprocessing image: Resize to {size}, {noise}.")
