class Preprocessor:
    def __init__(self, image_size=256):
        self.image_size = image_size
        return

    def print(self):
        print(f"The image size is set to {self.image_size}x{self.image_size}")
