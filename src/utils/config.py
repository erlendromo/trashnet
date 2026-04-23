import os
import time


class Config:
    def __init__(self):
        # Track time for program execution
        self.start_time = time.time()

        # Dataset
        self.dataset_path = os.getenv("DATASET_PATH", "./dataset")
        self.test_size = float(os.getenv("TEST_SIZE", "0.15"))
        self.val_size = float(os.getenv("VAL_SIZE", "0.10"))

        # Preprocessing
        self.noise = True if os.getenv("NOISE", False) == "True" else False
        self.segment = True if os.getenv("SEGMENT", False) == "True" else False

        # Feature extraction
        self.lbp = True if os.getenv("LBP", False) == "True" else False
        self.glcm = True if os.getenv("GLCM", False) == "True" else False
        self.hsv = True if os.getenv("HSV", False) == "True" else False
        self.gabor = True if os.getenv("GABOR", False) == "True" else False
        self.sift = True if os.getenv("SIFT", False) == "True" else False
        self.hu = True if os.getenv("HU", False) == "True" else False
        self.hog = True if os.getenv("HOG", False) == "True" else False
        self.superpixel = True if os.getenv("SUPERPIXEL", False) == "True" else False

        # Classification
        save_model = os.getenv("SAVE_MODEL", "")
        self.save_model = (
            save_model
            if save_model.endswith(".model") and len(save_model) >= 7
            else None
        )

        load_model = os.getenv("LOAD_MODEL", "")
        self.load_model = (
            load_model
            if load_model.endswith(".model") and len(load_model) >= 7
            else None
        )

        # Visualization
        self.visualize = True if os.getenv("VISUALIZE", False) == "True" else False


    def elapsed_time(self):
        elapsed_time = time.time() - self.start_time

        minutes = int(elapsed_time // 60)
        seconds = int(elapsed_time % 60)

        print(f"\nProgram completed in {minutes}min and {seconds}sec")
