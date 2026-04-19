import os
import time


class Config:
    def __init__(self):
        self.start_time = time.time()

        debug = os.getenv("DEBUG", "False")
        self.debug = True if debug == "True" else False

        self.dataset_path = os.getenv("DATASET_PATH", "./dataset")

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

        self.classifier_type = os.getenv("CLASSIFIER_TYPE", "svm")

    def print(self):
        print(
            f"Running program in the following configuration:\n\n-----\nDebug mode: {self.debug}\nDataset: {self.dataset_path}\nLoad from pretrained model: {'False' if self.load_model is None else 'True'}\nSave model: {'False' if self.save_model is None else 'True'}\nClassifier: {self.classifier_type}\n-----\n"
        )

    def elapsed_time(self):
        elapsed_time = time.time() - self.start_time

        minutes = int(elapsed_time // 60)
        seconds = int(elapsed_time % 60)

        print(f"\nProgram completed in {minutes}min and {seconds}sec")
