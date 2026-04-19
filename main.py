from dotenv import load_dotenv

from src.pipeline import Pipeline
from src.utils.config import Config


def main():
    load_dotenv("./.env")

    config = Config()
    config.print()

    pipeline = Pipeline(
        dataset_path=config.dataset_path,
        save_model=config.save_model,
        load_model=config.load_model,
        classifier_type=config.classifier_type,
        debug=config.debug,
    )
    pipeline.execute()

    config.elapsed_time()


if __name__ == "__main__":
    main()
