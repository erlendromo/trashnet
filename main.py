from dotenv import load_dotenv
from src.pipeline import Pipeline
from src.utils.config import Config


def main():
    load_dotenv("./.env")
    config = Config()

    pipeline = Pipeline(config)
    pipeline.execute()

    config.elapsed_time()


if __name__ == "__main__":
    main()
