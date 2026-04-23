# Trashnet
This is a project for the Computer Vision (IDIG4004) course at NTNU Gjøvik 2026.
The project is inspired by the original [trashnet](https://github.com/garythung/trashnet) repository, and leverages their dataset.
This project will however not utilize any deep learning, but rather use traditional methods to classify the images.

## Dataset
The dataset used for this project can be found [here](https://huggingface.co/datasets/garythung/trashnet). Download the original-dataset, and follow along the guide for the usage. The dataset is comprised of 2527 images:

- 501 glass
- 594 paper
- 403 cardboard
- 482 plastic
- 410 metal
- 137 trash `(recommended to omit, as the models doesn't recall them during the classification stage)`

## Installation
Install `Python` using your preferred package manager (Python 3.14.3 was used for the project).
For ease-of-use, download the `Makefile` extension for your OS.

## How to use
1. Clone the repository
2. Create a `.env` file based on the provided `.env.example` with your preferred configuration.
3. If you have the Makefile extension, simply run the `Makefile` commands:
    - Optional: `make test` (Currently not supported...)
    - `make run`
4. Alternatively copy the commands in the `Makefile` and execute them in your terminal. If you do this, make sure to run all the commands in the file in order from top to bottom.

## Results


## Acknowledgements
I want to give a shoutout to my fellow students, and the course coordinator!
Special thanks to the developers of the original [trashnet](https://github.com/garythung/trashnet) for providing the dataset.
