# CAPTCHA Solver

A Redis-integrated CAPTCHA solving system using TensorFlow and Keras.

## Overview

This project allows you to:
- Train a model to solve CAPTCHA images with randomized alphanumeric text.
- Serve predictions through a Redis-based system using a lightweight Python listener.
- Preprocess and normalize images from a `./cropped` directory.

## Features

- Character set excludes ambiguous symbols like `0`, `1`, `I`, `O`, and `l`.
- Model architecture: CNN + BiLSTM + CTC decoding.
- Integrated with Redis for queue-based request/response processing.
- TensorFlow 2.x compatible.

## Directory Structure

```
/cropped/            # Input CAPTCHA images, filenames are used as labels
trainer.py           # Trains the model and saves to captcha_model.h5
load.py              # Loads model, solves CAPTCHAs from Redis queue
requirements.txt     # Python dependencies
run.bash             # Optional startup script
index.php            # PHP interface to push CAPTCHA solving requests
```

## Usage

### Training

Place CAPTCHA images in the `./cropped` folder with correct labels as filenames:

```
./cropped/8TrbK.png
./cropped/Yz7fU.png
```

Then run:

```bash
python3 trainer.py
```

### Serving (Prediction)

Start the Redis server, then run:

```bash
python3 load.py
```

It will listen on the `captcha` Redis channel and write results using `r.set()`.

## Dependencies

Install Python packages:

```bash
pip install -r requirements.txt
```

## License

MIT License

## Author

Kaveh Sarkhanlou
