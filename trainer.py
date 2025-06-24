

import os
import numpy as np
from pathlib import Path
from PIL import Image, ImageChops

import tensorflow as tf
from tensorflow.keras import layers, models

# Parameters
data_dir = Path("./cropped")
characters = ['4', '5', '6', '7', '8', '9', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'J', 'K',
              'M', 'N', 'P', 'R', 'S', 'T', 'U', 'V', 'X', 'Y', 'Z', 'a', 'b', 'c', 'd', 'e',
              'f', 'g', 'h', 'j', 'k', 'm', 'n', 'p', 'q', 'r', 's', 't', 'u', 'v', 'x', 'y', 'z']
img_width = 100
img_height = 50
max_length = 5
batch_size = 16
epochs = 20

char_to_num = layers.StringLookup(vocabulary=list(characters), mask_token=None)
num_to_char = layers.StringLookup(vocabulary=char_to_num.get_vocabulary(), mask_token=None, invert=True)

def trim(im):
    bg = Image.new(im.mode, im.size, im.getpixel((0, 0)))
    diff = ImageChops.difference(im, bg)
    diff = ImageChops.add(diff, diff, 2.0, -100)
    bbox = diff.getbbox()
    newImage = im.crop(bbox)
    rect = Image.new('RGB', (img_width, img_height), (255, 255, 255))
    img_w, img_h = newImage.size
    bg_w, bg_h = rect.size
    offset = ((bg_w - img_w) // 2, (bg_h - img_h) // 2)
    rect.paste(newImage, offset)
    return rect

def encode_single_sample(img_path, label):
    img = trim(Image.open(img_path).convert("L"))
    img = np.array(img) / 255.0
    img = np.expand_dims(img, axis=-1)
    label = tf.strings.unicode_split(label, input_encoding="UTF-8")
    label = char_to_num(label)
    return {"image": img, "label": label}

def build_dataset(image_paths, labels):
    dataset = tf.data.Dataset.from_tensor_slices((image_paths, labels))
    dataset = dataset.map(lambda x, y: tf.py_function(encode_single_sample, inp=[x, y],
                                                      Tout={"image": tf.float32, "label": tf.int64}),
                          num_parallel_calls=tf.data.AUTOTUNE)
    return dataset.batch(batch_size).cache().prefetch(buffer_size=tf.data.AUTOTUNE)

def build_model():
    input_img = layers.Input(shape=(img_height, img_width, 1), name="image")
    labels = layers.Input(name="label", shape=(None,), dtype="int64")

    x = layers.Conv2D(32, (3, 3), activation="relu")(input_img)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Conv2D(64, (3, 3), activation="relu")(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Reshape(target_shape=((img_width // 4), 64))(x)
    x = layers.Dense(64, activation="relu")(x)
    x = layers.Bidirectional(layers.LSTM(128, return_sequences=True))(x)
    x = layers.Dense(len(characters) + 1, activation="softmax")(x)

    model = tf.keras.models.Model(inputs=input_img, outputs=x)
    return model

def ctc_loss_fn(y_true, y_pred):
    label_length = tf.math.count_nonzero(y_true, axis=-1, dtype=tf.int32)
    input_length = tf.fill([tf.shape(y_pred)[0]], tf.shape(y_pred)[1])
    return tf.keras.backend.ctc_batch_cost(y_true, y_pred, input_length, label_length)

def main():
    images = sorted(list(map(str, list(data_dir.glob("*.png")))))
    labels = [img.split(os.path.sep)[-1].split(".png")[0] for img in images]
    dataset = build_dataset(images, labels)

    model = build_model()
    model.compile(optimizer="adam", loss=ctc_loss_fn)
    model.fit(dataset, epochs=epochs)

    model.save("captcha_model.h5")

if __name__ == "__main__":
    main()