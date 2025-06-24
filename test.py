import base64
from genericpath import isfile
import os
from pyexpat import model
import numpy as np
import matplotlib.pyplot as plt

from pathlib import Path
from collections import Counter

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from keras.models import load_model

from PIL import Image , ImageChops

import time

import redis

import base64

# -*- coding: future_fstrings -*-

# r = redis.Redis(host='192.168.31.240', port=6379, db=0)
r = redis.Redis(host='10.0.0.11', port=6379, db=0)


data_dir = Path("./cropped")
# data_dir = Path("../captcha_images_v2")

def trim(im):
    
    bg = Image.new(im.mode, im.size, im.getpixel((0,0)))
    diff = ImageChops.difference(im, bg)
    diff = ImageChops.add(diff, diff, 2.0, -100)
    bbox = diff.getbbox()
    newImage = im.crop(bbox)
    rect = Image.new('RGB', (100, 50), (255, 255, 255))
    img_w, img_h = newImage.size
    bg_w, bg_h = rect.size
    offset = ((bg_w - img_w) // 2, (bg_h - img_h) // 2)
    rect.paste(newImage, offset)
    return rect


# # Get list of all the images
images = sorted(list(map(str, list(data_dir.glob("*.png")))))
labels = [img.split(os.path.sep)[-1].split(".png")[0] for img in images]
# characters = set(char for label in labels for char in label)
# characters = sorted(list(characters))

characters = ['4', '5', '6', '7', '8', '9', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'J', 'K', 'M', 'N', 'P', 'R', 'S', 'T', 'U', 'V', 'X', 'Y', 'Z', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'j', 'k', 'm', 'n', 'p', 'q', 'r', 's', 't', 'u', 'v', 'x', 'y', 'z']

# print("Number of images found: ", len(images))
# print("Number of labels found: ", len(labels))
# print("Number of unique characters: ", len(characters))
# print("Characters present: ", characters)

# # Batch size for training and validation
batch_size = 3


# # Desired image dimensions
img_width = 100
img_height = 50

# # Factor by which the image is going to be downsampled
# # by the convolutional blocks. We will be using two
# # convolution blocks and each block will have
# # a pooling layer which downsample the features by a factor of 2.
# # Hence total downsampling factor would be 4.
# downsample_factor = 4

# # Maximum length of any captcha in the dataset
# max_length = max([len(label) for label in labels])

max_length = 5



# # Mapping characters to integers
char_to_num = layers.StringLookup(
    vocabulary=list(characters), mask_token=None
)

# # Mapping integers back to original characters
num_to_char = layers.StringLookup(
    vocabulary=char_to_num.get_vocabulary(), mask_token=None, invert=True
)


# def split_data(images, labels, train_size=0.9, shuffle=True):
#     # 1. Get the total size of the dataset
#     size = len(images)
#     # 2. Make an indices array and shuffle it, if required
#     indices = np.arange(size)
#     if shuffle:
#         np.random.shuffle(indices)
#     # 3. Get the size of training samples
#     train_samples = int(size * train_size)
#     # 4. Split data into training and validation sets
#     x_train, y_train = images[indices[:train_samples]], labels[indices[:train_samples]]
#     x_valid, y_valid = images[indices[train_samples:]], labels[indices[train_samples:]]
#     return x_train, x_valid, y_train, y_valid


# # # Splitting data into training and validation sets
# x_train, x_valid, y_train, y_valid = split_data(np.array(images), np.array(labels))


# print(x_valid)
# print(y_valid)


def encode_single_sample(img_path, label):
    # 1. Read image
    img = tf.io.read_file(img_path)
    # 2. Decode and convert to grayscale
    img = tf.io.decode_png(img, channels=1)
    # 3. Convert to float32 in [0, 1] range
    img = tf.image.convert_image_dtype(img, tf.float32)
    # 4. Resize to the desired size
    img = tf.image.resize(img, [img_height, img_width])
    # 5. Transpose the image because we want the time
    # dimension to correspond to the width of the image.
    img = tf.transpose(img, perm=[1, 0, 2])
    # 6. Map the characters in label to numbers
    label = char_to_num(tf.strings.unicode_split(label, input_encoding="UTF-8"))
    # 7. Return a dict as our model is expecting two inputs
    return {"image": img, "label": label}


# train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
# train_dataset = (
#     train_dataset.map(
#         encode_single_sample, num_parallel_calls=tf.data.AUTOTUNE
#     )
#     .batch(batch_size)
#     .prefetch(buffer_size=tf.data.AUTOTUNE)
# )



# model

class CTCLayer(layers.Layer):
    def __init__(self, name=None):
        super().__init__(name=name)
        self.loss_fn = keras.backend.ctc_batch_cost

    def call(self, y_true, y_pred):
        # Compute the training-time loss value and add it
        # to the layer using `self.add_loss()`.
        batch_len = tf.cast(tf.shape(y_true)[0], dtype="int64")
        input_length = tf.cast(tf.shape(y_pred)[1], dtype="int64")
        label_length = tf.cast(tf.shape(y_true)[1], dtype="int64")

        input_length = input_length * tf.ones(shape=(batch_len, 1), dtype="int64")
        label_length = label_length * tf.ones(shape=(batch_len, 1), dtype="int64")

        loss = self.loss_fn(y_true, y_pred, input_length, label_length)
        self.add_loss(loss)

        # At test time, just return the computed predictions
        return y_pred


def build_model():
    # Inputs to the model
    input_img = layers.Input(
        shape=(img_width, img_height, 1), name="image", dtype="float32"
    )
    labels = layers.Input(name="label", shape=(None,), dtype="float32")

    # First conv block
    x = layers.Conv2D(
        32,
        (3, 3),
        activation="relu",
        kernel_initializer="he_normal",
        padding="same",
        name="Conv1",
    )(input_img)
    x = layers.MaxPooling2D((2, 2), name="pool1")(x)

    # Second conv block
    x = layers.Conv2D(
        64,
        (3, 3),
        activation="relu",
        kernel_initializer="he_normal",
        padding="same",
        name="Conv2",
    )(x)
    x = layers.MaxPooling2D((2, 2), name="pool2")(x)

    # We have used two max pool with pool size and strides 2.
    # Hence, downsampled feature maps are 4x smaller. The number of
    # filters in the last layer is 64. Reshape accordingly before
    # passing the output to the RNN part of the model
    new_shape = ((img_width // 4), (img_height // 4) * 64)
    x = layers.Reshape(target_shape=new_shape, name="reshape")(x)
    x = layers.Dense(64, activation="relu", name="dense1")(x)
    x = layers.Dropout(0.2)(x)

    # RNNs
    x = layers.Bidirectional(layers.LSTM(128, return_sequences=True, dropout=0.25))(x)
    x = layers.Bidirectional(layers.LSTM(64, return_sequences=True, dropout=0.25))(x)

    # Output layer
    x = layers.Dense(
        len(char_to_num.get_vocabulary()) + 1, activation="softmax", name="dense2"
    )(x)

    # Add CTC layer for calculating CTC loss at each step
    output = CTCLayer(name="ctc_loss")(labels, x)

    # Define the model
    model = keras.models.Model(
        inputs=[input_img, labels], outputs=output, name="ocr_model_v1"
    )
    # Optimizer
    opt = keras.optimizers.Adam()
    # Compile the model and return
    model.compile(optimizer=opt)
    return model


# Get the model
model = build_model()
# model.summary()


# model = load_model('first_working_model')
#model = load_model('model_captcha')
model = keras.layers.TFSMLayer('model_captcha', call_endpoint='serving_default')

# Get the prediction model by extracting layers till the output layer
prediction_model = keras.models.Model(
    model.get_layer(name="image").input, model.get_layer(name="dense2").output
)
# prediction_model.summary()

# A utility function to decode the output of the network
def decode_batch_predictions(pred):
    input_len = np.ones(pred.shape[0]) * pred.shape[1]
    # Use greedy search. For complex tasks, you can use beam search
    results = keras.backend.ctc_decode(pred, input_length=input_len, greedy=True)[0][0][
        :, :max_length
    ]
    # Iterate over the results and get back the text
    output_text = []
    for res in results:
        res = tf.strings.reduce_join(num_to_char(res)).numpy().decode("utf-8")
        output_text.append(res)
    return output_text

def saveAnswer(answer,filename):
    f = open(filename, "w+")
    f.write(answer)
    f.close()

p = r.pubsub()
p.subscribe('captcha')
print('Ready!')
while True:
    message = p.get_message()
    if message:
        if message['type'] == "message":
            print("received message string")
            filename = message['data'].decode("utf-8")
            filePath = Path("wrong/" + filename)
            if filePath.is_file():
                image_file = Image.open("wrong/" + filename) # open colour image
                image_file = image_file.convert('1') # convert image to black and white
                image_file.save("temp/" + filename)

                im = Image.open('temp/' + filename)
                im = trim(im)
                im.save('prediction.png')

                x_valid = [
                    'prediction.png'
                ]
                y_valid = [
                    'job'
                ]

                validation_dataset = tf.data.Dataset.from_tensor_slices((x_valid, y_valid))
                validation_dataset = (
                    validation_dataset.map(
                        encode_single_sample, num_parallel_calls=tf.data.AUTOTUNE
                    )
                    .batch(batch_size)
                    .prefetch(buffer_size=tf.data.AUTOTUNE)
                )

                #  Let's check results on some validation samples
                for batch in validation_dataset.take(1):
                    batch_images = batch["image"]
                    batch_labels = batch["label"]

                    preds = prediction_model.predict(batch_images)
                    pred_texts = decode_batch_predictions(preds)

                    orig_texts = []
                    for label in batch_labels:
                        label = tf.strings.reduce_join(num_to_char(label)).numpy().decode("utf-8")
                        orig_texts.append(label)

                    for i in range(len(pred_texts)):
                        #title = f"Prediction: {filename}/{pred_texts[i]}"
                        title = fromstr('Prediction: {}/{}').format(filename,pred_texts[i])
                        saveAnswer(pred_texts[i],'answers/' + filename)
                        # r.publish("captcha_answer", pred_texts[i])
                        r.set(filename,pred_texts[i])
                        print(title)
            else:
                print("no such file")
                # r.publish("captcha_answer", "00000")
    time.sleep(0.001)




# test_img = encode_single_sample('wrong/7pr.png','7pr')

# preds = prediction_model.predict(test_img)
