
import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, Conv2DTranspose, LeakyReLU, Dropout, Concatenate
from tensorflow.keras.models import Model
import os
import time
import numpy as np
import tensorflow as tf
from PIL import Image, ImageFile
from tensorflow.keras.preprocessing.image import img_to_array, array_to_img
import matplotlib.pyplot as plt
import tensorflow_hub as hub

ImageFile.LOAD_TRUNCATED_IMAGES = True

def preprocess_image(image_path):
    try:
        img = Image.open(image_path).convert('RGB')
        img = img.resize((256, 256))
        img_array = img_to_array(img)
        img_array = (img_array / 127.5) - 1  
        return np.expand_dims(img_array, axis=0)
    except (OSError, ValueError) as e:
        print(f"Error loading image {image_path}: {e}")
        return None

def postprocess_image(img_array):
    img_array = (img_array + 1) * 127.5  
    img_array = img_array.astype(np.uint8)
    return array_to_img(img_array[0])


def preprocess_hr_image(image):
    hr_image = tf.image.decode_image(tf.io.read_file(image))
    if hr_image.shape[-1] == 4:
        hr_image = hr_image[...,:-1]
    hr_size = (tf.convert_to_tensor(hr_image.shape[:-1]) // 4) * 4
    hr_image = tf.image.crop_to_bounding_box(hr_image, 0, 0, hr_size[0], hr_size[1])
    hr_image = tf.cast(hr_image, tf.float32)
    return tf.expand_dims(hr_image, 0)

def save_image(image, filename):
    if not isinstance(image, Image.Image):
        image = tf.clip_by_value(image, 0, 255)
        image = Image.fromarray(tf.cast(image, tf.uint8).numpy())
    image.save("%s.jpg" % filename)
    print("Saved as %s.jpg" % filename)

def plot_image(image, title=""):
    image = np.asarray(image)
    image = tf.clip_by_value(image, 0, 255)
    image = Image.fromarray(tf.cast(image, tf.uint8).numpy())
    plt.imshow(image)
    plt.axis("off")
    plt.title(title)


def build_generator():
    def conv_block(x, filters, batch_norm=True):
        x = tf.keras.layers.Conv2D(filters, kernel_size=4, strides=2, padding='same')(x)
        if batch_norm:
            x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.LeakyReLU(alpha=0.2)(x)
        return x

    def deconv_block(x, skip_input, filters, dropout_rate=0):
        x = tf.keras.layers.Conv2DTranspose(filters, kernel_size=4, strides=2, padding='same')(x)
        x = tf.keras.layers.BatchNormalization()(x)
        if dropout_rate:
            x = tf.keras.layers.Dropout(dropout_rate)(x)
        x = tf.keras.layers.Concatenate()([x, skip_input])
        x = tf.keras.layers.LeakyReLU(alpha=0.2)(x)
        return x

    inputs = tf.keras.layers.Input(shape=[256, 256, 3])
    e1 = conv_block(inputs, 64, batch_norm=False)
    e2 = conv_block(e1, 128)
    e3 = conv_block(e2, 256)
    e4 = conv_block(e3, 512)
    e5 = conv_block(e4, 512)
    e6 = conv_block(e5, 512)
    e7 = conv_block(e6, 512)
    e8 = conv_block(e7, 512, batch_norm=False)

    d1 = deconv_block(e8, e7, 512, dropout_rate=0.5)
    d2 = deconv_block(d1, e6, 512, dropout_rate=0.5)
    d3 = deconv_block(d2, e5, 512, dropout_rate=0.5)
    d4 = deconv_block(d3, e4, 512)
    d5 = deconv_block(d4, e3, 256)
    d6 = deconv_block(d5, e2, 128)
    d7 = deconv_block(d6, e1, 64)
    d8 = tf.keras.layers.Conv2DTranspose(3, kernel_size=4, strides=2, padding='same', activation='tanh')(d7)

    return tf.keras.models.Model(inputs, d8)

generator = build_generator()
generator.load_weights('/Users/satyasusarla/code/CNN-Project/models/generator_epoch_80.h5')


SAVED_MODEL_PATH = "https://tfhub.dev/captain-pool/esrgan-tf2/1"
esrgan_model = hub.load(SAVED_MODEL_PATH)


input_image_path = '/Users/satyasusarla/code/CNN-Project/Final-Photos/2.png'
intermediate_image_path = '/Users/satyasusarla/code/CNN-Project/output/intermediate_output.png'
output_image_path = '/Users/satyasusarla/code/CNN-Project/output/final_output.png'


input_image = preprocess_image(input_image_path)
if input_image is not None:
    intermediate_output_array = generator.predict(input_image)
    intermediate_output_image = postprocess_image(intermediate_output_array)
    intermediate_output_image.save(intermediate_image_path)
    print(f"Intermediate output image saved to {intermediate_image_path}")

    
    hr_image = preprocess_hr_image(intermediate_image_path)
    start = time.time()
    sr_image = esrgan_model(hr_image)
    sr_image = tf.squeeze(sr_image)
    print("Time Taken for Super Resolution: %f" % (time.time() - start))

    
    sr_image_pil = tf.clip_by_value(sr_image, 0, 255).numpy().astype(np.uint8)
    sr_image_pil = Image.fromarray(sr_image_pil)
    sr_image_pil.save(output_image_path)
    print(f"Final output image saved to {output_image_path}")

    
    input_img = Image.open(input_image_path).convert('RGB').resize((256, 256))
    fig, ax = plt.subplots(1, 3, figsize=(15, 5))

    ax[0].imshow(input_img)
    ax[0].set_title('Input Image')
    ax[0].axis('off')

    ax[1].imshow(intermediate_output_image)
    ax[1].set_title('Intermediate Output Image')
    ax[1].axis('off')

    ax[2].imshow(sr_image_pil)
    ax[2].set_title('Super Resolution Output Image')
    ax[2].axis('off')

    plt.show()
else:
    print("Failed to process the input image.")