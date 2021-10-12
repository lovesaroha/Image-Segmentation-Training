# Love Saroha
# lovesaroha1994@gmail.com (email address)
# https://www.lovesaroha.com (website)
# https://github.com/lovesaroha  (github)

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers.experimental import preprocessing 
import tensorflow_datasets as tfds
from tensorflow_examples.models.pix2pix import pix2pix
import matplotlib.pyplot as plt

# Oxford-IIIT Pet Dataset.
dataset, info = tfds.load('oxford_iiit_pet:3.*.*', with_info=True)

# Parameters.
epochs = 1
batchSize = 64
buffer_size = 1000
trainingLength = info.splits['train'].num_examples
steps = trainingLength // batchSize
validation_steps = info.splits['test'].num_examples//batchSize//5

# Normalize input image and labels.
def normalize(input_image, input_mask):
  input_image = tf.cast(input_image, tf.float32) / 255.0
  input_mask -= 1
  return input_image, input_mask

# Load images.
def load_image(datapoint):
  input_image = tf.image.resize(datapoint['image'], (128, 128))
  input_mask = tf.image.resize(datapoint['segmentation_mask'], (128, 128))
  input_image, input_mask = normalize(input_image, input_mask)
  return input_image, input_mask

# Set training and validation images.
training_images = dataset['train'].map(load_image, num_parallel_calls=tf.data.AUTOTUNE)
validation_images = dataset['test'].map(load_image, num_parallel_calls=tf.data.AUTOTUNE)

class Augment(keras.layers.Layer):
  def __init__(self, seed=42):
    super().__init__()
    self.augment_inputs = preprocessing.RandomFlip(mode="horizontal", seed=seed)
    self.augment_labels = preprocessing.RandomFlip(mode="horizontal", seed=seed)

  def call(self, inputs, labels):
    inputs = self.augment_inputs(inputs)
    labels = self.augment_labels(labels)
    return inputs, labels

# Make training batches.
training_batches = (
    training_images
    .cache()
    .shuffle(buffer_size)
    .batch(batchSize)
    .repeat()
    .map(Augment())
    .prefetch(buffer_size=tf.data.AUTOTUNE))

# Make validation batches.
validation_batches = validation_images.batch(batchSize)

# Show images.
def display(display_list):
  plt.figure(figsize=(15, 15))
  for i in range(len(display_list)):
    plt.subplot(1, len(display_list), i+1)
    plt.imshow(keras.preprocessing.image.array_to_img(display_list[i]))
    plt.axis('off')
  plt.show()

# Take base model.
base_model = keras.applications.MobileNetV2(input_shape=[128, 128, 3], include_top=False)

# Use the activations of these layers.
layer_names = [
    'block_1_expand_relu', 
    'block_3_expand_relu', 
    'block_6_expand_relu',  
    'block_13_expand_relu', 
    'block_16_project',     
]
base_model_outputs = [base_model.get_layer(name).output for name in layer_names]

# Create the feature extraction model.
feature_model = keras.Model(inputs=base_model.input, outputs=base_model_outputs)
feature_model.trainable = False

up_stack = [
    pix2pix.upsample(512, 3), 
    pix2pix.upsample(256, 3),
    pix2pix.upsample(128, 3),
    pix2pix.upsample(64, 3),   
]

# Custom model.
def smodel(output_channels:int):
  inputs = keras.layers.Input(shape=[128, 128, 3])

  # Downsampling through the model.
  skips = feature_model(inputs)
  x = skips[-1]
  skips = reversed(skips[:-1])

  # Upsampling and establishing the skip connections.
  for up, skip in zip(up_stack, skips):
    x = up(x)
    concat = keras.layers.Concatenate()
    x = concat([x, skip])

  # This is the last layer of the model.
  last = keras.layers.Conv2DTranspose(
      filters=output_channels, kernel_size=3, strides=2,
      padding='same') 
  x = last(x)
  return keras.Model(inputs=inputs, outputs=x)

# Create a model with 3 output channels.
model = smodel(output_channels=3)

# Set loss function and optimizer.
model.compile(optimizer='adam',
              loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

def create_mask(pred_mask):
  pred_mask = tf.argmax(pred_mask, axis=-1)
  pred_mask = pred_mask[..., tf.newaxis]
  return pred_mask[0]

# Show predictions.
def show_predictions(dataset=None, num=1):
  if dataset:
    for image, mask in dataset.take(num):
      pred_mask = model.predict(image)
      display([image[0], mask[0], create_mask(pred_mask)])

class myCallback(keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
      # Stop when validation accuracy is more than 98%.
        if logs.get('val_accuracy') is not None and logs.get('val_accuracy') > 0.98:
            print("\nTraining Stopped!")
            self.model.stop_training = True


# Callback function to check accuracy.
checkAccuracy = myCallback()

model_history = model.fit(training_batches, epochs=epochs,
                          steps_per_epoch=steps,
                          validation_steps=validation_steps,
                          validation_data=validation_batches,
                          callbacks=[checkAccuracy])

show_predictions(validation_batches, 3)
