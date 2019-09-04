"""
An Implementation of Fast Neural Style in TensorFlow 2.0.
"""

import functools

from absl import app, flags
import numpy as np
import tensorflow as tf
from tensorflow.keras import initializers  # pylint: disable=import-error
from tensorflow.keras import Model  # pylint: disable=import-error
from tensorflow.keras.applications import vgg19  # pylint: disable=import-error
from tensorflow.keras.layers import Activation, Add, Conv2D, Conv2DTranspose, Input, Lambda  # pylint: disable=import-error
from tensorflow_addons.layers import InstanceNormalization
import tensorflow_datasets as tfds

from .image_utils import gram_matrix, resize_min, vgg_preprocess_input


def _conv_block(net, filters, kernel_size, strides, padding, activation):
    # https://github.com/lengstrom/fast-style-transfer/blob/master/src/transform.py#L59-L67
    weights_initializer = initializers.TruncatedNormal(mean=0.0,
                                                       stddev=0.1,
                                                       seed=1)

    net = Conv2D(filters=filters,
                 kernel_size=kernel_size,
                 strides=strides,
                 padding=padding,
                 activation=None,
                 kernel_initializer=weights_initializer)(net)

    net = InstanceNormalization()(net)

    if activation != None:
        net = Activation(activation)(net)

    return net


def _residual_block(net, filters, kernel_size, strides, padding):
    tmp = _conv_block(net,
                      filters,
                      kernel_size,
                      strides,
                      padding,
                      activation="relu")
    tmp = _conv_block(tmp,
                      filters,
                      kernel_size,
                      strides,
                      padding,
                      activation=None)
    return Add()([net, tmp])


def _conv_transpose_block(net, filters, kernel_size, strides, padding,
                          activation):
    # https://github.com/lengstrom/fast-style-transfer/blob/master/src/transform.py#L59-L67
    weights_initializer = initializers.TruncatedNormal(mean=0.0,
                                                       stddev=0.1,
                                                       seed=1)

    net = Conv2DTranspose(filters=filters,
                          kernel_size=kernel_size,
                          strides=strides,
                          padding=padding,
                          kernel_initializer=weights_initializer)(net)

    net = InstanceNormalization()(net)

    if activation != None:
        net = Activation(activation)(net)

    return net


def build_transformation_network():
    """
    The returned model can take unnormalized image with colors in range 0-255
    Normalization is handled in the first layer.

    TODO: if not using estimators, consider building with the imperative API instead of this
    """
    inp = Input((None, None, 3), name="transform_input")

    net = Lambda(lambda t: tf.cast(t, tf.float32) / 255.0)(inp)

    net = _conv_block(net,
                      filters=32,
                      kernel_size=(9, 9),
                      strides=(1, 1),
                      padding="same",
                      activation="relu")
    net = _conv_block(net,
                      filters=64,
                      kernel_size=(3, 3),
                      strides=(2, 2),
                      padding="same",
                      activation="relu")
    net = _conv_block(net,
                      filters=128,
                      kernel_size=(3, 3),
                      strides=(2, 2),
                      padding="same",
                      activation="relu")

    net = _residual_block(net,
                          filters=128,
                          kernel_size=(3, 3),
                          strides=(1, 1),
                          padding="same")
    net = _residual_block(net,
                          filters=128,
                          kernel_size=(3, 3),
                          strides=(1, 1),
                          padding="same")
    net = _residual_block(net,
                          filters=128,
                          kernel_size=(3, 3),
                          strides=(1, 1),
                          padding="same")
    net = _residual_block(net,
                          filters=128,
                          kernel_size=(3, 3),
                          strides=(1, 1),
                          padding="same")
    net = _residual_block(net,
                          filters=128,
                          kernel_size=(3, 3),
                          strides=(1, 1),
                          padding="same")

    net = _conv_transpose_block(net,
                                filters=64,
                                kernel_size=(3, 3),
                                strides=(2, 2),
                                padding="same",
                                activation="relu")
    net = _conv_transpose_block(net,
                                filters=32,
                                kernel_size=(3, 3),
                                strides=(2, 2),
                                padding="same",
                                activation="relu")

    net = _conv_block(net,
                      filters=3,
                      kernel_size=(9, 9),
                      strides=(1, 1),
                      padding="same",
                      activation=None)

    # NOTE: in the lengstrom implementation, tanh is multiplied by 150
    # which seems to be wrong, since we want values between 0 and 255?
    net = Lambda(lambda t: tf.nn.tanh(t) * 127.5 + 127.5,
                 name="transform_output")(net)
    return Model(inp, net, name="transform_net")


@tf.function
def load_and_preprocess_image(path, max_dim=512):
    f = tf.io.read_file(path)
    img = tf.io.decode_image(f)

    scale = tf.constant(max_dim, dtype=tf.float32) / tf.cast(
        tf.reduce_max(tf.shape(img)), tf.float32)

    img = tf.image.resize_with_pad(
        img,
        tf.cast(tf.round(tf.cast(tf.shape(img)[0], tf.float32) * scale),
                tf.int32),
        tf.cast(tf.round(tf.cast(tf.shape(img)[1], tf.float32) * scale),
                tf.int32),
    )

    img = tf.expand_dims(img, axis=0)
    img = vgg_preprocess_input(img)
    return img


@tf.function
def train_step(batch, style_layers, content_layers, style_features,
               content_features, style_weight, content_weight,
               total_variation_weight, transform_net, loss_net, optimizer):
    with tf.GradientTape() as tape:
        predictions = transform_net(batch)

        predictions = vgg_preprocess_input(predictions)

        output_features = loss_net(predictions)

        style_outputs, content_outputs = output_features[:len(
            style_layers)], output_features[len(style_layers):]

        style_losses = [
            tf.reduce_mean(tf.square(feature - gram_matrix(output)))
            for feature, output in zip(style_features, style_outputs)
        ]

        content_losses = [
            tf.reduce_mean(tf.square(feature - output))
            for feature, output in zip(content_features, content_outputs)
        ]

        batch_size = tf.cast(tf.shape(batch)[0], tf.float32)

        style_score = (style_weight *
                       functools.reduce(tf.add, style_losses)) / batch_size
        content_score = (content_weight *
                         functools.reduce(tf.add, content_losses)) / batch_size

        total_variation_y = tf.reduce_mean(
            tf.square(predictions[:, 1:, :, :] - predictions[:, :-1, :, :]))
        total_variation_x = tf.reduce_mean(
            tf.square(predictions[:, :, 1:, :] - predictions[:, :, :-1, :]))

        total_variation_score = (
            total_variation_weight *
            (total_variation_x + total_variation_y)) / batch_size

        loss = style_score + content_score + total_variation_score

    grads = tape.gradient(loss, transform_net.trainable_variables)
    optimizer.apply_gradients(zip(grads, transform_net.trainable_variables))
    return loss, style_score, content_score, total_variation_score


def train(dataset, epochs, optimizer, style_layers, content_layers,
          style_features, style_weight, content_weight, total_variation_weight,
          transform_net, loss_net):
    step = 0
    for epoch in range(epochs):
        epoch_loss = 0
        epoch_style_loss = 0
        epoch_content_loss = 0
        epoch_total_variation_loss = 0
        epoch_steps = 0

        for batch in dataset:
            content_features = loss_net(
                vgg_preprocess_input(batch))[len(style_layers):]
            loss, style_score, content_score, total_variation_score = train_step(
                batch=batch,
                style_layers=style_layers,
                content_layers=content_layers,
                style_features=style_features,
                content_features=content_features,
                style_weight=style_weight,
                content_weight=content_weight,
                total_variation_weight=total_variation_weight,
                transform_net=transform_net,
                loss_net=loss_net,
                optimizer=optimizer)

            tf.summary.scalar("Batch Total Loss", loss, step=step)
            tf.summary.scalar("Batch Style Loss", style_score, step=step)
            tf.summary.scalar("Batch Content Loss", content_score, step=step)
            tf.summary.scalar("Batch Total Variation Loss",
                              total_variation_score,
                              step=step)

            step += 1
            epoch_steps += 1
            epoch_loss += loss
            epoch_style_loss += style_score
            epoch_content_loss += content_score
            epoch_total_variation_loss += total_variation_score

        epoch_loss /= epoch_steps
        epoch_style_loss /= epoch_steps
        epoch_content_loss /= epoch_steps
        epoch_total_variation_loss /= epoch_steps

        tf.summary.scalar("Total Loss", epoch_loss, step=epoch)
        tf.summary.scalar("Style Loss", epoch_style_loss, step=epoch)
        tf.summary.scalar("Content Loss", epoch_content_loss, step=epoch)
        tf.summary.scalar("Total Variation Loss",
                          epoch_total_variation_loss,
                          step=epoch)

        print(
            "Epoch: %d, Style Loss: %f, Content Loss: %f, Variation Loss: %f, Total Loss: %f"
            % (epoch + 1, epoch_style_loss, epoch_content_loss,
               epoch_total_variation_loss, epoch_loss))


def run(job_dir, style_image_path, dataset, epochs, learning_rate,
        style_layers, content_layers, style_weight, content_weight,
        total_variation_weight):
    with tf.summary.create_file_writer(job_dir).as_default():
        vgg = vgg19.VGG19(include_top=False, weights="imagenet")
        vgg.trainable = False
        for layer in vgg.layers:
            layer.trainable = False

        content_outputs = [vgg.get_layer(l).output for l in content_layers]
        style_outputs = [vgg.get_layer(l).output for l in style_layers]

        loss_net = Model(vgg.input, style_outputs + content_outputs)

        style_image = load_and_preprocess_image(style_image_path)
        style_features = loss_net(style_image)[:len(style_layers)]
        style_features = [gram_matrix(a) for a in style_features]

        transform_net = build_transformation_network()
        transform_net.summary()

        optimizer = tf.keras.optimizers.Adam(1e-3)

        train(dataset=dataset,
              epochs=epochs,
              optimizer=optimizer,
              style_layers=style_layers,
              content_layers=content_layers,
              style_features=style_features,
              style_weight=style_weight,
              content_weight=content_weight,
              total_variation_weight=total_variation_weight,
              transform_net=transform_net,
              loss_net=loss_net)

        return transform_net


def main(_):
    data_builder = tfds.builder("coco2014", data_dir=flags.FLAGS.tfds_data_dir)
    dataset = data_builder.as_dataset(split=tfds.Split.TRAIN)
    dataset = dataset.map(lambda i: i["image"])
    dataset = dataset.map(lambda i: resize_min(i, 256))
    dataset = dataset.batch(flags.FLAGS.batch_size)

    # https://github.com/lengstrom/fast-style-transfer/blob/master/src/optimize.py#L8-L9
    style_layers = [
        "block1_conv1", "block2_conv1", "block3_conv1", "block4_conv1",
        "block5_conv1"
    ]
    content_layers = ["block4_conv2"]

    transform_net = run(
        job_dir=flags.FLAGS["job-dir"].value,
        style_image_path=flags.FLAGS.style_image,
        dataset=dataset,
        epochs=flags.FLAGS.epochs,
        learning_rate=flags.FLAGS.learning_rate,
        style_layers=style_layers,
        content_layers=content_layers,
        style_weight=flags.FLAGS.style_weight,
        content_weight=flags.FLAGS.content_weight,
        total_variation_weight=flags.FLAGS.total_variation_weight)

    tf.keras.experimental.export_saved_model(transform_net,
                                             "%s/export/transform_net" %
                                             flags.FLAGS["job-dir"].value,
                                             serving_only=True)


if __name__ == "__main__":
    print(tf.version.VERSION)

    app.flags.DEFINE_float("content_weight", 7.5, "content weight")
    app.flags.DEFINE_float("style_weight", 100.0, "style weight")
    app.flags.DEFINE_float("total_variation_weight", 200.0, "content weight")
    app.flags.DEFINE_integer("epochs", 2, "epochs")
    app.flags.DEFINE_integer("batch_size", 4, "batch_size")
    app.flags.DEFINE_float("learning_rate", 1e-3, "learning rate")
    app.flags.DEFINE_string("style_image", "images/starry_night.jpg",
                            "style image")
    app.flags.DEFINE_string("tfds_data_dir", "~/tensorflow_datasets",
                            "tfds data dir")
    app.flags.DEFINE_string("job-dir", "runs/local", "job dir")

    app.run(main)
