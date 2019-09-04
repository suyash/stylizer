"""
This will train the 4 decoders with feature mse, pixel mse and total variation loss.
It exports both the encoder and the decoder as a saved_model
"""

import os

from absl import app, flags
import tensorflow as tf
from tensorflow.keras import Model  # pylint: disable=import-error
from tensorflow.keras.applications import vgg19  # pylint: disable=import-error
from tensorflow.keras.layers import Conv2D, Input, Lambda, Layer  # pylint: disable=import-error
import tensorflow_datasets as tfds

from stylizer.layers import Unpool
from stylizer.utils import resize_min


def build_encoder_1(vgg):
    inp = Input((None, None, 3))

    net = Lambda(lambda t: tf.pad(t, [[0, 0], [1, 1], [1, 1], [0, 0]],
                                  mode="REFLECT"))(inp)
    net = Conv2D(filters=64,
                 kernel_size=(3, 3),
                 strides=(1, 1),
                 padding="VALID",
                 activation="relu",
                 name="block1_conv1")(net)

    encoder1 = Model(inp, net, name="encoder1")
    encoder1.get_layer("block1_conv1").set_weights(
        vgg.get_layer("block1_conv1").get_weights())
    return encoder1


def build_decoder_1():
    inp = Input((None, None, 64))
    net = Lambda(lambda t: tf.pad(t, [[0, 0], [1, 1], [1, 1], [0, 0]],
                                  mode="REFLECT"))(inp)
    net = Conv2D(filters=3,
                 kernel_size=(3, 3),
                 strides=(1, 1),
                 activation=None,
                 padding="VALID")(net)
    return Model(inp, net, name="decoder1")


def build_encoder_2(vgg):
    inp = Input((None, None, 3))

    net = Lambda(lambda t: tf.pad(t, [[0, 0], [1, 1], [1, 1], [0, 0]],
                                  mode="REFLECT"))(inp)
    net = Conv2D(filters=64,
                 kernel_size=(3, 3),
                 strides=(1, 1),
                 padding="VALID",
                 activation="relu",
                 name="block1_conv1")(net)
    net = Lambda(lambda t: tf.pad(t, [[0, 0], [1, 1], [1, 1], [0, 0]],
                                  mode="REFLECT"))(net)
    net = Conv2D(filters=64,
                 kernel_size=(3, 3),
                 strides=(1, 1),
                 padding="VALID",
                 activation="relu",
                 name="block1_conv2")(net)

    net, mask1 = Lambda(lambda t: tf.nn.max_pool_with_argmax(
        t, ksize=2, strides=2, padding="VALID"))(net)

    net = Lambda(lambda t: tf.pad(t, [[0, 0], [1, 1], [1, 1], [0, 0]],
                                  mode="REFLECT"))(net)
    net = Conv2D(filters=128,
                 kernel_size=(3, 3),
                 strides=(1, 1),
                 padding="VALID",
                 activation="relu",
                 name="block2_conv1")(net)

    encoder2 = Model(inp, [net, mask1], name="encoder2")
    encoder2.get_layer("block1_conv1").set_weights(
        vgg.get_layer("block1_conv1").get_weights())
    encoder2.get_layer("block1_conv2").set_weights(
        vgg.get_layer("block1_conv2").get_weights())
    encoder2.get_layer("block2_conv1").set_weights(
        vgg.get_layer("block2_conv1").get_weights())
    return encoder2


def build_decoder_2():
    inp = Input((None, None, 128))
    mask1 = Input((None, None, 64), dtype=tf.int64)

    net = Lambda(lambda t: tf.pad(t, [[0, 0], [1, 1], [1, 1], [0, 0]],
                                  mode="REFLECT"))(inp)
    net = Conv2D(filters=64,
                 kernel_size=(3, 3),
                 strides=(1, 1),
                 activation="relu",
                 padding="VALID")(net)

    net = Unpool((2, 2))([net, mask1])

    net = Lambda(lambda t: tf.pad(t, [[0, 0], [1, 1], [1, 1], [0, 0]],
                                  mode="REFLECT"))(net)
    net = Conv2D(filters=64,
                 kernel_size=(3, 3),
                 strides=(1, 1),
                 activation="relu",
                 padding="VALID")(net)

    net = Lambda(lambda t: tf.pad(t, [[0, 0], [1, 1], [1, 1], [0, 0]],
                                  mode="REFLECT"))(net)
    net = Conv2D(filters=3,
                 kernel_size=(3, 3),
                 strides=(1, 1),
                 activation=None,
                 padding="VALID")(net)

    return Model([inp, mask1], net, name="decoder2")


def build_encoder_3(vgg):
    inp = Input((None, None, 3))
    net = Lambda(lambda t: tf.pad(t, [[0, 0], [1, 1], [1, 1], [0, 0]],
                                  mode="REFLECT"))(inp)
    net = Conv2D(filters=64,
                 kernel_size=(3, 3),
                 strides=(1, 1),
                 padding="VALID",
                 activation="relu",
                 name="block1_conv1")(net)
    net = Lambda(lambda t: tf.pad(t, [[0, 0], [1, 1], [1, 1], [0, 0]],
                                  mode="REFLECT"))(net)
    net = Conv2D(filters=64,
                 kernel_size=(3, 3),
                 strides=(1, 1),
                 padding="VALID",
                 activation="relu",
                 name="block1_conv2")(net)

    net, mask1 = Lambda(lambda t: tf.nn.max_pool_with_argmax(
        t, ksize=2, strides=2, padding="VALID"))(net)

    net = Lambda(lambda t: tf.pad(t, [[0, 0], [1, 1], [1, 1], [0, 0]],
                                  mode="REFLECT"))(net)
    net = Conv2D(filters=128,
                 kernel_size=(3, 3),
                 strides=(1, 1),
                 padding="VALID",
                 activation="relu",
                 name="block2_conv1")(net)
    net = Lambda(lambda t: tf.pad(t, [[0, 0], [1, 1], [1, 1], [0, 0]],
                                  mode="REFLECT"))(net)
    net = Conv2D(filters=128,
                 kernel_size=(3, 3),
                 strides=(1, 1),
                 padding="VALID",
                 activation="relu",
                 name="block2_conv2")(net)

    net, mask2 = Lambda(lambda t: tf.nn.max_pool_with_argmax(
        t, ksize=2, strides=2, padding="VALID"))(net)

    net = Lambda(lambda t: tf.pad(t, [[0, 0], [1, 1], [1, 1], [0, 0]],
                                  mode="REFLECT"))(net)
    net = Conv2D(filters=256,
                 kernel_size=(3, 3),
                 strides=(1, 1),
                 padding="VALID",
                 activation="relu",
                 name="block3_conv1")(net)

    encoder3 = Model(inp, [net, mask1, mask2], name="encoder3")
    encoder3.get_layer("block1_conv1").set_weights(
        vgg.get_layer("block1_conv1").get_weights())
    encoder3.get_layer("block1_conv2").set_weights(
        vgg.get_layer("block1_conv2").get_weights())
    encoder3.get_layer("block2_conv1").set_weights(
        vgg.get_layer("block2_conv1").get_weights())
    encoder3.get_layer("block2_conv2").set_weights(
        vgg.get_layer("block2_conv2").get_weights())
    encoder3.get_layer("block3_conv1").set_weights(
        vgg.get_layer("block3_conv1").get_weights())

    return encoder3


def build_decoder_3():
    inp = Input((None, None, 256))
    mask1 = Input((None, None, 64), dtype=tf.int64)
    mask2 = Input((None, None, 128), dtype=tf.int64)

    net = Lambda(lambda t: tf.pad(t, [[0, 0], [1, 1], [1, 1], [0, 0]],
                                  mode="REFLECT"))(inp)
    net = Conv2D(filters=128,
                 kernel_size=(3, 3),
                 strides=(1, 1),
                 activation="relu",
                 padding="VALID")(net)

    net = Unpool((2, 2))([net, mask2])

    net = Lambda(lambda t: tf.pad(t, [[0, 0], [1, 1], [1, 1], [0, 0]],
                                  mode="REFLECT"))(net)
    net = Conv2D(filters=128,
                 kernel_size=(3, 3),
                 strides=(1, 1),
                 activation="relu",
                 padding="VALID")(net)

    net = Lambda(lambda t: tf.pad(t, [[0, 0], [1, 1], [1, 1], [0, 0]],
                                  mode="REFLECT"))(net)
    net = Conv2D(filters=64,
                 kernel_size=(3, 3),
                 strides=(1, 1),
                 activation="relu",
                 padding="VALID")(net)

    net = Unpool((2, 2))([net, mask1])

    net = Lambda(lambda t: tf.pad(t, [[0, 0], [1, 1], [1, 1], [0, 0]],
                                  mode="REFLECT"))(net)
    net = Conv2D(filters=64,
                 kernel_size=(3, 3),
                 strides=(1, 1),
                 activation="relu",
                 padding="VALID")(net)

    net = Lambda(lambda t: tf.pad(t, [[0, 0], [1, 1], [1, 1], [0, 0]],
                                  mode="REFLECT"))(net)
    net = Conv2D(filters=3,
                 kernel_size=(3, 3),
                 strides=(1, 1),
                 activation=None,
                 padding="VALID")(net)

    return Model([inp, mask1, mask2], net, name="decoder3")


def build_encoder_4(vgg):
    inp = Input((None, None, 3))

    net = Lambda(lambda t: tf.pad(t, [[0, 0], [1, 1], [1, 1], [0, 0]],
                                  mode="REFLECT"))(inp)
    net = Conv2D(filters=64,
                 kernel_size=(3, 3),
                 strides=(1, 1),
                 padding="VALID",
                 activation="relu",
                 name="block1_conv1")(net)
    net = Lambda(lambda t: tf.pad(t, [[0, 0], [1, 1], [1, 1], [0, 0]],
                                  mode="REFLECT"))(net)
    net = Conv2D(filters=64,
                 kernel_size=(3, 3),
                 strides=(1, 1),
                 padding="VALID",
                 activation="relu",
                 name="block1_conv2")(net)

    net, mask1 = Lambda(lambda t: tf.nn.max_pool_with_argmax(
        t, ksize=2, strides=2, padding="VALID"))(net)

    net = Lambda(lambda t: tf.pad(t, [[0, 0], [1, 1], [1, 1], [0, 0]],
                                  mode="REFLECT"))(net)
    net = Conv2D(filters=128,
                 kernel_size=(3, 3),
                 strides=(1, 1),
                 padding="VALID",
                 activation="relu",
                 name="block2_conv1")(net)
    net = Lambda(lambda t: tf.pad(t, [[0, 0], [1, 1], [1, 1], [0, 0]],
                                  mode="REFLECT"))(net)
    net = Conv2D(filters=128,
                 kernel_size=(3, 3),
                 strides=(1, 1),
                 padding="VALID",
                 activation="relu",
                 name="block2_conv2")(net)

    net, mask2 = Lambda(lambda t: tf.nn.max_pool_with_argmax(
        t, ksize=2, strides=2, padding="VALID"))(net)

    net = Lambda(lambda t: tf.pad(t, [[0, 0], [1, 1], [1, 1], [0, 0]],
                                  mode="REFLECT"))(net)
    net = Conv2D(filters=256,
                 kernel_size=(3, 3),
                 strides=(1, 1),
                 padding="VALID",
                 activation="relu",
                 name="block3_conv1")(net)
    net = Lambda(lambda t: tf.pad(t, [[0, 0], [1, 1], [1, 1], [0, 0]],
                                  mode="REFLECT"))(net)
    net = Conv2D(filters=256,
                 kernel_size=(3, 3),
                 strides=(1, 1),
                 padding="VALID",
                 activation="relu",
                 name="block3_conv2")(net)
    net = Lambda(lambda t: tf.pad(t, [[0, 0], [1, 1], [1, 1], [0, 0]],
                                  mode="REFLECT"))(net)
    net = Conv2D(filters=256,
                 kernel_size=(3, 3),
                 strides=(1, 1),
                 padding="VALID",
                 activation="relu",
                 name="block3_conv3")(net)
    net = Lambda(lambda t: tf.pad(t, [[0, 0], [1, 1], [1, 1], [0, 0]],
                                  mode="REFLECT"))(net)
    net = Conv2D(filters=256,
                 kernel_size=(3, 3),
                 strides=(1, 1),
                 padding="VALID",
                 activation="relu",
                 name="block3_conv4")(net)

    net, mask3 = Lambda(lambda t: tf.nn.max_pool_with_argmax(
        t, ksize=2, strides=2, padding="VALID"))(net)

    net = Lambda(lambda t: tf.pad(t, [[0, 0], [1, 1], [1, 1], [0, 0]],
                                  mode="REFLECT"))(net)
    net = Conv2D(filters=512,
                 kernel_size=(3, 3),
                 strides=(1, 1),
                 padding="VALID",
                 activation="relu",
                 name="block4_conv1")(net)

    encoder4 = Model(inp, [net, mask1, mask2, mask3], name="encoder4")
    encoder4.get_layer("block1_conv1").set_weights(
        vgg.get_layer("block1_conv1").get_weights())
    encoder4.get_layer("block1_conv2").set_weights(
        vgg.get_layer("block1_conv2").get_weights())
    encoder4.get_layer("block2_conv1").set_weights(
        vgg.get_layer("block2_conv1").get_weights())
    encoder4.get_layer("block2_conv2").set_weights(
        vgg.get_layer("block2_conv2").get_weights())
    encoder4.get_layer("block3_conv1").set_weights(
        vgg.get_layer("block3_conv1").get_weights())
    encoder4.get_layer("block3_conv2").set_weights(
        vgg.get_layer("block3_conv2").get_weights())
    encoder4.get_layer("block3_conv3").set_weights(
        vgg.get_layer("block3_conv3").get_weights())
    encoder4.get_layer("block3_conv4").set_weights(
        vgg.get_layer("block3_conv4").get_weights())
    encoder4.get_layer("block4_conv1").set_weights(
        vgg.get_layer("block4_conv1").get_weights())
    return encoder4


def build_decoder_4():
    inp = Input((None, None, 512))
    mask1 = Input((None, None, 64), dtype=tf.int64)
    mask2 = Input((None, None, 128), dtype=tf.int64)
    mask3 = Input((None, None, 256), dtype=tf.int64)

    net = Lambda(lambda t: tf.pad(t, [[0, 0], [1, 1], [1, 1], [0, 0]],
                                  mode="REFLECT"))(inp)
    net = Conv2D(filters=256,
                 kernel_size=(3, 3),
                 strides=(1, 1),
                 activation="relu",
                 padding="VALID")(net)

    net = Unpool((2, 2))([net, mask3])

    net = Lambda(lambda t: tf.pad(t, [[0, 0], [1, 1], [1, 1], [0, 0]],
                                  mode="REFLECT"))(net)
    net = Conv2D(filters=256,
                 kernel_size=(3, 3),
                 strides=(1, 1),
                 activation="relu",
                 padding="VALID")(net)

    net = Lambda(lambda t: tf.pad(t, [[0, 0], [1, 1], [1, 1], [0, 0]],
                                  mode="REFLECT"))(net)
    net = Conv2D(filters=256,
                 kernel_size=(3, 3),
                 strides=(1, 1),
                 activation="relu",
                 padding="VALID")(net)

    net = Lambda(lambda t: tf.pad(t, [[0, 0], [1, 1], [1, 1], [0, 0]],
                                  mode="REFLECT"))(net)
    net = Conv2D(filters=256,
                 kernel_size=(3, 3),
                 strides=(1, 1),
                 activation="relu",
                 padding="VALID")(net)

    net = Lambda(lambda t: tf.pad(t, [[0, 0], [1, 1], [1, 1], [0, 0]],
                                  mode="REFLECT"))(net)
    net = Conv2D(filters=128,
                 kernel_size=(3, 3),
                 strides=(1, 1),
                 activation="relu",
                 padding="VALID")(net)

    net = Unpool((2, 2))([net, mask2])

    net = Lambda(lambda t: tf.pad(t, [[0, 0], [1, 1], [1, 1], [0, 0]],
                                  mode="REFLECT"))(net)
    net = Conv2D(filters=128,
                 kernel_size=(3, 3),
                 strides=(1, 1),
                 activation="relu",
                 padding="VALID")(net)

    net = Lambda(lambda t: tf.pad(t, [[0, 0], [1, 1], [1, 1], [0, 0]],
                                  mode="REFLECT"))(net)
    net = Conv2D(filters=64,
                 kernel_size=(3, 3),
                 strides=(1, 1),
                 activation="relu",
                 padding="VALID")(net)

    net = Unpool((2, 2))([net, mask1])

    net = Lambda(lambda t: tf.pad(t, [[0, 0], [1, 1], [1, 1], [0, 0]],
                                  mode="REFLECT"))(net)
    net = Conv2D(filters=64,
                 kernel_size=(3, 3),
                 strides=(1, 1),
                 activation="relu",
                 padding="VALID")(net)

    net = Lambda(lambda t: tf.pad(t, [[0, 0], [1, 1], [1, 1], [0, 0]],
                                  mode="REFLECT"))(net)
    net = Conv2D(filters=3,
                 kernel_size=(3, 3),
                 strides=(1, 1),
                 activation=None,
                 padding="VALID")(net)

    return Model([inp, mask1, mask2, mask3], net, name="decoder4")


@tf.function
def train_step(batch, encoder, decoder, feature_weight, pixel_weight,
               variation_weight, optimizer):
    with tf.GradientTape() as tape:
        encoded_features = encoder(batch)
        reconstructed_image = decoder(encoded_features)
        reconstruction_features = encoder(reconstructed_image)

        if isinstance(reconstruction_features, list):
            reconstruction_features = reconstruction_features[0]

        if isinstance(encoded_features, list):
            encoded_features = encoded_features[0]

        feature_loss = feature_weight * tf.reduce_mean(
            tf.square(reconstruction_features - encoded_features))
        pixel_loss = pixel_weight * tf.reduce_mean(
            tf.square(reconstructed_image - batch))
        total_variation_loss = variation_weight * tf.reduce_mean(
            tf.image.total_variation(reconstructed_image))

        loss = feature_loss + pixel_loss + total_variation_loss

    gradients = tape.gradient(loss, decoder.trainable_variables)
    optimizer.apply_gradients(zip(gradients, decoder.trainable_variables))
    return loss, feature_loss, pixel_loss, total_variation_loss


def train(job_dir, dataset, encoder, decoder, epochs, feature_weight,
          pixel_weight, variation_weight, optimizer):
    with tf.summary.create_file_writer(job_dir).as_default():
        step = 0
        for _ in range(epochs):
            for batch in dataset:
                loss, feature_loss, pixel_loss, total_variation_loss = train_step(
                    batch=batch,
                    encoder=encoder,
                    decoder=decoder,
                    feature_weight=feature_weight,
                    pixel_weight=pixel_weight,
                    variation_weight=variation_weight,
                    optimizer=optimizer)

                step += 1

                tf.summary.scalar("loss", loss, step=step)
                tf.summary.scalar("feature_loss", feature_loss, step=step)
                tf.summary.scalar("pixel_loss", pixel_loss, step=step)
                tf.summary.scalar("total_variation_loss",
                                  total_variation_loss,
                                  step=step)

                if step % 100 == 0 or step == 1:
                    print(
                        "Step: %d, loss: %f, feature_loss: %f, pixel_loss: %f, total_variation_loss: %f"
                        % (step, loss, feature_loss, pixel_loss,
                           total_variation_loss))


def run(job_dir, dataset, decoder, epochs, feature_weight, pixel_weight,
        variation_weight, learning_rate):
    vgg = vgg19.VGG19(include_top=False, weights="imagenet")

    if decoder == 1:
        encoder = build_encoder_1(vgg)
        encoder.trainable = False
        for layer in encoder.layers:
            layer.trainable = False

        decoder = build_decoder_1()
    elif decoder == 2:
        encoder = build_encoder_2(vgg)
        encoder.trainable = False
        for layer in encoder.layers:
            layer.trainable = False

        decoder = build_decoder_2()
    elif decoder == 3:
        encoder = build_encoder_3(vgg)
        encoder.trainable = False
        for layer in encoder.layers:
            layer.trainable = False

        decoder = build_decoder_3()
    else:
        encoder = build_encoder_4(vgg)
        encoder.trainable = False
        for layer in encoder.layers:
            layer.trainable = False

        decoder = build_decoder_4()

    encoder.summary()
    decoder.summary()

    optimizer = tf.keras.optimizers.Adam(learning_rate)

    train(job_dir=os.path.join(job_dir, "decoder"),
          dataset=dataset,
          encoder=encoder,
          decoder=decoder,
          epochs=epochs,
          feature_weight=feature_weight,
          pixel_weight=pixel_weight,
          variation_weight=variation_weight,
          optimizer=optimizer)

    return encoder, decoder


def main(_):
    data_builder = tfds.builder("coco2014", data_dir=flags.FLAGS.tfds_data_dir)
    dataset = data_builder.as_dataset(split=tfds.Split.TRAIN)
    dataset = dataset.map(lambda i: i["image"])
    dataset = dataset.map(lambda i: resize_min(i, 256))
    dataset = dataset.batch(flags.FLAGS.batch_size)

    encoder, decoder = run(job_dir=flags.FLAGS["job-dir"].value,
                           dataset=dataset,
                           decoder=flags.FLAGS.decoder,
                           epochs=flags.FLAGS.epochs,
                           feature_weight=flags.FLAGS.feature_weight,
                           pixel_weight=flags.FLAGS.pixel_weight,
                           variation_weight=flags.FLAGS.variation_weight,
                           learning_rate=flags.FLAGS.learning_rate)

    tf.keras.experimental.export_saved_model(encoder,
                                             os.path.join(
                                                 flags.FLAGS["job-dir"].value,
                                                 "export", "encoder"),
                                             serving_only=False)

    tf.keras.experimental.export_saved_model(decoder,
                                             os.path.join(
                                                 flags.FLAGS["job-dir"].value,
                                                 "export", "decoder"),
                                             serving_only=False)


if __name__ == "__main__":
    print(tf.version.VERSION)

    app.flags.DEFINE_integer("decoder", 4, "nth decoder to train")
    app.flags.DEFINE_integer("epochs", 1, "number of epochs")
    app.flags.DEFINE_float("learning_rate", 1e-3, "learning rate")
    app.flags.DEFINE_float("feature_weight", 1e-2, "features weight")
    app.flags.DEFINE_float("pixel_weight", 1.0, "pixel weight")
    app.flags.DEFINE_float("variation_weight", 1e-5, "variation weight")
    app.flags.DEFINE_integer("batch_size", 16, "batch size")
    app.flags.DEFINE_string("tfds_data_dir", "~/tensorflow_datasets",
                            "tfds data dir")
    app.flags.DEFINE_string("job-dir", "runs/local", "job dir")

    app.run(main)
