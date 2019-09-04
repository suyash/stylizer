"""
This implements a task that can build and train both CartoonGAN and ComixGAN.

ComixGAN is CartoonGAN with 4 changes

- non-saturating loss for generator
- sigmoid activation in last layer of discriminator
- 3 training steps for generator per 1 training step for discriminator
- pretrained discriminators on distinguishing real comic images with images that are edge-blurred.
"""

import os

from absl import app, flags
import tensorflow as tf
from tensorflow.keras import Model  # pylint: disable=import-error
from tensorflow.keras.applications import vgg19  # pylint: disable=import-error
from tensorflow.keras.layers import Input, Lambda  # pylint: disable=import-error
import tensorflow_datasets as tfds
from tensorflow_addons.layers import InstanceNormalization

from stylizer.datasets import Danbooru2017
from stylizer.utils import resize_min, vgg_preprocess_input

from .generator_pretraining import preprocess_generator_input, postprocess_generator_output
from .discriminator_pretraining import build_discriminator


@tf.function
def train_step(generator, discriminator, loss_net, generator_optimizer,
               discriminator_optimizer, crossentropy, mae, content_weight,
               real_batch, comics_batch, comics_edge_blurred_batch,
               train_discriminator):
    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        real_generator_input = preprocess_generator_input(real_batch)
        real_output = generator(real_generator_input, training=True)
        real_output = postprocess_generator_output(real_output)

        real_output_score = discriminator(real_output, training=True)
        comics_score = discriminator(comics_batch, training=True)
        comics_edge_blurred_score = discriminator(comics_edge_blurred_batch,
                                                  training=True)

        real_content_output = loss_net(vgg_preprocess_input(real_batch),
                                       training=False)
        generated_content_output = loss_net(vgg_preprocess_input(real_output),
                                            training=False)

        generator_gan_loss = crossentropy(
            y_true=tf.ones_like(real_output_score), y_pred=real_output_score)

        content_loss = content_weight * mae(y_true=real_content_output,
                                            y_pred=generated_content_output)

        discriminator_loss_real_output = crossentropy(
            y_true=tf.zeros_like(real_output_score), y_pred=real_output_score)
        discriminator_loss_comics = crossentropy(
            y_true=tf.ones_like(comics_score), y_pred=comics_score)
        discriminator_loss_comics_edge_blurred = crossentropy(
            y_true=tf.zeros_like(comics_edge_blurred_score),
            y_pred=comics_edge_blurred_score)

        discriminator_loss = discriminator_loss_real_output + discriminator_loss_comics + discriminator_loss_comics_edge_blurred

        generator_loss = generator_gan_loss + content_loss

    gen_gradients = gen_tape.gradient(generator_loss,
                                      generator.trainable_variables)
    # NOTE: disc_gradients needs to be initialized in advance, and cannot be done conditionally
    disc_gradients = disc_tape.gradient(discriminator_loss,
                                        discriminator.trainable_variables)

    generator_optimizer.apply_gradients(
        zip(gen_gradients, generator.trainable_variables))

    if train_discriminator:
        discriminator_optimizer.apply_gradients(
            zip(disc_gradients, discriminator.trainable_variables))

    return generator_loss, discriminator_loss, generator_gan_loss, content_loss, discriminator_loss_comics, discriminator_loss_comics_edge_blurred, discriminator_loss_real_output


def train(real_dataset, comics_dataset, comics_edge_blurred_dataset, generator,
          discriminator, loss_net, learning_rate, content_weight, max_steps,
          save_summary_steps, discriminator_training_interval):
    generator_optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    discriminator_optimizer = tf.keras.optimizers.Adam(
        learning_rate=learning_rate)

    crossentropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)
    mae = tf.keras.losses.MeanAbsoluteError()

    step = 0

    generator_loss_ = tf.keras.metrics.Mean(name="Loss")
    discriminator_loss_ = tf.keras.metrics.Mean(name="Loss")
    generator_gan_loss_ = tf.keras.metrics.Mean(name="Loss")
    content_loss_ = tf.keras.metrics.Mean(name="Loss")
    discriminator_loss_comics_ = tf.keras.metrics.Mean(name="Loss")
    discriminator_loss_comics_edge_blurred_ = tf.keras.metrics.Mean(
        name="Loss")
    discriminator_loss_real_output_ = tf.keras.metrics.Mean(name="Loss")

    while step < max_steps:
        for real_batch, comics_batch, comics_edge_blurred_batch in tf.data.Dataset.zip(
            (real_dataset, comics_dataset, comics_edge_blurred_dataset)):

            generator_loss, discriminator_loss, generator_gan_loss, content_loss, discriminator_loss_comics, discriminator_loss_comics_edge_blurred, discriminator_loss_real_output = train_step(
                generator=generator,
                discriminator=discriminator,
                loss_net=loss_net,
                generator_optimizer=generator_optimizer,
                discriminator_optimizer=discriminator_optimizer,
                crossentropy=crossentropy,
                mae=mae,
                content_weight=content_weight,
                real_batch=real_batch,
                comics_batch=comics_batch,
                comics_edge_blurred_batch=comics_edge_blurred_batch,
                train_discriminator=tf.constant(
                    (step + 1) % discriminator_training_interval == 0,
                    dtype=tf.bool),
            )

            step += 1
            generator_loss_(generator_loss)
            discriminator_loss_(discriminator_loss)
            generator_gan_loss_(generator_gan_loss)
            content_loss_(content_loss)
            discriminator_loss_comics_(discriminator_loss_comics)
            discriminator_loss_comics_edge_blurred_(
                discriminator_loss_comics_edge_blurred)
            discriminator_loss_real_output_(discriminator_loss_real_output)

            if step == 1 or step % save_summary_steps == 0:
                # summaries

                tf.summary.scalar("Generator Loss",
                                  generator_loss_.result(),
                                  step=step)
                generator_loss_.reset_states()

                tf.summary.scalar("Discriminator Loss",
                                  discriminator_loss_.result(),
                                  step=step)
                discriminator_loss_.reset_states()

                tf.summary.scalar("Generator GAN Loss",
                                  generator_gan_loss_.result(),
                                  step=step)
                generator_gan_loss_.reset_states()

                tf.summary.scalar("Content Loss",
                                  content_loss_.result(),
                                  step=step)
                content_loss_.reset_states()

                tf.summary.scalar("Discriminator Loss Comic Inputs",
                                  discriminator_loss_comics_.result(),
                                  step=step)
                discriminator_loss_comics_.reset_states()

                tf.summary.scalar(
                    "Discriminator Loss Edge Blurred Inputs",
                    discriminator_loss_comics_edge_blurred_.result(),
                    step=step)
                discriminator_loss_comics_edge_blurred_.reset_states()

                tf.summary.scalar("Discriminator Loss Real Inputs",
                                  discriminator_loss_real_output_.result(),
                                  step=step)
                discriminator_loss_real_output_.reset_states()

            if step >= max_steps:
                return


def run(job_dir,
        generator_dir,
        discriminator_dir,
        content_layers,
        real_dataset,
        comics_dataset,
        comics_edge_blurred_dataset,
        learning_rate,
        content_weight,
        max_steps,
        save_summary_steps,
        discriminator_training_interval=1):
    with tf.summary.create_file_writer(job_dir).as_default():
        generator = tf.keras.experimental.load_from_saved_model(
            generator_dir,
            custom_objects={"InstanceNormalization": InstanceNormalization},
        )
        generator.summary()

        discriminator = tf.keras.experimental.load_from_saved_model(
            discriminator_dir,
            custom_objects={"InstanceNormalization": InstanceNormalization},
        ) if discriminator_dir != None else build_discriminator()
        discriminator.summary()

        vgg = vgg19.VGG19(include_top=False, weights="imagenet")
        output_layers = [
            vgg.get_layer(layer).output for layer in content_layers
        ]
        loss_net = Model(vgg.input, output_layers, name="loss_net")

        train(real_dataset=real_dataset,
              comics_dataset=comics_dataset,
              comics_edge_blurred_dataset=comics_edge_blurred_dataset,
              generator=generator,
              discriminator=discriminator,
              loss_net=loss_net,
              learning_rate=learning_rate,
              content_weight=content_weight,
              max_steps=max_steps,
              save_summary_steps=save_summary_steps,
              discriminator_training_interval=discriminator_training_interval)

        return generator


def build_cmle_generator(generator):
    """
    Wrap the generator so it can be used by CMLE Online Prediction Service

    Unable to get it to work with arbitrary batch sizes, see https://stackoverflow.com/q/55822349/3673043

    This basically will transform the first image and return it.
    """

    inp = Input(shape=(), dtype=tf.string, name="image_bytes")

    net = Lambda(lambda t: preprocess_generator_input(
        tf.cast(tf.expand_dims(tf.io.decode_jpeg(t[0]), 0), tf.float32)))(inp)

    net = generator(net)

    net = Lambda(lambda t: tf.cast(tf.round(postprocess_generator_output(t)),
                                   tf.uint8))(net)
    net = Lambda(lambda t: tf.expand_dims(tf.io.encode_jpeg(t[0]), 0),
                 name="output_bytes")(net)

    return Model(inp, net, name="generator_cmle")


def main(_):
    real_builder = tfds.builder("coco2014", data_dir=flags.FLAGS.tfds_data_dir)
    real_dataset = real_builder.as_dataset(split=tfds.Split.TRAIN)
    # taking a 256x256 image by first scaling then taking center portion
    real_dataset = real_dataset.map(lambda i: resize_min(i["image"], 256))
    real_dataset = real_dataset.batch(flags.FLAGS.batch_size)
    # repeating since has less items than danbooru (~80000 to ~330000)
    real_dataset = real_dataset.repeat()

    comics_builder = Danbooru2017(config="danbooru-images",
                                  data_dir=flags.FLAGS.tfds_data_dir)
    comics_dataset = comics_builder.as_dataset(split=tfds.Split.TRAIN)
    # danbooru images are 512x512 with transparent padding, taking a 256x256 portion in the center
    comics_dataset = comics_dataset.map(
        lambda i: tf.cast(tf.image.central_crop(i["image"], 0.5), tf.float32))
    comics_dataset = comics_dataset.batch(flags.FLAGS.batch_size)

    comics_edge_blurred_builder = Danbooru2017(
        config="danbooru-images-edge-blurred",
        data_dir=flags.FLAGS.tfds_data_dir)
    comics_edge_blurred_dataset = comics_edge_blurred_builder.as_dataset(
        split=tfds.Split.TRAIN)
    # danbooru images are 512x512 with transparent padding, taking a 256x256 portion in the center
    comics_edge_blurred_dataset = comics_edge_blurred_dataset.map(
        lambda i: tf.cast(tf.image.central_crop(i["image"], 0.5), tf.float32))
    comics_edge_blurred_dataset = comics_edge_blurred_dataset.batch(
        flags.FLAGS.batch_size)
    # TODO: repeating since has less items than danbooru regular version (incomplete dataset)
    # remove once full version is parsed and done.
    comics_edge_blurred_dataset = comics_edge_blurred_dataset.repeat()

    content_layers = ["block4_conv1"]

    generator = run(job_dir=flags.FLAGS["job-dir"].value,
                    generator_dir=flags.FLAGS.generator,
                    discriminator_dir=flags.FLAGS.discriminator,
                    content_layers=content_layers,
                    real_dataset=real_dataset,
                    comics_dataset=comics_dataset,
                    comics_edge_blurred_dataset=comics_edge_blurred_dataset,
                    learning_rate=flags.FLAGS.learning_rate,
                    content_weight=flags.FLAGS.content_weight,
                    max_steps=flags.FLAGS.max_steps,
                    save_summary_steps=flags.FLAGS.save_summary_steps,
                    discriminator_training_interval=flags.FLAGS.
                    discriminator_training_interval)

    tf.keras.experimental.export_saved_model(generator,
                                             os.path.join(
                                                 flags.FLAGS["job-dir"].value,
                                                 "export", "generator"),
                                             serving_only=False)

    tf.keras.experimental.export_saved_model(
        generator,
        os.path.join(flags.FLAGS["job-dir"].value, "export_serving",
                     "generator"),
        serving_only=True)

    cmle_generator = build_cmle_generator(generator)

    tf.keras.experimental.export_saved_model(
        cmle_generator,
        os.path.join(flags.FLAGS["job-dir"].value, "export_serving_cmle",
                     "generator"),
        serving_only=True)


if __name__ == "__main__":
    print(tf.version.VERSION)

    app.flags.DEFINE_integer("batch_size", 4, "batch size")
    app.flags.DEFINE_float("learning_rate", 0.001, "learning rate")
    app.flags.DEFINE_float("content_weight", 1e-3, "content weight")
    app.flags.DEFINE_integer("max_steps", 10000, "maximum training steps")
    app.flags.DEFINE_integer("save_summary_steps", 100,
                             "intervals at which a summary is saved")
    app.flags.DEFINE_integer(
        "discriminator_training_interval", 1,
        "intervals at which a discriminator is trained, so 3 (ComixGAN) means there will be 3 updates to generator per update to discriminator"
    )
    app.flags.DEFINE_string("generator",
                            "runs/generator_pretrained/export/generator",
                            "location of pretrained generator")
    app.flags.DEFINE_string("discriminator", None,
                            "location of pretrained discriminator")
    app.flags.DEFINE_string("tfds_data_dir", "~/tensorflow_datasets",
                            "tfds data dir")
    app.flags.DEFINE_string("job-dir", "runs/local", "job dir")

    app.run(main)
