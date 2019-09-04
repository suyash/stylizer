"""
ComixGAN paper, Section 3.2.5

Pretrain the discriminator to differentiate between smooth edge comic images
and edge blurred images.
"""

from absl import app, flags
import tensorflow as tf
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.layers import Activation, Conv2D, Input, LeakyReLU
from tensorflow.keras.models import Model
import tensorflow_datasets as tfds

from trainer.generator_pretraining_task import InstanceNormalization
from trainer.danbooru2017 import Danbooru2017


def build_discriminator(name="discriminator"):
    inp = Input((None, None, 3))

    net = Conv2D(
        filters=32,
        kernel_size=(3, 3),
        strides=(1, 1),
        padding="SAME",
        activation=None)(inp)
    net = LeakyReLU(0.2)(net)

    net = Conv2D(
        filters=64,
        kernel_size=(3, 3),
        strides=(2, 2),
        padding="SAME",
        activation=None)(net)
    net = LeakyReLU(0.2)(net)
    net = Conv2D(
        filters=128,
        kernel_size=(3, 3),
        strides=(1, 1),
        padding="SAME",
        activation=None)(net)
    net = InstanceNormalization()(net)
    net = LeakyReLU(0.2)(net)

    net = Conv2D(
        filters=128,
        kernel_size=(3, 3),
        strides=(2, 2),
        padding="SAME",
        activation=None)(net)
    net = LeakyReLU(0.2)(net)
    net = Conv2D(
        filters=256,
        kernel_size=(3, 3),
        strides=(1, 1),
        padding="SAME",
        activation=None)(net)
    net = InstanceNormalization()(net)
    net = LeakyReLU(0.2)(net)

    net = Conv2D(
        filters=256,
        kernel_size=(3, 3),
        strides=(1, 1),
        padding="SAME",
        activation=None)(net)
    net = InstanceNormalization()(net)
    net = LeakyReLU(0.2)(net)

    net = Conv2D(
        filters=1,
        kernel_size=(3, 3),
        strides=(1, 1),
        padding="SAME",
        activation=None)(net)

    return Model(inp, net, name=name)


def run(job_dir, dataset, epochs, steps_per_epoch, learning_rate):
    net = build_discriminator()

    loss = tf.keras.losses.BinaryCrossentropy(from_logits=True)
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

    net.compile(loss=loss, optimizer=optimizer, metrics=["acc"])

    net.fit(
        dataset,
        epochs=epochs,
        steps_per_epoch=steps_per_epoch,
        callbacks=[TensorBoard(log_dir=job_dir)])

    return net


def create_mixed_dataset(tfds_data_dir, batch_size):
    """
    take the smooth and edge blurred images, and return a dataset
    returning a mix of them in random order.
    """

    smooth_builder = Danbooru2017(
        config="danbooru-images", data_dir=tfds_data_dir)
    smooth_dataset = smooth_builder.as_dataset(split=tfds.Split.ALL)
    smooth_dataset = smooth_dataset.map(lambda i: (
        tf.cast(tf.image.central_crop(i["image"], 0.5), tf.float32),
        tf.ones((64, 64, 1), tf.float32)))

    edge_blurred_builder = Danbooru2017(
        config="danbooru-images-edge-blurred", data_dir=tfds_data_dir)
    edge_blurred_dataset = edge_blurred_builder.as_dataset(
        split=tfds.Split.ALL)
    edge_blurred_dataset = edge_blurred_dataset.map(lambda i: (
        tf.cast(tf.image.central_crop(i["image"], 0.5), tf.float32),
        tf.zeros((64, 64, 1), tf.float32)))

    dataset = tf.data.Dataset.zip((smooth_dataset, edge_blurred_dataset))
    dataset = dataset.map(lambda a, b: (tf.stack([a[0], b[0]]),
                                        tf.stack([a[1], b[1]])))
    dataset = dataset.apply(tf.data.experimental.unbatch())
    dataset = dataset.shuffle(buffer_size=1000)
    dataset = dataset.batch(batch_size)

    return dataset


def main(_):
    dataset = create_mixed_dataset(flags.FLAGS.tfds_data_dir,
                                   flags.FLAGS.batch_size)

    net = run(
        job_dir=flags.FLAGS["job-dir"].value,
        dataset=dataset,
        epochs=flags.FLAGS.epochs,
        steps_per_epoch=flags.FLAGS.steps_per_epoch,
        learning_rate=flags.FLAGS.learning_rate)

    tf.keras.experimental.export_saved_model(
        net,
        "%s/export/discriminator" % flags.FLAGS["job-dir"].value,
        serving_only=False)


if __name__ == "__main__":
    print(tf.__version__)

    app.flags.DEFINE_integer("batch_size", 4, "batch size")
    app.flags.DEFINE_float("learning_rate", 0.001, "learning rate")
    app.flags.DEFINE_integer("epochs", 100, "epochs")
    app.flags.DEFINE_integer("steps_per_epoch", 1000, "steps_per_epoch")
    app.flags.DEFINE_string("tfds_data_dir", "~/tensorflow_datasets",
                            "tfds data dir")
    app.flags.DEFINE_string("job-dir", "runs/local", "job dir")

    app.run(main)
