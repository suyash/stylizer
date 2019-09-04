"""
CartoonGAN paper, Section 3.3

Basically pretrain the generator with L1 reconstruction loss on content images
to reconstruct the content as is.
"""

from absl import app, flags
import tensorflow as tf
from tensorflow.keras import Model  # pylint: disable=import-error
from tensorflow.keras.applications import vgg19  # pylint: disable=import-error
from tensorflow.keras.callbacks import TensorBoard  # pylint: disable=import-error
from tensorflow.keras.layers import Add, Conv2D, Conv2DTranspose, Input, Lambda, ReLU, SeparableConv2D, UpSampling2D  # pylint: disable=import-error
from tensorflow_addons.layers import InstanceNormalization
import tensorflow_datasets as tfds

from stylizer.utils import resize_min, vgg_preprocess_input


def build_generator(num_residual_blocks, use_upsampling, name="generator"):
    """
    https://github.com/maciej3031/comixify/blob/master/CartoonGAN/network/Transformer.py
    """
    inp = Input((None, None, 3))

    net = Lambda(lambda t: tf.pad(t, [[0, 0], [3, 3], [3, 3], [0, 0]],
                                  mode="REFLECT"))(inp)
    net = Conv2D(filters=64,
                 kernel_size=(7, 7),
                 strides=(1, 1),
                 padding="VALID",
                 activation=None)(net)
    net = InstanceNormalization()(net)
    net = ReLU()(net)

    net = Conv2D(filters=128,
                 kernel_size=(3, 3),
                 strides=(2, 2),
                 padding="SAME",
                 activation=None)(net)
    net = Conv2D(filters=128,
                 kernel_size=(3, 3),
                 strides=(1, 1),
                 padding="SAME",
                 activation=None)(net)
    net = InstanceNormalization()(net)
    net = ReLU()(net)

    net = Conv2D(filters=256,
                 kernel_size=(3, 3),
                 strides=(2, 2),
                 padding="SAME",
                 activation=None)(net)
    net = Conv2D(filters=256,
                 kernel_size=(3, 3),
                 strides=(1, 1),
                 padding="SAME",
                 activation=None)(net)
    net = InstanceNormalization()(net)
    net = ReLU()(net)

    # 8 residual blocks
    for _ in range(num_residual_blocks):
        tmp = Lambda(lambda t: tf.pad(t, [[0, 0], [1, 1], [1, 1], [0, 0]],
                                      mode="REFLECT"))(net)
        tmp = Conv2D(filters=256,
                     kernel_size=(3, 3),
                     strides=(1, 1),
                     padding="VALID",
                     activation=None)(tmp)
        tmp = InstanceNormalization()(tmp)
        tmp = ReLU()(tmp)
        tmp = Lambda(lambda t: tf.pad(t, [[0, 0], [1, 1], [1, 1], [0, 0]],
                                      mode="REFLECT"))(tmp)
        tmp = Conv2D(filters=256,
                     kernel_size=(3, 3),
                     strides=(1, 1),
                     padding="VALID",
                     activation=None)(tmp)
        tmp = InstanceNormalization()(tmp)
        net = Add()([net, tmp])

    if use_upsampling:
        net = UpSampling2D((2, 2), interpolation="nearest")(net)
        net = Conv2D(filters=128,
                     kernel_size=(3, 3),
                     strides=(1, 1),
                     padding="SAME",
                     activation=None)(net)
    else:
        net = Conv2DTranspose(filters=128,
                              kernel_size=(3, 3),
                              strides=(2, 2),
                              padding="SAME",
                              activation=None)(net)

    net = Conv2D(filters=128,
                 kernel_size=(3, 3),
                 strides=(1, 1),
                 padding="SAME",
                 activation=None)(net)
    net = InstanceNormalization()(net)
    net = ReLU()(net)

    if use_upsampling:
        net = UpSampling2D((2, 2), interpolation="nearest")(net)
        net = Conv2D(filters=64,
                     kernel_size=(3, 3),
                     strides=(1, 1),
                     padding="SAME",
                     activation=None)(net)
    else:
        net = Conv2DTranspose(filters=64,
                              kernel_size=(3, 3),
                              strides=(2, 2),
                              padding="SAME",
                              activation=None)(net)

    net = Conv2D(filters=64,
                 kernel_size=(3, 3),
                 strides=(1, 1),
                 padding="SAME",
                 activation=None)(net)
    net = InstanceNormalization()(net)
    net = ReLU()(net)

    net = Lambda(lambda t: tf.pad(t, [[0, 0], [3, 3], [3, 3], [0, 0]],
                                  mode="REFLECT"))(net)
    net = Conv2D(filters=3,
                 kernel_size=(7, 7),
                 strides=(1, 1),
                 padding="VALID",
                 activation="tanh")(net)

    return Model(inp, net, name=name)


@tf.function
def preprocess_generator_input(img):
    img = img / 127.5
    img -= 1
    return img


@tf.function
def postprocess_generator_output(img):
    img = img + 1
    img *= 127.5
    return img


@tf.function
def train_step(batch, transform_net, loss_net, optimizer, criterion):
    with tf.GradientTape() as tape:
        generator_inp = preprocess_generator_input(batch)
        generator_out = transform_net(generator_inp)
        generator_out = postprocess_generator_output(generator_out)

        loss_net_real_inp = vgg_preprocess_input(batch)
        loss_net_fake_inp = vgg_preprocess_input(generator_out)

        loss_net_real_out = loss_net(loss_net_real_inp)
        loss_net_fake_out = loss_net(loss_net_fake_inp)

        loss = criterion(y_true=loss_net_real_out, y_pred=loss_net_fake_out)

    grads = tape.gradient(loss, transform_net.trainable_variables)
    optimizer.apply_gradients(zip(grads, transform_net.trainable_variables))
    return loss


def train(dataset, transform_net, loss_net, learning_rate, max_steps,
          save_summary_steps):
    # L1 Reconstruction Loss
    criterion = tf.losses.MeanAbsoluteError()

    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

    loss_ = tf.keras.metrics.Mean(name="Loss")
    step = 0

    while step < max_steps:
        for batch in dataset:
            loss = train_step(batch=batch,
                              transform_net=transform_net,
                              loss_net=loss_net,
                              optimizer=optimizer,
                              criterion=criterion)

            step += 1
            loss_(loss)

            if step == 1 or step % save_summary_steps == 0:
                loss_val = loss_.result()
                print("step: %d, loss: %f" % (step, loss_val))
                tf.summary.scalar("Loss", loss_val, step=step)
                loss_.reset_states()

            if step == max_steps:
                return


def run(job_dir, dataset, content_layers, num_residual_blocks, use_upsampling,
        learning_rate, max_steps, save_summary_steps):
    with tf.summary.create_file_writer(job_dir).as_default():
        vgg = vgg19.VGG19(include_top=False, weights="imagenet")
        content_outputs = [vgg.get_layer(l).output for l in content_layers]
        loss_net = Model(vgg.input, content_outputs, name="loss_net")

        transform_net = build_generator(
            num_residual_blocks=num_residual_blocks,
            use_upsampling=use_upsampling)
        transform_net.summary()

        train(dataset=dataset,
              transform_net=transform_net,
              loss_net=loss_net,
              learning_rate=learning_rate,
              max_steps=max_steps,
              save_summary_steps=save_summary_steps)

        return transform_net


def main(_):
    dataset_builder = tfds.builder("coco2014",
                                   data_dir=flags.FLAGS.tfds_data_dir)
    dataset = dataset_builder.as_dataset(split=tfds.Split.TRAIN)
    dataset = dataset.map(lambda i: tf.cast(i["image"], tf.float32))
    dataset = dataset.map(lambda i: resize_min(i, 256))
    dataset = dataset.batch(flags.FLAGS.batch_size)

    content_layers = ["block4_conv1"]

    transform_net = run(job_dir=flags.FLAGS["job-dir"].value,
                        dataset=dataset,
                        content_layers=content_layers,
                        num_residual_blocks=flags.FLAGS.num_residual_blocks,
                        use_upsampling=flags.FLAGS.use_upsampling,
                        learning_rate=flags.FLAGS.learning_rate,
                        max_steps=flags.FLAGS.max_steps,
                        save_summary_steps=flags.FLAGS.save_summary_steps)

    tf.keras.experimental.export_saved_model(transform_net,
                                             "%s/export/generator" %
                                             flags.FLAGS["job-dir"].value,
                                             serving_only=False)


if __name__ == "__main__":
    print(tf.version.VERSION)

    app.flags.DEFINE_integer("batch_size", 8, "batch size")
    app.flags.DEFINE_integer("num_residual_blocks", 8,
                             "number of residual blocks")
    app.flags.DEFINE_boolean("use_upsampling", False,
                             "use upsampling instead of transposed conv")
    app.flags.DEFINE_float("learning_rate", 0.001, "learning rate")
    app.flags.DEFINE_integer("max_steps", 80000, "maximum training steps")
    app.flags.DEFINE_integer("save_summary_steps", 1,
                             "intervals at which a summary is saved")
    app.flags.DEFINE_string("tfds_data_dir", "~/tensorflow_datasets",
                            "tfds data dir")
    app.flags.DEFINE_string("job-dir", "runs/local", "job dir")

    app.run(main)
