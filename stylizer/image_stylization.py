"""
A Learned Representation of Artistic Style, Domulin et.al.

Currently only a single node training loop is implemented.
Using the standard_v100 config, 40000 steps can be trained in roughly ~3 hours

NOTE: the magenta implementation has a batch size of 16, and trains for 40000 steps. Here, each batch is trained with each style image
effectively multiplying the batch size with the number of styles for a single train step. So training for 40000 steps with batch size of 16,
converts to training for 20000 steps with a batch size of 1, for 32 styles.

COCO has ~80K images in the training set, so training for 20K images is enough.

TODO: implemented distributed training as described at
https://www.tensorflow.org/alpha/tutorials/distribute/training_loops.
"""

import functools
import json
import sys

from absl import app, flags
import tensorflow as tf
from tensorflow.keras import Model  # pylint: disable=import-error
from tensorflow.keras import initializers  # pylint: disable=import-error
from tensorflow.keras.applications import vgg19  # pylint: disable=import-error
from tensorflow.keras.layers import Activation, Add, Conv2D, Input, Lambda, Layer, SeparableConv2D, UpSampling2D  # pylint: disable=import-error
import tensorflow_datasets as tfds

from .image_utils import gram_matrix, resize_min, vgg_preprocess_input
from .normalization import ConditionalInstanceNormalization


def conv_block(net, style_weights, filters, kernel_size, strides, activation,
               depthwise_separable_conv):
    """
    first applies padding with mode=REFLECT,
    then a valid conv2d,
    then conditional instance norm using style_weights
    and finally, activation

    TODO: add support to choose between regular and depthwise separable convolutions
    """
    pad_0 = (kernel_size[0] - 1) // 2
    pad_1 = (kernel_size[1] - 1) // 2

    net = Lambda(
        lambda t: tf.pad(t, [[0, 0], [pad_0, pad_0], [pad_1, pad_1], [0, 0]],
                         mode="REFLECT"))(net)

    weight_initializer = initializers.TruncatedNormal(mean=0.0,
                                                      stddev=0.1,
                                                      seed=1)

    if depthwise_separable_conv:
        net = SeparableConv2D(
            filters=filters,
            kernel_size=kernel_size,
            strides=strides,
            padding="valid",
            activation=None,
            depthwise_initializer=weight_initializer,
            pointwise_initializer=weight_initializer,
        )(net)
    else:
        net = Conv2D(
            filters=filters,
            kernel_size=kernel_size,
            strides=strides,
            padding="valid",
            activation=None,
            kernel_initializer=weight_initializer,
        )(net)

    net = ConditionalInstanceNormalization(
        style_weights.shape[-1])([net, style_weights])

    if activation != None:
        net = Activation(activation)(net)

    return net


def residual_block(net, style_weights, filters, kernel_size, strides,
                   depthwise_separable_conv):
    """
    basically a relu activated conv block followed by a non-activated conv block
    """
    tmp = conv_block(net,
                     style_weights,
                     filters=filters,
                     kernel_size=kernel_size,
                     strides=strides,
                     activation="relu",
                     depthwise_separable_conv=depthwise_separable_conv)
    tmp = conv_block(tmp,
                     style_weights,
                     filters=filters,
                     kernel_size=kernel_size,
                     strides=strides,
                     activation=None,
                     depthwise_separable_conv=depthwise_separable_conv)
    return Add()([net, tmp])


def upsampling_block(net, style_weights, interpolation_factor, filters,
                     kernel_size, strides, activation,
                     depthwise_separable_conv):
    """
    first resizing the image by interpolation factor using nearest neighbor sampling
    then a conv block
    """

    net = UpSampling2D(interpolation_factor, interpolation="nearest")(net)
    net = conv_block(net,
                     style_weights,
                     filters=filters,
                     kernel_size=kernel_size,
                     strides=strides,
                     activation=activation,
                     depthwise_separable_conv=depthwise_separable_conv)

    return net


def build_transformation_network(n_styles, depthwise_separable_conv):
    """
    This __has__ to be built using the functional API, as the task requires exporting both serving and non-serving
    variants of the model.

    The non-serving variant is required in order to support the usecase of adding a N+1th style, given we have a model
    capable of generating N styles.

    The model has two inputs, the first is the image itself, the second is a 1-D Tensor of floats, representing the
    weights of individual styles. Being able to control weights of arbitrary styles in a single pass gives the
    ability to combine styles at runtime. See Figure 7 in the paper, also https://www.youtube.com/watch?v=6ZHiARZmiUI
    """

    image_input = Input((None, None, 3), name="image")
    style_weights = Input((n_styles, ), name="style_weights")

    net = conv_block(image_input,
                     style_weights,
                     filters=32,
                     kernel_size=(9, 9),
                     strides=(1, 1),
                     activation="relu",
                     depthwise_separable_conv=depthwise_separable_conv)

    net = conv_block(net,
                     style_weights,
                     filters=64,
                     kernel_size=(3, 3),
                     strides=(2, 2),
                     activation="relu",
                     depthwise_separable_conv=depthwise_separable_conv)

    net = conv_block(net,
                     style_weights,
                     filters=128,
                     kernel_size=(3, 3),
                     strides=(2, 2),
                     activation="relu",
                     depthwise_separable_conv=depthwise_separable_conv)

    net = residual_block(net,
                         style_weights,
                         filters=128,
                         kernel_size=(3, 3),
                         strides=(1, 1),
                         depthwise_separable_conv=depthwise_separable_conv)

    net = residual_block(net,
                         style_weights,
                         filters=128,
                         kernel_size=(3, 3),
                         strides=(1, 1),
                         depthwise_separable_conv=depthwise_separable_conv)

    net = residual_block(net,
                         style_weights,
                         filters=128,
                         kernel_size=(3, 3),
                         strides=(1, 1),
                         depthwise_separable_conv=depthwise_separable_conv)

    net = residual_block(net,
                         style_weights,
                         filters=128,
                         kernel_size=(3, 3),
                         strides=(1, 1),
                         depthwise_separable_conv=depthwise_separable_conv)

    net = residual_block(net,
                         style_weights,
                         filters=128,
                         kernel_size=(3, 3),
                         strides=(1, 1),
                         depthwise_separable_conv=depthwise_separable_conv)

    net = upsampling_block(net,
                           style_weights,
                           interpolation_factor=2,
                           filters=64,
                           kernel_size=(3, 3),
                           strides=(1, 1),
                           activation="relu",
                           depthwise_separable_conv=depthwise_separable_conv)

    net = upsampling_block(net,
                           style_weights,
                           interpolation_factor=2,
                           filters=32,
                           kernel_size=(3, 3),
                           strides=(1, 1),
                           activation="relu",
                           depthwise_separable_conv=depthwise_separable_conv)

    net = conv_block(net,
                     style_weights,
                     filters=3,
                     kernel_size=(9, 9),
                     strides=(1, 1),
                     activation="sigmoid",
                     depthwise_separable_conv=depthwise_separable_conv)

    net = Lambda(lambda t: t * 255.0, name="output")(net)

    return Model([image_input, style_weights], net, name="transform_net")


@tf.function
def load_and_preprocess_image(path, max_dim=512):
    """
    load an image from the specified path
    then resize and centrally crop it so it is `max_dim` x `max_dim`
    then add the batch dimension
    then do the VGG preprocessing steps.
    """
    f = tf.io.read_file(path)
    img = tf.io.decode_image(f)
    img = resize_min(img, max_dim)
    img = tf.expand_dims(img, axis=0)
    img = vgg_preprocess_input(img)
    return img


@tf.function
def tile_content_feature(feature, n_styles):
    shape = tf.shape(feature)
    feature = tf.tile(feature, [1, n_styles, 1, 1])
    feature = tf.reshape(feature, [-1, shape[1], shape[2], shape[3]])
    return feature


@tf.function
def train_step(batch, n_styles, style_layers, style_features, style_weight,
               content_weight, transform_net, loss_net, optimizer):
    """
    repeat every item in the batch n_styles number of times
    repeat all styles in order batch_size number of times
    repeat all style features in order batch_size number of times
    repeat every content feature n_styles number of times

    overall, the batch size for a single step is "batch_size * n_styles"
    """

    with tf.GradientTape() as tape:
        batch = tile_content_feature(batch, n_styles)
        batch_size = tf.cast(tf.shape(batch)[0], tf.float32)

        style_inputs = tf.eye(n_styles, dtype=tf.float32)
        style_inputs = tf.tile(style_inputs, [batch_size / n_styles, 1])

        content_features = loss_net(batch)[len(style_layers):]

        predictions = transform_net([batch, style_inputs])

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

        style_score = (style_weight *
                       functools.reduce(tf.add, style_losses)) / batch_size
        content_score = (content_weight *
                         functools.reduce(tf.add, content_losses)) / batch_size

        # NOTE: according to the paper, total variation loss is no longer needed
        # due to improvements in the network architecture

        loss = style_score + content_score

    grads = tape.gradient(loss, transform_net.trainable_variables)
    optimizer.apply_gradients(zip(grads, transform_net.trainable_variables))

    return loss, style_score, content_score


def train(dataset, max_steps, save_summary_steps, optimizer, n_styles,
          style_layers, style_features, style_weight, content_weight,
          transform_net, loss_net):
    step = 0

    loss_ = tf.keras.metrics.Mean(name="Loss")
    style_loss_ = tf.keras.metrics.Mean(name="Style Loss")
    content_loss_ = tf.keras.metrics.Mean(name="Content Loss")

    while step < max_steps:
        for batch in dataset:
            loss, style_score, content_score = train_step(
                batch=batch,
                n_styles=n_styles,
                style_features=style_features,
                style_layers=style_layers,
                style_weight=style_weight,
                content_weight=content_weight,
                transform_net=transform_net,
                loss_net=loss_net,
                optimizer=optimizer)

            step += 1

            loss_(loss)
            style_loss_(style_score)
            content_loss_(content_score)

            if step == 1 or step % save_summary_steps == 0:
                tf.summary.scalar("Total Loss", loss_.result(), step=step)
                loss_.reset_states()

                tf.summary.scalar("Style Loss",
                                  style_loss_.result(),
                                  step=step)
                style_loss_.reset_states()

                tf.summary.scalar("Content Loss",
                                  content_loss_.result(),
                                  step=step)
                content_loss_.reset_states()

            if step >= max_steps:
                return


def run(job_dir, style_targets, style_targets_root, dataset, max_steps,
        save_summary_steps, batch_size, depthwise_separable_conv,
        learning_rate, style_layers, content_layers, style_weight,
        content_weight):
    with tf.summary.create_file_writer(job_dir).as_default():
        transform_net = build_transformation_network(
            n_styles=len(style_targets),
            depthwise_separable_conv=depthwise_separable_conv)
        transform_net.summary()

        vgg = vgg19.VGG19(include_top=False, weights="imagenet")
        vgg.trainable = False
        for layer in vgg.layers:
            layer.trainable = False

        content_outputs = [vgg.get_layer(l).output for l in content_layers]
        style_outputs = [vgg.get_layer(l).output for l in style_layers]

        loss_net = Model(vgg.input, style_outputs + content_outputs)

        style_images_processed = [
            load_and_preprocess_image("%s/%s" %
                                      (style_targets_root, style_target))
            for style_target in style_targets
        ]

        style_images = tf.squeeze(tf.stack(style_images_processed))

        style_features = loss_net(style_images)[:len(style_layers)]

        style_features = [gram_matrix(a) for a in style_features]
        style_features = [
            tf.tile(style_feature, [batch_size, 1, 1])
            for style_feature in style_features
        ]

        optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate,
                                             beta_1=0.9,
                                             beta_2=0.999)

        train(dataset=dataset,
              max_steps=max_steps,
              save_summary_steps=save_summary_steps,
              optimizer=optimizer,
              n_styles=len(style_targets),
              style_layers=style_layers,
              style_features=style_features,
              style_weight=style_weight,
              content_weight=content_weight,
              transform_net=transform_net,
              loss_net=loss_net)

        return transform_net, style_images_processed


def build_cmle_transform_net(transform_net, n_styles):
    """
    Wrap the transform net so it can be used by CMLE Online Prediction Service

    Unable to get it to work with arbitrary batch sizes, see https://stackoverflow.com/q/55822349/3673043

    This basically will transform the first image and return it.
    """

    inp = Input(shape=(), dtype=tf.string, name="image_bytes")
    style_weights = Input((n_styles, ), name="style_weights")

    net = Lambda(lambda t: tf.cast(tf.expand_dims(tf.io.decode_jpeg(t[0]), 0),
                                   tf.float32))(inp)

    net = transform_net([net, style_weights])

    net = Lambda(lambda t: tf.cast(tf.round(t), tf.uint8))(net)
    net = Lambda(lambda t: tf.expand_dims(tf.io.encode_jpeg(t[0]), 0),
                 name="output_bytes")(net)

    return Model([inp, style_weights], net, name="transform_net_cmle")


def main(_):
    artistic_style_targets = [
        "claude_monet__poppy_field_in_argenteuil.jpg",
        "edvard_munch__the_scream.jpg",
        "egon_schiele__edith_with_striped_dress.jpg",
        "frederic_edwin_church__eruption_at_cotopaxi.jpg",
        "henri_de_toulouse-lautrec__divan_japonais.jpg",
        "hokusai__the_great_wave_off_kanagawa.jpg",
        "joseph_william_turner__the_shipwreck_of_the_minotaur.jpg",
        "leonid_afremov__rain_princess.jpg",
        "louvre_udnie.jpg",
        "nicolas_poussin__landscape_with_a_calm.jpg",
        "pablo_picasso__la_muse.jpg",
        "paul_signac__cassis_cap_lombard.jpg",
        "pillars_of_creation.jpg",
        "vincent_van_gogh__the_starry_night.jpg",
        "wassily_kandinsky__white_zig_zags.jpg",
        "wolfgang_lettl__the_trial.jpg",
    ]

    comic_style_targets = [
        "archer__adam_reed.jpg",
        "batman_the_animated_series__bruce_timm.jpg",
        "ben_10__joe_casey_joe_kelly__cartoon_network.jpg",
        "dexters_laboratory__grenndy_taratkovsky.jpg",
        "dora_the_explorer__chris_glifford.jpg",
        "dragon_ball_z__akira_toriyama.jpg",
        "family_guy__seth_macfarlane.jpg",
        "g_i_joe__sunbow.jpg",
        "garfield__jim_davis.jpg",
        "johnny_bravo__van_partible.jpg",
        "popeye__max_fleischer.jpg",
        "powerpuff_girls__cartoon_network.jpg",
        "south_park__matt_parker_trey_stone.jpg",
        "the_simpsons__matt_groening_20th_Century_Fox.jpg",
        "thundercats__rankin_bass.jpg",
        "transformers__sunbow.jpg",
    ]

    if not flags.FLAGS.use_artistic_styles and not flags.FLAGS.use_comic_styles:
        print(
            "expected either --use_artistic_styles or --use_comic_styles or both"
        )
        sys.exit(1)

    style_targets = []

    if flags.FLAGS.use_artistic_styles:
        style_targets += artistic_style_targets

    if flags.FLAGS.use_comic_styles:
        style_targets += comic_style_targets

    data_builder = tfds.builder("coco2014", data_dir=flags.FLAGS.tfds_data_dir)
    dataset = data_builder.as_dataset(split=tfds.Split.TRAIN)
    dataset = dataset.map(lambda i: i["image"])
    dataset = dataset.map(lambda i: resize_min(i, 256))
    # TODO: shouldn't have to do drop_remainder, fix batch size dependent logic
    dataset = dataset.batch(flags.FLAGS.batch_size, drop_remainder=True)

    # https://github.com/suyash/stylizer/blob/master/tasks/fast_style_transfer/trainer/task.py#L412-L417
    style_layers = [
        "block1_conv1", "block2_conv1", "block3_conv1", "block4_conv1",
        "block5_conv1"
    ]
    content_layers = ["block4_conv2"]

    # TODO: consider also saving the cropped versions of the style images.
    # to show the __exact__ style inputs used for training
    transform_net, _ = run(
        job_dir=flags.FLAGS["job-dir"].value,
        style_targets=style_targets,
        style_targets_root=flags.FLAGS.style_targets_root,
        dataset=dataset,
        max_steps=flags.FLAGS.max_steps,
        save_summary_steps=flags.FLAGS.save_summary_steps,
        batch_size=flags.FLAGS.batch_size,
        depthwise_separable_conv=flags.FLAGS.use_depthwise_separable_conv,
        learning_rate=flags.FLAGS.learning_rate,
        style_layers=style_layers,
        content_layers=content_layers,
        style_weight=flags.FLAGS.style_weight,
        content_weight=flags.FLAGS.content_weight,
    )

    cmle_transform_net = build_cmle_transform_net(transform_net,
                                                  len(style_targets))

    tf.keras.experimental.export_saved_model(transform_net,
                                             "%s/export/transform_net" %
                                             flags.FLAGS["job-dir"].value,
                                             serving_only=False)

    tf.keras.experimental.export_saved_model(
        transform_net,
        "%s/export_serving/transform_net" % flags.FLAGS["job-dir"].value,
        serving_only=True)

    tf.keras.experimental.export_saved_model(
        cmle_transform_net,
        "%s/export_serving_cmle/transform_net" % flags.FLAGS["job-dir"].value,
        serving_only=True)

    with open("style_targets.json", "w") as f:
        json.dump(style_targets, f)

    tf.io.gfile.copy("style_targets.json",
                     "%s/style_targets.json" % flags.FLAGS["job-dir"].value)


if __name__ == "__main__":
    print(tf.version.VERSION)

    app.flags.DEFINE_float("content_weight", 1.0, "content weight")
    app.flags.DEFINE_float("style_weight", 1.0, "style weight")
    app.flags.DEFINE_integer("batch_size", 1, "batch size")
    app.flags.DEFINE_integer("max_steps", 20000, "maximum training steps")
    app.flags.DEFINE_integer("save_summary_steps", 1,
                             "intervals at which a summary is saved")
    app.flags.DEFINE_float("learning_rate", 0.001, "learning rate")
    app.flags.DEFINE_boolean(
        "use_depthwise_separable_conv", False,
        "use depthwise separable convolutions in image transformation net, instead of regular convolutions"
    )
    app.flags.DEFINE_boolean("use_artistic_styles", False,
                             "use artistic styles")
    app.flags.DEFINE_boolean("use_comic_styles", False, "use comic styles")
    app.flags.DEFINE_string("tfds_data_dir", "~/tensorflow_datasets",
                            "tfds data dir")
    app.flags.DEFINE_string("style_targets_root", "../images/style_targets",
                            "style targets root")
    app.flags.DEFINE_string("job-dir", "runs/local", "job dir")

    app.run(main)
