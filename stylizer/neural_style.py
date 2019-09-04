"""
https://colab.research.google.com/github/tensorflow/models/blob/master/research/nst_blogpost/4_Neural_Style_Transfer_with_Eager_Execution.ipynb
"""

from absl import app, flags
import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow.keras import Model  # pylint: disable=import-error
from tensorflow.keras.applications import vgg19  # pylint: disable=import-error

from .utils import gram_matrix, vgg_preprocess_input


def train(content_image, style_image, max_steps, content_weight, style_weight,
          job_dir):
    with tf.summary.create_file_writer(job_dir).as_default():
        content_layers = ["block5_conv2"]
        style_layers = [
            "block1_conv1", "block2_conv1", "block3_conv1", "block4_conv1",
            "block5_conv1"
        ]

        model = create_model(content_layers, style_layers)

        for layer in model.layers:
            layer.trainable = False

        style_features, content_features = feature_representations(
            model, content_image, style_image, len(style_layers))

        gram_matrices = [
            gram_matrix(style_feature) for style_feature in style_features
        ]

        input_image = tf.Variable(load_and_preprocess_image(content_image),
                                  dtype=tf.float32)

        x_outputs = model(input_image)[len(style_layers):]

        optimizer = tf.keras.optimizers.Adam(learning_rate=5,
                                             beta_1=0.99,
                                             epsilon=1e-1)

        norm_means = np.array([103.939, 116.779, 123.68])
        min_vals = -norm_means
        max_vals = 255 - norm_means

        best_loss, best_img = float("inf"), None

        for i in range(max_steps + 1):
            total_loss, content_loss, style_loss = train_step(
                model, optimizer, input_image, content_features, gram_matrices,
                style_weight, content_weight, len(style_layers),
                len(content_layers))

            clipped = tf.clip_by_value(input_image, min_vals, max_vals)
            input_image.assign(clipped)

            tf.summary.scalar("Total Loss", total_loss, step=i)
            tf.summary.scalar("Style Loss", style_loss, step=i)
            tf.summary.scalar("Content Loss", content_loss, step=i)

            current_image = convert_to_image(input_image.numpy())

            if i % 100 == 0:
                print(
                    "Iteration: %d, style loss: %.4f, content_loss: %.4f, total_loss: %.4f"
                    % (i, style_loss, content_loss, total_loss))
                tf.summary.image("Step %d" % i,
                                 np.expand_dims(current_image, 0),
                                 step=i)

            if total_loss < best_loss:
                best_loss = total_loss
                best_img = current_image

        return best_img


def create_model(content_layers, style_layers):
    vgg = vgg19.VGG19(include_top=False, weights="imagenet")
    vgg.trainable = False

    style_outputs = [vgg.get_layer(name).output for name in style_layers]
    content_outputs = [vgg.get_layer(name).output for name in content_layers]
    model_outputs = style_outputs + content_outputs

    return Model(vgg.input, model_outputs)


def feature_representations(model, content_path, style_path, num_style_layers):
    content_image = load_and_preprocess_image(content_path)
    style_image = load_and_preprocess_image(style_path)

    content_outputs = model(content_image)
    style_outputs = model(style_image)

    style_features = [
        style_layer for style_layer in style_outputs[:num_style_layers]
    ]

    content_features = [
        content_layer for content_layer in content_outputs[num_style_layers:]
    ]

    return style_features, content_features


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
def train_step(model, optimizer, input_image, content_features, gram_matrices,
               style_weight, content_weight, num_style_layers,
               num_content_layers):
    with tf.GradientTape() as tape:
        total_loss, content_loss, style_loss = compute_loss(
            model, input_image, content_features, gram_matrices, style_weight,
            content_weight, num_style_layers, num_content_layers)

    grads = tape.gradient(total_loss, input_image)
    optimizer.apply_gradients([(grads, input_image)])
    return total_loss, content_loss, style_loss


@tf.function
def compute_loss(model, input_image, content_features, gram_matrices,
                 style_weight, content_weight, num_style_layers,
                 num_content_layers):
    model_outputs = model(input_image)

    style_output_features = model_outputs[:num_style_layers]
    content_output_features = model_outputs[num_style_layers:]

    style_score = 0
    content_score = 0

    weight_per_style_layer = 1.0 / float(num_style_layers)
    for target_style, output_style in zip(gram_matrices,
                                          style_output_features):
        style_score += weight_per_style_layer * compute_style_loss(
            output_style, target_style)

    weight_per_content_layer = 1.0 / float(num_content_layers)
    for target_content, output_content in zip(content_features,
                                              content_output_features):
        content_score += weight_per_content_layer * compute_content_loss(
            output_content, target_content)

    style_score *= style_weight
    content_score *= content_weight

    total_loss = style_score + content_score

    return total_loss, content_score, style_score


@tf.function
def compute_style_loss(target_style, gram_input):
    gram_style = gram_matrix(target_style)
    return tf.reduce_mean(tf.square(gram_style - gram_input))


@tf.function
def compute_content_loss(target_content, input_content):
    return tf.reduce_mean(tf.square(target_content - input_content))


def convert_to_image(processed_image):
    x = processed_image.copy()
    if len(x.shape) == 4:
        x = np.squeeze(x, 0)

    x[:, :, 0] += 103.939
    x[:, :, 1] += 116.779
    x[:, :, 2] += 123.68

    x = x[:, :, ::-1]
    x = np.clip(x, 0, 255).astype("uint8")
    return x


def main(_):
    best_image = train(flags.FLAGS.content_image, flags.FLAGS.style_image,
                       flags.FLAGS.max_steps, flags.FLAGS.content_weight,
                       flags.FLAGS.style_weight, flags.FLAGS["job-dir"].value)
    Image.fromarray(best_image).save("best.jpg")
    tf.io.gfile.copy("best.jpg",
                     "%s/best.jpg" % flags.FLAGS["job-dir"].value,
                     overwrite=True)


if __name__ == "__main__":
    print(tf.version.VERSION)

    app.flags.DEFINE_string("content_image", None, "content image")
    app.flags.DEFINE_string("style_image", None, "style image")
    app.flags.DEFINE_integer("max_steps", 1000, "max steps")
    app.flags.DEFINE_float("content_weight", 1e3, "content weight")
    app.flags.DEFINE_float("style_weight", 1e-2, "style_weight")
    app.flags.DEFINE_string("job-dir", "runs/local", "model dir")

    app.run(main)
