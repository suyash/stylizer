"""
This script takes in the locations of 4 decoders and an export location
and will export a serving compatible saved_model at the location.

python fast_photo_style_model.py \
    --decoder_1 PATH_TO_DECODER_1_SAVED_MODEL \
    --decoder_2 PATH_TO_DECODER_2_SAVED_MODEL \
    --decoder_3 PATH_TO_DECODER_3_SAVED_MODEL \
    --decoder_4 PATH_TO_DECODER_4_SAVED_MODEL \
    --export_dir runs/local
"""

import os

from absl import app, flags
import tensorflow as tf
from tensorflow.keras import Model  # pylint: disable=import-error
from tensorflow.keras.layers import Input, Lambda  # pylint: disable=import-error

from stylizer.layers import Unpool


@tf.function
def wct_core(inputs):
    """
    https://github.com/NVIDIA/FastPhotoStyle/blob/master/photo_wct.py#L122
    """
    cont_feat, styl_feat = inputs

    # Converting from channels last to channels first
    cont_feat = tf.transpose(cont_feat, [2, 0, 1])
    styl_feat = tf.transpose(styl_feat, [2, 0, 1])

    cont_feat_shape = tf.shape(cont_feat)

    cont_feat = tf.reshape(cont_feat, [tf.shape(cont_feat)[0], -1])
    styl_feat = tf.reshape(styl_feat, [tf.shape(styl_feat)[0], -1])

    cFSize = tf.shape(cont_feat)
    sFSize = tf.shape(styl_feat)

    c_mean = tf.reduce_mean(cont_feat, axis=1, keepdims=True)
    c_mean_expanded = tf.tile(c_mean, [1, cFSize[1]])
    cont_feat = cont_feat - c_mean_expanded

    iden = tf.eye(cFSize[0])

    cont_feat_t = tf.transpose(cont_feat)
    contentConv = (tf.matmul(cont_feat, cont_feat_t) /
                   tf.cast(cFSize[1] - 1, tf.float32)) + iden

    c_e, c_u, c_v = tf.linalg.svd(contentConv, full_matrices=True)

    k_c = cFSize[0]
    for i in range(cFSize[0] - 1, -1, -1):
        if c_e[i] >= 0.00001:
            k_c = i + 1
            break

    s_mean = tf.reduce_mean(styl_feat, axis=1, keepdims=True)
    s_mean_expanded = tf.tile(s_mean, [1, tf.shape(styl_feat)[1]])
    styl_feat = styl_feat - s_mean_expanded

    styl_feat_t = tf.transpose(styl_feat)
    styleConv = tf.matmul(styl_feat, styl_feat_t) / tf.cast(
        sFSize[1] - 1, tf.float32)

    s_e, s_u, s_v = tf.linalg.svd(styleConv, full_matrices=True)

    k_s = sFSize[0]
    for i in range(sFSize[0] - 1, -1, -1):
        if s_e[i] >= 0.00001:
            k_s = i + 1
            break

    c_d = tf.pow(c_e[0:k_c], -0.5)
    step1 = tf.matmul(c_v[:, 0:k_c], tf.linalg.tensor_diag(c_d))
    step2 = tf.matmul(step1, tf.transpose(c_v[:, 0:k_c]))
    whiten_cF = tf.matmul(step2, cont_feat)

    s_d = tf.pow(s_e[0:k_s], 0.5)
    targetFeature = tf.matmul(
        tf.matmul(tf.matmul(s_v[:, 0:k_s], tf.linalg.tensor_diag(s_d)),
                  tf.transpose(s_v[:, 0:k_s])), whiten_cF)

    targetFeature += tf.tile(s_mean, [1, tf.shape(targetFeature)[1]])

    targetFeature = tf.reshape(targetFeature, cont_feat_shape)

    # bring back [h, w, c]
    targetFeature = tf.transpose(targetFeature, [1, 2, 0])
    return targetFeature


@tf.function
def feature_wct(inp):
    cont_feat, styl_feat = inp
    # TODO: figure out the best strategy for parallel_iterations here
    return tf.map_fn(wct_core, [cont_feat, styl_feat],
                     dtype=tf.float32,
                     parallel_iterations=None)


def build(encoder1, decoder1, encoder2, decoder2, encoder3, decoder3, encoder4,
          decoder4):
    """
    https://github.com/NVIDIA/FastPhotoStyle/blob/master/photo_wct.py#L25
    """
    cont_img = Input((None, None, 3), name="content_image")
    styl_img = Input((None, None, 3), name="style_image")

    sF4, _, _, _ = encoder4(styl_img)
    cF4, cmask1, cmask2, cmask3 = encoder4(cont_img)
    csF4 = Lambda(feature_wct, name="feature_wct_4")([cF4, sF4])
    Im4 = decoder4([csF4, cmask1, cmask2, cmask3])

    sF3, _, _ = encoder3(styl_img)
    cF3, cmask1, cmask2 = encoder3(Im4)
    csF3 = Lambda(feature_wct, name="feature_wct_3")([cF3, sF3])
    Im3 = decoder3([csF3, cmask1, cmask2])

    sF2, _ = encoder2(styl_img)
    cF2, cmask1 = encoder2(Im3)
    csF2 = Lambda(feature_wct, name="feature_wct_2")([cF2, sF2])
    Im2 = decoder2([csF2, cmask1])

    sF1 = encoder1(styl_img)
    cF1 = encoder1(Im2)
    csF1 = Lambda(feature_wct, name="feature_wct_1")([cF1, sF1])
    Im1 = decoder1(csF1)

    no_smoothing_model = Model([cont_img, styl_img], Im1)

    return no_smoothing_model


def main(_):
    encoder1 = tf.keras.experimental.load_from_saved_model(
        flags.FLAGS.encoder_1)
    decoder1 = tf.keras.experimental.load_from_saved_model(
        flags.FLAGS.decoder_1, custom_objects={"Unpool": Unpool})

    encoder2 = tf.keras.experimental.load_from_saved_model(
        flags.FLAGS.encoder_2)
    decoder2 = tf.keras.experimental.load_from_saved_model(
        flags.FLAGS.decoder_2, custom_objects={"Unpool": Unpool})

    encoder3 = tf.keras.experimental.load_from_saved_model(
        flags.FLAGS.encoder_3)
    decoder3 = tf.keras.experimental.load_from_saved_model(
        flags.FLAGS.decoder_3, custom_objects={"Unpool": Unpool})

    encoder4 = tf.keras.experimental.load_from_saved_model(
        flags.FLAGS.encoder_4)
    decoder4 = tf.keras.experimental.load_from_saved_model(
        flags.FLAGS.decoder_4, custom_objects={"Unpool": Unpool})

    no_smoothing_model = build(encoder1, decoder1, encoder2, decoder2,
                               encoder3, decoder3, encoder4, decoder4)

    no_smoothing_model.summary()

    tf.keras.experimental.export_saved_model(no_smoothing_model,
                                             os.path.join(
                                                 flags.FLAGS.export_dir,
                                                 "no_smoothing"),
                                             serving_only=True)


if __name__ == "__main__":
    print(tf.version.VERSION)

    app.flags.DEFINE_string(
        "encoder_1", "runs/fast_photo_style_decoder_1_5/export/encoder",
        "location of encoder 1 saved model")
    app.flags.DEFINE_string(
        "decoder_1", "runs/fast_photo_style_decoder_1_5/export/decoder",
        "location of decoder 1 saved model")

    app.flags.DEFINE_string(
        "encoder_2", "runs/fast_photo_style_decoder_2_15/export/encoder",
        "location of encoder 2 saved model")
    app.flags.DEFINE_string(
        "decoder_2", "runs/fast_photo_style_decoder_2_15/export/decoder",
        "location of decoder 2 saved model")

    app.flags.DEFINE_string(
        "encoder_3", "runs/fast_photo_style_decoder_3_21/export/encoder",
        "location of encoder 3 saved model")
    app.flags.DEFINE_string(
        "decoder_3", "runs/fast_photo_style_decoder_3_21/export/decoder",
        "location of decoder 3 saved model")

    app.flags.DEFINE_string(
        "encoder_4", "runs/fast_photo_style_decoder_4_22/export/encoder",
        "location of encoder 4 saved model")
    app.flags.DEFINE_string(
        "decoder_4", "runs/fast_photo_style_decoder_4_22/export/decoder",
        "location of decoder 4 saved model")

    app.flags.DEFINE_string("export_dir", "runs/local", "job dir")
    app.run(main)
