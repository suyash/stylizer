import tensorflow as tf


@tf.function
def resize_min(img, min_dim=512):
    """
    Scale the image preserving aspect ratio so that the minimum dimension is `min_dim`.
    Then, take a square crop in the middle to get a `min_dim` x `min_dim` image.
    """
    scale = tf.constant(min_dim, dtype=tf.float32) / tf.cast(
        tf.reduce_min(tf.shape(img)[0:2]), tf.float32)
    img = tf.image.resize_with_pad(
        img,
        tf.cast(tf.round(tf.cast(tf.shape(img)[0], tf.float32) * scale),
                tf.int32),
        tf.cast(tf.round(tf.cast(tf.shape(img)[1], tf.float32) * scale),
                tf.int32),
    )
    img = tf.image.resize_image_with_crop_or_pad(img, min_dim, min_dim)
    return img


@tf.function
def gram_matrix(feature):
    """
    https://github.com/tensorflow/magenta/blob/master/magenta/models/image_stylization/learning.py#L196
    """
    shape = tf.shape(feature)
    channels = shape[-1]
    batch_size = shape[0]
    a = tf.reshape(feature, [batch_size, -1, channels])
    a_T = tf.transpose(a, [0, 2, 1])
    n = shape[1] * shape[2]
    return tf.matmul(a_T, a) / tf.cast(n, tf.float32)


@tf.function
def vgg_preprocess_input(img):
    """
    VGG19 preprocessing steps for an image
    RGB -> BGR
    followed by normalization
    """
    img = img[..., ::-1]
    return tf.cast(img, tf.float32) - [103.939, 116.779, 123.68]
