import tensorflow as tf
from tensorflow.keras.layers import Layer  # pylint: disable=import-error


class Unpool(Layer):
    """
    https://github.com/tensorflow/tensorflow/pull/16885/files
    """
    def __init__(self, pool_size, **kwargs):
        super(Unpool, self).__init__(**kwargs)
        self.pool_size = pool_size

    def call(self, inp):
        val, mask = inp
        shape = tf.shape(val)

        flat_shape = [tf.size(val)]
        val_ = tf.reshape(val, flat_shape)

        batch_range = tf.reshape(tf.cast(tf.range(shape[0]), tf.int64),
                                 (shape[0], 1, 1, 1))
        b = tf.ones_like(mask) * batch_range
        b = tf.reshape(b, (tf.size(val), 1))

        ind = tf.reshape(mask, (tf.size(val), 1))
        ind = ind - b * tf.cast(
            (shape[1] * shape[2] * shape[3] * self.pool_size[0] *
             self.pool_size[1]), tf.int64)
        ind = tf.concat([b, ind], axis=1)

        ans = tf.scatter_nd(ind,
                            val_,
                            shape=(shape[0], shape[1] * shape[2] * shape[3] *
                                   self.pool_size[0] * self.pool_size[1]))
        ans = tf.reshape(ans,
                         (shape[0], shape[1] * self.pool_size[0],
                          shape[2] * self.pool_size[1], val.get_shape()[3]))
        return ans

    def get_config(self):
        config = super(Unpool, self).get_config()
        config["pool_size"] = self.pool_size
        return config
