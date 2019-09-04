import tensorflow as tf
from tensorflow.keras import initializers, regularizers, constraints  # pylint: disable=import-error
from tensorflow.keras.layers import Layer  # pylint: disable=import-error


class ConditionalInstanceNormalization(Layer):
    """
    A Conditional Instance Normalization Layer Implementation
    """
    def __init__(self,
                 n_styles,
                 epsilon=1e-3,
                 beta_initializer="zeros",
                 gamma_initializer="ones",
                 beta_regularizer=None,
                 gamma_regularizer=None,
                 beta_constraint=None,
                 gamma_constraint=None,
                 **kwargs):
        super(ConditionalInstanceNormalization, self).__init__(**kwargs)

        self.n_styles = n_styles
        self.epsilon = epsilon
        self.beta_initializer = initializers.get(beta_initializer)
        self.gamma_initializer = initializers.get(gamma_initializer)
        self.beta_regularizer = regularizers.get(beta_regularizer)
        self.gamma_regularizer = regularizers.get(gamma_regularizer)
        self.beta_constraint = constraints.get(beta_constraint)
        self.gamma_constraint = constraints.get(gamma_constraint)

    def build(self, input_shape):
        self.gamma = self.add_weight(shape=[self.n_styles, input_shape[0][-1]],
                                     name="gamma",
                                     initializer=self.gamma_initializer,
                                     regularizer=self.gamma_regularizer,
                                     constraint=self.gamma_constraint)

        self.beta = self.add_weight(shape=[self.n_styles, input_shape[0][-1]],
                                    name="beta",
                                    initializer=self.beta_initializer,
                                    regularizer=self.beta_regularizer,
                                    constraint=self.beta_constraint)

    def call(self, inputs):
        """
        inputs:
            image: incoming activation
            styles: a __column__ tensor (i.e. [N, 1]) of weights for individual styles.

        first normalizes the input image,
        and then multiplies it with a weighted sum of the gamma and beta parameters
        """

        image, style_weights = inputs

        mu, sigma_sq = tf.nn.moments(image, axes=[1, 2], keepdims=True)
        normalized = (image - mu) / tf.sqrt(sigma_sq + self.epsilon)

        gamma = tf.expand_dims(style_weights, -1) * self.gamma
        beta = tf.expand_dims(style_weights, -1) * self.beta

        gamma = tf.reduce_sum(gamma, axis=1)
        beta = tf.reduce_sum(beta, axis=1)

        gamma = tf.expand_dims(tf.expand_dims(gamma, axis=1), axis=1)
        beta = tf.expand_dims(tf.expand_dims(beta, axis=1), axis=1)

        return gamma * normalized + beta

    def get_config(self):
        base_config = super(ConditionalInstanceNormalization,
                            self).get_config()
        base_config["n_styles"] = self.n_styles
        base_config["epsilon"] = self.epsilon
        base_config["beta_initializer"] = self.beta_initializer
        base_config["gamma_initializer"] = self.gamma_initializer
        base_config["beta_regularizer"] = self.beta_regularizer
        base_config["gamma_regularizer"] = self.gamma_regularizer
        base_config["beta_constraint"] = self.beta_constraint
        base_config["gamma_constraint"] = self.gamma_constraint
        return base_config
