import tensorflow as tf
import math


def concat(x, y):
    """From Dr. Galkin(2018, personal communication, 10 July)"""
    return tf.concat([x, y], axis=3)


def conv(inputs, filters, kernel_size, stride):
    """From Dr. Galkin(2018, personal communication, 10 July)"""
    out = tf.layers.conv2d(
        inputs, filters=filters, kernel_size=kernel_size,
        kernel_initializer=tf.random_normal_initializer(stddev=0.02),
        strides=stride, padding='SAME')

    return out


def deconv(inputs, filters, kernel_size, stride):
    """From Dr. Galkin(2018, personal communication, 10 July)"""
    out = tf.layers.conv2d_transpose(
        inputs, filters=filters, kernel_size=kernel_size,
        kernel_initializer=tf.random_normal_initializer(stddev=0.02),
        strides=stride, padding='SAME')

    return out


def batch_norm(inputs, is_training, reuse):
    """From Dr. Galkin(2018, personal communication, 10 July)"""
    return tf.contrib.layers.batch_norm(
        inputs,
        decay=0.9,
        updates_collections=None,
        epsilon=1e-5,
        scale=True,
        is_training=is_training,
        reuse=reuse)


def dropout(inputs, rate):
    """From Dr. Galkin(2018, personal communication, 10 July)"""
    return tf.nn.dropout(inputs, keep_prob=1 - rate)


def relu(inputs):
    """From Dr. Galkin(2018, personal communication, 10 July)"""
    return tf.nn.relu(inputs)


def tanh(inputs):
    """From Dr. Galkin(2018, personal communication, 10 July)"""
    return tf.nn.tanh(inputs)


def lrelu(x, leak=0.2):
    """From Dr. Galkin(2018, personal communication, 10 July)"""
    with tf.variable_scope('lrelu'):
        f1 = 0.5 * (1 + leak)
        f2 = 0.5 * (1 - leak)
        return f1 * x + f2 * abs(x)


def l1_loss(x, y):
    """From Dr. Galkin(2018, personal communication, 10 July)"""
    return tf.reduce_mean(tf.abs(x - y))


def approx_min(X):
    rho = 10
    sum_term = tf.map_fn(lambda x: tf.exp(-rho*x), X)
    approximate_min = -(1/rho)*tf.log(tf.reduce_sum(sum_term, axis=0))
    return approximate_min


def l1_phase_loss(y, y_hat, approximate=True, loss_masking=False, mag_y_hat=None):
    """
    Calculates the l1 loss between two phase spectrograms, correcting for the circularity of phase. The true difference
    between each element of y_hat (the estimate) and y (the true value) is the closest to 0 of y_hat - y, y_hat - (y + 2pi) and y_hat - (y - 2pi).
    :param y_hat: 2D tensor, a phase spectrogram in radians
    :param y: 2D tensor, a phase spectrogram in radians
    :param approximate:
    :param loss_masking:
    :param mag_y_hat:
    :return: l1 loss between y_hat and y
    """
    pi = tf.constant(math.pi)
    original_diff = tf.abs(y_hat - y)
    add_2_pi_diff = tf.abs(y_hat - (y + 2 * pi))
    minus_2_pi_diff = tf.abs(y_hat - (y - 2 * pi))
    if approximate and loss_masking:
        return tf.reduce_mean(mag_y_hat * approx_min(tf.stack([original_diff, add_2_pi_diff, minus_2_pi_diff])))
    elif approximate:
        return tf.reduce_mean(approx_min(tf.stack([original_diff, add_2_pi_diff, minus_2_pi_diff])))
    elif loss_masking:
        return tf.reduce_mean(mag_y_hat * tf.minimum(original_diff, tf.minimum(add_2_pi_diff, minus_2_pi_diff)))
    else:
        return tf.reduce_mean(tf.minimum(original_diff, tf.minimum(add_2_pi_diff, minus_2_pi_diff)))


def phase_difference(x, y):
    """
    Calculates the difference between two phase spectrograms, correcting for the circularity of phase. The true difference
    between each element of x and y is the closest to 0 of x - y, x - (y + 2pi) and x - (y - 2pi).
    :param x: 2D tensor, a phase spectrogram in radians
    :param y: 2D tensor, a phase spectrogram in radians
    :return: difference between x and y
    """
    pi = tf.constant(math.pi)
    original_diff = x - y
    add_2_pi_diff = x - (y + 2 * pi)
    minus_2_pi_diff = x - (y - 2 * pi)
    first_corrected_diff = tf.where(tf.less(tf.abs(original_diff), tf.abs(add_2_pi_diff)),
                                    original_diff,
                                    add_2_pi_diff)
    second_corrected_diff = tf.where(tf.less(tf.abs(first_corrected_diff), tf.abs(minus_2_pi_diff)),
                                     first_corrected_diff,
                                     minus_2_pi_diff)

    return second_corrected_diff

