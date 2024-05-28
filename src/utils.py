import tensorflow as tf
from tensorflow import keras
import numpy as np

def pad_mask(x, pad_id : int):
    pad = [x!=pad_id][:, tf.newaxis, tf.newaxis, :]
    return pad

def sequence_mask(length, max_length=None):
  if max_length is None:
    max_length = length.max()
  x = keras.backend.arange(max_length, dtype=length.dtype, device=length.device)
  return x[tf.newaxis, :] < length[:, tf.newaxis]

def log_likelihood(z, mu, sigma):
    with tf.device("/CPU:0"):
        return -0.5 * tf.reduce_sum(tf.square((z - mu) / sigma), axis=-1) - tf.reduce_sum(tf.math.log(sigma), axis=-1)

# MAS 동적 계획법 구현
def monotonic_alignment_search(z, mu, sigma):
    with tf.device("/CPU:0"):
        batch_size, seq_len = tf.shape(z)[0], tf.shape(z)[1]
        dp = tf.TensorArray(tf.float32, size=seq_len+1, dynamic_size=False, clear_after_read=False)
        dp = dp.write(0, tf.zeros((batch_size,), dtype=tf.float32))
        
        for j in tf.range(1, seq_len+1):
            current_ll = log_likelihood(z[:, j-1], mu[j-1], sigma[j-1])
            max_prev_ll = tf.maximum(dp.read(j-1), dp.read(j-1))
            current_q = max_prev_ll + current_ll
            dp = dp.write(j, current_q)
        
        # backtracking
        alignment = []
        i = seq_len
        while i > 0:
            alignment.append(i)
            if dp.read(i) == dp.read(i-1):
                i -= 1
            i -= 1
        
        alignment = tf.reverse(alignment, axis=[0])
        return alignment

DEFAULT_MIN_BIN_WIDTH = 1e-3
DEFAULT_MIN_BIN_HEIGHT = 1e-3
DEFAULT_MIN_DERIVATIVE = 1e-3

def piecewise_rational_quadratic_transform(inputs, 
                                           unnormalized_widths,
                                           unnormalized_heights,
                                           unnormalized_derivatives,
                                           inverse=False,
                                           tails=None, 
                                           tail_bound=1.,
                                           min_bin_width=DEFAULT_MIN_BIN_WIDTH,
                                           min_bin_height=DEFAULT_MIN_BIN_HEIGHT,
                                           min_derivative=DEFAULT_MIN_DERIVATIVE):

    if tails is None:
        spline_fn = rational_quadratic_spline
        spline_kwargs = {}
    else:
        spline_fn = unconstrained_rational_quadratic_spline
        spline_kwargs = {
            'tails': tails,
            'tail_bound': tail_bound
        }

    outputs, logabsdet = spline_fn(
            inputs=inputs,
            unnormalized_widths=unnormalized_widths,
            unnormalized_heights=unnormalized_heights,
            unnormalized_derivatives=unnormalized_derivatives,
            inverse=inverse,
            min_bin_width=min_bin_width,
            min_bin_height=min_bin_height,
            min_derivative=min_derivative,
            **spline_kwargs
    )
    return outputs, logabsdet

def searchsorted(bin_locations, inputs, eps=1e-6):
    bin_locations[..., -1] += eps
    return keras.backend.sum(
        inputs[..., None] >= bin_locations,
        dim=-1
    ) - 1

def unconstrained_rational_quadratic_spline(inputs,
                                            unnormalized_widths,
                                            unnormalized_heights,
                                            unnormalized_derivatives,
                                            inverse=False,
                                            tails='linear',
                                            tail_bound=1.,
                                            min_bin_width=DEFAULT_MIN_BIN_WIDTH,
                                            min_bin_height=DEFAULT_MIN_BIN_HEIGHT,
                                            min_derivative=DEFAULT_MIN_DERIVATIVE):
    inside_interval_mask = (inputs >= -tail_bound) & (inputs <= tail_bound)
    outside_interval_mask = ~inside_interval_mask

    outputs = keras.backend.zeros_like(inputs)
    logabsdet = keras.backend.zeros_like(inputs)

    if tails == 'linear':
        unnormalized_derivatives = tf.pad(unnormalized_derivatives, pad=(1, 1))
        constant = np.log(np.exp(1 - min_derivative) - 1)
        unnormalized_derivatives[..., 0] = constant
        unnormalized_derivatives[..., -1] = constant

        outputs[outside_interval_mask] = inputs[outside_interval_mask]
        logabsdet[outside_interval_mask] = 0
    else:
        raise RuntimeError('{} tails are not implemented.'.format(tails))

    outputs[inside_interval_mask], logabsdet[inside_interval_mask] = rational_quadratic_spline(
        inputs=inputs[inside_interval_mask],
        unnormalized_widths=unnormalized_widths[inside_interval_mask, :],
        unnormalized_heights=unnormalized_heights[inside_interval_mask, :],
        unnormalized_derivatives=unnormalized_derivatives[inside_interval_mask, :],
        inverse=inverse,
        left=-tail_bound, right=tail_bound, bottom=-tail_bound, top=tail_bound,
        min_bin_width=min_bin_width,
        min_bin_height=min_bin_height,
        min_derivative=min_derivative
    )

    return outputs, logabsdet

def rational_quadratic_spline(inputs,
                              unnormalized_widths,
                              unnormalized_heights,
                              unnormalized_derivatives,
                              inverse=False,
                              left=0., right=1., bottom=0., top=1.,
                              min_bin_width=DEFAULT_MIN_BIN_WIDTH,
                              min_bin_height=DEFAULT_MIN_BIN_HEIGHT,
                              min_derivative=DEFAULT_MIN_DERIVATIVE):
    if tf.minimum(inputs) < left or tf.maximum(inputs) > right:
        raise ValueError('Input to a transform is not within its domain')

    num_bins = unnormalized_widths.shape[-1]

    if min_bin_width * num_bins > 1.0:
        raise ValueError('Minimal bin width too large for the number of bins')
    if min_bin_height * num_bins > 1.0:
        raise ValueError('Minimal bin height too large for the number of bins')

    widths = keras.activations.softmax(unnormalized_widths, dim=-1)
    widths = min_bin_width + (1 - min_bin_width * num_bins) * widths
    cumwidths = tf.cumsum(widths, dim=-1)
    cumwidths = tf.pad(cumwidths, pad=(1, 0), mode='constant', value=0.0)
    cumwidths = (right - left) * cumwidths + left
    cumwidths[..., 0] = left
    cumwidths[..., -1] = right
    widths = cumwidths[..., 1:] - cumwidths[..., :-1]

    derivatives = min_derivative + keras.activations.softplus(unnormalized_derivatives)

    heights = keras.activations.softmax(unnormalized_heights, dim=-1)
    heights = min_bin_height + (1 - min_bin_height * num_bins) * heights
    cumheights = tf.cumsum(heights, dim=-1)
    cumheights = tf.pad(cumheights, pad=(1, 0), mode='constant', value=0.0)
    cumheights = (top - bottom) * cumheights + bottom
    cumheights[..., 0] = bottom
    cumheights[..., -1] = top
    heights = cumheights[..., 1:] - cumheights[..., :-1]

    if inverse:
        bin_idx = searchsorted(cumheights, inputs)[..., None]
    else:
        bin_idx = searchsorted(cumwidths, inputs)[..., None]

    input_cumwidths = cumwidths.gather(-1, bin_idx)[..., 0]
    input_bin_widths = widths.gather(-1, bin_idx)[..., 0]

    input_cumheights = cumheights.gather(-1, bin_idx)[..., 0]
    delta = heights / widths
    input_delta = delta.gather(-1, bin_idx)[..., 0]

    input_derivatives = derivatives.gather(-1, bin_idx)[..., 0]
    input_derivatives_plus_one = derivatives[..., 1:].gather(-1, bin_idx)[..., 0]

    input_heights = heights.gather(-1, bin_idx)[..., 0]

    if inverse:
        a = (((inputs - input_cumheights) * (input_derivatives
                                             + input_derivatives_plus_one
                                             - 2 * input_delta)
              + input_heights * (input_delta - input_derivatives)))
        b = (input_heights * input_derivatives
             - (inputs - input_cumheights) * (input_derivatives
                                              + input_derivatives_plus_one
                                              - 2 * input_delta))
        c = - input_delta * (inputs - input_cumheights)

        discriminant = b.pow(2) - 4 * a * c
        assert (discriminant >= 0).all()

        root = (2 * c) / (-b - tf.sqrt(discriminant))
        outputs = root * input_bin_widths + input_cumwidths

        theta_one_minus_theta = root * (1 - root)
        denominator = input_delta + ((input_derivatives + input_derivatives_plus_one - 2 * input_delta)
                                     * theta_one_minus_theta)
        derivative_numerator = input_delta.pow(2) * (input_derivatives_plus_one * root.pow(2)
                                                     + 2 * input_delta * theta_one_minus_theta
                                                     + input_derivatives * (1 - root).pow(2))
        logabsdet = tf.math.log(derivative_numerator) - 2 * tf.math.log(denominator)

        return outputs, -logabsdet
    else:
        theta = (inputs - input_cumwidths) / input_bin_widths
        theta_one_minus_theta = theta * (1 - theta)

        numerator = input_heights * (input_delta * theta.pow(2)
                                     + input_derivatives * theta_one_minus_theta)
        denominator = input_delta + ((input_derivatives + input_derivatives_plus_one - 2 * input_delta)
                                     * theta_one_minus_theta)
        outputs = input_cumheights + numerator / denominator

        derivative_numerator = input_delta.pow(2) * (input_derivatives_plus_one * theta.pow(2)
                                                     + 2 * input_delta * theta_one_minus_theta
                                                     + input_derivatives * (1 - theta).pow(2))
        logabsdet = tf.math.log(derivative_numerator) - 2 * tf.math.log(denominator)

        return outputs, logabsdet

