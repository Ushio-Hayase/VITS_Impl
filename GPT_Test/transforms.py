import tensorflow as tf
import numpy as np

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
        raise RuntimeError('{} tails are not implemented.'.format(tails))

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
    bin_locations = bin_locations + eps
    return tf.reduce_sum(
        tf.cast(inputs[..., tf.newaxis] >= bin_locations, dtype=tf.int32),
        axis=-1
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
    inside_interval_mask = tf.logical_and(inputs >= -tail_bound, inputs <= tail_bound)
    outside_interval_mask = ~inside_interval_mask

    outputs = tf.zeros_like(inputs)
    logabsdet = tf.zeros_like(inputs)

    if tails == 'linear':
        unnormalized_derivatives = tf.pad(unnormalized_derivatives, paddings=[[0, 0], [1, 1]])
        constant = np.log(np.exp(1 - min_derivative) - 1)
        unnormalized_derivatives[..., 0] = constant
        unnormalized_derivatives[..., -1] = constant

        outputs = tf.where(outside_interval_mask, inputs, outputs)
    else:
        raise RuntimeError('{} tails are not implemented.'.format(tails))

    outputs_inside, logabsdet_inside = rational_quadratic_spline(
        inputs=tf.boolean_mask(inputs, inside_interval_mask),
        unnormalized_widths=tf.boolean_mask(unnormalized_widths, inside_interval_mask),
        unnormalized_heights=tf.boolean_mask(unnormalized_heights, inside_interval_mask),
        unnormalized_derivatives=tf.boolean_mask(unnormalized_derivatives, inside_interval_mask),
        inverse=inverse,
        min_bin_width=min_bin_width,
        min_bin_height=min_bin_height,
        min_derivative=min_derivative
    )

    outputs = tf.tensor_scatter_nd_update(outputs, tf.where(inside_interval_mask), outputs_inside)
    logabsdet = tf.tensor_scatter_nd_update(logabsdet, tf.where(inside_interval_mask), logabsdet_inside)

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
    if tf.reduce_min(inputs) < left or tf.reduce_max(inputs) > right:
        raise ValueError('Input to a transform is not within its domain')

    num_bins = unnormalized_widths.shape[-1]

    if min_bin_width * num_bins > 1.0:
        raise ValueError('Minimal bin width too large for the number of bins')
    if min_bin_height * num_bins > 1.0:
        raise ValueError('Minimal bin height too large for the number of bins')

    widths = tf.nn.softmax(unnormalized_widths, axis=-1)
    widths = min_bin_width + (1 - min_bin_width * num_bins) * widths
    cumwidths = tf.cumsum(widths, axis=-1)
    cumwidths = tf.pad(cumwidths, paddings=[[0, 0], [1, 0]])
    cumwidths = (right - left) * cumwidths + left
    cumwidths = tf.tensor_scatter_nd_update(cumwidths, [[0, 0]], [left])
    cumwidths = tf.tensor_scatter_nd_update(cumwidths, [[0, num_bins]], [right])
    widths = cumwidths[..., 1:] - cumwidths[..., :-1]

    derivatives = min_derivative + tf.nn.softplus(unnormalized_derivatives)

    heights = tf.nn.softmax(unnormalized_heights, axis=-1)
    heights = min_bin_height + (1 - min_bin_height * num_bins) * heights
    cumheights = tf.cumsum(heights, axis=-1)
    cumheights = tf.pad(cumheights, paddings=[[0, 0], [1, 0]])
    cumheights = (top - bottom) * cumheights + bottom
    cumheights = tf.tensor_scatter_nd_update(cumheights, [[0, 0]], [bottom])
    cumheights = tf.tensor_scatter_nd_update(cumheights, [[0, num_bins]], [top])
    heights = cumheights[..., 1:] - cumheights[..., :-1]

    if inverse:
        bin_idx = searchsorted(cumheights, inputs)[..., tf.newaxis]
    else:
        bin_idx = searchsorted(cumwidths, inputs)[..., tf.newaxis]

    input_cumwidths = tf.gather_nd(cumwidths, bin_idx)
    input_bin_widths = tf.gather_nd(widths, bin_idx)

    input_cumheights = tf.gather_nd(cumheights, bin_idx)
    delta = heights / widths
    input_delta = tf.gather_nd(delta, bin_idx)

    input_derivatives = tf.gather_nd(derivatives, bin_idx)
    input_derivatives_plus_one = tf.gather_nd(derivatives[..., 1:], bin_idx)

    input_heights = tf.gather_nd(heights, bin_idx)

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

        discriminant = tf.pow(b, 2) - 4 * a * c
        assert tf.reduce_all