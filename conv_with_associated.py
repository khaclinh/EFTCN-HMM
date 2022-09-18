import tensorflow.keras as keras

layers = keras.layers


def Conv1d_with_associated(x, out_filters, kernel_size, stride=1, padding='valid',
                           droprate=0., use_spatial_drop=True,
                           dilation=1, activation='relu', use_batchnorm=True):
    """
    Conv1d layer with activation, batchnorm, dropout.
    :param x: input tensor
    :param padding: can be string like 'same', 'valid', integer or list-like
    :param use_batchnorm: whether to use Batchnorm after Conv
    :return: output tensor
    """

    # Conv layer
    if type(padding) is str:
        x = layers.Conv1D(out_filters, kernel_size=kernel_size, strides=stride, padding=padding,
                          activation=activation, dilation_rate=dilation)(x)

    elif type(padding) in [int, tuple, list]:
        x = layers.ZeroPadding1D(padding)(x)
        x = layers.Conv1D(out_filters, kernel_size=kernel_size, strides=stride,
                          activation=activation, dilation_rate=dilation)(x)

    else:
        raise ValueError('padding must be int, list-like, or string')

    # Batchnorm layer
    if use_batchnorm:
        x = layers.BatchNormalization(axis=-1)(x)

    # Dropout layer
    if use_spatial_drop:
        x = layers.SpatialDropout1D(droprate)(x)
    else:
        x = layers.Dropout(droprate)(x)

    return x
