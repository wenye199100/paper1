import tensorflow as tf

def normailize(inputs,
               epsilon=1e-8,
               scope="ln",
               reuse=None):
    """

    :param inputs:
    :param epsilon:
    :param scoper:
    :param reuser:
    :return:
    """
    with tf.variable_scope(scope, reuse=reuse):
        inputs_shape = inputs.get_shape()
        param_shape = inputs_shape[-1:]

        mean, variance = tf.nn.moments(inputs, [-1], keep_dims=True)
        beta = tf.Variable(tf.zeros(params_shape))
        gamma = tf.Variable(tf.ones(params_shape))
        normalized = (inputs - mean) / ((variance + epsilon) ** (.5))
        outputs = gamma * normalized + beta


def embedding(inputs,
              vocab_size,
              num_units,
              zero_pad=False,
              scale=True,
              scope='embedding',
              reuse=None):
    '''
    Embeds a given tensor.

        Args:
          inputs: A `Tensor` with type `int32` or `int64` containing the ids
             to be looked up in `lookup table`.
          vocab_size: An int. Vocabulary size.
          num_units: An int. Number of embedding hidden units.
          zero_pad: A boolean. If True, all the values of the fist row (id 0)
            should be constant zeros.
          scale: A boolean. If True. the outputs is multiplied by sqrt num_units.
          scope: Optional scope for `variable_scope`.
          reuse: Boolean, whether to reuse the weights of a previous layer
            by the same name.

        Returns:
          A `Tensor` with one more rank than inputs's. The last dimensionality
            should be `num_units`.
    '''
    with tf.variable_scope(scope, reuse=reuse):
        lookup_table = tf.get_variable('lookup_table',
                                       shape=[vocab_size, num_units],
                                       initializer=tf.contrib.layers.xavier_initializer())
        if zero_pad:
            lookup_table = tf.concat((tf.zeros(shape=[1, num_units]),
                                      lookup_table[1:, :]), 0)
        outputs = tf.nn.embedding_lookup(lookup_table, inputs)

        if scale:
            outputs = outputs * (num_units ** 0.5)
    return outputs


def relative_postions(length, max_relative_position):
    """

    :param length:
    :param max_relative_position:
    :return:
    """
    range_vec = tf.range(length)
    range_mat = tf.reshape(tf.tile(range_vec, [length]), [length, length])
    distance_mat = range_mat - tf.transpose(range_mat)
    distance_mat_clipped = tf.clip_by_value(distance_mat, -max_relative_position, max_relative_position)
    final_mat = distance_mat_clipped + max_relative_position
    return final_mat

def relative_position_embeddings(input, max_relative_position, name):
    """

    :param input:
    :param max_relative_position:
    :param name:
    :return:
    """
    with tf.variable_scope(name):
        length = input.get_shape().as_list[1]
        depth = input.get_shape().as_list[-1]
        relative_postions_matrix = relative_postions(length, max_relative_position)
        vocab_size = max_relative_position * 2 + 1
        embeddings_table = tf.get_variable("embeddings", [vocab_size, depth])
        embedding_relative = tf.gather(embeddings_table, relative_postions_matrix)
        return embedding_relative

def relative_inner(x, y, z, transpose):
    """

    :param x: Tensor with shape [h*N, T_q, C/h]
    :param y: Tensor with shape [h*N, T_k, C/h]
    :param z: Tensor with shape [max_len, max_len, C/h]
    :param transpose:
    :return:
    """
    xy_matmul = tf.matmul(x, y, transpose_b = transpose)
    x_t = tf.transpose(x, [1, 0, 2])
    x_tz_matmul = tf.matmul(x_t, z, transpose_b = transpose)
    x_tz_matmul_t = tf.transpose(x_tz_matmul, [1,0,2])
    return xy_matmul + x_tz_matmul_t




def multihead_attention(queries,
                        keys,
                        num_units=None,
                        num_heads=8,
                        dropout_rate=0,
                        is_training=True,
                        causality=False,
                        scope="multihead_attention",
                        relative_mode=False,
                        max_relative_position=2,
                        reuse=None):
    """

    :param queries:
    :param keys:
    :param num_units:
    :param num_heads:
    :param dropout_rate:
    :param is_training:
    :param causality:
    :param scpoe:
    :param relative_mode:
    :param max_relative_position:
    :param reuse:
    :return:
    """

    with tf.variable_scope(scope, reuse=reuse):
        if num_units is None:
            num_units = queries.get_shape().as_list[-1]

        Q = tf.layers.dense(queries, num_units, activation=tf.nn.relu)
        K = tf.layers.dense(keys, num_units, activation=tf.nn.relu)
        V = tf.layers.dense(keys, num_units, activation=tf.nn.relu)

        Q_Muti = tf.concat(tf.split(Q, num_heads, axis=2), axis=0)
        K_Muti = tf.concat(tf.split(K, num_heads, axis=2), axis=0)
        V_Muti = tf.concat(tf.split(V, num_heads, axis=2), axis=0)

        if relative_mode:
            K_Relative = relative_position_embeddings(keys, max_relative_position, "relative_position_keys")
            weights = relative_inner(Q_Muti, K_Muti, K_Relative, True)
        else:
            weights = tf.matmul(Q_Muti, tf.transpose(K_Muti, [0, 2, 1]))

        weights = weights / (K_Muti.get_shape().as_list()[-1] ** 0.5)

        weights = tf.nn.softmax(weights)

        if relative_mode:
            V_Relative = relative_position_embeddings(keys, max_relative_position, "relative_position_values")
            outputs = relative_inner(weights, V_Muti, V_Relative, False)
        else:
            outputs = tf.matmul(weights, V_Muti)

        outputs = tf.concat(tf.split(outputs, num_heads, axis=0), axis=2 )
        outputs += queries
        outputs = normailize(outputs)

    return outputs

def feedforward(inputs,
                num_units=[2048,512],
                scope="multihead_attention",
                reuse=None):
    """

    :param inputs:
    :param num_uits:
    :param soope:
    :param reuse:
    :return:
    """
    with tf.variable_scope(scope, reuse=reuse):
        # Inner layer
        params = {"inputs": inputs, "filters": num_units[0], "kernel_size": 1,
                  "activation": tf.nn.relu, "use_bias": True}
        outputs = tf.layers.conv1d(**params)

        # Readout layer
        params = {"inputs": outputs, "filters": num_units[1], "kernel_size": 1,
                  "activation": None, "use_bias": True}
        outputs = tf.layers.conv1d(**params)

        # Residual connection
        outputs += inputs

        # Normalize
        outputs = normailize(outputs)

    return outputs




