
import tensorflow as tf


class BasicNLSTMCell(tf.nn.rnn_cell.BasicLSTMCell):
    """Basic Nested LSTM recurrent network cell.

    Adapted from `tf.nn.rnn_cell.BasicLSTMCell`

    The implementation is based the paper:
        https://arxiv.org/abs/1801.10308
        JRA. Moniz, D. Krueger.
        "Nested LSTMs"
        ACML, PMLR 77:530-544, 2017
    And is a simplified version of:
        https://github.com/hannw/nlstm/blob/master/rnn_cell.py

    We add forget_bias (default: 1) to the biases of the forget gate in order
    to reduce the scale of forgetting in the beginning of the training.

    It does not allow cell clipping, a projection layer, and does not
    use peep-hole connections: it is the basic baseline.
    """

    def __init__(self, num_units, depth=2, **kwargs):
        """Initialize the basic LSTM cell.
        Args:
          num_units: int, The number of units in the LSTM cell.
          depth:
          forget_bias: float, The bias added to forget gates (see above).
            Must set to `0.0` manually when restoring from CudnnLSTM-trained
            checkpoints.
          activation: Activation function of the inner states. Default: `tanh`.
          reuse: (optional) Python boolean describing whether to reuse
            variables in an existing scope.  If not `True`, and the existing
            scope already has the given variables, an error is raised.
          name: String, the name of the layer. Layers with the same name will
            share weights, but to avoid mistakes we require reuse=True in such
            cases.
        """
        super(BasicNLSTMCell, self).__init__(num_units, **kwargs)
        self._depth = depth

        if not self._state_is_tuple:
            raise NotImplementedError('state_is_tuple=False is not supported')

    @property
    def state_size(self):
        return tuple([self._num_units] * (self._depth + 1))

    @property
    def output_size(self):
        return self._num_units

    def build(self, inputs_shape):
        if inputs_shape[1].value is None:
            raise ValueError(f"Expected inputs.shape[-1] to be known,"
                             f"saw shape: {inputs_shape}")

        input_depth = inputs_shape[1].value
        h_depth = self._num_units
        self._kernels = []
        self._biases = []

        for i in range(self._depth):
            if i == 0:
                self._kernels.append(
                    self.add_variable(
                        f"kernel_{i}",
                        shape=[input_depth + h_depth, 4 * self._num_units]))
            else:
                self._kernels.append(
                    self.add_variable(
                        f"kernel_{i}",
                        shape=[2 * h_depth, 4 * self._num_units]))

            self._biases.append(
                self.add_variable(
                    f"bias_{i}",
                    shape=[4 * self._num_units],
                    initializer=tf.zeros_initializer(dtype=self.dtype)))

        self.built = True

    def _recurrence(self, inputs, hidden_state, cell_states, depth):
        """use recurrence to traverse the nested structure
        Args:
          inputs: A 2D `Tensor` of [batch_size x input_size] shape.
          hidden_state: A 2D `Tensor` of [batch_size x num_units] shape.
          cell_states: A `list` of 2D `Tensor` of [batch_size x num_units]
            shape.
          depth: `int`
            the current depth in the nested structure, begins at 0.
        Returns:
          new_h: A 2D `Tensor` of [batch_size x num_units] shape.
            the latest hidden state for current step.
          new_cs: A `list` of 2D `Tensor` of [batch_size x num_units] shape.
            The accumulated cell states for current step.
        """
        one = tf.constant(1, dtype=tf.int32)
        # Parameters of gates are concatenated into one multiply for efficiency
        c = cell_states[depth]
        h = hidden_state

        gate_inputs = tf.matmul(
            tf.concat([inputs, h], 1), self._kernels[depth])
        gate_inputs = tf.nn.bias_add(gate_inputs, self._biases[depth])

        # i = input_gate, j = new_input, f = forget_gate, o = output_gate
        i, j, f, o = tf.split(
            value=gate_inputs, num_or_size_splits=4, axis=one)

        forget_bias_tensor = tf.constant(self._forget_bias, dtype=f.dtype)
        # Note that using `tf.add` and `tf.multiply` instead of `+` and `*`
        # gives a performance improvement. So using those at the cost of
        # readability.
        inner_hidden = tf.multiply(
            c,
            tf.sigmoid(tf.add(f, forget_bias_tensor))
        )
        if depth == 0:
            inner_input = tf.multiply(tf.sigmoid(i), j)
        else:
            inner_input = tf.multiply(tf.sigmoid(i), self._activation(j))

        # For the final layer, just do what a normal LSTM cell would do
        if depth == (self._depth - 1):
            new_c = tf.add(inner_hidden, inner_input)
            new_cs = [new_c]
        # For the other layers, calculate the cell state using a nested LSTM
        else:
            new_c, new_cs = self._recurrence(
                inputs=inner_input,
                hidden_state=inner_hidden,
                cell_states=cell_states,
                depth=depth + 1)

        new_h = tf.multiply(self._activation(new_c), tf.sigmoid(o))
        new_cs = [new_h] + new_cs

        return new_h, new_cs

    def call(self, inputs, states):
        """Nested long short-term memory cell (NLSTM).
        Args:
          inputs: `2-D` tensor with shape `[batch_size, input_size]`.
          states: An `tuple` of state tensors, each shaped
            `[batch_size, self.state_size]`.
        Returns:
          A pair containing the new hidden state, and the new state (either a
            `tuple` or a concatenated state, depending on `state_is_tuple`).
        """

        hidden_state = states[0]
        cell_states = states[1:]

        outputs, next_state = self._recurrence(
            inputs=inputs,
            hidden_state=hidden_state,
            cell_states=cell_states,
            depth=0)

        return outputs, tuple(next_state)
