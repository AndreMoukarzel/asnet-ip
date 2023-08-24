from typing import List

import tensorflow as tf
from keras.layers import Layer, Lambda, Dense
from keras import backend as K


def build_connections_layer(relevant_indexes: List[int]) -> Lambda:
        """Returns an intermediary Lambda layer that receives all inputs from the
        previous layer and outputs only the values in specified relevant_indexes.

        Essentially serves as a mask to filter only the desired outputs from a layer.
        """
        return Lambda(lambda x: tf.gather(x, relevant_indexes, axis=1), trainable=False)


class Output(Layer):
    """Output layer for an ASNet.
    """
    def __init__(self, input_action_sizes: List[int], **kwargs):
        super(Output, self).__init__(**kwargs)
        # Creates a 'mask' with values 1.0 for applicable actions and 0.0 for non-applicable actions
        action_sizes: List[int] = list(input_action_sizes.values())
        sizes_sum: int = 0
        applicable_indexes: List[int] = [] # Indexes from the input layer that specify if an action is applicable or not
        for act_size in action_sizes:
             sizes_sum += act_size
             applicable_indexes.append(sizes_sum - 1)
        self.lambda_mask = build_connections_layer(applicable_indexes)

    def call(self, inputs):
        if not isinstance(inputs, list) or len(inputs) != 2:
            raise ValueError('The output layer should be called on a list of inputs, being the first element the previous layer and the second element the input layer.')
        
        prev_layer = inputs[0]
        input_layer = inputs[1] # Must be the input layer from the whole ASNet!

        output = tf.multiply(prev_layer, self.lambda_mask(input_layer))
        return tf.nn.softmax(output)


class ActionModule(Layer):
    """Action Module representing a single action from an action layer from an
    ASNet.
    """
    def __init__(self, related_prep_indexes: List[int], **kwargs):
        super(ActionModule, self).__init__(**kwargs)
        self.filter_shape = (None, len(related_prep_indexes))
        self.filter = build_connections_layer(related_prep_indexes)
        self.neuron = Dense(1)

    def call(self, input):
        x = self.filter(input)
        return self.neuron(x)

    def build_weights(self):
        self.neuron.build(self.filter_shape)

    def get_traineable_weights(self) -> tuple:
        return self.neuron.kernel, self.neuron.bias
    
    def set_traineable_weights(self, kernel, bias) -> tuple:
        self.neuron.kernel = kernel
        self.neuron.bias = bias
        self.neuron._trainable_weights = []
        self.neuron._trainable_weights.append(kernel)
        self.neuron._trainable_weights.append(bias)
