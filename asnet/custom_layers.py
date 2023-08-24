from typing import List

import tensorflow as tf
from keras.layers import Layer, Lambda
from keras import backend as K


def builds_connections_layer(relevant_indexes: List[int]) -> Lambda:
        """Returns an intermediary Lambda layer that receives all inputs from the
        previous layer and outputs only the values in specified relevant_indexes.

        Essentially serves as a mask to filter only the desired outputs from a layer.
        """
        return Lambda(lambda x: tf.gather(x, relevant_indexes, axis=1), trainable=False)


class Output(Layer):
    """Output layer for an ASNet.
    """
    def __init__(self, input_action_sizes: int, **kwargs):
        super(Output, self).__init__(**kwargs)
        #self.input_action_sizes: List[int] = list(input_action_sizes.values())#K.cast(input_action_sizes, "int16")
        action_sizes: List[int] = list(input_action_sizes.values())
        sizes_sum: int = 0
        applicable_indexes: List[int] = [] # Indexes from the input layer that specify if an action is applicable or not
        for act_size in action_sizes:
             sizes_sum += act_size
             applicable_indexes.append(sizes_sum - 1)
        self.lambda_mask = builds_connections_layer(applicable_indexes)

    def call(self, inputs):
        if not isinstance(inputs, list):
            raise ValueError('The output layer should be called on a list of inputs, being the first element the previous layer and the second element the input layer.')
        
        prev_layer = inputs[0]
        input_layer = inputs[1] # Must be the input layer from the whole ASNet!

        # Creates a mask with values 1.0 for applicable actions and 0.0 for non-applicable actions
        """
        app_actions_mask = [0.0] * len(self.input_action_sizes)
        input_layer_pos: int = 0
        for mask_index, act_size in enumerate(self.input_action_sizes):
            input_layer_pos += act_size
            app_actions_mask[mask_index] = input_layer[input_layer_pos]
        """

        #changed = Multiply()([prev_layer, app_actions_mask])
        #return K.sigmoid(self.beta * inputs) * inputs

        #output = tf.matmul(prev_layer, app_actions_mask)
        output = tf.multiply(prev_layer, self.lambda_mask(input_layer))
        return tf.nn.softmax(output)

    #def get_config(self):
    #    config = {'input_action_size': int(self.input_action_size)}
    #    base_config = super(Output, self).get_config()
    #    return dict(list(base_config.items()) + list(config.items()))

    #def compute_output_shape(self, input_shape):
    #    return input_shape