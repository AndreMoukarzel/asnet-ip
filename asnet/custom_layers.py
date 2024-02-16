"""Contains custom layers used in building an ASNet """
from typing import List

import tensorflow as tf
from keras.layers import Layer, Lambda, Dense, Flatten
from keras.regularizers import L2
from keras import backend as K


DEFAULT_HIDDEN_DIMENSION: int = 16


def build_connections_layer(relevant_indexes: List[int]) -> Lambda:
        """Returns an intermediary Lambda layer that receives all inputs from the
        previous layer and outputs only the values in specified relevant_indexes.

        Essentially serves as a mask to filter only the desired outputs from a layer.

        Parameters
        ----------
        relevant_indexes: List[int]
            List of the positions, or indexes, of outputs to be filtered and 
            passed through the created Lambda layer.
        """
        return Lambda(lambda x: tf.gather(x, relevant_indexes, axis=1), trainable=False)


class Output(Layer):
    """
    Output Layer of an ASNet.

    This layer outputs the probabilities of each of the VALID actions being
    chosen given the received proposition values received by the previous
    Proposition layer, composed of the concatenation of PropositionModules.
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
        self.lambda_mask: Lambda = build_connections_layer(applicable_indexes)

    def call(self, inputs):
        if not isinstance(inputs, list) or len(inputs) != 2:
            raise ValueError('The output layer should be called on a list of inputs, being the first element the previous layer and the second element the input layer.')
        
        prev_layer = inputs[0]
        input_layer = inputs[1] # Must be the input layer from the whole ASNet!
        output = tf.multiply(prev_layer, self.lambda_mask(input_layer)) 
        return tf.nn.softmax(output)


class ActionModule(Layer):
    """
    Action Module representing a single action from an Action Layer in an ASNet.

    Methods
    -------
    build_weights()
        Builds the layers weights, so its kernel and biases can be transfered
        to other Action Modules of the same format.
    get_trainable_weights()
        Returns the kernel and bias of the ActionModule
    set_trainable_weights(kernel, bias)
        Overwrides the kernel and bias of the ActionModule
    """
    def __init__(self, related_prep_indexes: List[int], hidden_dimension: int=DEFAULT_HIDDEN_DIMENSION, **kwargs):
        super(ActionModule, self).__init__(**kwargs)
        self.filter_shape = (None, len(related_prep_indexes))
        self.filter = build_connections_layer(related_prep_indexes)
        self.neuron = Dense(hidden_dimension, kernel_regularizer=L2(1e-4), input_shape=self.filter_shape)

    def call(self, input):
        x = self.filter(input)
        return self.neuron(x)

    def build_weights(self):
        self.neuron.build(self.filter_shape)

    def get_trainable_weights(self) -> tuple:
        return self.neuron.kernel, self.neuron.bias
    
    def set_trainable_weights(self, kernel, bias) -> tuple:
        """
        Overwrides the kernel and bias of the ActionModule.

        Also ensures that this ActionModule's weights and biases are shared with
        the original kernel and bias, forcing any training that affects this
        module to also reflect in the values of the original.
        """
        self.neuron.kernel = kernel
        self.neuron.bias = bias
        self.neuron._trainable_weights = []
        self.neuron._trainable_weights = [kernel, bias]


class PropositionModule(Layer):
    """
    Proposition Module representing a single proposition from an Proposition
    Layer in an ASNet.

    Methods
    -------
    build_weights()
        Builds the layers weights, so its kernel and biases can be transfered
        to other Action Modules of the same format.
    get_trainable_weights()
        Returns the kernel and bias of the PropositionModule
    set_trainable_weights(kernel, bias)
        Overwrides the kernel and bias of the PropositionModule
    """
    def __init__(self, related_connections: List[List[int]], hidden_dimension: int=DEFAULT_HIDDEN_DIMENSION, **kwargs):
        super(PropositionModule, self).__init__(**kwargs)
        self.pooling_filters: List[Lambda] = []
        self.hidden_dimension: int = hidden_dimension
        for connections in related_connections:
            # When multiple predicates are related, we pool them into a single value with max pooling
            self.pooling_filters.append(build_connections_layer(connections))
        self.flatten = Flatten()
        self.concat_shape = (None, self.hidden_dimension * len(related_connections))
        self.neuron = Dense(hidden_dimension, kernel_regularizer=L2(1e-4))

    def call(self, input):
        pooled_inputs: list = []

        for filter_layer in self.pooling_filters:
            # Pools maximum value (element-wise) of propositions with related predicates into a single output
            pool = filter_layer(input)
            input_size: int = int(pool.shape[1]/self.hidden_dimension) # Size of input from each origin
            # Reshapes input into vectors representing each input origin
            reshape = tf.reshape(pool, [-1, input_size, self.hidden_dimension])
            # Max Pools values from all origins element-wise
            pool = tf.nn.max_pool(reshape, ksize=input_size, strides=input_size, padding="SAME")
            # Flattens result into single array
            pool = self.flatten(pool)

            pooled_inputs.append(pool)
        
        x = tf.concat(pooled_inputs, axis=-1)

        return self.neuron(x)

    def build_weights(self):
        self.neuron.build(self.concat_shape)

    def get_trainable_weights(self) -> tuple:
        return self.neuron.kernel, self.neuron.bias
    
    def set_trainable_weights(self, kernel, bias) -> tuple:
        """
        Overwrides the kernel and bias of the PropositionModule.

        Also ensures that this PropositionModule's weights and biases are shared
        with the original kernel and bias, forcing any training that affects this
        module to also reflect in the values of the original.
        """
        self.neuron.kernel = kernel
        self.neuron.bias = bias
        self.neuron._trainable_weights = [kernel, bias]


class AltPropositionModule(PropositionModule):
    """
    Alternative Proposition Module representing a single proposition from an
    Proposition Layer in an Alt ASNet. Differently than the "traditional"
    ASNet's Module, this one pools all values from each of its 'related' inputs.

    Methods
    -------
    build_weights()
        Builds the layers weights, so its kernel and biases can be transfered
        to other Action Modules of the same format.
    get_trainable_weights()
        Returns the kernel and bias of the PropositionModule
    set_trainable_weights(kernel, bias)
        Overwrides the kernel and bias of the PropositionModule
    """
    def __init__(self, related_connections: List[List[int]], unrelated_connections: List[int], hidden_dimension: int=DEFAULT_HIDDEN_DIMENSION, **kwargs):
        super(PropositionModule, self).__init__(**kwargs)
        self.pooling_filters: List[Lambda] = []
        for connections in related_connections:
            # When multiple predicates are related, we pool them into a single value with max pooling
            self.pooling_filters.append(build_connections_layer(connections))
        self.solo_filter = None
        if unrelated_connections:
            self.solo_filter = build_connections_layer(unrelated_connections) # We also filter out all relevant predicates with no relations to others
        self.concat_shape = (None, len(related_connections) + len(unrelated_connections))
        self.neuron = Dense(hidden_dimension, kernel_regularizer=L2(1e-4))


    def call(self, input):
        pooled_inputs: list = []

        for filter_layer in self.pooling_filters:
            # Pools maximum value of propositions with related predicates into a single output
            pool = filter_layer(input)
            pool = K.max(pool, axis=-1)
            pool = tf.convert_to_tensor(pool)
            pool = tf.reshape(pool, (-1, 1))

            pooled_inputs.append(pool)
        
        if self.solo_filter:
            pooled_inputs.append(self.solo_filter(input))
        x = tf.concat(pooled_inputs, axis=-1)
        return self.neuron(x)





class ExperimentalPropositionModule(PropositionModule):
    """
    Proposition Module representing a single proposition from an Proposition
    Layer in an ASNet.

    Methods
    -------
    build_weights()
        Builds the layers weights, so its kernel and biases can be transfered
        to other Action Modules of the same format.
    get_trainable_weights()
        Returns the kernel and bias of the PropositionModule
    set_trainable_weights(kernel, bias)
        Overwrides the kernel and bias of the PropositionModule
    """
    def __init__(self, related_connections: List[List[int]], unrelated_connections: List[int], hidden_dimension: int=DEFAULT_HIDDEN_DIMENSION, **kwargs):
        super(PropositionModule, self).__init__(**kwargs)
        self.pooling_filters: List[Lambda] = []
        self.hidden_dimension: int = hidden_dimension
        for connections in related_connections:
            # When multiple predicates are related, we pool them into a single value with max pooling
            self.pooling_filters.append(build_connections_layer(connections))
        self.solo_filter = None
        if unrelated_connections:
            self.solo_filter = build_connections_layer(unrelated_connections) # We also filter out all relevant predicates with no relations to others
        self.flatten = Flatten()
        self.concat_shape = (None, self.hidden_dimension * len(related_connections) + len(unrelated_connections))
        self.neuron = Dense(hidden_dimension, kernel_regularizer=L2(1e-4))

    def call(self, input):
        pooled_inputs: list = []

        for filter_layer in self.pooling_filters:
            # Pools maximum value (element-wise) of propositions with related predicates into a single output
            pool = filter_layer(input)
            input_size: int = int(pool.shape[1]/self.hidden_dimension) # Size of input from each origin
            # Reshapes input into vectors representing each input origin
            reshape = tf.reshape(pool, [1, input_size, self.hidden_dimension])
            # Max Pools values from all origins element-wise
            pool = tf.nn.max_pool(reshape, ksize=input_size, strides=input_size, padding="SAME")
            # Flattens result into single array
            pool = self.flatten(pool)

            pooled_inputs.append(pool)
        
        if self.solo_filter:
            solo = self.solo_filter(input)
            #pooled_inputs.append(tf.reshape(solo, [solo.shape[1]]))
            pooled_inputs.append(self.flatten(solo))
        x = tf.concat(pooled_inputs, axis=-1)

        return self.neuron(x)